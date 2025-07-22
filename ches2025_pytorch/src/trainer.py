import torch
import time
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau # Import LR scheduler
import os # For saving best model

# Import your model classes
from src.net import MLP, CNN, weight_init, CNN_LSTM_SCA 


def trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device, writer=None, trial_id=None):
    """
    Trains a deep learning model with specified configuration.
    Includes Automatic Mixed Precision (AMP), Learning Rate Scheduling, and Early Stopping.

    Args:
        config (dict): Hyperparameter configuration for the model.
        num_epochs (int): Maximum number of training epochs.
        num_sample_pts (int): Length of input traces.
        dataloaders (dict): Dictionary of 'train' and 'val' PyTorch DataLoaders.
        dataset_sizes (dict): Dictionary of 'train' and 'val' dataset sizes.
        model_type (str): Type of model ("mlp", "cnn", "cnn_lstm_sca").
        classes (int): Number of output classes (e.g., 9 for HW, 256 for ID).
        device (torch.device): Device to train on (CPU or CUDA).
        writer (torch.utils.tensorboard.SummaryWriter, optional): TensorBoard writer object.
        trial_id (int, optional): Unique ID for the current trial, used for best model saving.
    Returns:
        torch.nn.Module: The trained model (with weights from the best validation epoch).
    """

    # --- 1. Model Initialization ---
    if model_type == "mlp":
        model = MLP(config, num_sample_pts, classes).to(device)
    elif model_type == "cnn":
        model = CNN(config, num_sample_pts, classes).to(device)
    elif model_type == "cnn_lstm_sca": 
        model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Apply global weight_init if the model doesn't have its own (like CNN_LSTM_SCA does)
    if model_type not in ["cnn_lstm_sca"]: 
        weight_init(model, config['kernel_initializer']) 
    
    # --- 2. Optimizer Initialization ---
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0) # Get weight_decay from config

    # Dynamically select optimizer based on config
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "RMSprop":
        # RMSprop typically doesn't have a direct `weight_decay` parameter in its constructor.
        # L2 regularization for RMSprop is often handled differently (e.g., via `alpha` or manually in loss).
        # For simplicity, we ignore `weight_decay` for RMSprop here if present in config.
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr) 
        if weight_decay > 0.0:
            print(f"Warning: weight_decay ({weight_decay}) is ignored for RMSprop as it's not a direct constructor argument.")
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # --- 3. Loss Function & Learning Rate Scheduler ---
    criterion = nn.CrossEntropyLoss()
    # Monitors 'val_loss', reduces LR by `factor` if no improvement for `patience` epochs
    # Removed `verbose=True` for compatibility with newer PyTorch versions.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # --- 4. Early Stopping Parameters ---
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20 # Number of epochs to wait for improvement on val_loss before stopping
    
    # Define a temporary path to save the best model weights during training
    temp_best_model_path = f"temp_best_model_trial_{trial_id}.pth" # Use trial_id for uniqueness

    # --- 5. Automatic Mixed Precision (AMP) Scaler ---
    scaler = torch.amp.GradScaler('cuda') # Updated API for mixed precision training

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data batches
            for batch_data in dataloaders[phase]:
                if len(batch_data) == 3:  # (trace, label, plaintext)
                    inputs, labels, _ = batch_data
                else:  # (trace, label) - for backward compatibility
                    inputs, labels = batch_data
                    
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                # Enable gradient calculation only for training phase
                with torch.set_grad_enabled(phase == 'train'):
                    # --- AMP: Autocast for mixed precision forward pass ---
                    with torch.amp.autocast('cuda'):
                        outputs = model(inputs)
                        # Extract outputs if the model returns a dictionary (like CNN_LSTM_SCA)
                        if isinstance(outputs, dict) and 'output' in outputs:
                            outputs = outputs['output']

                        # Get predicted class indices
                        _, preds = torch.max(outputs, dim=1)
                        # Calculate loss
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        # --- AMP: Scale the loss and call backward on scaled loss ---
                        scaler.scale(loss).backward()
                        # --- AMP: Unscale gradients and call optimizer.step() ---
                        scaler.step(optimizer)
                        scaler.update() # Update the scaler for the next iteration

                # Accumulate statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Calculate epoch-level statistics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Epoch Loss: {epoch_loss:.4f} Epoch Acc: {epoch_acc:.4f}')
            
            # --- TensorBoard Logging ---
            if writer: 
                writer.add_scalar(f'Loss/{phase}', epoch_loss, global_step=epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, global_step=epoch)

            # --- Learning Rate Scheduler Step (on validation loss) ---
            if phase == 'val':
                scheduler.step(epoch_loss)

                # --- Early Stopping Logic ---
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0 # Reset patience if validation loss improves
                    # Save the best model state found so far
                    torch.save(model.state_dict(), temp_best_model_path) 
                    print(f"Validation loss improved. Saving best model to {temp_best_model_path}")
                else:
                    patience_counter += 1
                    print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")

        # Check early stopping condition after the validation phase of each epoch
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Validation loss not improved for {early_stopping_patience} epochs.")
            break # Exit the training loop

    print("Finished Training Model")
    
    # --- Load the best model weights before returning ---
    # This ensures that the model returned is the one that performed best on the validation set.
    if os.path.exists(temp_best_model_path):
        model.load_state_dict(torch.load(temp_best_model_path))
        print(f"Loaded best model from {temp_best_model_path}")
        # Clean up the temporary best model file
        os.remove(temp_best_model_path)
    else:
        print("Warning: Best model file not found. Returning model from last epoch.")
        
    return model