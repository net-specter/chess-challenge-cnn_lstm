import torch
import time
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau 
import os 

from src.net import MLP, CNN, weight_init, CNN_LSTM_SCA 


def trainer(config, num_epochs, num_sample_pts, dataloaders, dataset_sizes, model_type, classes, device, writer=None, trial_id=None):
    """
    Trains a deep learning model with specified configuration.
    Includes Automatic Mixed Precision (AMP), Learning Rate Scheduling, Early Stopping, and Gradient Accumulation.

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
    
    if model_type not in ["cnn_lstm_sca"]: 
        weight_init(model, config['kernel_initializer']) 
    
    # --- 2. Optimizer Initialization ---
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 0.0) 

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "RAdam":
        optimizer = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "NAdam":
        optimizer = torch.optim.NAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr) 
        if weight_decay > 0.0:
            print(f"Warning: weight_decay ({weight_decay}) is ignored for RMSprop as it's not a direct constructor argument.")
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    # --- 3. Loss Function & Learning Rate Scheduler ---
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # --- 4. Early Stopping Parameters ---
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20 
    
    temp_best_model_path = f"temp_best_model_trial_{trial_id}.pth" 

    # --- 5. Automatic Mixed Precision (AMP) Scaler ---
    # FIX: Use torch.amp.grad_scaler.GradScaler() and explicitly enable
    scaler = torch.amp.grad_scaler.GradScaler(enabled=True) 

    # --- 6. Gradient Accumulation Parameter ---
    gradient_accumulation_steps = 1 # Consider increasing if OOM occurs even with smaller batch sizes

    start_time = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            running_loss = 0.0
            running_corrects = 0

            for i, batch_data in enumerate(dataloaders[phase]): 
                inputs, labels, _ = batch_data 
                    
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # FIX: Use torch.amp.autocast with device_type and enabled=True
                    with torch.amp.autocast(device_type=device.type, enabled=True): 
                        outputs = model(inputs)
                        if isinstance(outputs, dict) and 'output' in outputs:
                            outputs = outputs['output']

                        _, preds = torch.max(outputs, dim=1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss = loss / gradient_accumulation_steps 
                        scaler.scale(loss).backward() 
                        scaler.step(optimizer) 
                        scaler.update() 

                running_loss += loss.item() * inputs.size(0) 
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and (i + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

            if phase == 'train' and (i + 1) % gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase.capitalize()} Epoch Loss: {epoch_loss:.4f} Epoch Acc: {epoch_acc:.4f}')
            
            if writer: 
                writer.add_scalar(f'Loss/{phase}', epoch_loss, global_step=epoch)
                writer.add_scalar(f'Accuracy/{phase}', epoch_acc, global_step=epoch)

            if phase == 'val':
                scheduler.step(epoch_loss)

                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    patience_counter = 0 
                    torch.save(model.state_dict(), temp_best_model_path) 
                    print(f"Validation loss improved. Saving best model to {temp_best_model_path}")
                else:
                    patience_counter += 1
                    print(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Validation loss not improved for {early_stopping_patience} epochs.")
            break 

    print("Finished Training Model")
    
    if os.path.exists(temp_best_model_path):
        model.load_state_dict(torch.load(temp_best_model_path))
        print(f"Loaded best model from {temp_best_model_path}")
        os.remove(temp_best_model_path)
    else:
        print("Warning: Best model file not found. Returning model from last epoch.")
        
    return model