import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import time

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, weight_init
from src.cnn_lstm_sca import CNN_LSTM_SCA, CNN_LSTM_Light, create_cnn_lstm_sca_config
from src.utils import AES_Sbox, calculate_HW, evaluate,evaluate_fast
from src.early_stopping import EarlyStoppingAccuracy

def train_cnn_lstm_model(model, train_loader, val_loader, config, device, num_epochs, model_root, model_index, writer=None):
    """
    Custom training function for CNN-LSTM models with early stopping
    """
    # Setup optimizer
    lr = config["lr"]
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()    # Early stopping setup
    patience = config.get("patience", 30)  # Default patience of 30 epochs
    early_stopping = EarlyStoppingAccuracy(
        patience=patience, 
        verbose=False,  # Disable default verbose output
        path=f"{model_root}model_{model_index}.pth",
        trace_func=print
    )
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (traces, labels) in enumerate(train_loader):
            traces = traces.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, CNN_LSTM_SCA):
                outputs = model(traces)['output']
            else:
                outputs = model(traces)
            
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            
            # Forward pass and backward pass for training
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * traces.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += traces.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f'Epoch {epoch + 1}/{num_epochs} __ Loss: {epoch_loss:.4f}')
        print('-' * 10)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for traces, labels in val_loader:
                traces = traces.to(device)
                labels = labels.to(device)
                
                if isinstance(model, CNN_LSTM_SCA):
                    outputs = model(traces)['output']
                else:
                    outputs = model(traces)
                
                _, preds = torch.max(outputs, dim=1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * traces.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
                val_total_samples += traces.size(0)
        
        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_acc = val_running_corrects.double() / val_total_samples
        
        # Log to TensorBoard
        if writer:
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Val', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_epoch_acc, epoch)
        
        # Early stopping check
        early_stopping(val_epoch_acc, model)
        
        # Display early stopping counter
        if early_stopping.counter > 0:
            print(f"EarlyStopping counter: {early_stopping.counter} out of {patience}")
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
            
        # Update best accuracy for logging
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
        
        print()
    
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Load the best model weights
    model.load_state_dict(torch.load(f"{model_root}model_{model_index}.pth"))
    
    return model

def main():
    # Configuration
    dataset = "CHES_2025"
    leakage = "HW"
    model_type = "cnn_lstm_light"
    nb_traces_attacks = 100000
    total_nb_traces_attacks = 100000
    num_epochs = 100
    num_models = 200 

    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataloadertest = Custom_Dataset(
        root='./../', 
        dataset=dataset, 
        leakage="ID",
        transform=transforms.Compose([ToTensor_trace()])
    )

    # Set up leakage function
    if leakage == 'ID':
        def leakage_fn(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        def leakage_fn(att_plt, k):
            hw = [bin(x).count("1") for x in range(256)]
            return hw[AES_Sbox[k ^ int(att_plt)]]
        classes = 9
        dataloadertest.Y_attack = calculate_HW(dataloadertest.Y_attack)

    # Prepare data splits
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("train")
    
    X_train = dataloadertest.X_attack
    Y_train = dataloadertest.Y_attack
    plt_train = dataloadertest.plt_attack
    correct_key = dataloadertest.correct_key
    num_sample_pts = X_train.shape[-1]

    # Validation data
    dataloadertest.choose_phase("val")
    X_val = dataloadertest.X_attack
    Y_val = dataloadertest.Y_attack
    plt_val = dataloadertest.plt_attack
    correct_key_val = dataloadertest.correct_key

    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {classes}")
    print(f"Number of sample points: {num_sample_pts}")

    # Set up directories
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    
    os.makedirs(model_root, exist_ok=True)
    
    # Set up TensorBoard
    writer = SummaryWriter(log_dir=save_root + "tensorboard/")

    # Train multiple models
    best_models = []
    
    for model_idx in range(num_models):
        print(f"\n{'='*50}")
        print(f"Training Model {model_idx + 1}/{num_models}")
        print(f"{'='*50}")
        
        # Create model configuration for each model
        if model_type == "cnn_lstm_sca":
            config = create_hyperparameter_space(model_type)
            print("Using CNN-LSTM-SCA model with attention")
        else:
            # Use hyperparameter space for consistent random configuration
            config = create_hyperparameter_space(model_type)
            print("Using CNN-LSTM-Light model")

        print(f"Model {model_idx + 1} configuration: {config}")

        # Create model
        if model_type == "cnn_lstm_sca":
            model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
        else:
            model = CNN_LSTM_Light(config, num_sample_pts, classes).to(device)

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        model.apply(init_weights)

        print(f"Model {model_idx + 1} created with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")

        # Create data loaders
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), 
                                     torch.tensor(Y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                   torch.tensor(Y_val, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

        # Training
        print(f"Starting training for model {model_idx + 1}...")
        
        # Create a separate writer for each model
        model_writer = SummaryWriter(log_dir=save_root + f"tensorboard/model_{model_idx}/")
        
        trained_model = train_cnn_lstm_model(
            model, train_loader, val_loader, config, device, 
            num_epochs, model_root, model_idx, model_writer
        )

        # Save model configuration
        np.save(f"{model_root}model_configuration_{model_idx}.npy", config)

        # Quick evaluation on test set
        dataloadertest.choose_phase("test")
        X_test = dataloadertest.X_attack[:total_nb_traces_attacks]  # Use total available traces for prediction
        plt_test = dataloadertest.plt_attack[:total_nb_traces_attacks]
        
        try:
            # Use the imported evaluate_fast function with proper parameters
            GE, NTGE = evaluate_fast(
                device, trained_model, X_test, plt_test, correct_key, leakage_fn,
                nb_attacks=10000,  # Reduce for faster training evaluation
                total_nb_traces_attacks=10000,
                nb_traces_attacks=100,  # Use available traces
                # batch_size=25
            )
            
            print(f"Model: {model_idx + 1}")
            print(f"GE: {GE[-1] if len(GE) > 0 else 'N/A'}, NTGE: {NTGE}")
            print()
            
            # Store model performance
            final_rank = GE[-1] if len(GE) > 0 else float('inf')
            best_models.append({
                'model_idx': model_idx,
                'config': config,
                'GE': final_rank,  # Store final rank for sorting
                'NTGE': NTGE
            })
            # Log model performance
            model_writer.add_scalar('Performance/Final_GE_Rank', final_rank, 0)
            model_writer.add_scalar('Performance/NTGE', NTGE if NTGE != float('inf') else 999999, 0)
            
        except Exception as e:
            print(f"Error evaluating model {model_idx + 1}: {e}")
            best_models.append({
                'model_idx': model_idx,
                'config': config,
                'GE': float('inf'),
                'NTGE': float('inf')
            })
        
        model_writer.close()
        
        # Clean up memory
        del model, trained_model, train_loader, val_loader, train_dataset, val_dataset
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Sort models by performance (lower GE is better)
    best_models.sort(key=lambda x: x['GE'])
    
    print(f"\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"Trained {num_models} models")
    print("\nTop 10 Model:")
    
    for i, model_info in enumerate(best_models[:10]):
        print(f"Model: {model_info['model_idx']} -> GE: {model_info['GE']:.2f}, NTGE: {model_info['NTGE']}")
    
    # Save summary
    summary = {
        'model_type': model_type,
        'dataset': dataset,
        'leakage': leakage,
        'num_models_trained': num_models,
        'best_models': best_models
    }
    
    np.save(f"{save_root}training_summary.npy", summary)
    
    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Models saved in: {model_root}")
    print(f"TensorBoard logs in: {save_root}tensorboard/")
    print(f"Training summary saved in: {save_root}training_summary.npy")
    print("To view TensorBoard: tensorboard --logdir=./Result/")

if __name__ == "__main__":
    main()
