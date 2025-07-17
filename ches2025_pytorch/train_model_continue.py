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
from copy import deepcopy 

from src.dataloader_continue import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space 
from src.cnn_lstm_sca import CNN_LSTM_SCA, CNN_LSTM_Light, create_cnn_lstm_sca_config 
from src.utils import AES_Sbox, calculate_HW, evaluate,evaluate_fast
from src.early_stopping import EarlyStoppingAccuracy

def get_leakage_fn(leakage_type):
    if leakage_type == 'ID':
        def leakage_fn_id(att_plt, k):
            return AES_Sbox[k ^ int(att_plt)]
        return leakage_fn_id, 256
    elif leakage_type == 'HW':
        hw_values = [bin(x).count("1") for x in range(256)]
        def leakage_fn_hw(att_plt, k):
            return hw_values[AES_Sbox[k ^ int(att_plt)]]
        return leakage_fn_hw, 9
    else:
        raise ValueError("Unsupported leakage type.")

# --- MODIFIED train_cnn_lstm_model (No changes here from previous fix) ---
def train_cnn_lstm_model(model_to_train, train_loader, val_loader, config, device, num_epochs, model_root, model_index, writer=None):
    model = model_to_train 
    lr = config["lr"] 
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif config["optimizer"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else: 
        raise ValueError(f"Optimizer '{config['optimizer']}' not supported.")
    
    criterion = nn.CrossEntropyLoss()    
    patience = config.get("patience", 30)  
    early_stopping = EarlyStoppingAccuracy(
        patience=patience, 
        verbose=False,  
        path=f"{model_root}model_{model_index}.pth", 
        trace_func=print
    )
    
    best_val_acc = 0.0 
    # start_time = time.time() # This was defined here, but not needed within this function's scope
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (traces, labels, plaintexts) in enumerate(train_loader): 
            traces = traces.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if isinstance(model, CNN_LSTM_SCA):
                outputs = model(traces)['output']
            else:
                outputs = model(traces)
            
            _, preds = torch.max(outputs, dim=1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * traces.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += traces.size(0)
        
        epoch_loss = running_loss / total_samples
        epoch_acc = running_corrects.double() / total_samples

        print(f'Epoch {epoch + 1}/{num_epochs} __ Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}') 
        print('-' * 10)
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for traces, labels, plaintexts in val_loader: 
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
        
        if writer:
            writer.add_scalar('Loss/Train', epoch_loss, epoch)
            writer.add_scalar('Loss/Val', val_epoch_loss, epoch)
            writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_epoch_acc, epoch)
        
        early_stopping(val_epoch_acc, model)
        
        if early_stopping.counter > 0:
            print(f"EarlyStopping counter: {early_stopping.counter} out of {patience}")
        
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
            
        if val_epoch_acc > best_val_acc: 
            best_val_acc = val_epoch_acc
        
        print()
    
    # total_time = time.time() - start_time # <-- REMOVED this line. The total_time calculation is now in main()
    # print(f"Training completed in {total_time:.2f}s") # <-- REMOVED
    print(f"Best validation accuracy for this phase: {best_val_acc:.4f}") 
    
    model.load_state_dict(torch.load(f"{model_root}model_{model_index}.pth")) 
    
    return model 


def main():
    # Configuration
    dataset = "CHES_2025"
    leakage = "HW"
    model_type = "cnn_lstm_light" 
    nb_traces_attacks = 100000 
    total_nb_traces_attacks = 100000 
    num_epochs_per_chunk = 10 
    total_profiling_traces = 500000
    chunk_size = 100000
    num_chunks = total_profiling_traces // chunk_size

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

    # Set up leakage function
    leakage_fn, classes = get_leakage_fn(leakage)

    # --- MODIFIED START ---
    # Initial load to get num_sample_pts and correct_key from the full profiling set context
    # This also sets up the validation data (X_attack_val, Y_attack_val, plt_attack_val)
    initial_dataset_loader = Custom_Dataset(
        root='./../', 
        dataset=dataset, 
        leakage=leakage, # Use the actual leakage from main config
        transform=transforms.Compose([ToTensor_trace()]),
        # Load the full 500k profiling data just to determine its shape and get the correct_key
        # We also need the full 100k attack data loaded for validation splitting
        custom_train_begin=0, custom_train_end=total_profiling_traces, 
        custom_test_begin=0, custom_test_end=total_nb_traces_attacks 
    )
    # Correct key for the attack dataset (this is the final secret we are looking for)
    correct_key = initial_dataset_loader.correct_key 
    num_sample_pts = initial_dataset_loader.X_profiling.shape[-1] # Get num_sample_pts from profiling data
    
    # Prepare Validation Loader once using the data loaded by initial_dataset_loader
    initial_dataset_loader.split_attack_set_validation_test()
    initial_dataset_loader.choose_phase("val") 
    
    X_val = initial_dataset_loader.X
    Y_val = initial_dataset_loader.Y
    plt_val = initial_dataset_loader.plt 
    correct_key_val = initial_dataset_loader.correct_key 
    
    val_tensor_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                       torch.tensor(Y_val, dtype=torch.long),
                                       torch.tensor(plt_val, dtype=torch.long))
    val_loader = DataLoader(val_tensor_dataset, batch_size=config["batch_size"], shuffle=False) # config must be defined before here
    
    # Need to load a dummy config first to get the batch_size for the DataLoader
    # This is a common pattern for initial setup when config is defined later in a loop.
    dummy_config = create_hyperparameter_space(model_type)
    val_loader = DataLoader(val_tensor_dataset, batch_size=dummy_config["batch_size"], shuffle=False) # Use dummy config for batch_size

    print(f"Validation data shape: {X_val.shape}")
    print(f"Number of classes: {classes}")
    print(f"Number of sample points: {num_sample_pts}")
    # --- END MODIFIED START ---

    # Specify the index of the model to load and continue training
    model_to_continue_idx = 8 

    # Set up directories
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    
    os.makedirs(model_root, exist_ok=True)
    
    # Setup TensorBoard for the entire continuous training session
    writer = SummaryWriter(log_dir=save_root + f"tensorboard/continued_model_{model_to_continue_idx}/")

    # Load specific model and config to continue training
    config_path = f"{model_root}model_configuration_{model_to_continue_idx}.npy"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found for model index {model_to_continue_idx}. Please ensure it was trained and saved.")
    
    config = np.load(config_path, allow_pickle=True).item()
    print(f"Loaded configuration for model index {model_to_continue_idx}: {config}")

    # Create the model instance based on its original config
    if model_type == "cnn_lstm_sca":
        model = CNN_LSTM_SCA(config, num_sample_pts=num_sample_pts, classes=classes).to(device) 
    elif model_type == "cnn_lstm_light": 
        model = CNN_LSTM_Light(config, num_sample_pts=num_sample_pts, classes=classes).to(device)
    # Assuming `TransNetModel` is correctly imported and defined for `model_type`
    # elif model_type == "transnet": 
    #     model = TransNetModel(config, num_sample_pts=num_sample_pts, classes=classes).to(device)
    else:
        raise ValueError(f"Unsupported model_type for loading: {model_type}")

    # Load the saved weights (brain) of the best model
    model_path = f"{model_root}model_{model_to_continue_idx}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found for model index {model_to_continue_idx}. Please ensure it was trained and saved.")
    
    model.load_state_dict(torch.load(model_path))
    print(f"Loaded weights for model index {model_to_continue_idx} from {model_path}")
    
    # --- MODIFIED: Loop through profiling data chunks ---
    total_training_time_s = 0.0 # Initialize total training time
    for chunk_idx in range(num_chunks):
        print(f"\n{'='*50}")
        print(f"TRAINING CHUNK {chunk_idx + 1}/{num_chunks}")
        print(f"{'='*50}")
        
        train_begin = chunk_idx * chunk_size
        train_end = train_begin + chunk_size
        
        print(f"Loading training data from {train_begin} to {train_end}...")
        
        # Load profiling data for the current chunk
        train_dataset_loader = Custom_Dataset(
            root='./../', 
            dataset=dataset, 
            leakage=leakage, # Use actual leakage
            transform=transforms.Compose([ToTensor_trace()]),
            custom_train_begin=train_begin, 
            custom_train_end=train_end
        )
        # if leakage == "HW": # No need, Custom_Dataset __init__ handles this if passed leakage=leakage
        #     train_dataset_loader.Y_profiling = np.array(calculate_HW(train_dataset_loader.Y_profiling))
        
        train_dataset_loader.choose_phase("train") # Select train phase for current chunk
        
        X_train_chunk = train_dataset_loader.X
        Y_train_chunk = train_dataset_loader.Y
        plt_train_chunk = train_dataset_loader.plt 
        
        train_tensor_dataset = TensorDataset(torch.tensor(X_train_chunk, dtype=torch.float32), 
                                             torch.tensor(Y_train_chunk, dtype=torch.long),
                                             torch.tensor(plt_train_chunk, dtype=torch.long)) 
        train_loader = DataLoader(train_tensor_dataset, batch_size=config["batch_size"], shuffle=True)
        
        print(f"Chunk training data shape: {X_train_chunk.shape}")

        # Training
        print(f"Starting training for chunk {chunk_idx + 1}...")
        
        chunk_start_time = time.time() # Track time for this chunk
        model = train_cnn_lstm_model( 
            model, train_loader, val_loader, config, device, 
            num_epochs_per_chunk, model_root, model_to_continue_idx, writer 
        )
        chunk_end_time = time.time()
        chunk_duration = chunk_end_time - chunk_start_time
        total_training_time_s += chunk_duration # Add to total
        print(f"Chunk {chunk_idx + 1} training completed in {chunk_duration:.2f}s")
        
        torch.save(model.state_dict(), f"{model_root}model_{model_to_continue_idx}_after_chunk_{chunk_idx}.pth")
        
        del train_dataset_loader, train_tensor_dataset, X_train_chunk, Y_train_chunk, plt_train_chunk
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- END MODIFIED: Loop through profiling data chunks ---

    # --- MODIFIED: Final evaluation after all chunks are processed ---
    print(f"\n{'='*50}")
    print("FINAL EVALUATION AFTER ALL CHUNKS")
    print(f"{'='*50}")
    
    # Get the full X_attack and plt_attack for final evaluation
    final_eval_dataloader = Custom_Dataset(
        root='./../', dataset=dataset, leakage=leakage, # Use actual leakage
        transform=transforms.Compose([ToTensor_trace()]),
        custom_test_begin=0, custom_test_end=total_nb_traces_attacks 
    )
    # if leakage == "HW": # No need, Custom_Dataset __init__ handles this
    #     final_eval_dataloader.Y_attack = np.array(calculate_HW(final_eval_dataloader.Y_attack))
    final_eval_dataloader.choose_phase("test")

    X_final_attack = final_eval_dataloader.X
    plt_final_attack = final_eval_dataloader.plt
    
    GE_final, NTGE_final = evaluate_fast(
        device, model, X_final_attack, plt_final_attack, correct_key, leakage_fn,
        nb_attacks=10000, 
        total_nb_traces_attacks=total_nb_traces_attacks,
        nb_traces_attacks=nb_traces_attacks,
    )
    
    print(f"\nFinal Model (Index {model_to_continue_idx}): GE={GE_final[-1]:.2f}, NTGE={NTGE_final}")
    print(f"Total continuous training time: {total_training_time_s:.2f}s ({total_training_time_s / 3600:.2f} hours)")
    
    final_summary_path = f"{model_root}final_training_summary_{model_to_continue_idx}.npy"
    final_summary = {
        'model_idx': model_to_continue_idx,
        'config': config,
        'GE': GE_final[-1],
        'NTGE': NTGE_final,
        'total_epochs': num_epochs_per_chunk * num_chunks,
        'total_training_time_s': total_training_time_s
    }
    np.save(final_summary_path, final_summary)
    print(f"Final summary saved to: {final_summary_path}")

    writer.close()
    
    print(f"\nTraining completed!")
    print(f"Models saved in: {model_root}")
    print(f"TensorBoard logs in: {save_root}tensorboard/")

if __name__ == "__main__":
    main()