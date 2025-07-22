import os
import random
from copy import deepcopy
import numpy as np
import torch
import time # For unique TensorBoard log directories

from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter # Import SummaryWriter

# Import your custom modules
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN, CNN_LSTM_SCA # Keep CNN_LSTM_SCA
from src.trainer import trainer # Ensure trainer is updated to accept writer and trial_id
from src.utils import evaluate_fast, AES_Sbox, calculate_HW, HW_lookup_table # Import HW_lookup_table

if __name__ == "__main__":
    # --- Configuration Constants ---
    dataset = "CHES_2025"
    model_type = "cnn_lstm_sca" # Keep as cnn_lstm_sca
    leakage = "HW" # "ID" for Sbox output (256 classes), "HW" for Hamming Weight (9 classes)
    
    train_models = True # Set to False to only evaluate pre-trained models
    num_epochs = 100 # Maximum number of epochs for training each model configuration (early stopping will cut this short)
    total_num_models = 50 # Number of random hyperparameter configurations to test

    # --- Best Model Loading/Evaluation Control ---
    # Set to True to load a specific best model for final evaluation/submission (skips search)
    use_specific_best_model = False 
    # If use_specific_best_model is True, specify its path and config:
    SPECIFIC_MODEL_PATH = "Result/CHES_2025_cnn_lstm_sca_HW/models/model_X.pth" # Replace X with actual model_id
    SPECIFIC_CONFIG_PATH = "Result/CHES_2025_cnn_lstm_sca_HW/models/model_configuration_X.npy" # Replace X with actual model_id

    # --- SCA Preprocessing Parameters ---
    # These parameters are passed to Custom_Dataset for raw trace processing.
    # Experiment with these to optimize performance for your dataset.
    TRACE_START_POINT = 0 # Start index for trace cropping (POI). E.g., 500 for a specific window.
    TRACE_END_POINT = None # End index for trace cropping (POI). None for full trace after start_point. E.g., 1500 for 500:1500 window.
    
    DENOISING_FILTER_TYPE = 'gaussian' # Options: None, 'gaussian', 'moving_average'
    DENOISING_FILTER_WINDOW = 5 # Window size for the denoising filter (odd integer recommended for gaussian/moving_average)
    
    ALIGN_TRACES = True # Whether to perform trace alignment (cross-correlation based)
    # alignment_ref_trace can be manually set to a specific strong trace,
    # or if None, the first profiling trace will be used by Custom_Dataset.
    ALIGNMENT_REF_TRACE = None 

    # --- Attack Evaluation Parameters (as per CHES Challenge rules) ---
    nb_attacks_for_eval = 100 # Number of attack experiments for GE/NTGE calculation
    nb_traces_for_final_eval = 1700 # Max traces to use for GE/NTGE evaluation
    total_attack_traces_available = 2000 # Total attack traces loaded from dataset

    # --- Setup Directories ---
    log_base_dir = './runs/' # Base directory for TensorBoard logs
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    results_base_dir = './Result/'
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)

    save_root = os.path.join(results_base_dir, f"{dataset}_{model_type}_{leakage}/")
    model_root = os.path.join(save_root, "models/")
    
    print(f"Results will be saved to: {save_root}")
    print(f"Models will be saved to: {model_root}")
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # --- Set Random Seeds for Reproducibility ---
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to False for deterministic behavior

    # --- Device Configuration ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Leakage Model and Classes Definition ---
    if leakage == 'ID':
        def leakage_fn(att_plt, k_guess):
            return AES_Sbox[k_guess ^ int(att_plt)]
        classes = 256
    elif leakage == 'HW':
        def leakage_fn(att_plt, k_guess):
            # HW_lookup_table is imported from utils.py
            return HW_lookup_table[AES_Sbox[k_guess ^ int(att_plt)]]
        classes = 9
    else:
        raise ValueError("Invalid leakage model. Choose 'ID' or 'HW'.")

    # --- Data Loading and Preparation ---
    # Custom_Dataset loads data, handles preprocessing, and can split into train/val/test
    # The `leakage` parameter passed here ensures the labels are generated correctly.
    dataloadertrain = Custom_Dataset(root='./../', dataset=dataset, leakage=leakage, 
                                     transform=transforms.Compose([ToTensor_trace()]))

    # Split the loaded attack set into validation and test sets (from public attack traces)
    # This is used for local evaluation of model performance during hyperparameter search.
    dataloadertrain.split_attack_set_validation_test() 
    
    # Set the dataset to expose profiling data for training phase
    dataloadertrain.choose_phase("train") 
    
    # Create deep copies and set phases for validation and testing
    dataloadertest = deepcopy(dataloadertrain)
    dataloadertest.choose_phase("test") 
    
    dataloaderval = deepcopy(dataloadertrain)
    dataloaderval.choose_phase("validation")

    # Extract common data needed for evaluation (from the full attack set loaded by Custom_Dataset)
    correct_key = dataloadertrain.correct_key 
    # Use the processed X_attack from the dataset for evaluation
    X_attack = dataloadertrain.X_attack 
    Y_attack = dataloadertrain.Y_attack # This Y_attack is used for accuracy metrics if calculated
    plt_attack = dataloadertrain.plt_attack 
    # Get actual trace length after all preprocessing (cropping)
    num_sample_pts = X_attack.shape[-1] 

    # --- Main Execution Logic ---
    if train_models:
        # --- Hyperparameter Search Loop (Random Search) ---
        for num_models_trial_id in range(total_num_models):
            print(f"\n--- Running Model Trial {num_models_trial_id+1}/{total_num_models} ---")
            
            # --- Clear CUDA cache at the start of each trial to free up GPU memory ---
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")

            # --- TensorBoard Setup for Current Trial ---
            current_time_str = time.strftime("%Y%m%d-%H%M%S")
            trial_log_dir = os.path.join(log_base_dir, f"{model_type}_{leakage}_trial_{num_models_trial_id}_{current_time_str}")
            writer = SummaryWriter(log_dir=trial_log_dir)

            # --- Model Training Phase ---
            # Generate a random hyperparameter configuration for this trial
            config = create_hyperparameter_space(model_type)
            print(f"Trial {num_models_trial_id+1} Config: {config}")
            # Save the configuration to disk
            np.save(os.path.join(model_root, f"model_configuration_{num_models_trial_id}.npy"), config)
            
            # Prepare DataLoaders for training and validation with the current config's batch_size
            batch_size = config["batch_size"]
            num_workers = 0 # Set to 0 for Windows compatibility; for Linux/macOS, consider os.cpu_count() // 2
            
            dataloaders = {
                "train": torch.utils.data.DataLoader(dataloadertrain, batch_size=batch_size,
                                                    shuffle=True, num_workers=num_workers),
                "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                   shuffle=False, num_workers=num_workers) # No need to shuffle validation
            }
            dataset_sizes = {"train": len(dataloadertrain), "val": len(dataloaderval)}

            # Train the model, passing the TensorBoard writer and trial ID for best model saving
            model = trainer(config, num_epochs, num_sample_pts, dataloaders, 
                            dataset_sizes, model_type, classes, device, 
                            writer=writer, trial_id=num_models_trial_id) # Pass writer and trial_id
            
            # Save the trained model's state dictionary (which is the best validation model due to early stopping)
            torch.save(model.state_dict(), os.path.join(model_root, f"model_{num_models_trial_id}.pth"))
            
            # --- Model Evaluation Phase (Guessing Entropy & NTGE Calculation) ---
            print(f"Evaluating model for Trial {num_models_trial_id+1} on attack traces...")
            GE_curve, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key,
                                           leakage_fn=leakage_fn, nb_attacks=nb_attacks_for_eval, 
                                           total_nb_traces_attacks=total_attack_traces_available, 
                                           nb_traces_attacks=nb_traces_for_final_eval,
                                           batch_size=config["batch_size"]) # Use batch_size from config for evaluation
            
            # --- Log Attack Metrics to TensorBoard ---
            writer.add_scalar('Attack_Metrics/Final_GE_at_Max_Traces', GE_curve[-1], global_step=num_models_trial_id)
            writer.add_scalar('Attack_Metrics/NTGE', NTGE, global_step=num_models_trial_id)
            
            # Log the GE evolution curve (GE vs. number of traces)
            config_plot_tag = f"lr_{config['lr']}_bs_{config['batch_size']}_opt_{config['optimizer']}_attn_{config.get('use_attention', 'N/A')}"
            writer.add_custom_scalars(layout={
                'Attack GE Evolution': {
                    f'Trial_{num_models_trial_id}_{config_plot_tag}': ['Multiline', f'GE_Evolution/Trial_{num_models_trial_id}_GE_vs_Traces']
                }
            })
            for i, ge_val in enumerate(GE_curve):
                writer.add_scalar(f'GE_Evolution/Trial_{num_models_trial_id}_GE_vs_Traces', ge_val, global_step=i+1)

            # Save GE curve and NTGE to a numpy file for later analysis
            np.save(os.path.join(model_root, f"result_{num_models_trial_id}.npy"), {"GE_curve": GE_curve, "NTGE": NTGE})
            
            # --- Close TensorBoard Writer for Current Trial ---
            writer.close()
            print(f"Trial {num_models_trial_id+1} completed. Final GE: {GE_curve[-1]:.2f}, NTGE: {NTGE:.0f}")

        print("\n--- All model trials completed! ---")
        print(f"To view results in TensorBoard, run 'tensorboard --logdir {log_base_dir}' in your terminal.")

    else: # --- Evaluate a Specific Best Model ---
        print(f"\n--- Loading and Evaluating Specific Best Model ---")
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

        # Load configuration of the specific model
        if not os.path.exists(SPECIFIC_CONFIG_PATH):
            raise FileNotFoundError(f"Specific config file not found: {SPECIFIC_CONFIG_PATH}")
        config = np.load(SPECIFIC_CONFIG_PATH, allow_pickle=True).item()
        print(f"Loaded Config: {config}")

        # Instantiate model based on loaded config
        if model_type == "mlp":
            model = MLP(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn":
            model = CNN(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn_lstm_sca":
            model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
        else:
            raise ValueError(f"Unknown model type for specific best model: {model_type}")

        # Load the saved weights
        if not os.path.exists(SPECIFIC_MODEL_PATH):
            raise FileNotFoundError(f"Specific model weights file not found: {SPECIFIC_MODEL_PATH}")
        model.load_state_dict(torch.load(SPECIFIC_MODEL_PATH))
        model.to(device)
        model.eval() # Set to evaluation mode
        print(f"Model loaded from: {SPECIFIC_MODEL_PATH}")

        # --- Perform Final Evaluation ---
        print(f"Performing final evaluation of the best model...")
        GE_curve, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key,
                                       leakage_fn=leakage_fn, nb_attacks=nb_attacks_for_eval, 
                                       total_nb_traces_attacks=total_attack_traces_available, 
                                       nb_traces_attacks=nb_traces_for_final_eval,
                                       batch_size=config["batch_size"]) # Use its own batch_size
        
        print(f"\n--- Final Best Model Evaluation Results ---")
        print(f"Model Type: {model_type}")
        print(f"Leakage: {leakage}")
        print(f"Final GE (averaged over {nb_attacks_for_eval} attacks): {GE_curve[-1]:.2f}")
        print(f"NTGE: {NTGE:.0f}")
        print(f"Raw GE Curve (up to {nb_traces_for_final_eval} traces):\n{GE_curve}")