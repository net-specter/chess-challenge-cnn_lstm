# main_pytorch.py

import os
import random
from copy import deepcopy
import numpy as np
import torch
import time 

from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter 
import matplotlib.pyplot as plt 


# Import custom modules
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN, CNN_LSTM_SCA 
from src.trainer import trainer 
from src.utils import evaluate_fast, AES_Sbox, calculate_HW, HW_lookup_table 


if __name__ == "__main__":
    # --- GLOBAL CHALLENGE & MODEL CONFIGURATION ---
    dataset = "CHES_2025" 
    model_type = "cnn_lstm_sca" 
    leakage = "HW" 
    
    # --- TRAINING & HYPERPARAMETER SEARCH CONTROL ---
    run_hyperparameter_search = True 
    total_num_models_to_search = 50 
    num_epochs_per_trial = 100 

    # --- SPECIFIC BEST MODEL LOADING & EVALUATION CONTROL ---
    use_specific_best_model = False 
    BEST_MODEL_TRIAL_ID = 0 
    SPECIFIC_MODEL_PATH = os.path.join("Result", f"{dataset}_{model_type}_{leakage}", "models", f"model_{BEST_MODEL_TRIAL_ID}.pth")
    SPECIFIC_CONFIG_PATH = os.path.join("Result", f"{dataset}_{model_type}_{leakage}", "models", f"model_configuration_{BEST_MODEL_TRIAL_ID}.npy")


    # --- SCA PREPROCESSING PARAMETERS ---
    PROCESSED_DATA_FILEPATH = "./processed_ches_data.h5" 
    TRACE_START_POINT = 0 
    TRACE_END_POINT = None 
    DENOISING_FILTER_TYPE = 'gaussian' 
    DENOISING_FILTER_WINDOW = 5 
    ALIGN_TRACES = True 
    ALIGNMENT_REF_TRACE = None 

    # --- ATTACK EVALUATION PARAMETERS ---
    nb_attacks_for_eval = 100 
    nb_traces_for_final_eval = 100000 
    total_attack_traces_available = 100000 

    # --- SETUP DIRECTORIES ---
    log_base_dir = './runs/' 
    if not os.path.exists(log_base_dir):
        os.makedirs(log_base_dir)

    results_base_dir = './Result/'
    if not os.path.exists(results_base_dir):
        os.makedirs(results_base_dir)

    save_root = os.path.join(results_base_dir, f"{dataset}_{model_type}_{leakage}/")
    model_root = os.path.join(save_root, "models/")
    
    print(f"Results for this run will be saved to: {save_root}")
    print(f"Models will be saved to: {model_root}")
    
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(model_root):
        os.makedirs(model_root)

    # --- SET RANDOM SEEDS FOR REPRODUCIBILITY ---
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    # --- DEVICE CONFIGURATION ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training will use device: {device}")

    # --- LEAKAGE MODEL AND CLASSES DEFINITION ---
    if leakage == 'ID':
        def leakage_fn(att_plt, k_guess):
            return AES_Sbox[k_guess ^ int(att_plt)]
        classes = 256 
    elif leakage == 'HW':
        def leakage_fn(att_plt, k_guess):
            return HW_lookup_table[AES_Sbox[k_guess ^ int(att_plt)]]
        classes = 9 
    else:
        raise ValueError("Invalid leakage model specified. Choose 'ID' or 'HW'.")

    # --- Data Loading and Preparation (Initial Load/Preprocess) ---
    dataset_h5_path = './../Dataset/CHES_2025/CHES_Challenge.h5' 

    # The first Custom_Dataset instance will trigger the full data loading and preprocessing,
    # and save it to PROCESSED_DATA_FILEPATH.
    dataloadertrain_master = Custom_Dataset(root='./../', dataset=dataset, leakage=leakage, 
                                     transform=transforms.Compose([ToTensor_trace()]),
                                     trace_start_point=TRACE_START_POINT,
                                     trace_end_point=TRACE_END_POINT,
                                     denoising_filter_type=DENOISING_FILTER_TYPE,
                                     denoising_filter_window=DENOISING_FILTER_WINDOW,
                                     align_traces=ALIGN_TRACES,
                                     alignment_ref_trace=ALIGNMENT_REF_TRACE, 
                                     processed_data_filepath=PROCESSED_DATA_FILEPATH
                                     )

    # Extract the fully processed NumPy arrays from the first Custom_Dataset instance.
    # These arrays will be passed to subsequent Custom_Dataset instances to avoid re-loading/re-processing.
    X_profiling_processed = dataloadertrain_master.X_profiling
    X_attack_processed = dataloadertrain_master.X_attack
    Y_profiling_processed = dataloadertrain_master.Y_profiling
    Y_attack_processed = dataloadertrain_master.Y_attack
    P_profiling_processed = dataloadertrain_master.plt_profiling
    P_attack_processed = dataloadertrain_master.plt_attack
    correct_key_processed = dataloadertrain_master.correct_key
    
    # --- Perform the split on the master dataset instance ---
    dataloadertrain_master.split_attack_set_validation_test() 

    # --- Create Custom_Dataset instances for each phase, sharing the processed data ---
    # The 'train' DataLoader will use the profiling data
    # (No need to create a new instance for dataloadertrain_master, just ensure its phase is set)
    dataloadertrain_master.choose_phase("train") 
    
    # Create new instances for test and validation, explicitly passing the pre-loaded data
    # AND performing the split on each new instance.
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage=leakage, 
                                    transform=transforms.Compose([ToTensor_trace()]),
                                    X_profiling_preloaded=X_profiling_processed, X_attack_preloaded=X_attack_processed,
                                    Y_profiling_preloaded=Y_profiling_processed, Y_attack_preloaded=Y_attack_processed,
                                    P_profiling_preloaded=P_profiling_processed, P_attack_preloaded=P_attack_processed,
                                    correct_key_preloaded=correct_key_processed)
    # *** FIX: Call split_attack_set_validation_test on the new dataloadertest instance ***
    dataloadertest.split_attack_set_validation_test() # Perform split on this instance
    dataloadertest.choose_phase("test") 
    
    dataloaderval = Custom_Dataset(root='./../', dataset=dataset, leakage=leakage, 
                                   transform=transforms.Compose([ToTensor_trace()]),
                                   X_profiling_preloaded=X_profiling_processed, X_attack_preloaded=X_attack_processed,
                                   Y_profiling_preloaded=Y_profiling_processed, Y_attack_preloaded=Y_attack_processed,
                                   P_profiling_preloaded=P_profiling_processed, P_attack_preloaded=P_attack_processed,
                                   correct_key_preloaded=correct_key_processed)
    # *** FIX: Call split_attack_set_validation_test on the new dataloaderval instance ***
    dataloaderval.split_attack_set_validation_test() # Perform split on this instance
    dataloaderval.choose_phase("validation")

    # Extract common data needed for evaluation (from the processed data, now accessible via any dataloader)
    correct_key = correct_key_processed # Use the processed key
    X_attack = X_attack_processed # Use the processed attack traces
    plt_attack = P_attack_processed # Use the processed attack plaintexts
    num_sample_pts = X_profiling_processed.shape[1] # Actual trace length after preprocessing


    # --- MAIN EXECUTION LOGIC: HYPERPARAMETER SEARCH OR SPECIFIC MODEL EVALUATION ---
    if run_hyperparameter_search:
        best_overall_ntge = float('inf') 
        best_overall_final_ge = float('inf') 
        best_model_trial_id = -1 
        best_model_config = None 

        for num_models_trial_id in range(total_num_models_to_search):
            print(f"\n--- Running Model Trial {num_models_trial_id+1}/{total_num_models_to_search} ---")
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("CUDA cache cleared.")

            current_time_str = time.strftime("%Y%m%d-%H%M%S")
            trial_log_dir = os.path.join(log_base_dir, f"{model_type}_{leakage}_trial_{num_models_trial_id}_{current_time_str}")
            writer = SummaryWriter(log_dir=trial_log_dir)

            config = create_hyperparameter_space(model_type)
            print(f"Trial {num_models_trial_id+1} Config: {config}")
            np.save(os.path.join(model_root, f"model_configuration_{num_models_trial_id}.npy"), config)
            
            batch_size = config["batch_size"]
            num_workers = 0 
            
            dataloaders = {
                "train": torch.utils.data.DataLoader(dataloadertrain_master, batch_size=batch_size, # Use dataloadertrain_master for training
                                                    shuffle=True, num_workers=num_workers),
                "val": torch.utils.data.DataLoader(dataloaderval, batch_size=batch_size,
                                                   shuffle=False, num_workers=num_workers) 
            }
            dataset_sizes = {"train": len(dataloadertrain_master), "val": len(dataloaderval)} # Use lengths from correct datasets

            model = trainer(config, num_epochs_per_trial, num_sample_pts, dataloaders, 
                            dataset_sizes, model_type, classes, device, 
                            writer=writer, trial_id=num_models_trial_id) 
            
            torch.save(model.state_dict(), os.path.join(model_root, f"model_{num_models_trial_id}.pth"))
            
            print(f"Evaluating model for Trial {num_models_trial_id+1} on attack traces...")
            GE_curve, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key,
                                           leakage_fn=leakage_fn, nb_attacks=nb_attacks_for_eval, 
                                           total_nb_traces_attacks=total_attack_traces_available, 
                                           nb_traces_attacks=nb_traces_for_final_eval,
                                           batch_size=config["batch_size"]) 
            
            writer.add_scalar('Attack_Metrics/Final_GE_at_Max_Traces', GE_curve[-1], global_step=num_models_trial_id)
            writer.add_scalar('Attack_Metrics/NTGE', NTGE, global_step=num_models_trial_id)
            
            config_plot_tag = f"lr_{config['lr']}_bs_{config['batch_size']}_opt_{config['optimizer']}_attn_{config.get('use_attention', 'N/A')}"
            writer.add_custom_scalars(layout={
                'Attack GE Evolution': { 
                    f'Trial_{num_models_trial_id}_{config_plot_tag}': ['Multiline', f'GE_Evolution/Trial_{num_models_trial_id}_GE_vs_Traces']
                }
            })
            for i, ge_val in enumerate(GE_curve):
                writer.add_scalar(f'GE_Evolution/Trial_{num_models_trial_id}_GE_vs_Traces', ge_val, global_step=i+1)

            np.save(os.path.join(model_root, f"result_{num_models_trial_id}.npy"), {"GE_curve": GE_curve, "NTGE": NTGE})
            
            writer.close()
            print(f"Trial {num_models_trial_id+1} completed. Final GE: {GE_curve[-1]:.2f}, NTGE: {NTGE:.0f}")

        print("\n" + "="*80)
        print("HYPERPARAMETER SEARCH COMPLETE")
        print(f"Total trials executed: {total_num_models_to_search}")
        if best_model_trial_id != -1:
            print(f"\n🏆 BEST MODEL FOUND DURING SEARCH:")
            print(f"   Trial ID: {best_model_trial_id}")
            print(f"   NTGE: {best_overall_ntge:.0f}")
            print(f"   Final GE (at {nb_traces_for_final_eval} traces): {best_overall_final_ge:.2f}")
            print(f"   Configuration: {best_model_config}")
            print(f"   To re-evaluate this specific model, set `use_specific_best_model = True` and `BEST_MODEL_TRIAL_ID = {best_model_trial_id}` at the top of this script.")
        else:
            print("No models achieved a finite NTGE (all attacks were unsuccessful in reaching GE=0).")
        print("="*80)
        print(f"To view detailed results in TensorBoard, run 'tensorboard --logdir {log_base_dir}' in your terminal.")

    else: # --- Evaluate a Specific Best Model ---
        print(f"\n--- Loading and Evaluating Specific Best Model (Trial ID: {BEST_MODEL_TRIAL_ID}) ---")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

        if not os.path.exists(SPECIFIC_CONFIG_PATH):
            raise FileNotFoundError(f"Specific config file not found: {SPECIFIC_CONFIG_PATH}. "
                                    f"Please ensure BEST_MODEL_TRIAL_ID is correct and the file exists.")
        config = np.load(SPECIFIC_CONFIG_PATH, allow_pickle=True).item()
        print(f"Loaded Config: {config}")

        if model_type == "mlp":
            model = MLP(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn":
            model = CNN(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn_lstm_sca":
            model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
        else:
            raise ValueError(f"Unknown model type for specific best model: {model_type}. Check `model_type` global variable.")

        if not os.path.exists(SPECIFIC_MODEL_PATH):
            raise FileNotFoundError(f"Specific model weights file not found: {SPECIFIC_MODEL_PATH}. "
                                    f"Please ensure BEST_MODEL_TRIAL_ID is correct and the file exists.")
        model.load_state_dict(torch.load(SPECIFIC_MODEL_PATH))
        model.to(device)
        model.eval() 
        print(f"Model loaded successfully from: {SPECIFIC_MODEL_PATH}")

        print(f"Performing final evaluation of the best model...")
        GE_curve, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key,
                                       leakage_fn=leakage_fn, nb_attacks=nb_attacks_for_eval, 
                                       total_nb_traces_attacks=total_attack_traces_available, 
                                       nb_traces_attacks=nb_traces_for_final_eval,
                                       batch_size=config["batch_size"]) 
        
        print(f"\n--- FINAL BEST MODEL EVALUATION RESULTS ---")
        print(f"Model Type: {model_type}")
        print(f"Leakage: {leakage}")
        print(f"Final GE (averaged over {nb_attacks_for_eval} attacks): {GE_curve[-1]:.2f}")
        print(f"NTGE: {NTGE:.0f}")
        print(f"Raw GE Curve (up to {nb_traces_for_final_eval} traces):\n{GE_curve}")
        print("="*80)