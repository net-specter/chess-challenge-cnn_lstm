import os
import random
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import transforms
import glob
import pandas as pd
import matplotlib.pyplot as plt

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.cnn_lstm_sca import CNN_LSTM_SCA, CNN_LSTM_Light
from src.utils import evaluate_fast, AES_Sbox, calculate_HW

def evaluate_cnn_lstm_batch(device, model, X_attack, plt_attack, correct_key, leakage_fn, 
                           nb_attacks=100, total_nb_traces_attacks=10000, nb_traces_attacks=10000, batch_size=25):
    """
    Custom evaluation function for CNN-LSTM models with batch processing
    """
    try_cpu = False
    
    try:
        # Convert to tensor
        attack_traces = torch.from_numpy(X_attack[:total_nb_traces_attacks]).float()
        
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            # Process in batches to avoid memory issues
            for i in range(0, len(attack_traces), batch_size):
                batch = attack_traces[i:i+batch_size].to(device)
                
                try:
                    if isinstance(model, CNN_LSTM_SCA):
                        batch_output = model(batch)['output']
                    else:
                        batch_output = model(batch)
                    
                    batch_predictions = F.softmax(batch_output, dim=1).cpu()
                    all_predictions.append(batch_predictions)
                    
                except torch.cuda.OutOfMemoryError:
                    print("GPU out of memory, switching to CPU...")
                    try_cpu = True
                    break
                    
                # Clear cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if try_cpu:
            print("Processing on CPU due to memory constraints...")
            # Move model to CPU and retry with smaller batches
            model = model.cpu()
            device_cpu = torch.device("cpu")
            attack_traces = attack_traces.to(device_cpu)
            
            all_predictions = []
            batch_size = 10  # Even smaller batches for CPU
            
            with torch.no_grad():
                for i in range(0, len(attack_traces), batch_size):
                    batch = attack_traces[i:i+batch_size]
                    
                    if isinstance(model, CNN_LSTM_SCA):
                        batch_output = model(batch)['output']
                    else:
                        batch_output = model(batch)
                    
                    batch_predictions = F.softmax(batch_output, dim=1)
                    all_predictions.append(batch_predictions)
        
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=0).numpy()
        
    except Exception as e:
        print(f"Error during batch processing: {e}")
        raise e
    
    # Use the existing attack performance evaluation
    from src.utils import perform_attacks
    GE, key_prob = perform_attacks(nb_traces_attacks, predictions, plt_attack, correct_key,
                                   nb_attacks=nb_attacks, shuffle=True, leakage_fn=leakage_fn)
    
    # Calculate NTGE
    NTGE = float('inf')
    for i in range(GE.shape[0] - 1, -1, -1):
        if GE[i] > 0:
            break
        elif GE[i] == 0:
            NTGE = i
    
    return GE, NTGE

def analyze_single_model(model_path, config_path, model_type, dataset, leakage, device, 
                        X_attack, plt_attack, correct_key, leakage_fn, classes, num_sample_pts):
    """
    Analyze a single model
    """
    try:
        # Load configuration
        config = np.load(config_path, allow_pickle=True).item()
        
        # Create model
        if model_type == "cnn_lstm_sca":
            model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn_lstm_light":
            model = CNN_LSTM_Light(config, num_sample_pts, classes).to(device)
        elif model_type == "mlp":
            model = MLP(config, num_sample_pts, classes).to(device)
        elif model_type == "cnn":
            model = CNN(config, num_sample_pts, classes).to(device)
        else:
            print(f"Unknown model type: {model_type}")
            return None
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate model
        if "cnn_lstm" in model_type:
            GE, NTGE = evaluate_fast(
                device, model, X_attack, plt_attack, correct_key, leakage_fn,
                nb_attacks=100, total_nb_traces_attacks=100000, nb_traces_attacks=100000, batch_size=25
            )
        else:
            GE, NTGE = evaluate_fast(
                device, model, X_attack, plt_attack, correct_key, leakage_fn,
                nb_attacks=100, total_nb_traces_attacks=100000, nb_traces_attacks=100000
            )
        
        # Calculate final metrics
        final_rank = GE[-1] if len(GE) > 0 else float('inf')
        success = final_rank == 0
        
        return {
            'model_path': model_path,
            'config': config,
            'GE': GE,
            'NTGE': NTGE,
            'final_rank': final_rank,
            'success': success,
            'model_type': model_type
        }
        
    except Exception as e:
        print(f"Error analyzing {model_path}: {e}")
        return None

def find_all_models(result_root="./Result/"):
    """
    Find all trained models in the result directory
    """
    models_info = []
    
    # Search for all model directories
    for dir_name in os.listdir(result_root):
        dir_path = os.path.join(result_root, dir_name)
        if os.path.isdir(dir_path):
            models_dir = os.path.join(dir_path, "models")
            if os.path.exists(models_dir):
                # Extract model type from directory name
                parts = dir_name.split("_")
                if len(parts) >= 3:
                    dataset = parts[0]
                    if "cnn_lstm" in dir_name:
                        if "sca" in dir_name:
                            model_type = "cnn_lstm_sca"
                        else:
                            model_type = "cnn_lstm_light"
                    else:
                        model_type = parts[1]
                    leakage = parts[-1]
                    
                    # Find all model files
                    model_files = glob.glob(os.path.join(models_dir, "model_*.pth"))
                    for model_file in model_files:
                        # Extract model index
                        filename = os.path.basename(model_file)
                        model_idx = filename.replace("model_", "").replace(".pth", "")
                        
                        # Find corresponding config file
                        config_file = os.path.join(models_dir, f"model_configuration_{model_idx}.npy")
                        
                        if os.path.exists(config_file):
                            models_info.append({
                                'model_path': model_file,
                                'config_path': config_file,
                                'model_type': model_type,
                                'dataset': dataset,
                                'leakage': leakage,
                                'model_idx': int(model_idx),
                                'dir_name': dir_name
                            })
    
    return models_info

def create_analysis_report(results, output_dir="./Analysis_Results/"):
    """
    Create comprehensive analysis report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter successful analyses
    successful_results = [r for r in results if r is not None]
    
    if not successful_results:
        print("No successful analyses to report")
        return
    
    # Create summary DataFrame
    summary_data = []
    for result in successful_results:
        summary_data.append({
            'Model_Path': os.path.basename(result['model_path']),
            'Model_Type': result['model_type'],
            'Final_Rank': result['final_rank'],
            'Success': result['success'],
            'NTGE': result['NTGE'] if result['NTGE'] != float('inf') else 'No Success',
            'Batch_Size': result['config'].get('batch_size', 'N/A'),
            'Learning_Rate': result['config'].get('lr', 'N/A'),
            'Optimizer': result['config'].get('optimizer', 'N/A')
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save summary CSV
    df.to_csv(os.path.join(output_dir, "model_analysis_summary.csv"), index=False)
    
    # Print summary
    print(f"/n{'='*80}")
    print("COMPREHENSIVE MODEL ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total models analyzed: {len(successful_results)}")
    print(f"Successful attacks: {sum(1 for r in successful_results if r['success'])}")
    print(f"Success rate: {sum(1 for r in successful_results if r['success']) / len(successful_results) * 100:.1f}%")
    
    # Group by model type
    print(f"/n{'Model Type':<15} {'Count':<8} {'Success':<8} {'Best Rank':<12} {'Best NTGE':<12}")
    print("-" * 60)
    
    for model_type in df['Model_Type'].unique():
        type_results = [r for r in successful_results if r['model_type'] == model_type]
        success_count = sum(1 for r in type_results if r['success'])
        best_rank = min(r['final_rank'] for r in type_results)
        best_ntge = min(r['NTGE'] for r in type_results if r['NTGE'] != float('inf'))
        if best_ntge == float('inf'):
            best_ntge = "No Success"
        
        print(f"{model_type:<15} {len(type_results):<8} {success_count:<8} {best_rank:<12} {best_ntge:<12}")
    
    # Show top 10 models
    print(f"/n{'='*80}")
    print("TOP 10 MODELS BY PERFORMANCE")
    print(f"{'='*80}")
    
    # Sort by final rank, then by NTGE
    sorted_results = sorted(successful_results, key=lambda x: (x['final_rank'], x['NTGE'] if x['NTGE'] != float('inf') else 99999))
    
    print(f"{'Rank':<6} {'Model':<20} {'Type':<15} {'Final_Rank':<12} {'NTGE':<12} {'Config'}")
    print("-" * 100)
    
    for i, result in enumerate(sorted_results[:10]):
        model_name = os.path.basename(result['model_path'])
        config_str = f"bs={result['config'].get('batch_size', 'N/A')}, lr={result['config'].get('lr', 'N/A')}"
        ntge_str = str(result['NTGE']) if result['NTGE'] != float('inf') else "No Success"
        
        print(f"{i+1:<6} {model_name:<20} {result['model_type']:<15} {result['final_rank']:<12} {ntge_str:<12} {config_str}")
    
    # Create visualization
    try:
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Success rate by model type
        plt.subplot(2, 2, 1)
        success_by_type = df.groupby('Model_Type')['Success'].mean()
        success_by_type.plot(kind='bar')
        plt.title('Success Rate by Model Type')
        plt.ylabel('Success Rate')
        plt.xticks(rotation=45)
        
        # Plot 2: Final rank distribution
        plt.subplot(2, 2, 2)
        final_ranks = [r['final_rank'] for r in successful_results if r['final_rank'] < 50]  # Limit for visualization
        plt.hist(final_ranks, bins=20)
        plt.title('Distribution of Final Key Ranks')
        plt.xlabel('Final Rank')
        plt.ylabel('Count')
        
        # Plot 3: NTGE distribution for successful attacks
        plt.subplot(2, 2, 3)
        ntge_values = [r['NTGE'] for r in successful_results if r['NTGE'] != float('inf')]
        if ntge_values:
            plt.hist(ntge_values, bins=20)
            plt.title('NTGE Distribution (Successful Attacks)')
            plt.xlabel('NTGE')
            plt.ylabel('Count')
        
        # Plot 4: Performance by hyperparameters
        plt.subplot(2, 2, 4)
        lr_values = [float(r['config'].get('lr', 0)) for r in successful_results if 'lr' in r['config']]
        final_ranks = [r['final_rank'] for r in successful_results if 'lr' in r['config']]
        if lr_values and final_ranks:
            plt.scatter(lr_values, final_ranks)
            plt.xlabel('Learning Rate')
            plt.ylabel('Final Rank')
            plt.title('Performance vs Learning Rate')
            plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "analysis_plots.png"), dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    print(f"/nDetailed results saved to: {output_dir}")
    print(f"Summary CSV: {os.path.join(output_dir, 'model_analysis_summary.csv')}")

def main():
    # Configuration
    dataset = "CHES_2025"
    leakage = "HW"
    
    # Set random seeds
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset (keeping the "do not touch" sections)
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID",
                                   transform=transforms.Compose([ToTensor_trace()]))
    
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

    # Prepare test data
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    
    print(f"Correct key: {correct_key}")
    print(f"Test data shape: {X_attack.shape}")
    print(f"Number of classes: {classes}")

    # Find all models
    print("/nSearching for trained models...")
    models_info = find_all_models()
    
    if not models_info:
        print("No trained models found in ./Result/ directory")
        return
    
    print(f"Found {len(models_info)} models to analyze")
    
    # Group by model type
    for model_type in set(m['model_type'] for m in models_info):
        count = sum(1 for m in models_info if m['model_type'] == model_type)
        print(f"  {model_type}: {count} models")
    
    # Analyze all models
    print(f"/n{'='*80}")
    print("STARTING COMPREHENSIVE ANALYSIS")
    print(f"{'='*80}")
    
    results = []
    
    for i, model_info in enumerate(models_info):
        print(f"/nAnalyzing model {i+1}/{len(models_info)}: {os.path.basename(model_info['model_path'])}")
        print(f"Type: {model_info['model_type']}, Dataset: {model_info['dataset']}, Leakage: {model_info['leakage']}")
        
        result = analyze_single_model(
            model_info['model_path'], 
            model_info['config_path'],
            model_info['model_type'],
            model_info['dataset'],
            model_info['leakage'],
            device,
            X_attack, plt_attack, correct_key, leakage_fn, classes, num_sample_pts
        )
        
        if result:
            print(f"  Final rank: {result['final_rank']}, NTGE: {result['NTGE']}, Success: {result['success']}")
            results.append(result)
        else:
            print(f"  Analysis failed")
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Create comprehensive report
    create_analysis_report(results)

if __name__ == "__main__":
    main()
