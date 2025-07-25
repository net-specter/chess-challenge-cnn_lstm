import os
import random
import glob
import re
from copy import deepcopy
import numpy as np
import torch
import pandas as pd  # Add pandas import for CSV functionality
from datetime import datetime

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
from src.utils import evaluate_fast, AES_Sbox, calculate_HW


def find_all_models(result_root="./Result/"):
    """Find all trained models"""
    models_info = []
    
    for dir_name in os.listdir(result_root):
        dir_path = os.path.join(result_root, dir_name)
        if os.path.isdir(dir_path):
            models_dir = os.path.join(dir_path, "models")
            if os.path.exists(models_dir):
                # Extract model info from directory name
                parts = dir_name.split("_")
                if len(parts) >= 3:
                    dataset = parts[0]
                    leakage = parts[-1]
                
                    if "mlp" in dir_name:
                        model_type = "mlp"
                    elif "cnn" in dir_name:
                        model_type = "cnn"
                    else:
                        continue
                    
                    # Find model files (prioritize final models)
                    final_models = glob.glob(os.path.join(models_dir, "model_*_final.pth"))
                    regular_models = glob.glob(os.path.join(models_dir, "model_*.pth"))
                    regular_models = [f for f in regular_models if "_final" not in f]
                    
                    # Process final models first
                    for model_file in final_models:
                        filename = os.path.basename(model_file)
                        match = re.search(r'model_(\d+)_final\.pth', filename)
                        if match:
                            model_idx = int(match.group(1))
                            config_file = os.path.join(models_dir, f"model_configuration_{model_idx}_final.npy")
                            if os.path.exists(config_file):
                                models_info.append({
                                    'model_path': model_file,
                                    'config_path': config_file,
                                    'model_type': model_type,
                                    'dataset': dataset,
                                    'leakage': leakage,
                                    'model_idx': model_idx,
                                    'dir_name': dir_name,
                                    'is_final': True
                                })
                    
                    # Process regular models (only if no final version)
                    for model_file in regular_models:
                        filename = os.path.basename(model_file)
                        match = re.search(r'model_(\d+)\.pth', filename)
                        if match:
                            model_idx = int(match.group(1))
                            
                            # Check if final version exists
                            final_exists = any(info['model_idx'] == model_idx and info['is_final'] 
                                             for info in models_info if info['dir_name'] == dir_name)
                            
                            if not final_exists:
                                config_file = os.path.join(models_dir, f"model_configuration_{model_idx}.npy")
                                if os.path.exists(config_file):
                                    models_info.append({
                                        'model_path': model_file,
                                        'config_path': config_file,
                                        'model_type': model_type,
                                        'dataset': dataset,
                                        'leakage': leakage,
                                        'model_idx': model_idx,
                                        'dir_name': dir_name,
                                        'is_final': False
                                    })
    
    return models_info

if __name__=="__main__":
    dataset = "CHES_2025"
    leakage = "ID"
    nb_traces_attacks = 1700
    total_nb_traces_attacks = 2000

    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_attacks = 100

    ##################please do not touch this code below###################
    dataloadertest = Custom_Dataset(root='./../', dataset=dataset, leakage="ID", #change root to where you download your dataset.
                                                 transform=transforms.Compose([ToTensor_trace()]))
    #########################################################################
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
    else:
        ####TODO: You can change the code here if you want to create your own leakage model.
        pass

    ##################please do not touch this code here###################
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]
    #########################################################################

    # Find all trained models
    print("🔍 Finding all trained models...")
    models_info = find_all_models()
    
    if not models_info:
        print("❌ No trained models found!")
        exit()
    
    print(f"✅ Found {len(models_info)} trained models")
    print("=" * 80)
    
    # Store results
    all_results = []
    
    # Loop through all models
    for i, model_info in enumerate(models_info, 1):
        print(f"\n[{i}/{len(models_info)}] Analyzing: {model_info['dir_name']}/model_{model_info['model_idx']}{'_final' if model_info['is_final'] else ''}")
        print("-" * 60)
        
        try:
            model_type = model_info['model_type']
            config = np.load(model_info['config_path'], allow_pickle=True).item()
            
            # Try different data preprocessing configurations to match the trained model
            data_configs = [
                {'clean_data': False, 'desc': 'raw data'},
                {'clean_data': True, 'roi_extraction': True, 'roi_width': 200, 'desc': 'cleaned data with ROI'},
                {'clean_data': True, 'roi_extraction': False, 'desc': 'cleaned data without ROI'},
            ]
            
            model_loaded = False
            for data_config in data_configs:
                try:
                    print(f"🔄 Trying to load model with {data_config['desc']}...")
                    
                    # Create dataloader with specific configuration
                    dataloadertest_model = Custom_Dataset(
                        root='./../', 
                        dataset=dataset, 
                        leakage="ID",
                        transform=transforms.Compose([ToTensor_trace()]),
                        **{k: v for k, v in data_config.items() if k != 'desc'}
                    )
                    
                    if leakage == 'HW':
                        dataloadertest_model.Y_attack = calculate_HW(dataloadertest_model.Y_attack)
                    
                    dataloadertest_model.split_attack_set_validation_test()
                    dataloadertest_model.choose_phase("test")
                    
                    X_attack_model = dataloadertest_model.X_attack
                    plt_attack_model = dataloadertest_model.plt_attack
                    correct_key_model = dataloadertest_model.correct_key
                    num_sample_pts_model = X_attack_model.shape[-1]
                    
                    # Create model with current data dimensions
                    if model_type == "mlp":
                        model = MLP(config, num_sample_pts_model, classes).to(device)
                    elif model_type == "cnn":
                        model = CNN(config, num_sample_pts_model, classes).to(device)
                    else:
                        print(f"❌ Unsupported model type: {model_type}")
                        break
                    
                    # Try to load the model weights
                    model.load_state_dict(torch.load(model_info['model_path'], map_location=device))
                    model.eval()
                    
                    print(f"✅ Model loaded successfully with {data_config['desc']}")
                    print(f"📏 Input dimensions: {num_sample_pts_model}")
                    model_loaded = True
                    break
                    
                except (RuntimeError, KeyError) as e:
                    if "size mismatch" in str(e):
                        print(f"⚠️  Dimension mismatch with {data_config['desc']}")
                        continue
                    else:
                        raise e
            
            if not model_loaded:
                raise RuntimeError("Could not load model with any data configuration")
            
            print(f"📁 Config: {model_info['config_path']}")
            print(f"🧠 Weights: {model_info['model_path']}")

            # Evaluate model with the correct data preprocessing
            print(f"📊 Evaluating model...")
            GE, NTGE = evaluate_fast(device, model, X_attack_model, plt_attack_model, correct_key_model, 
                                   leakage_fn=leakage_fn, nb_attacks=100,
                                   total_nb_traces_attacks=min(2000, len(X_attack_model)), 
                                   nb_traces_attacks=min(1700, len(X_attack_model)))
            
            # Store results
            result = {
                'model_idx': model_info['model_idx'],
                'model_type': model_type,
                'dir_name': model_info['dir_name'],
                'is_final': model_info['is_final'],
                'GE_final': GE[-1],
                'NTGE': NTGE,
                'input_size': num_sample_pts_model,
                'data_config': data_config['desc'],
                'success': True
            }
            all_results.append(result)
            
            print(f"✅ Evaluation completed!")
            print(f"📈 Final GE: {GE[-1]:.2f}")
            print(f"🎯 NTGE: {NTGE}")
            
            # Clean up GPU memory
            del model
            del dataloadertest_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing model: {str(e)}")
            result = {
                'model_idx': model_info['model_idx'],
                'model_type': model_info['model_type'],
                'dir_name': model_info['dir_name'],
                'is_final': model_info['is_final'],
                'GE_final': None,
                'NTGE': None,
                'input_size': None,
                'data_config': None,
                'success': False,
                'error': str(e)
            }
            all_results.append(result)
            continue

    # Print final summary
    print(f"\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    successful_results = [r for r in all_results if r['success']]
    failed_results = [r for r in all_results if not r['success']]
    
    print(f"📊 Total models analyzed: {len(all_results)}")
    print(f"✅ Successful: {len(successful_results)}")
    print(f"❌ Failed: {len(failed_results)}")
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if all_results:
        # Create DataFrame from all results
        df_results = pd.DataFrame(all_results)
        
        # Create results directory if it doesn't exist
        results_dir = "./Result/analysis_results/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Save all results
        all_results_file = os.path.join(results_dir, f"all_models_analysis_{timestamp}.csv")
        df_results.to_csv(all_results_file, index=False)
        print(f"\n💾 All results saved to: {all_results_file}")
        
        # Save only successful results
        if successful_results:
            df_successful = pd.DataFrame(successful_results)
            # Sort by NTGE (lower is better)
            df_successful = df_successful.sort_values(['NTGE', 'GE_final'])
            
            successful_results_file = os.path.join(results_dir, f"successful_models_{timestamp}.csv")
            df_successful.to_csv(successful_results_file, index=False)
            print(f"💾 Successful results saved to: {successful_results_file}")
            
            # Create a summary CSV with key metrics
            summary_data = {
                'total_models': len(all_results),
                'successful_models': len(successful_results),
                'failed_models': len(failed_results),
                'best_NTGE': df_successful['NTGE'].min(),
                'best_GE': df_successful['GE_final'].min(),
                'best_model_idx': df_successful.iloc[0]['model_idx'],
                'best_model_type': df_successful.iloc[0]['model_type'],
                'best_model_config': df_successful.iloc[0]['data_config'],
                'analysis_timestamp': timestamp
            }
            
            summary_df = pd.DataFrame([summary_data])
            summary_file = os.path.join(results_dir, f"analysis_summary_{timestamp}.csv")
            summary_df.to_csv(summary_file, index=False)
            print(f"💾 Analysis summary saved to: {summary_file}")
        
        # Save failed results separately if any
        if failed_results:
            df_failed = pd.DataFrame(failed_results)
            failed_results_file = os.path.join(results_dir, f"failed_models_{timestamp}.csv")
            df_failed.to_csv(failed_results_file, index=False)
            print(f"💾 Failed results saved to: {failed_results_file}")
    
    if successful_results:
        print(f"\n✅ SUCCESSFUL MODELS:")
        print(f"{'Model':<25} {'Type':<10} {'Final':<6} {'Input':<6} {'Data Config':<20} {'GE':<8} {'NTGE'}")
        print("-" * 85)
        
        # Sort by NTGE (lower is better)
        successful_results.sort(key=lambda x: x['NTGE'] if x['NTGE'] is not None else float('inf'))
        
        for result in successful_results:
            model_name = f"model_{result['model_idx']}"
            model_type = result['model_type']
            is_final = "Yes" if result['is_final'] else "No"
            input_size = f"{result.get('input_size', 'N/A')}"
            data_config = result.get('data_config', 'N/A')[:18]
            ge = f"{result['GE_final']:.2f}"
            ntge = f"{result['NTGE']}"
            
            print(f"{model_name:<25} {model_type:<10} {is_final:<6} {input_size:<6} {data_config:<20} {ge:<8} {ntge}")

        # Find best model
        best_model = min(successful_results, key=lambda x: x['NTGE'])
        print(f"\n🏆 BEST MODEL:")
        print(f"   Model {best_model['model_idx']} ({'Final' if best_model['is_final'] else 'Regular'})")
        print(f"   Type: {best_model['model_type']}")
        print(f"   NTGE: {best_model['NTGE']}")
        print(f"   Final GE: {best_model['GE_final']:.2f}")
        print(f"   Data Config: {best_model['data_config']}")
        print(f"   Input Size: {best_model['input_size']}")
    
    if failed_results:
        print(f"\n❌ FAILED MODELS:")
        for result in failed_results:
            print(f"   Model {result['model_idx']} ({result['model_type']}): {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Analysis completed!")
    if all_results:
        print(f"📂 Results saved in: ./Result/analysis_results/")