import os
import random
import glob
import re
from copy import deepcopy
import numpy as np
import torch

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN, CNN_LSTM_Light
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
                
                    # Better model type detection
                    if "cnn_lstm_light" in dir_name:
                        model_type = "cnn_lstm_light"
                    elif "mlp" in dir_name:
                        model_type = "mlp"
                    elif "cnn" in dir_name:
                        model_type = "cnn"
                    else:
                        print(f"⚠️ Unknown model type in directory: {dir_name}")
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
    leakage = "HW"
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
            ##TODO: Load your model (note, you have to create your model in this file and new function should be in this file.) ########################
            ############## Load model based on type ############################################
            model_type = model_info['model_type']
            config = np.load(model_info['config_path'], allow_pickle=True).item()
            
            print(f"🔧 Model type: {model_type}")
            print(f"📋 Config keys: {list(config.keys())}")
            
            # Create model instance based on type
            if model_type == "mlp":
                model = MLP(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn":
                model = CNN(config, num_sample_pts, classes).to(device)
            elif model_type == "cnn_lstm_light":
                model = CNN_LSTM_Light(config, num_sample_pts, classes).to(device)
            else:
                print(f"❌ Unsupported model type: {model_type}")
                continue
                
            # Load model weights
            model.load_state_dict(torch.load(model_info['model_path'], map_location=device))
            model.eval()
            
            print(f"✅ Model loaded: {model_type}")
            print(f"📁 Config: {model_info['config_path']}")
            print(f"🧠 Weights: {model_info['model_path']}")
            print(f"🔢 Parameters: {model.number_of_parameters():,}")
            ###############################################################################

            ####All model will be evaluated based on this function, if it does not adhere to the following, it will be eliminated. ##################
            print(f"📊 Evaluating model...")
            GE, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                                total_nb_traces_attacks=2000, nb_traces_attacks=1700)
            
            # Store results
            result = {
                'model_idx': model_info['model_idx'],
                'model_type': model_type,
                'dir_name': model_info['dir_name'],
                'is_final': model_info['is_final'],
                'GE_final': GE[-1],
                'NTGE': NTGE,
                'success': True
            }
            all_results.append(result)
            
            print(f"✅ Evaluation completed!")
            print(f"📈 Final GE: {GE[-1]:.2f}")
            print(f"🎯 NTGE: {NTGE}")
            
            # Clean up GPU memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Error processing model: {str(e)}")
            print(f"📋 Config content: {config if 'config' in locals() else 'Could not load config'}")
            result = {
                'model_idx': model_info['model_idx'],
                'model_type': model_info['model_type'],
                'dir_name': model_info['dir_name'],
                'is_final': model_info['is_final'],
                'GE_final': None,
                'NTGE': None,
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
    
    if successful_results:
        print(f"\n✅ SUCCESSFUL MODELS:")
        print(f"{'Model':<25} {'Type':<15} {'Final':<7} {'GE':<10} {'NTGE'}")
        print("-" * 70)
        
        # Sort by NTGE (lower is better)
        successful_results.sort(key=lambda x: x['NTGE'] if x['NTGE'] is not None else float('inf'))
        
        for result in successful_results:
            model_name = f"model_{result['model_idx']}"
            model_type = result['model_type']
            is_final = "Yes" if result['is_final'] else "No"
            ge = f"{result['GE_final']:.2f}"
            ntge = f"{result['NTGE']}"
            
            print(f"{model_name:<25} {model_type:<15} {is_final:<7} {ge:<10} {ntge}")
        
        # Find best model
        best_model = min(successful_results, key=lambda x: x['NTGE'])
        print(f"\n🏆 BEST MODEL:")
        print(f"   Model {best_model['model_idx']} ({'Final' if best_model['is_final'] else 'Regular'})")
        print(f"   Type: {best_model['model_type']}")
        print(f"   NTGE: {best_model['NTGE']}")
        print(f"   Final GE: {best_model['GE_final']:.2f}")
    
    if failed_results:
        print(f"\n❌ FAILED MODELS:")
        for result in failed_results:
            print(f"   Model {result['model_idx']} ({result['model_type']}): {result.get('error', 'Unknown error')}")
    
    print(f"\n🎉 Analysis completed!")
