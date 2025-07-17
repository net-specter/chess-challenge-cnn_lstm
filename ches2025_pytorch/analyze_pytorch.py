import os
import random
from copy import deepcopy
import numpy as np
import torch

from torchvision.transforms import transforms
from src.dataloader import ToTensor_trace, Custom_Dataset
from src.net import create_hyperparameter_space, MLP, CNN
from src.trainer import trainer
from src.utils import evaluate, AES_Sbox, calculate_HW,evaluate_fast

if __name__=="__main__":
    dataset = "CHES_2025"
    leakage = "HW"
    nb_traces_attacks = 100000  # Reduced to match available data
    total_nb_traces_attacks = 100000  # Match actual dataset size

    seed = 0
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
    
    # Check actual dataset size and adjust parameters
    actual_traces = X_attack.shape[0]
    print(f"Actual number of test traces available: {actual_traces}")
    
    # Adjust parameters based on available data
    total_nb_traces_attacks = min(total_nb_traces_attacks, actual_traces)
    nb_traces_attacks = min(nb_traces_attacks, total_nb_traces_attacks)
    
    print(f"Using {total_nb_traces_attacks} total traces, {nb_traces_attacks} traces for attacks")
    #########################################################################


    ##TODO: Load your model (note, you have to create your model in this file and new function should be in this file.) ########################
    ############## Below is an example ############################################
    # Import CNN-LSTM models
    from src.cnn_lstm_sca import CNN_LSTM_SCA, CNN_LSTM_Light
    import torch.nn.functional as F
    
    def evaluate_cnn_lstm_batch(device, model, X_attack, plt_attack, correct_key, leakage_fn, 
                               nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks=1700, batch_size=50):
        """
        Custom evaluation function for CNN-LSTM models with batch processing to handle memory constraints
        """
        # Use CPU if GPU runs out of memory
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
                batch_size = 25  # Even smaller batches for CPU
                
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
    
    model_type = "cnn_lstm_light"  # Change to "cnn_lstm_sca" for full model
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    
    # Load model configuration
    config = np.load(model_root + "model_configuration_0.npy", allow_pickle=True).item()
    print(f"Loaded configuration: {config}")
    
    # Create and load CNN-LSTM model
    if model_type == "cnn_lstm_sca":
        model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
    else:
        model = CNN_LSTM_Light(config, num_sample_pts, classes).to(device)
    
    model.load_state_dict(torch.load(model_root + "model_0.pth", map_location=device))
    print(f"Model loaded from: {model_root}model_0.pth")
    ###############################################################################


    ####All model will be evaluated based on this function, if it does not adhere to the following, it will be eliminated. ##################
    GE, NTGE = evaluate_fast(device, model, X_attack, plt_attack, correct_key, leakage_fn=leakage_fn, nb_attacks=100,
                        total_nb_traces_attacks=total_nb_traces_attacks, nb_traces_attacks=nb_traces_attacks, batch_size=25)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Correct key: {correct_key}")
    print(f"Guessing Entropy (GE): {GE}")
    print(f"Number of Traces to Guess Entropy (NTGE): {NTGE}")
    
    if NTGE != float('inf'):
        print(f"Attack successful! NTGE = {NTGE}")
    else:
        print("Attack not successful within the given number of traces")
        
    # Show final GE rank
    if len(GE) > 0:
        final_rank = GE[-1]
        print(f"Final key rank: {final_rank}")
        if final_rank == 0:
            print("✅ Key recovery successful!")
        else:
            print(f"❌ Key not recovered. Final rank: {final_rank}")
    print(f"{'='*50}")

