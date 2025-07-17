import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms import transforms

from src.dataloader import ToTensor_trace, Custom_Dataset
from src.cnn_lstm_sca import CNN_LSTM_SCA, CNN_LSTM_Light, create_cnn_lstm_sca_config, create_cnn_lstm_light_config
from src.utils import AES_Sbox, calculate_HW, perform_attacks

def evaluate_cnn_lstm(device, model, X_attack, plt_attack, correct_key, leakage_fn, 
                     nb_attacks=100, total_nb_traces_attacks=2000, nb_traces_attacks=1700):
    """
    Custom evaluation function for CNN-LSTM models that handles input shapes properly
    """
    # Convert to tensor without adding extra dimensions that cause issues
    attack_traces = torch.from_numpy(X_attack[:total_nb_traces_attacks]).to(device).float()
    
    model.eval()
    with torch.no_grad():
        if isinstance(model, CNN_LSTM_SCA):
            predictions_wo_softmax = model(attack_traces)['output']
        else:
            predictions_wo_softmax = model(attack_traces)
        
        predictions = F.softmax(predictions_wo_softmax, dim=1)
        predictions = predictions.cpu().detach().numpy()
    
    # Use the existing attack performance evaluation
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

def analyze_attention_weights(model, x_sample, device):
    """
    Analyze and visualize attention weights for CNN-LSTM-SCA model
    """
    model.eval()
    
    with torch.no_grad():
        x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0).to(device)
        
        if isinstance(model, CNN_LSTM_SCA):
            outputs = model(x_tensor)
            
            # Extract attention weights
            spatial_weights = outputs['spatial_weights']
            channel_weights = outputs['channel_weights']
            lstm_attention = outputs['lstm_attention']
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot original trace
            axes[0, 0].plot(x_sample)
            axes[0, 0].set_title('Original Power Trace')
            axes[0, 0].set_xlabel('Time Samples')
            axes[0, 0].set_ylabel('Power')
            
            # Plot LSTM attention weights
            if lstm_attention is not None:
                attention_weights = lstm_attention.cpu().numpy().flatten()
                time_steps = np.arange(len(attention_weights))
                axes[0, 1].plot(time_steps, attention_weights)
                axes[0, 1].set_title('LSTM Attention Weights')
                axes[0, 1].set_xlabel('LSTM Time Steps')
                axes[0, 1].set_ylabel('Attention Weight')
            
            # Plot spatial attention for last CNN layer
            if spatial_weights and spatial_weights[-1] is not None:
                spatial_attn = spatial_weights[-1].cpu().numpy().squeeze()
                axes[1, 0].plot(spatial_attn)
                axes[1, 0].set_title(f'Spatial Attention (CNN Layer {len(spatial_weights)})')
                axes[1, 0].set_xlabel('Spatial Position')
                axes[1, 0].set_ylabel('Attention Weight')
            
            # Plot channel attention for last CNN layer
            if channel_weights and channel_weights[-1] is not None:
                channel_attn = channel_weights[-1].cpu().numpy().squeeze()
                channels = np.arange(len(channel_attn))
                axes[1, 1].bar(channels, channel_attn)
                axes[1, 1].set_title(f'Channel Attention (CNN Layer {len(channel_weights)})')
                axes[1, 1].set_xlabel('Channel')
                axes[1, 1].set_ylabel('Attention Weight')
            
            plt.tight_layout()
            return fig
        
        else:
            print("Attention analysis only available for CNN_LSTM_SCA model")
            return None

def main():
    # Configuration
    dataset = "CHES_2025"
    leakage = "HW"
    model_type = "cnn_lstm_light"  # Change this to test different models
    nb_traces_attacks = 500
    total_nb_traces_attacks = 1000

    # Set random seeds
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nb_attacks = 100

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

    # Prepare test data
    dataloadertest.split_attack_set_validation_test()
    dataloadertest.choose_phase("test")
    correct_key = dataloadertest.correct_key
    X_attack = dataloadertest.X_attack
    Y_attack = dataloadertest.Y_attack
    plt_attack = dataloadertest.plt_attack
    num_sample_pts = X_attack.shape[-1]

    print(f"Test data shape: {X_attack.shape}")
    print(f"Number of classes: {classes}")

    # Load trained model
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    model_root = save_root + "models/"
    
    try:
        # Load configuration
        config_path = model_root + "model_configuration_1.npy"
        if os.path.exists(config_path):
            config = np.load(config_path, allow_pickle=True).item()
            print(f"Loaded configuration: {config}")
        else:
            print(f"Configuration file not found at {config_path}")
            print("Using default configuration...")
            if model_type == "cnn_lstm_sca":
                config = create_cnn_lstm_sca_config()
            else:
                config = create_cnn_lstm_light_config()

        # Create and load model
        if model_type == "cnn_lstm_sca":
            model = CNN_LSTM_SCA(config, num_sample_pts, classes).to(device)
        else:
            model = CNN_LSTM_Light(config, num_sample_pts, classes).to(device)

        # Load trained weights
        model_path = model_root + "model_1.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Model loaded from: {model_path}")
        else:
            print(f"Model file not found at {model_path}")
            print("Please train the model first using train_cnn_lstm.py")
            return

        # Evaluate model
        print("Evaluating model...")
        GE, NTGE = evaluate_cnn_lstm(
            device, model, X_attack, plt_attack, correct_key, 
            leakage_fn=leakage_fn, nb_attacks=nb_attacks,
            total_nb_traces_attacks=total_nb_traces_attacks, 
            nb_traces_attacks=nb_traces_attacks
        )

        print(f"Guessing Entropy (GE): {GE}")
        print(f"Number of Traces to Guess Entropy (NTGE): {NTGE}")

        # Analyze attention weights for CNN-LSTM-SCA model
        if model_type == "cnn_lstm_sca":
            print("Analyzing attention weights...")
            
            # Select a random sample for attention analysis
            sample_idx = random.randint(0, len(X_attack) - 1)
            sample_trace = X_attack[sample_idx]
            
            fig = analyze_attention_weights(model, sample_trace, device)
            
            if fig is not None:
                # Save attention analysis
                attention_save_path = save_root + f"attention_analysis_sample_{sample_idx}.png"
                fig.savefig(attention_save_path, dpi=300, bbox_inches='tight')
                print(f"Attention analysis saved to: {attention_save_path}")
                plt.show()

        # Model complexity analysis
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nModel Statistics:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Memory usage estimation (rough)
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        print(f"Model size: {total_size / 1024**2:.2f} MB")

    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
