import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def analyze_training_results():
    """
    Analyze the results from CNN-LSTM training
    """
    # Configuration
    dataset = "CHES_2025"
    model_type = "cnn_lstm_light"
    leakage = "HW"
    
    root = "./Result/"
    save_root = root + dataset + "_" + model_type + "_" + leakage + "/"
    
    # Load training summary
    summary_path = save_root + "training_summary.npy"
    
    if os.path.exists(summary_path):
        summary = np.load(summary_path, allow_pickle=True).item()
        
        print("="*60)
        print("CNN-LSTM TRAINING RESULTS ANALYSIS")
        print("="*60)
        
        print(f"Dataset: {summary['dataset']}")
        print(f"Model Type: {summary['model_type']}")
        print(f"Leakage Model: {summary['leakage']}")
        print(f"Number of models trained: {summary['num_models_trained']}")
        
        best_models = summary['best_models']
        
        print(f"\nTOP 10 BEST MODELS (by Guessing Entropy):")
        print("-" * 50)
        print(f"{'Rank':<5} {'Model':<8} {'GE':<8} {'NTGE':<8} {'Config'}")
        print("-" * 50)
        
        for i, model_info in enumerate(best_models[:10]):
            config_str = f"bs={model_info['config']['batch_size']}, lr={model_info['config']['lr']:.0e}, opt={model_info['config']['optimizer']}"
            print(f"{i+1:<5} {model_info['model_idx']:<8} {model_info['GE']:<8} {model_info['NTGE']:<8} {config_str}")
        
        # Statistics
        all_ges = [model['GE'] for model in best_models if model['GE'] != float('inf')]
        
        if all_ges:
            print(f"\nSTATISTICS:")
            print("-" * 30)
            print(f"Best GE: {min(all_ges)}")
            print(f"Worst GE: {max(all_ges)}")
            print(f"Average GE: {np.mean(all_ges):.2f}")
            print(f"Median GE: {np.median(all_ges):.2f}")
            print(f"Std GE: {np.std(all_ges):.2f}")
            
            # Count successful models (GE < 128)
            successful_models = [ge for ge in all_ges if ge < 128]
            print(f"Models with GE < 128: {len(successful_models)}/{len(all_ges)}")
            
            # Hyperparameter analysis
            print(f"\nHYPERPARAMETER ANALYSIS:")
            print("-" * 30)
            
            # Analyze batch sizes
            batch_sizes = {}
            learning_rates = {}
            optimizers = {}
            
            for model in best_models:
                if model['GE'] != float('inf'):
                    bs = model['config']['batch_size']
                    lr = model['config']['lr']
                    opt = model['config']['optimizer']
                    
                    if bs not in batch_sizes:
                        batch_sizes[bs] = []
                    batch_sizes[bs].append(model['GE'])
                    
                    if lr not in learning_rates:
                        learning_rates[lr] = []
                    learning_rates[lr].append(model['GE'])
                    
                    if opt not in optimizers:
                        optimizers[opt] = []
                    optimizers[opt].append(model['GE'])
            
            print("Average GE by Batch Size:")
            for bs, ges in sorted(batch_sizes.items()):
                print(f"  Batch size {bs}: {np.mean(ges):.2f} ± {np.std(ges):.2f} (n={len(ges)})")
            
            print("Average GE by Learning Rate:")
            for lr, ges in sorted(learning_rates.items()):
                print(f"  LR {lr:.0e}: {np.mean(ges):.2f} ± {np.std(ges):.2f} (n={len(ges)})")
            
            print("Average GE by Optimizer:")
            for opt, ges in sorted(optimizers.items()):
                print(f"  {opt}: {np.mean(ges):.2f} ± {np.std(ges):.2f} (n={len(ges)})")
            
            # Create visualization
            create_performance_plots(best_models, save_root)
        
        else:
            print("No successful models found!")
    
    else:
        print(f"Training summary not found at: {summary_path}")
        print("Please run the training script first.")
    
    # Check for individual model files
    model_root = save_root + "models/"
    if os.path.exists(model_root):
        model_files = [f for f in os.listdir(model_root) if f.endswith('.pth')]
        config_files = [f for f in os.listdir(model_root) if f.endswith('.npy')]
        
        print(f"\nMODEL FILES:")
        print("-" * 20)
        print(f"Model files (.pth): {len(model_files)}")
        print(f"Config files (.npy): {len(config_files)}")
        print(f"Models directory: {model_root}")
    
    # TensorBoard instructions
    print(f"\nTENSORBOARD:")
    print("-" * 20)
    print(f"To view training logs, run:")
    print(f"tensorboard --logdir=\"{save_root}tensorboard/\"")
    print(f"Then open: http://localhost:6006/")

def create_performance_plots(best_models, save_root):
    """
    Create visualization plots for model performance
    """
    try:
        import matplotlib.pyplot as plt
        
        # Filter out failed models
        valid_models = [model for model in best_models if model['GE'] != float('inf')]
        
        if len(valid_models) == 0:
            print("No valid models to plot.")
            return
        
        # Extract data
        model_indices = [model['model_idx'] for model in valid_models]
        ges = [model['GE'] for model in valid_models]
        batch_sizes = [model['config']['batch_size'] for model in valid_models]
        learning_rates = [model['config']['lr'] for model in valid_models]
        optimizers = [model['config']['optimizer'] for model in valid_models]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: GE vs Model Index
        axes[0, 0].scatter(model_indices, ges, alpha=0.6)
        axes[0, 0].set_xlabel('Model Index')
        axes[0, 0].set_ylabel('Guessing Entropy (GE)')
        axes[0, 0].set_title('Model Performance Overview')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: GE vs Batch Size
        unique_bs = sorted(set(batch_sizes))
        bs_ges = [np.mean([ges[i] for i in range(len(ges)) if batch_sizes[i] == bs]) for bs in unique_bs]
        bs_stds = [np.std([ges[i] for i in range(len(ges)) if batch_sizes[i] == bs]) for bs in unique_bs]
        
        axes[0, 1].errorbar(unique_bs, bs_ges, yerr=bs_stds, marker='o', capsize=5)
        axes[0, 1].set_xlabel('Batch Size')
        axes[0, 1].set_ylabel('Average GE')
        axes[0, 1].set_title('Performance vs Batch Size')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: GE vs Learning Rate
        unique_lr = sorted(set(learning_rates))
        lr_ges = [np.mean([ges[i] for i in range(len(ges)) if learning_rates[i] == lr]) for lr in unique_lr]
        lr_stds = [np.std([ges[i] for i in range(len(ges)) if learning_rates[i] == lr]) for lr in unique_lr]
        
        axes[1, 0].errorbar(range(len(unique_lr)), lr_ges, yerr=lr_stds, marker='o', capsize=5)
        axes[1, 0].set_xticks(range(len(unique_lr)))
        axes[1, 0].set_xticklabels([f'{lr:.0e}' for lr in unique_lr])
        axes[1, 0].set_xlabel('Learning Rate')
        axes[1, 0].set_ylabel('Average GE')
        axes[1, 0].set_title('Performance vs Learning Rate')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: GE Distribution by Optimizer
        adam_ges = [ges[i] for i in range(len(ges)) if optimizers[i] == 'Adam']
        rmsprop_ges = [ges[i] for i in range(len(ges)) if optimizers[i] == 'RMSprop']
        
        data_to_plot = []
        labels = []
        if adam_ges:
            data_to_plot.append(adam_ges)
            labels.append('Adam')
        if rmsprop_ges:
            data_to_plot.append(rmsprop_ges)
            labels.append('RMSprop')
        
        if data_to_plot:
            axes[1, 1].boxplot(data_to_plot, labels=labels)
            axes[1, 1].set_ylabel('Guessing Entropy (GE)')
            axes[1, 1].set_title('Performance Distribution by Optimizer')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = save_root + "performance_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nPerformance plots saved to: {plot_path}")
        
        # Show the plot
        plt.show()
        
    except ImportError:
        print("Matplotlib not available. Skipping plot generation.")
    except Exception as e:
        print(f"Error creating plots: {e}")

if __name__ == "__main__":
    analyze_training_results()
