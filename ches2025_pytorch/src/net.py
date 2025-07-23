# src/net.py

import math
import random

import torch.nn as nn
import torch.nn.functional as F
# Import CNN_LSTM_SCA model
from .cnn_lstm_sca import CNN_LSTM_SCA


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model.
    Takes raw traces (or flattened features) as input and processes them through fully connected layers.
    """
    def __init__(self, search_space, num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"] # Number of hidden MLP layers
        self.neurons = search_space["neurons"] # Number of neurons per hidden layer
        self.activation = search_space["activation"]
        self.dropout_rate = search_space.get("dropout_rate", 0.0) # Dropout rate, defaults to 0.0

        self.layers = nn.ModuleList() # To store sequential layers

        for layer_index in range(0, self.num_layers):
            # Input dimension for the first layer is the trace length (num_sample_pts),
            # otherwise it's the number of neurons from the previous layer.
            input_dim = num_sample_pts if layer_index == 0 else self.neurons
            self.layers.append(nn.Linear(input_dim, self.neurons))

            # Add activation function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            elif self.activation == 'leaky_relu': # Added LeakyReLU for better performance
                self.layers.append(nn.LeakyReLU())

            # Apply dropout after activation, but not on the last hidden layer
            # (as it's often applied directly before the final classification layer).
            if self.dropout_rate > 0 and layer_index < self.num_layers -1:
                self.layers.append(nn.Dropout(self.dropout_rate))

        # Output layer for classification (maps to number of classes)
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        # Ensure input 'x' is 2D (batch_size, features) for MLP.
        # This handles inputs from DataLoader which might add channel dimensions.
        if x.dim() == 3: # If (batch, 1, num_sample_pts), squeeze channel
            x = x.squeeze(1)
        elif x.dim() == 4: # If (batch, 1, 1, num_sample_pts), squeeze channel and extra dim
            x = x.squeeze(1).squeeze(1)

        # Pass input through all defined layers
        for layer in self.layers:
            x = layer(x)
        return x # Return logits (raw scores), CrossEntropyLoss expects this


class CNN(nn.Module):
    """
    Simple Convolutional Neural Network (CNN) model for 1D data (traces).
    Combines 1D convolutional layers with pooling and batch normalization,
    followed by an MLP for classification.
    """
    def __init__(self, search_space, num_sample_pts, classes):
        super(CNN, self).__init__()
        self.num_mlp_layers = search_space["layers"] # Number of hidden MLP layers after CNN
        self.mlp_neurons = search_space["neurons"] # Number of neurons per hidden layer in MLP part
        self.activation = search_space["activation"] # Activation function for CNN and MLP
        self.conv_layers_count = search_space["conv_layers"] # Number of convolutional layers
        self.dropout_rate = search_space.get("dropout_rate", 0.0) # Dropout rate

        self.layers = nn.ModuleList() # To hold all CNN and MLP layers

        # --- CNN Feature Extraction Part ---
        # Calls a helper function to generate specific CNN hyperparameters based on the search space.
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)

        current_num_features = num_sample_pts # Starting length of the trace (after POI cropping)
        prev_out_channels = 1 # Input channel for the first conv layer (1 for grayscale trace)

        for layer_index in range(0, self.conv_layers_count):
            new_out_channels = self.filters[layer_index]
            conv1d_kernel = self.kernels[layer_index]
            conv1d_stride = self.strides[layer_index] # Corrected: Use the stride calculated by create_cnn_hp

            # Calculate output features after convolution.
            # This is important to ensure the dimensions are valid before creating the layer.
            next_num_features = cal_num_features_conv1d(current_num_features, kernel_size=conv1d_kernel,
                                                       stride=conv1d_stride, padding=self.paddings[layer_index])

            # --- Robustness against invalid CNN configurations (dimensions collapse) ---
            # If the calculated output length is zero or negative, it means the kernel/stride is too aggressive.
            if next_num_features <=0:
                print(f"Warning: Calculated features <= 0 ({next_num_features}) before pooling at CNN layer {layer_index}. Adjusting kernel/stride.")
                # Attempt to find a smaller, valid kernel/stride to avoid crash.
                # This ensures the model can at least be instantiated for hyperparameter search trials.
                conv1d_kernel = max(1, min(conv1d_kernel, current_num_features // 2 if current_num_features > 0 else 1))
                conv1d_stride = max(1, min(conv1d_stride, current_num_features // 2 if current_num_features > 0 else 1))
                next_num_features = cal_num_features_conv1d(current_num_features, kernel_size=conv1d_kernel,
                                                           stride=conv1d_stride, padding=self.paddings[layer_index])
                if next_num_features <=0: # Fallback if adjustment still leads to 0
                    next_num_features = 1
                    print(f"Warning: Forced next_num_features to 1 for layer {layer_index}. This config will likely perform poorly.")

            current_num_features = next_num_features # Update length for the next layer

            # Add Convolutional Layer
            self.layers.append(nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels,
                                         kernel_size=conv1d_kernel, stride=conv1d_stride,
                                         padding=self.paddings[layer_index]))

            # Add Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            elif self.activation == 'leaky_relu': # Added LeakyReLU for better performance
                self.layers.append(nn.LeakyReLU())

            # Add Pooling Layer
            current_pool_type = self.pooling_type[layer_index] # Get pooling type for this layer
            pool_size = self.pooling_sizes[layer_index]
            pool_stride = self.pooling_strides[layer_index]

            # Calculate output features after pooling
            if current_pool_type == "max_pool":
                next_num_features = cal_num_features_maxpool1d(current_num_features, pool_size, pool_stride)
                if next_num_features <= 0: # Robustness check
                    print(f"Warning: Pooled features <= 0 ({next_num_features}) at CNN layer {layer_index}. Adjusting pool size/stride.")
                    pool_size = max(1, min(pool_size, current_num_features // 2 if current_num_features > 0 else 1))
                    pool_stride = max(1, min(pool_stride, current_num_features // 2 if current_num_features > 0 else 1))
                    next_num_features = cal_num_features_maxpool1d(current_num_features, pool_size, pool_stride)
                    if next_num_features <=0: next_num_features = 1
                current_num_features = next_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))
            elif current_pool_type == "average_pool":
                next_num_features = cal_num_features_avgpool1d(current_num_features, pool_size, pool_stride)
                if next_num_features <= 0: # Robustness check
                    print(f"Warning: Pooled features <= 0 ({next_num_features}) at CNN layer {layer_index}. Adjusting pool size/stride.")
                    pool_size = max(1, min(pool_size, current_num_features // 2 if current_num_features > 0 else 1))
                    pool_stride = max(1, min(pool_stride, current_num_features // 2 if current_num_features > 0 else 1))
                    next_num_features = cal_num_features_avgpool1d(current_num_features, pool_size, pool_stride)
                    if next_num_features <=0: next_num_features = 1
                current_num_features = next_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))

            # Add Batch Normalization
            self.layers.append(nn.BatchNorm1d(new_out_channels))

            # Add Dropout after each CNN block if specified (for regularization)
            if self.dropout_rate > 0:
                self.layers.append(nn.Dropout(self.dropout_rate))

            prev_out_channels = new_out_channels # Update channels for the next layer

        # --- MLP Part after CNN Features ---
        self.layers.append(nn.Flatten()) # Flatten the 3D CNN output (batch, channels, length) to 2D (batch, features)

        # Calculate input neurons for MLP after flattening
        flatten_neurons = prev_out_channels * current_num_features
        if flatten_neurons <= 0:
            raise ValueError(f"Flattened features input to MLP is zero or negative ({flatten_neurons}). "
                             f"This implies CNN layers are too aggressive or initial trace length is too short.")

        for layer_index in range(0, self.num_mlp_layers): # Iterate for MLP hidden layers
            input_dim_mlp = flatten_neurons if layer_index == 0 else self.mlp_neurons
            self.layers.append(nn.Linear(input_dim_mlp, self.mlp_neurons))

            # Add Activation layer
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            elif self.activation == 'leaky_relu':
                self.layers.append(nn.LeakyReLU())

            # Add Dropout to MLP hidden layers if specified
            if self.dropout_rate > 0 and layer_index < self.num_mlp_layers - 1:
                self.layers.append(nn.Dropout(self.dropout_rate))

        # Output layer for classification
        self.softmax_layer = nn.Linear(self.mlp_neurons, classes)

    def number_of_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        # Ensure input is 3D: (batch_size, 1, num_sample_pts) for Conv1d
        if x.dim() == 2: # If (batch, num_sample_pts), add channel dim
            x = x.unsqueeze(1)
        elif x.dim() == 4: # If (batch, 1, 1, num_sample_pts), squeeze extra dim
            x = x.squeeze(2)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        return x # Return logits


# --- Helper functions for calculating output sizes after conv/pool layers ---
def cal_num_features_conv1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    """Calculates output length after a 1D convolution."""
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    """Calculates output length after a 1D max pooling."""
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_avgpool1d(n_sample_points, kernel_size, stride, padding=0):
    """Calculates output length after a 1D average pooling."""
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    """
    Generates specific CNN hyperparameters (kernels, strides, filters, pooling)
    for a single CNN model instance based on the provided search space config.
    This function scales hyperparameters for deeper layers.
    """
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"]
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"] # Initial number of filters for the first layer
    init_kernels = search_space["kernels"] # Initial kernel size for the first layer
    init_padding = search_space["padding"]

    kernels = []
    strides = []
    filters = []
    paddings = []
    pooling_types = []
    pooling_sizes = []
    pooling_strides = []

    for layer_idx in range(conv_layers): # Iterate for each convolutional layer
        # Filter scaling: typically doubles channels with depth for more complex feature extraction
        current_filters = init_filters * (2 ** layer_idx)
        # Cap filters at a reasonable maximum to prevent excessive memory/computation
        if current_filters > 512: current_filters = 512
        filters.append(current_filters)

        # Kernel and stride scaling: often decrease kernel/stride size with depth
        # to focus on finer details or ensure dimensions don't collapse too quickly.
        current_kernel = max(3, init_kernels // (2 ** layer_idx)) # Minimum kernel size 3
        current_stride = max(1, current_kernel // 2) # Stride is often kernel_size / 2 for overlapping convolutions

        kernels.append(current_kernel)
        strides.append(current_stride)
        paddings.append(init_padding)

        pooling_sizes.append(pool_size)
        pooling_strides.append(pool_size)
        pooling_types.append(pooling_type)

    return kernels, strides, filters, pooling_type, pooling_sizes, pooling_strides, paddings


def weight_init(m, type='kaiming_uniform_'):
    """
    Initializes weights of convolutional and linear layers using specified methods.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            # Xavier (Glorot) initialization, good for tanh/sigmoid, but also usable with ReLU (with 'relu' gain)
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        elif type == 'he_uniform':
            # Kaiming (He) initialization, generally preferred for ReLU and its variants
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            # Simple uniform random initialization
            nn.init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias) # Initialize biases to zero


def create_hyperparameter_space(model_type):
    """
    Defines the hyperparameter search space for different model types.
    This function is used by the random search process to sample configurations.
    """
    if model_type == "mlp":
        search_space = {
            "batch_size": random.choice([64, 128, 256, 512]), # Common batch sizes, powers of 2
            "lr": random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]), # Learning rates
            "optimizer": random.choice(["AdamW", "Adam", "RMSprop"]), # Optimizer choices
            "weight_decay": random.choice([0.0, 1e-5, 1e-4, 1e-3]), # L2 regularization strength
            "dropout_rate": random.choice([0.0, 0.1, 0.2, 0.3, 0.4]), # Dropout rate for hidden layers
            "layers": random.randrange(2, 6), # Number of hidden MLP layers (2 to 5)
            "neurons": random.choice([256, 512, 768, 1024]), # Number of neurons per hidden layer
            "activation": random.choice(["relu", "selu", "elu", "tanh", "leaky_relu"]), # Activation functions
            "kernel_initializer": random.choice(["he_uniform", "glorot_uniform"]), # Weight initialization strategy
        }
        return search_space
    elif model_type == "cnn":
        search_space = {
            "batch_size": random.choice([64, 128, 256, 512]),
            "lr": random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
            "optimizer": random.choice(["AdamW", "RAdam", "Adam"]),
            "weight_decay": random.choice([0.0, 1e-5, 1e-4, 1e-3]),
            "dropout_rate": random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),

            "layers": random.randrange(1, 4), # Number of MLP hidden layers after CNN (1 to 3)
            "neurons": random.choice([128, 256, 512]), # Neurons in MLP part
            "activation": random.choice(["relu", "selu", "elu", "leaky_relu"]), # Activation for CNN/MLP
            "kernel_initializer": random.choice(["he_uniform", "glorot_uniform"]),

            "pooling_types": random.choice(["max_pool"]), # Max pooling generally preferred for feature robustness
            "pooling_sizes": random.choice([2, 3, 4]), # Kernel size for pooling
            "conv_layers": random.choice([2, 3, 4, 5]), # Number of convolutional layers
            "filters": random.choice([16, 32, 64, 96, 128]), # Initial number of filters (channels)
            "kernels": random.choice([11, 15, 21, 25, 31]), # Initial kernel size for convolutions
            "padding": random.choice([0, 1, 2, 3, 5]), # Padding for convolutions
        }
        return search_space
    elif model_type == "cnn_lstm_sca":
        # Define pre-defined matching CNN channel/kernel configurations as tuples or dicts.
        # This ensures that 'cnn_channels' and 'cnn_kernels' always have the same length,
        # preventing IndexError during model instantiation.
        cnn_arch_options = [
            {'cnn_channels': [16, 32, 64],          'cnn_kernels': [7, 5, 3]},
            {'cnn_channels': [32, 64, 128],         'cnn_kernels': [11, 7, 5]},
            {'cnn_channels': [32, 64, 128, 256],    'cnn_kernels': [11, 7, 5, 3]},
            {'cnn_channels': [64, 128, 256],        'cnn_kernels': [15, 11, 7]},
            {'cnn_channels': [64, 128, 256, 512],   'cnn_kernels': [9, 7, 5, 3]},
        ]

        # Select one of these matched CNN architecture configurations randomly for a trial.
        selected_cnn_arch = random.choice(cnn_arch_options)

        search_space = {
            "batch_size": random.choice([16, 32, 64, 128]), # Smaller batch sizes for memory-intensive models (with AMP)
            "lr": random.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]), # Learning rates
            "optimizer": random.choice(["AdamW", "RAdam", "Adam"]), # Optimizer choices (AdamW is often best)
            "weight_decay": random.choice([0.0, 1e-5, 1e-4]), # L2 regularization strength

            # Use the selected matching CNN architecture configuration
            "cnn_channels": selected_cnn_arch['cnn_channels'],
            "cnn_kernels": selected_cnn_arch['cnn_kernels'],

            "use_attention": random.choice([True, False]), # Whether to use Spatial and Channel Attention (True often better)

            "lstm_hidden_size": random.choice([32, 64, 128]), # LSTM hidden state size
            "lstm_num_layers": random.choice([1, 2]), # Number of stacked LSTM layers
            "lstm_dropout": random.choice([0.0, 0.1, 0.2]), # Dropout for LSTM layers (if num_layers > 1)
            "bidirectional": random.choice([True, False]), # Whether to use Bidirectional LSTM

            "fc_hidden": random.choice([128, 256]), # Hidden layer size in the final classifier MLP
            "dropout": random.choice([0.3, 0.4, 0.5]), # Dropout rate for the classifier MLP

            "kernel_initializer": random.choice(["he_uniform", "glorot_uniform"]), # Weight initialization
        }
        return search_space