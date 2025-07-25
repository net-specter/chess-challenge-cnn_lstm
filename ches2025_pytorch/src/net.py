import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(MLP, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]

        self.layers = nn.ModuleList()

        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(num_sample_pts, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))

            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x



class CNN(nn.Module):
    def __init__(self, search_space,num_sample_pts, classes):
        super(CNN, self).__init__()
        self.num_layers = search_space["layers"]
        self.neurons = search_space["neurons"]
        self.activation = search_space["activation"]
        self.conv_layers = search_space["conv_layers"]

        self.layers = nn.ModuleList()
        #CNN
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)
        num_features = num_sample_pts
        for layer_index in range(0, self.conv_layers):
            #Convolution layer
            new_out_channels = self.filters[layer_index]
            if layer_index == 0:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features,kernel_size = self.kernels[layer_index], stride = self.kernels[layer_index], padding = self.paddings[layer_index])
                if new_num_features <=0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=1, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))

            else:
                conv1d_kernel = self.kernels[layer_index]
                conv1d_stride = self.kernels[layer_index]
                new_num_features = cal_num_features_conv1d(num_features, kernel_size=self.kernels[layer_index],
                                                       stride=self.kernels[layer_index],
                                                       padding=self.paddings[layer_index])
                if new_num_features <= 0:
                    conv1d_kernel = 1
                    conv1d_stride = 1
                    new_num_features = cal_num_features_conv1d(num_features, kernel_size=1,
                                                               stride=1,
                                                               padding=self.paddings[layer_index])
                num_features = new_num_features
                self.layers.append(nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, kernel_size=conv1d_kernel,
                                             stride=conv1d_stride, padding=self.paddings[layer_index]))
            #Activation Function
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
            #Pooling Layer
            if self.pooling_type[layer_index] == "max_pool":
                layer_pool_size = self.pooling_sizes[layer_index]
                layer_pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_maxpool1d(num_features, layer_pool_size, layer_pool_stride)

                if new_num_features <= 0:
                    layer_pool_size = 1
                    layer_pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.MaxPool1d(kernel_size=layer_pool_size, stride=layer_pool_stride))
            elif self.pooling_type[layer_index] == "average_pool":
                pool_size = self.pooling_sizes[layer_index]
                pool_stride = self.pooling_strides[layer_index]
                new_num_features = cal_num_features_avgpool1d(num_features, pool_size, pool_stride)
                if new_num_features <= 0:
                    pool_size = 1
                    pool_stride = 1
                    new_num_features = cal_num_features_maxpool1d(num_features, 1, 1)
                num_features = new_num_features
                self.layers.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            #BatchNorm
            self.layers.append(nn.BatchNorm1d(new_out_channels))
            prev_out_channels = new_out_channels
        #MLP
        self.layers.append(nn.Flatten())
        #Flatten
        flatten_neurons =prev_out_channels*num_features
        for layer_index in range(0, self.num_layers):
            if layer_index == 0:
                self.layers.append(nn.Linear(flatten_neurons, self.neurons))
            else:
                self.layers.append(nn.Linear(self.neurons, self.neurons))
            #Activation layer
            if self.activation == 'relu':
                self.layers.append(nn.ReLU())
            elif self.activation == 'selu':
                self.layers.append(nn.SELU())
            elif self.activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif self.activation == 'elu':
                self.layers.append(nn.ELU())
        self.softmax_layer = nn.Linear(self.neurons, classes)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.softmax_layer(x) #F.softmax()
        x = x.squeeze(1)
        return x


class DeepCNN_GlobalPool(nn.Module):
    """
    Deep CNN with Global Pooling. Extracts features with multiple 1D CNN layers,
    then applies global pooling to summarize features across the trace length,
    followed by a final MLP classifier.
    """
    def __init__(self, search_space, num_sample_pts, classes):
        super(DeepCNN_GlobalPool, self).__init__()
        
        self.conv_layers_count = search_space["conv_layers"] # Number of CNN layers
        self.activation = search_space["activation"] # Activation function for CNN and FC parts
        self.dropout_rate = search_space.get("dropout_rate", 0.0) # Dropout rate
        self.fc_hidden = search_space["fc_hidden_size"] # FIX: Use correct key for FC hidden size
        self.kernel_initializer_type = search_space["kernel_initializer"] # For custom initialization
        
        # Get CNN hyperparameters from helper function
        self.kernels, self.strides, self.filters, self.pooling_type, self.pooling_sizes, self.pooling_strides, self.paddings = create_cnn_hp(search_space)

        self.cnn_blocks = nn.ModuleList() # To hold CNN layers and their components
        current_num_features = num_sample_pts # Starting trace length
        prev_out_channels = 1 # Input channel for the first conv layer

        for layer_index in range(0, self.conv_layers_count):
            new_out_channels = self.filters[layer_index]
            conv1d_kernel = self.kernels[layer_index]
            conv1d_stride = self.strides[layer_index] # Correctly use stride from create_cnn_hp
            padding = self.paddings[layer_index]

            # Calculate output features after convolution, with robustness checks
            next_num_features = cal_num_features_conv1d(current_num_features, kernel_size=conv1d_kernel, stride=conv1d_stride, padding=padding)
            if next_num_features <= 0:
                print(f"Warning: Calculated conv features <= 0 ({next_num_features}) at layer {layer_index}. Adjusting parameters.")
                conv1d_kernel = max(1, current_num_features // 2 if current_num_features > 0 else 1)
                conv1d_stride = max(1, conv1d_kernel // 2)
                next_num_features = cal_num_features_conv1d(current_num_features, kernel_size=conv1d_kernel, stride=conv1d_stride, padding=padding)
                if next_num_features <=0: raise ValueError(f"CNN layer {layer_index} features collapsed to 0 even after adjustment.")

            # Add Conv -> BatchNorm -> Activation
            self.cnn_blocks.append(
                nn.Conv1d(in_channels=prev_out_channels, out_channels=new_out_channels, 
                          kernel_size=conv1d_kernel, stride=conv1d_stride, padding=padding, bias=False)
            )
            self.cnn_blocks.append(nn.BatchNorm1d(new_out_channels))
            if self.activation == 'relu': self.cnn_blocks.append(nn.ReLU(inplace=True))
            elif self.activation == 'selu': self.cnn_blocks.append(nn.SELU(inplace=True))
            elif self.activation == 'tanh': self.cnn_blocks.append(nn.Tanh())
            elif self.activation == 'elu': self.cnn_blocks.append(nn.ELU(inplace=True))
            elif self.activation == 'leaky_relu': self.cnn_blocks.append(nn.LeakyReLU(inplace=True)) # Added LeakyReLU

            # Add Pooling Layer
            current_pool_type = self.pooling_type[layer_index]
            pool_size = self.pooling_sizes[layer_index]
            pool_stride = self.pooling_strides[layer_index]

            next_pool_features = current_num_features
            if current_pool_type == "max_pool":
                next_pool_features = cal_num_features_maxpool1d(current_num_features, pool_size, pool_stride)
                if next_pool_features <= 0: # Robustness check
                    print(f"Warning: Pooled features <= 0 ({next_pool_features}) at layer {layer_index}. Adjusting pool params.")
                    pool_size = max(1, current_num_features // 2 if current_num_features > 0 else 1)
                    pool_stride = max(1, pool_size // 2)
                    next_pool_features = cal_num_features_maxpool1d(current_num_features, pool_size, pool_stride)
                    if next_pool_features <=0: raise ValueError(f"Pool layer {layer_index} features collapsed to 0 even after adjustment.")
                self.cnn_blocks.append(nn.MaxPool1d(kernel_size=pool_size, stride=pool_stride))
            elif current_pool_type == "average_pool": # Added average pool type
                next_pool_features = cal_num_features_avgpool1d(current_num_features, pool_size, pool_stride)
                if next_pool_features <= 0: # Robustness check
                    print(f"Warning: Pooled features <= 0 ({next_pool_features}) at layer {layer_index}. Adjusting pool params.")
                    pool_size = max(1, current_num_features // 2 if current_num_features > 0 else 1)
                    pool_stride = max(1, pool_size // 2)
                    next_pool_features = cal_num_features_avgpool1d(current_num_features, pool_size, pool_stride)
                    if next_pool_features <=0: raise ValueError(f"Pool layer {layer_index} features collapsed to 0 even after adjustment.")
                self.cnn_blocks.append(nn.AvgPool1d(kernel_size=pool_size, stride=pool_stride))
            
            current_num_features = next_pool_features

            if self.dropout_rate > 0: # Add dropout after each CNN block
                self.cnn_blocks.append(nn.Dropout(self.dropout_rate))

            prev_out_channels = new_out_channels

        # Global Pooling Layer: Summarizes features across the entire length dimension.
        # Output will be (batch_size, prev_out_channels, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Using Global Average Pooling (or AdaptiveMaxPool1d)

        # Classifier (Fully Connected Layers)
        # Input to the first FC layer is the number of channels from the last CNN layer.
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate), # Dropout before first FC layer
            nn.Linear(prev_out_channels, self.fc_hidden), # First FC layer
            nn.ReLU(inplace=True), # Activation for hidden FC layer
            nn.Dropout(self.dropout_rate), # Dropout before second FC layer
            nn.Linear(self.fc_hidden, classes) # Output layer, maps to number of classes
        )
        
        # Initialize weights for all layers in the model
        self._initialize_weights(self.kernel_initializer_type)


    def _initialize_weights(self, init_type):
        """Initializes model weights based on type (Kaiming/Xavier) for Conv, BN, Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                if init_type == 'xavier_uniform_':
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                elif init_type == 'he_uniform':
                    nn.init.kaiming_uniform_(m.weight)
                elif init_type == 'random_uniform':
                    nn.init.uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def number_of_parameters(self):
        return (sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        # Ensure input is 3D: (batch_size, 1, num_sample_pts) for Conv1d
        if x.dim() == 2: x = x.unsqueeze(1)
        elif x.dim() == 4: x = x.squeeze(2)

        # Pass through CNN blocks
        for layer in self.cnn_blocks:
            x = layer(x)
            # Check for collapsed dimensions during forward pass
            if x.size(-1) <= 0:
                raise ValueError(f"DeepCNN_GlobalPool: CNN output length collapsed during forward pass after {layer}. Output size: {x.size()}")
        
        x = self.global_pool(x) # Apply Global Average Pooling: (batch, channels, length) -> (batch, channels, 1)
        x = x.squeeze(-1) # Remove the last dimension: (batch, channels)
        
        x = self.classifier(x) # Pass through the final classifier
        return x # Return logits


# --- Helper functions for calculating output sizes after conv/pool layers ---
# (These remain the same as they are standard calculations)
def cal_num_features_conv1d(n_sample_points,kernel_size, stride,padding = 0, dilation = 1):
        L_in = n_sample_points
        L_out = math.floor(((L_in +(2*padding) - dilation *(kernel_size -1 )-1)/stride )+1)
        return L_out

def cal_num_features_maxpool1d(n_sample_points, kernel_size, stride, padding=0, dilation=1):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - dilation * (kernel_size - 1) - 1) / stride) + 1)
    return L_out

def cal_num_features_avgpool1d(n_sample_points,kernel_size, stride, padding = 0):
    L_in = n_sample_points
    L_out = math.floor(((L_in + (2 * padding) - kernel_size ) / stride) + 1)
    return L_out


def create_cnn_hp(search_space):
    """
    Generates specific CNN hyperparameters (kernels, strides, filters, pooling)
    for a single CNN model instance based on the provided search space configuration.
    This function implements a scaling strategy for filters, kernels, and strides with depth.
    """
    pooling_type = search_space["pooling_types"]
    pool_size = search_space["pooling_sizes"] 
    conv_layers = search_space["conv_layers"]
    init_filters = search_space["filters"]
    init_kernels = search_space["kernels"] 
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
        # Cap filters at a reasonable maximum to prevent excessive memory/computation.
        if current_filters > 512: current_filters = 512 
        filters.append(current_filters)

        # Kernel and stride scaling: often decrease kernel/stride size with depth
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
    Args:
        m (torch.nn.Module): The module (layer) to initialize.
        type (str): Type of initialization ('kaiming_uniform_', 'xavier_uniform_', 'random_uniform', 'he_uniform').
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type == 'xavier_uniform_':
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu')) 
        elif type == 'he_uniform':
            nn.init.kaiming_uniform_(m.weight)
        elif type == 'random_uniform':
            nn.init.uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)




def create_hyperparameter_space(model_type):
    if model_type == "mlp":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                                   "lr": random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                                    "optimizer": random.choice( ["RMSprop", "Adam"]),
                                                    "layers": random.randrange(1, 8, 1),
                                                    "neurons": random.choice( [10, 20, 50, 100, 200, 300, 400, 500]),
                                                    "activation": random.choice(  ["relu", "selu", "elu", "tanh"]),
                                                    "kernel_initializer": random.choice(["random_uniform", "glorot_uniform", "he_uniform"]),
                                                }
        return search_space
    elif model_type == "cnn":
        search_space = {"batch_size": random.randrange(100, 1001, 100),
                                              "lr":random.choice( [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),  # 1e-3, 5e-3, 1e-4, 5e-4
                                              "optimizer":random.choice(["RMSprop", "Adam"]),
                                              "layers": random.randrange(1, 4, 1), # Reduced range for layers
                                              "neurons": random.choice( [50, 100, 200, 300]), #Reduced range for neurons
                                              "activation": random.choice( ["relu", "selu", "elu", "tanh"]),
                                              "kernel_initializer": random.choice( ["random_uniform", "glorot_uniform", "he_uniform"]),
                                              "pooling_types": random.choice(["max_pool", "average_pool"]),
                                              "pooling_sizes":random.choice(  [2,4,6]), # Reduced range for pooling sizes
                                              "conv_layers": random.choice( [1,2,3]), # Reduced range for conv layers
                                              "filters": random.choice( [8,12,16,24]), #Added 24
                                                "kernels": random.choice( [i for i in range(10,30,2)]), # Reduced range for kernels                                              "kernels": random.choice( [i for i in range(10,30,2)]), # Reduced range for kernels
                                              "padding": random.choice(  [0,4,8]), # Reduced range for padding
                                        }

        return search_space
    elif model_type == "deepcnn_globalpool":
        search_space = {
            "batch_size": random.choice([32, 64, 128,256, 512]),
            "lr": random.choice([1e-5, 5e-6, 1e-6, 5e-7, 1e-7]),
            "optimizer": random.choice(["AdamW", "RAdam", "Adam", "SGD"]),
            "weight_decay": random.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2]), 
            "dropout_rate": random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            
            "conv_layers": random.choice([2, 3, 4]),
            "filters": random.choice([32, 64, 96]), 
            "kernels": random.choice([5, 7, 9, 11]), 
            "strides": random.choice([1, 2]),
            "padding": random.choice([0, 1]),
            
            "pooling_types": random.choice(["max_pool", "average_pool"]),
            "pooling_sizes": random.choice([2, 3]),
            "fc_hidden_size": random.choice([128, 256]),
            "activation": random.choice(["relu", "leaky_relu", "elu", "tanh"]),
            "kernel_initializer": random.choice(["he_uniform", "glorot_uniform"]),
        }
        return search_space