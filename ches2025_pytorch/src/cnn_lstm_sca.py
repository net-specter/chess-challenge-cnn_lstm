# src/cnn_lstm_sca.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important temporal regions.
    Applies 1D convolutions to generate attention weights across the trace length.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        # First 1x1 Conv to reduce channels (e.g., from 64 to 8)
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        # Second 1x1 Conv to map back to 1 channel (attention score per position)
        self.conv2 = nn.Conv1d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid() # Sigmoid to produce attention weights between 0 and 1

    def forward(self, x):
        # x: (batch_size, channels, length)
        attention = self.conv1(x)
        attention = F.relu(attention) # ReLU non-linearity
        attention = self.conv2(attention)
        attention_weights = self.sigmoid(attention)  # (batch_size, 1, length)

        # Apply attention weights by element-wise multiplication.
        # This re-weights each position across all channels.
        attended_features = x * attention_weights
        return attended_features, attention_weights # Return attended features and the weights for analysis

class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism for focusing on important feature maps.
    Uses global average and max pooling to create channel descriptors, then FC layers.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Global average pooling across length dimension
        self.max_pool = nn.AdaptiveMaxPool1d(1) # Global max pooling across length dimension

        # Shared MLP for channel attention, implemented as 1x1 convolutions
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False), # Reduce channels
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False) # Expand channels back
        )

        self.sigmoid = nn.Sigmoid() # Sigmoid to produce attention weights between 0 and 1

    def forward(self, x):
        # x: (batch_size, channels, length)
        # Apply pooling to get channel-wise statistics, then pass through shared MLP
        avg_out = self.fc(self.avg_pool(x)) # (batch_size, channels, 1)
        max_out = self.fc(self.max_pool(x)) # (batch_size, channels, 1)

        # Combine average and max pooled features and apply sigmoid
        attention_weights = self.sigmoid(avg_out + max_out)  # (batch_size, channels, 1)

        # Apply attention weights by element-wise multiplication.
        # This re-weights each feature map (channel).
        attended_features = x * attention_weights
        return attended_features, attention_weights # Return attended features and the weights

class CNNBlock(nn.Module):
    """
    A single CNN block with Convolution, BatchNorm, ReLU, optional Spatial and Channel Attention, and Max Pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_attention=True):
        super(CNNBlock, self).__init__()
        self.use_attention = use_attention

        # Convolutional layer (padding keeps output size same for stride 1 for center-aligned kernel)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=kernel_size//2, bias=False) # Bias False as BatchNorm handles bias
        self.bn = nn.BatchNorm1d(out_channels) # Batch Normalization for stability
        self.relu = nn.ReLU(inplace=True) # ReLU activation, inplace for memory efficiency

        # Attention mechanisms (initialized only if use_attention is True)
        if use_attention:
            self.spatial_attention = SpatialAttention(out_channels)
            self.channel_attention = ChannelAttention(out_channels)

        # Max Pooling layer (reduces trace length by factor of 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply Conv -> BatchNorm -> ReLU
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        spatial_weights = None
        channel_weights = None

        # Apply attention if enabled
        if self.use_attention:
            # Common order for attention: Channel-wise first, then Spatial
            out, channel_weights = self.channel_attention(out)
            out, spatial_weights = self.spatial_attention(out)

        out = self.pool(out) # Apply pooling

        return out, spatial_weights, channel_weights # Return output and attention weights (for potential visualization)


class CNN_LSTM_SCA(nn.Module):
    """
    CNN-LSTM model for Side-Channel Analysis incorporating Spatial and Channel-wise Attention.
    Designed for robust feature extraction from traces and temporal modeling for classification.
    """
    def __init__(self, config, num_sample_pts, classes):
        super(CNN_LSTM_SCA, self).__init__()

        self.classes = classes
        self.num_sample_pts = num_sample_pts # Initial trace length after preprocessing (cropping)

        # --- CNN Feature Extractor ---
        # A list of CNNBlock layers to build the convolutional part of the model
        self.cnn_layers = nn.ModuleList()

        # First CNN layer: input channels=1 (for grayscale trace), output channels=config['cnn_channels'][0]
        self.cnn_layers.append(
            CNNBlock(1, config['cnn_channels'][0],
                    kernel_size=config['cnn_kernels'][0],
                    use_attention=config.get('use_attention', True)) # Use attention setting from config
        )

        # Subsequent CNN layers: input channels from previous block, output channels from config['cnn_channels']
        for i in range(1, len(config['cnn_channels'])):
            self.cnn_layers.append(
                CNNBlock(config['cnn_channels'][i-1], config['cnn_channels'][i],
                        kernel_size=config['cnn_kernels'][i], # Use corresponding kernel size from config
                        use_attention=config.get('use_attention', True))
            )

        # Calculate the resulting trace length after all CNN layers and pooling
        # This is crucial to correctly determine the input size for the LSTM layer.
        self.cnn_output_length = self._calculate_cnn_output_length()
        if self.cnn_output_length <= 0:
            raise ValueError(f"CNN output length is {self.cnn_output_length}. "
                             f"This model configuration (cnn_channels, cnn_kernels) leads to an invalid output size. "
                             f"Adjust cnn_channels/cnn_kernels or initial num_sample_pts (cropping).")

        # --- LSTM Layers for Temporal Modeling ---
        # Input to LSTM is the number of channels from the *last* CNN layer.
        # This is the feature dimension at each timestep for the LSTM.
        self.lstm = nn.LSTM(
            input_size=config['cnn_channels'][-1], # Features per timestep (channels of last CNN layer)
            hidden_size=config['lstm_hidden_size'], # LSTM hidden state size
            num_layers=config['lstm_num_layers'], # Number of stacked LSTM layers
            batch_first=True, # Input/output tensors are (batch_size, seq_len, features)
            # Apply dropout only if there's more than one LSTM layer (standard practice)
            dropout=config.get('lstm_dropout', 0.1) if config['lstm_num_layers'] > 1 else 0,
            bidirectional=config.get('bidirectional', False) # Whether to use bidirectional LSTM
        )

        # Calculate LSTM output size for the classifier (depends on bidirectionality)
        lstm_output_size = config['lstm_hidden_size']
        if config.get('bidirectional', False):
            lstm_output_size *= 2 # Bidirectional LSTMs concatenate forward and backward hidden states

        # --- Attention Mechanism for LSTM Outputs ---
        # This learns to weigh the importance of different time steps from the LSTM's sequence output.
        # A simple MLP with Tanh activation, followed by a linear layer to get a single attention score per timestep.
        self.lstm_attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2), # Reduce dimensionality
            nn.Tanh(), # Tanh non-linearity
            nn.Linear(lstm_output_size // 2, 1) # Output a single score per time step
        )

        # --- Classification Layers (Fully Connected MLP on top of aggregated LSTM features) ---
        self.classifier = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.5)), # Dropout for regularization
            nn.Linear(lstm_output_size, config.get('fc_hidden', 128)), # First fully connected layer
            nn.ReLU(inplace=True), # ReLU activation
            nn.Dropout(config.get('dropout', 0.5)), # Another dropout layer
            nn.Linear(config.get('fc_hidden', 128), classes) # Final classification layer (outputs logits)
        )

        # Initialize model weights (specific initialization strategies)
        self._initialize_weights()

    def _calculate_cnn_output_length(self):
        """
        Helper method to calculate the effective trace length after all CNN layers and pooling.
        This is crucial for setting up the LSTM input's sequence length.
        """
        # Create a dummy tensor with the expected input dimensions
        dummy_input = torch.randn(1, 1, self.num_sample_pts) # (batch_size, channels=1, initial_trace_length)

        # Pass the dummy tensor through all CNN layers to determine the final length
        for layer in self.cnn_layers:
            dummy_input, _, _ = layer(dummy_input) # CNNBlock returns (output, spatial_weights, channel_weights)

        return dummy_input.size(-1) # Return the final length dimension (seq_len for LSTM)

    def _initialize_weights(self):
        """Initializes model weights using Kaiming/Xavier for different layer types."""
        for m in self.modules(): # Iterate over all modules (layers) in the model
            if isinstance(m, nn.Conv1d):
                # Kaiming (He) initialization for Conv layers, good with ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0) # Initialize biases to zero
            elif isinstance(m, nn.BatchNorm1d):
                # BatchNorm weights to 1, biases to 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Normal initialization for Linear layers (common default)
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM): # Initialize LSTM weights specifically
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: # Input-to-hidden weights
                        nn.init.xavier_uniform_(param.data) # Xavier uniform for gates
                    elif 'weight_hh' in name: # Hidden-to-hidden weights
                        nn.init.orthogonal_(param.data) # Orthogonal initialization often good for recurrent layers
                    elif 'bias' in name: # Biases for LSTM gates
                        nn.init.constant_(param.data, 0) # Initialize biases to zero

    def number_of_parameters(self):
        """Counts total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Ensure input tensor 'x' is 3D: (batch_size, 1, length) for Conv1d layer
        if x.dim() == 4:
            x = x.squeeze(2)  # Remove an extra dimension if input is like (batch, channel, 1, length)
        elif x.dim() == 2:
            x = x.unsqueeze(1) # Add a channel dimension if input is like (batch, length)

        # --- CNN Feature Extraction with Attention ---
        features = x
        for i, layer in enumerate(self.cnn_layers):
            features, _, _ = layer(features) # CNNBlock returns (output, spatial_weights, channel_weights)
                                             # We discard weights here as they're not needed for forward pass calculation
            # Check for collapsed dimensions (trace length becoming 0 or negative) due to bad hyperparameters
            if features.size(-1) <= 0:
                raise ValueError(f"CNN output length became {features.size(-1)} after CNN layer {i}. "
                                 f"This is often caused by excessively large kernels or strides "
                                 f"relative to the input trace length. Review cnn_kernels and/or initial trace_length.")

        # --- Prepare features for LSTM ---
        # Current features shape: (batch_size, channels_last_cnn_layer, length_after_cnn)
        # LSTM expects (batch_size, seq_len, features) -> (batch_size, length_after_cnn, channels_last_cnn_layer)
        features = features.transpose(1, 2)  # Transpose dimensions

        # --- LSTM Forward Pass ---
        # lstm_out: (batch_size, seq_len, num_directions * hidden_size)
        lstm_out, (hidden, cell) = self.lstm(features)

        # --- Apply Attention to LSTM Outputs ---
        # Learn a weighted sum of LSTM outputs across the sequence length.
        attention_weights = self.lstm_attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1) # Apply softmax across sequence length dimension

        # Weighted sum: multiply LSTM outputs by attention weights and sum along sequence length
        attended_lstm = torch.sum(lstm_out * attention_weights, dim=1)  # (batch_size, num_directions * hidden_size)

        # --- Classification Layers ---
        output = self.classifier(attended_lstm)

        return output # Return logits (raw scores), CrossEntropyLoss expects this for classification