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
        self.conv1 = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels // 8, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, channels, length)
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention_weights = self.sigmoid(attention)  # (batch_size, 1, length)
        
        # Apply attention weights by element-wise multiplication
        attended_features = x * attention_weights
        return attended_features, attention_weights

class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism for focusing on important feature maps.
    Uses global average and max pooling to create channel descriptors, then FC layers.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1) # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool1d(1) # Global max pooling
        
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, channels, length)
        avg_out = self.fc(self.avg_pool(x)) # (batch_size, channels, 1)
        max_out = self.fc(self.max_pool(x)) # (batch_size, channels, 1)
        
        attention_weights = self.sigmoid(avg_out + max_out)  # (batch_size, channels, 1)
        
        # Apply channel attention by element-wise multiplication
        attended_features = x * attention_weights
        return attended_features, attention_weights

class CNNBlock(nn.Module):
    """
    A single CNN block with Convolution, BatchNorm, ReLU, optional Attention, and Max Pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_attention=True):
        super(CNNBlock, self).__init__()
        self.use_attention = use_attention
        
        # Convolutional layer (padding keeps output size same for stride 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False) # Bias False with BatchNorm
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace saves memory
        
        # Attention mechanisms
        if use_attention:
            self.spatial_attention = SpatialAttention(out_channels)
            self.channel_attention = ChannelAttention(out_channels)
            
        # Pooling layer (reduces length by factor of 2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        spatial_weights = None
        channel_weights = None
        
        if self.use_attention:
            # Order: Channel -> Spatial (as per common practice CBAM/SCA-CNN)
            out, channel_weights = self.channel_attention(out)
            out, spatial_weights = self.spatial_attention(out)
        
        out = self.pool(out)
        
        return out, spatial_weights, channel_weights


class CNN_LSTM_SCA(nn.Module):
    """
    CNN-LSTM model for Side-Channel Analysis incorporating Spatial and Channel-wise Attention.
    Designed for feature extraction from traces and temporal modeling for classification.
    """
    def __init__(self, config, num_sample_pts, classes):
        super(CNN_LSTM_SCA, self).__init__()
        
        self.classes = classes
        self.num_sample_pts = num_sample_pts # Initial trace length
        
        # CNN feature extractor with multiple CNNBlock layers
        self.cnn_layers = nn.ModuleList()
        
        # First CNN layer (input channels=1, output channels=config['cnn_channels'][0])
        self.cnn_layers.append(
            CNNBlock(1, config['cnn_channels'][0], 
                    kernel_size=config['cnn_kernels'][0], 
                    use_attention=config.get('use_attention', True))
        )
        
        # Subsequent CNN layers
        for i in range(1, len(config['cnn_channels'])):
            self.cnn_layers.append(
                CNNBlock(config['cnn_channels'][i-1], config['cnn_channels'][i],
                        kernel_size=config['cnn_kernels'][i], # Use corresponding kernel
                        use_attention=config.get('use_attention', True))
            )
        
        # Calculate the resulting trace length after all CNN layers
        self.cnn_output_length = self._calculate_cnn_output_length()
        if self.cnn_output_length <= 0:
            raise ValueError(f"CNN output length is {self.cnn_output_length}. "
                             f"This model configuration (channels, kernels) leads to an invalid output size. "
                             f"Adjust cnn_channels/cnn_kernels/trace_length.")

        # LSTM layers for temporal modeling
        # Input to LSTM is the number of channels from the *last* CNN layer
        self.lstm = nn.LSTM(
            input_size=config['cnn_channels'][-1], # Features per timestep for LSTM
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True, # (batch, seq_len, features)
            dropout=config.get('lstm_dropout', 0.1) if config['lstm_num_layers'] > 1 else 0,
            bidirectional=config.get('bidirectional', False)
        )
        
        # Calculate LSTM output size for the classifier
        lstm_output_size = config['lstm_hidden_size']
        if config.get('bidirectional', False):
            lstm_output_size *= 2 # Bidirectional LSTMs output concatenation of forward and backward hidden states
            
        # Attention mechanism for LSTM outputs (e.g., to weight different time steps from LSTM)
        # This is not from the original SCA-CNN paper but a common addition for LSTMs
        self.lstm_attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(), # Tanh non-linearity
            nn.Linear(lstm_output_size // 2, 1) # Output 1 score per time step
        )
        
        # Classification layers (MLP on top of aggregated LSTM features)
        self.classifier = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.5)), # Dropout before first FC layer
            nn.Linear(lstm_output_size, config.get('fc_hidden', 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)), # Dropout before second FC layer
            nn.Linear(config.get('fc_hidden', 128), classes) # Final classification layer
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _calculate_cnn_output_length(self):
        """
        Calculates the effective trace length after all CNN layers and pooling.
        This is crucial for setting up the LSTM input.
        """
        # Use a dummy tensor to trace the dimensions through CNN layers
        dummy_input = torch.randn(1, 1, self.num_sample_pts) # (batch, channels, length)
        
        for layer in self.cnn_layers:
            # CNNBlock returns (out, spatial_weights, channel_weights)
            dummy_input, _, _ = layer(dummy_input)
            
        return dummy_input.size(-1) # Return the final length dimension

    def _initialize_weights(self):
        """Initializes model weights using Kaiming/Xavier for different layer types."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # Small normal for linear layers
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM): # Initialize LSTM weights
                for name, param in m.named_parameters():
                    if 'weight_ih' in name: # Input-hidden weights
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name: # Hidden-hidden weights
                        nn.init.orthogonal_(param.data) # Orthogonal initialization often good for recurrent layers
                    elif 'bias' in name: # Biases
                        nn.init.constant_(param.data, 0) # Initialize biases to zero

    def number_of_parameters(self):
        """Counts total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Ensure input is 3D: (batch_size, 1, length) for Conv1d
        if x.dim() == 4:
            x = x.squeeze(2)  # Removes an extra dimension if like (batch, channel, 1, length)
        elif x.dim() == 2:
            x = x.unsqueeze(1) # Adds channel dimension if like (batch, length)
            
        # CNN feature extraction with attention
        features = x
        for i, layer in enumerate(self.cnn_layers):
            features, _, _ = layer(features) # We don't necessarily need the attention weights here
            # Add a check for collapsed dimensions to catch bad hyperparameters early in forward pass
            if features.size(-1) <= 0:
                raise ValueError(f"CNN output length became {features.size(-1)} after CNN layer {i}. "
                                 f"This is often caused by excessively large kernels or strides "
                                 f"relative to the trace length. Review cnn_kernels and input trace_length.")
        
        # Prepare features for LSTM: (batch, seq_len, features)
        # Current features shape: (batch, channels, length_after_cnn)
        features = features.transpose(1, 2)  # Transpose to (batch, length_after_cnn, channels)
        
        # LSTM forward pass
        # lstm_out: (batch, seq_len, num_directions * hidden_size)
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Apply attention to LSTM outputs to get a single context vector per batch
        attention_weights = self.lstm_attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1) # Softmax across seq_len
        
        # Weighted sum of LSTM outputs (across seq_len dimension)
        attended_lstm = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, num_directions * hidden_size)
        
        # Classification layers
        output = self.classifier(attended_lstm)
        
        return output # Return logits (raw scores), CrossEntropyLoss expects this