import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism for focusing on important temporal regions
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
        
        # Apply attention weights
        attended_features = x * attention_weights
        return attended_features, attention_weights

class ChannelAttention(nn.Module):
    """
    Channel-wise attention mechanism for focusing on important feature maps
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch_size, channels, length)
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        attention_weights = self.sigmoid(avg_out + max_out)  # (batch_size, channels, 1)
        
        # Apply channel attention
        attended_features = x * attention_weights
        return attended_features, attention_weights

class CNNBlock(nn.Module):
    """
    CNN block with spatial and channel attention
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, use_attention=True):
        super(CNNBlock, self).__init__()
        self.use_attention = use_attention
        
        # Convolutional layer
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention mechanisms
        if use_attention:
            self.spatial_attention = SpatialAttention(out_channels)
            self.channel_attention = ChannelAttention(out_channels)
            
        # Pooling
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Convolution + BatchNorm + ReLU
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        spatial_weights = None
        channel_weights = None
        
        if self.use_attention:
            # Apply channel attention first
            out, channel_weights = self.channel_attention(out)
            
            # Then apply spatial attention
            out, spatial_weights = self.spatial_attention(out)
        
        # Pooling
        out = self.pool(out)
        
        return out, spatial_weights, channel_weights

class CNN_LSTM_SCA(nn.Module):
    """
    CNN-LSTM model for Side-Channel Analysis with attention mechanisms
    Based on the SCA-CNN paper concepts adapted for power traces
    """
    def __init__(self, config, num_sample_pts, classes):
        super(CNN_LSTM_SCA, self).__init__()
        
        self.classes = classes
        self.num_sample_pts = num_sample_pts
        
        # CNN feature extractor with multiple layers
        self.cnn_layers = nn.ModuleList()
        
        # First layer: 1 -> config['cnn_channels'][0]
        self.cnn_layers.append(
            CNNBlock(1, config['cnn_channels'][0], 
                    kernel_size=config['cnn_kernels'][0], 
                    use_attention=config.get('use_attention', True))
        )
        
        # Subsequent layers
        for i in range(1, len(config['cnn_channels'])):
            self.cnn_layers.append(
                CNNBlock(config['cnn_channels'][i-1], config['cnn_channels'][i],
                        kernel_size=config['cnn_kernels'][i],
                        use_attention=config.get('use_attention', True))
            )
        
        # Calculate the size after CNN layers
        self.cnn_output_size = self._calculate_cnn_output_size()
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=config['cnn_channels'][-1],
            hidden_size=config['lstm_hidden_size'],
            num_layers=config['lstm_num_layers'],
            batch_first=True,
            dropout=config.get('lstm_dropout', 0.1) if config['lstm_num_layers'] > 1 else 0,
            bidirectional=config.get('bidirectional', False)
        )
        
        # Calculate LSTM output size
        lstm_output_size = config['lstm_hidden_size']
        if config.get('bidirectional', False):
            lstm_output_size *= 2
            
        # Attention mechanism for LSTM outputs
        self.lstm_attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(lstm_output_size, config.get('fc_hidden', 128)),
            nn.ReLU(inplace=True),
            nn.Dropout(config.get('dropout', 0.5)),
            nn.Linear(config.get('fc_hidden', 128), classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _calculate_cnn_output_size(self):
        """Calculate the output size after CNN layers"""
        x = torch.randn(1, 1, self.num_sample_pts)
        
        for layer in self.cnn_layers:
            x, _, _ = layer(x)
            
        return x.size(-1)  # Return the length dimension
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Handle input shape: if 4D [batch, 1, 1, length], squeeze to 3D [batch, 1, length]
        if x.dim() == 4:
            x = x.squeeze(2)  # Remove the extra dimension
        # If 2D [batch, length], add channel dimension to make it 3D [batch, 1, length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # CNN feature extraction with attention
        features = x  # Already has channel dimension
        
        all_spatial_weights = []
        all_channel_weights = []
        
        for i, layer in enumerate(self.cnn_layers):
            features, spatial_weights, channel_weights = layer(features)
            all_spatial_weights.append(spatial_weights)
            all_channel_weights.append(channel_weights)
        
        # Prepare for LSTM: (batch, seq_len, features)
        # features shape: (batch, channels, length)
        features = features.transpose(1, 2)  # (batch, length, channels)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(features)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Apply attention to LSTM outputs
        attention_weights = self.lstm_attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum of LSTM outputs
        attended_lstm = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size)
        
        # Classification
        output = self.classifier(attended_lstm)
        
        return {
            'output': output,
            'spatial_weights': all_spatial_weights,
            'channel_weights': all_channel_weights,
            'lstm_attention': attention_weights.squeeze(-1),  # (batch, seq_len)
            'lstm_features': attended_lstm
        }

def create_cnn_lstm_sca_config():
    """
    Create default configuration for CNN-LSTM-SCA model
    """
    config = {
        # CNN configuration
        'cnn_channels': [32, 64, 128, 256],  # Number of channels in each CNN layer
        'cnn_kernels': [11, 7, 5, 3],        # Kernel sizes for each CNN layer
        'use_attention': True,                # Whether to use spatial and channel attention
        
        # LSTM configuration
        'lstm_hidden_size': 128,              # LSTM hidden state size
        'lstm_num_layers': 2,                 # Number of LSTM layers
        'lstm_dropout': 0.1,                  # LSTM dropout
        'bidirectional': True,                # Bidirectional LSTM
        
        # Classification configuration
        'fc_hidden': 256,                     # Hidden layer size in classifier
        'dropout': 0.5,                       # Dropout rate
    }
    
    return config

class CNN_LSTM_Light(nn.Module):
    """
    Lightweight version of CNN-LSTM for faster training
    """
    def __init__(self, config, num_sample_pts, classes):
        super(CNN_LSTM_Light, self).__init__()
        
        self.classes = classes
        
        # Get configuration parameters with defaults
        cnn_channels = config.get('cnn_channels', [32, 64, 128])
        cnn_kernels = config.get('cnn_kernels', [11, 7, 5])
        lstm_hidden = config.get('lstm_hidden_size', 64)
        lstm_layers = config.get('lstm_num_layers', 1)
        dropout_rate = config.get('dropout', 0.3)
        
        # Build CNN layers dynamically based on config
        cnn_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_channels, cnn_kernels)):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        with torch.no_grad():
            x = torch.randn(1, 1, num_sample_pts)
            x = self.cnn(x)
            cnn_output_size = x.size(-1)
        
        # LSTM with configurable parameters
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # Last CNN channel count
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=config.get('lstm_dropout', 0.0) if lstm_layers > 1 else 0.0
        )
        
        # Classifier with configurable dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden, classes)
        )
    
    def forward(self, x):
        # Handle input shape: if 4D [batch, 1, 1, length], squeeze to 3D [batch, 1, length]
        if x.dim() == 4:
            x = x.squeeze(2)  # Remove the extra dimension
        # If 2D [batch, length], add channel dimension to make it 3D [batch, 1, length]
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        # CNN
        features = self.cnn(x)
        
        # Prepare for LSTM
        features = features.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(features)
        
        # Use last hidden state
        output = self.classifier(hidden[-1])
        
        return output
