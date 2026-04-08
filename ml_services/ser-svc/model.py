import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.block(x)

class AttnPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape from LSTM: [batch, seq_len, hidden_dim * num_directions]
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.norm(x)
        # Global Average Pooling over temporal dimension
        return x.mean(dim=1)

class SERModel(nn.Module):
    def __init__(self):
        super().__init__()
        # CNN Front-end for local feature extraction
        self.cnn = nn.Sequential(
            CNNBlock(3, 32),   # cnn.0
            CNNBlock(32, 64),  # cnn.1
            CNNBlock(64, 128), # cnn.2
            CNNBlock(128, 256) # cnn.3
        )
        
        # BiLSTM for temporal context
        # Input size: 256 channels * 8 height (assuming 128/16 input) = 2048
        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=256, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Attention Pooling to compress temporal sequence into a vector
        self.attn_pool = AttnPool(512) # 256 * 2 (bidirectional)
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),    # classifier.0
            nn.BatchNorm1d(256),    # classifier.1
            nn.ReLU(),
            nn.Linear(256, 8)       # classifier.3
        )

    def forward(self, x):
        # x: [Batch, 3, 128, 400]
        x = self.cnn(x)             # [Batch, 256, 8, 25]
        b, c, h, w = x.shape
        # Permute and reshape for LSTM: [Batch, Width(Time), Channels * Height]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, w, c * h) # [B, 25, 2048]
        x, _ = self.lstm(x)         # [Batch, 25, 512]
        x = self.attn_pool(x)       # [Batch, 512]
        return self.classifier(x)
