import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.2)
        )
    def forward(self, x): return self.block(x)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        attn_out, weights = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        w = weights.mean(dim=1, keepdim=True).transpose(1, 2)
        return (x * w).sum(dim=1)

class SERModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        h = self.config.get("LSTM_HIDDEN", 256)
        self.cnn = nn.Sequential(ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128), ConvBlock(128, 256))
        self.lstm = nn.LSTM(2048, h, 2, batch_first=True, bidirectional=True)
        self.attn_pool = AttentionPool(h * 2)
        self.classifier = nn.Sequential(nn.LayerNorm(h*2), nn.Linear(h*2, 256), nn.GELU(), nn.Linear(256, 7))

    def forward(self, features, labels=None):
        x = self.cnn(features)
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Freq)
        x, _ = self.lstm(x)
        x = self.attn_pool(x)
        logits = self.classifier(x)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}
