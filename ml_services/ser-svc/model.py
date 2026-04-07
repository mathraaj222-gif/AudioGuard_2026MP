import torch
import torch.nn as nn

class SERModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.lstm = nn.LSTM(
            input_size=32 * 32,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(256, 8)

    def forward(self, x):
        x = self.conv(x)                     # [B, 32, H, W]
        b, c, h, w = x.shape
        x = x.view(b, w, c * h)             # [B, T, Features]
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)
