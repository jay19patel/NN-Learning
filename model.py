import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch,
            kernel_size=3,
            padding=dilation,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        res = x if self.residual is None else self.residual(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x + res

class StockTCN(nn.Module):
    def __init__(self, input_size, num_classes, num_layers=2):
        super().__init__()
        
        layers = []
        in_channels = input_size
        
        # Define layer sizes based on user request or standard growth
        # For simplicity, we can use 64, 128, 256... or fixed sizes as in app.py
        channel_sizes = [64, 128, 256, 512]
        
        for i in range(num_layers):
            out_channels = channel_sizes[i] if i < len(channel_sizes) else channel_sizes[-1]
            dilation = 2 ** (i + 1) # 2, 4, 8, 16...
            layers.append(TCNBlock(in_channels, out_channels, dilation))
            in_channels = out_channels
            
        self.tcn_layers = nn.Sequential(*layers)
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.tcn_layers(x)
        return self.head(x)
