import torch
import torch.nn as nn

class DigitNet(nn.Module):
    """
    Tiny CNN for Sudoku digit classification.
    Classes: 0..9 where 0 = blank.
    Input: (N,1,H,W) grayscale (default 28x28).
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
