import torch
import torch.nn as nn

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_layers=17):
        super(DnCNN, self).__init__()

        layers = []

        # First layer
        layers.append(nn.Conv2d(channels, 64, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(64, affine=True))
            layers.append(nn.ReLU(inplace=True))

        # Last layer
        layers.append(nn.Conv2d(64, channels, 3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.dncnn(x)
        return x - noise
