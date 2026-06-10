"""
dncnn_model.py
--------------
DnCNN architecture for residual noise estimation.
Compatible with Apple M1 (MPS), CUDA, and CPU.

Architecture: 17-layer Conv-BN-ReLU stack.
Input  : noisy/blurred image  (B, 1, H, W)  float32  range [0, 1]
Output : estimated noise residual            (B, 1, H, W)
Clean  : input - output
"""

import torch
import torch.nn as nn


class DnCNN(nn.Module):
    def __init__(self, depth: int = 17, n_filters: int = 64, kernel_size: int = 3):
        """
        Args:
            depth      : total number of conv layers (default 17, standard DnCNN).
            n_filters  : number of feature channels in hidden layers.
            kernel_size: spatial size of every conv kernel.
        """
        super().__init__()

        padding = kernel_size // 2
        layers: list[nn.Module] = []

        # ── Layer 1: Conv + ReLU (no BN on first layer) ─────────────────────
        layers += [
            nn.Conv2d(1, n_filters, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        ]

        # ── Layers 2 … depth-1: Conv + BN + ReLU ────────────────────────────
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(n_filters, n_filters, kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(n_filters),
                nn.ReLU(inplace=True),
            ]

        # ── Last layer: Conv only (outputs noise residual) ───────────────────
        layers.append(
            nn.Conv2d(n_filters, 1, kernel_size, padding=padding, bias=False)
        )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ── Weight initialisation (He normal for conv, const for BN) ────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the denoised image (input minus predicted residual)."""
        residual = self.net(x)
        return torch.clamp(x - residual, 0.0, 1.0)


# ── Convenience: select best available device ────────────────────────────────
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
