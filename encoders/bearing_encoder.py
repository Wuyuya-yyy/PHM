"""PyTorch interface for future XJTU-SY bearing degradation features."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class BearingEncoder(nn.Module):
    """Map bearing vibration features into the shared latent degradation space."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(last_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, latent_dim))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Encode bearing features such as RMS, kurtosis, FFT, wavelet, and STFT descriptors."""
        return self.network(features)
