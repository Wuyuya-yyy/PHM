"""Reserved physics-informed constraints for latent degradation models."""

from __future__ import annotations

import torch


class PhysicsInformedConstraint:
    """Interface for monotonicity, temporal continuity, and physical consistency losses."""

    def monotonicity_loss(self, health_index: torch.Tensor) -> torch.Tensor:
        """Penalize negative increments in a degradation-oriented HI sequence."""
        if health_index.numel() < 2:
            return torch.zeros((), dtype=health_index.dtype, device=health_index.device)
        diff = health_index[1:] - health_index[:-1]
        return torch.relu(-diff).mean()

    def temporal_continuity_loss(self, latent: torch.Tensor) -> torch.Tensor:
        """Penalize abrupt latent-state jumps."""
        if latent.shape[0] < 2:
            return torch.zeros((), dtype=latent.dtype, device=latent.device)
        return torch.mean((latent[1:] - latent[:-1]) ** 2)

    def physical_consistency_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Placeholder physical consistency term implemented as MSE for the prototype."""
        return torch.mean((prediction - target) ** 2)
