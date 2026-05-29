"""Reserved shared latent degradation representation model."""

from __future__ import annotations

import numpy as np


class SharedLatentDegradationSpace:
    """Base API for future cross-modal shared latent degradation learning."""

    def fit(self, features: np.ndarray, health_index: np.ndarray | None = None) -> "SharedLatentDegradationSpace":
        """Fit the latent degradation model.

        Args:
            features: Input degradation feature matrix.
            health_index: Optional supervision signal.

        Returns:
            Fitted model.
        """
        self.feature_dim = features.shape[1]
        self.latent_dim = min(8, self.feature_dim)
        return self

    def encode(self, features: np.ndarray) -> np.ndarray:
        """Encode features into the shared latent degradation space.

        Args:
            features: Input feature matrix.

        Returns:
            Latent matrix.
        """
        return features[:, : getattr(self, "latent_dim", min(8, features.shape[1]))]
