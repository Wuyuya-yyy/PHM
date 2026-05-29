"""Reserved multimodal fusion module for vibration, current, and temperature data."""

from __future__ import annotations

from typing import Dict

import numpy as np


class MultimodalFusion:
    """Base interface for modality-aware degradation feature fusion."""

    def fuse(self, modality_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse multiple modality feature matrices by sample-wise concatenation.

        Args:
            modality_features: Mapping from modality name to feature matrix.

        Returns:
            Concatenated multimodal feature matrix.
        """
        if not modality_features:
            raise ValueError("No modality features were provided.")
        return np.concatenate(list(modality_features.values()), axis=1)
