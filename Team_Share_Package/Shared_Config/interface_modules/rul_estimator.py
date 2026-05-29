"""Reserved RUL prediction module."""

from __future__ import annotations

import numpy as np


class RULEstimator:
    """Baseline RUL estimator using a health-index failure threshold."""

    def __init__(self, failure_threshold: float = 1.0) -> None:
        """Initialize the estimator.

        Args:
            failure_threshold: HI threshold regarded as end of useful life.
        """
        self.failure_threshold = failure_threshold

    def estimate(self, time: np.ndarray, health_index: np.ndarray) -> float:
        """Estimate RUL by linear extrapolation of the recent HI slope.

        Args:
            time: Time array.
            health_index: Health-index array.

        Returns:
            Estimated remaining useful life in the same unit as `time`.
        """
        if len(time) < 3:
            return float("nan")
        window = min(10, len(time))
        slope, intercept = np.polyfit(time[-window:], health_index[-window:], deg=1)
        if slope <= 0:
            return float("inf")
        eol_time = (self.failure_threshold - intercept) / slope
        return float(max(0.0, eol_time - time[-1]))
