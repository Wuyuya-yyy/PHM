"""Automatic degradation stage segmentation methods."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.plot_utils import save_figure, set_ieee_style


class StageSegmenter:
    """Segment health, slow degradation, and accelerated degradation stages."""

    def __init__(self, project_root: Path, dpi: int, logger: logging.Logger) -> None:
        """Initialize the stage segmenter.

        Args:
            project_root: PHM project root.
            dpi: Figure resolution.
            logger: Project logger.
        """
        self.project_root = project_root
        self.dpi = dpi
        self.logger = logger
        self.fig_dir = project_root / "figures" / "stages"
        set_ieee_style()

    def run(self, hi_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run all stage segmentation methods.

        Args:
            hi_results: Output from HealthIndexBuilder.

        Returns:
            Stage segmentation results.
        """
        output: Dict[str, Any] = {}
        for name, info in hi_results.items():
            hi_df = pd.read_csv(info["processed_path"])
            result = self.segment(hi_df)
            result_path = self.project_root / "results" / f"{name}_stage_boundaries.json"
            with result_path.open("w", encoding="utf-8") as file:
                json.dump(result["boundaries"], file, indent=2, ensure_ascii=False)
            figures = self.plot_segments(hi_df, result, name)
            output[name] = {"boundaries": result["boundaries"], "figures": figures, "result_path": str(result_path)}
        return output

    def segment(self, hi_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute threshold, derivative, and Bayesian-style change point segmentation.

        Args:
            hi_df: Health-index DataFrame.

        Returns:
            Segment labels and boundaries for each method.
        """
        hi = hi_df["HI_smooth"].to_numpy()
        der = hi_df["HI_derivative"].to_numpy()
        threshold_bounds = self._threshold_bounds(hi)
        derivative_bounds = self._derivative_bounds(der)
        bayesian_bounds = self._bayesian_change_points(hi)
        return {
            "labels": {
                "threshold": self._labels(len(hi), threshold_bounds),
                "derivative": self._labels(len(hi), derivative_bounds),
                "bayesian": self._labels(len(hi), bayesian_bounds),
            },
            "boundaries": {
                "threshold": threshold_bounds,
                "derivative": derivative_bounds,
                "bayesian": bayesian_bounds,
            },
        }

    @staticmethod
    def _threshold_bounds(hi: np.ndarray) -> List[int]:
        """Find stage boundaries from HI quantile thresholds.

        Args:
            hi: Smoothed health index.

        Returns:
            Two boundary indices.
        """
        b1 = int(np.argmax(hi >= np.quantile(hi, 0.33)))
        b2 = int(np.argmax(hi >= np.quantile(hi, 0.70)))
        return sorted([max(1, b1), max(b1 + 1, b2)])

    @staticmethod
    def _derivative_bounds(derivative: np.ndarray) -> List[int]:
        """Find boundaries from derivative intensity.

        Args:
            derivative: HI derivative array.

        Returns:
            Two boundary indices.
        """
        smooth = pd.Series(derivative).rolling(max(3, len(derivative) // 20), min_periods=1).mean().to_numpy()
        positive = np.maximum(smooth, 0)
        q1, q2 = np.quantile(positive, [0.55, 0.85])
        b1_candidates = np.where(positive >= q1)[0]
        b2_candidates = np.where(positive >= q2)[0]
        b1 = int(b1_candidates[0]) if len(b1_candidates) else len(derivative) // 3
        b2 = int(b2_candidates[0]) if len(b2_candidates) else 2 * len(derivative) // 3
        return sorted([max(1, b1), max(b1 + 1, b2)])

    @staticmethod
    def _bayesian_change_points(hi: np.ndarray) -> List[int]:
        """Approximate Bayesian change points using Gaussian segment evidence.

        Args:
            hi: Smoothed health index.

        Returns:
            Two most plausible change-point indices.
        """
        n = len(hi)
        if n < 9:
            return [max(1, n // 3), max(2, 2 * n // 3)]
        min_seg = max(3, n // 10)
        best_score = np.inf
        best: Tuple[int, int] = (n // 3, 2 * n // 3)
        for b1 in range(min_seg, n - 2 * min_seg + 1):
            for b2 in range(b1 + min_seg, n - min_seg + 1):
                score = 0.0
                for segment in [hi[:b1], hi[b1:b2], hi[b2:]]:
                    var = np.var(segment) + 1e-8
                    score += len(segment) * np.log(var)
                penalty = 2 * np.log(n)
                score += penalty
                if score < best_score:
                    best_score = score
                    best = (b1, b2)
        return [int(best[0]), int(best[1])]

    @staticmethod
    def _labels(n: int, bounds: List[int]) -> List[str]:
        """Build stage labels from two boundaries.

        Args:
            n: Number of samples.
            bounds: Two boundary indices.

        Returns:
            Stage label list.
        """
        labels = []
        for idx in range(n):
            if idx < bounds[0]:
                labels.append("Healthy")
            elif idx < bounds[1]:
                labels.append("Slow Degradation")
            else:
                labels.append("Accelerated Degradation")
        return labels

    def plot_segments(self, hi_df: pd.DataFrame, result: Dict[str, Any], name: str) -> List[str]:
        """Plot segmentation labels and stage statistics.

        Args:
            hi_df: HI DataFrame.
            result: Segmentation result.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        x_col = hi_df.columns[0]
        x = hi_df[x_col].to_numpy()
        hi = hi_df["HI_smooth"].to_numpy()
        paths: List[str] = []
        colors = {"threshold": "#D62728", "derivative": "#2B6CB0", "bayesian": "#2F855A"}
        plt.figure(figsize=(6.8, 3.4))
        plt.plot(x, hi, color="#2D3748", linewidth=1.2, label="Smoothed HI")
        for method, bounds in result["boundaries"].items():
            for b in bounds:
                plt.axvline(x[b], color=colors[method], linestyle="--", linewidth=0.9, label=method if b == bounds[0] else None)
        plt.xlabel(x_col)
        plt.ylabel("Health Index")
        plt.title("Degradation Stage Segmentation")
        plt.legend(ncol=2)
        paths.append(save_figure(self.fig_dir / f"{name}_stage_labels.png", self.dpi))

        methods = []
        healthy = []
        slow = []
        accelerated = []
        for method, labels in result["labels"].items():
            methods.append(method)
            counts = pd.Series(labels).value_counts()
            healthy.append(int(counts.get("Healthy", 0)))
            slow.append(int(counts.get("Slow Degradation", 0)))
            accelerated.append(int(counts.get("Accelerated Degradation", 0)))
        bottom = np.zeros(len(methods))
        plt.figure(figsize=(6.2, 3.2))
        for values, label, color in [
            (healthy, "Healthy", "#4C78A8"),
            (slow, "Slow Degradation", "#F2C94C"),
            (accelerated, "Accelerated Degradation", "#D62728"),
        ]:
            plt.bar(methods, values, bottom=bottom, label=label, color=color)
            bottom += np.asarray(values)
        plt.ylabel("Sample Count")
        plt.title("Segment Statistics Across Methods")
        plt.legend()
        paths.append(save_figure(self.fig_dir / f"{name}_stage_statistics.png", self.dpi))
        return paths
