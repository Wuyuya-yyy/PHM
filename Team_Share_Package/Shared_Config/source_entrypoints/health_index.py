"""Health index construction for flywheel degradation assessment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from utils.plot_utils import save_figure, set_ieee_style


class AutoEncoderHI:
    """Reserved AutoEncoder health-index interface for later deep fusion."""

    def __init__(self) -> None:
        """Initialize the placeholder AutoEncoder HI model."""
        self.is_fitted = False

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        """Reserve the future AutoEncoder fusion API.

        Args:
            values: Normalized feature matrix.

        Raises:
            NotImplementedError: Always raised until the deep model is implemented.
        """
        raise NotImplementedError("AutoEncoder HI fusion is reserved for phase-2 deep representation learning.")


class HealthIndexBuilder:
    """Build interpretable health indicators using Min-Max and PCA fusion."""

    def __init__(self, project_root: Path, dpi: int, logger: logging.Logger) -> None:
        """Initialize the HI builder.

        Args:
            project_root: PHM project root.
            dpi: Figure resolution.
            logger: Project logger.
        """
        self.project_root = project_root
        self.dpi = dpi
        self.logger = logger
        self.fig_dir = project_root / "figures" / "hi"
        set_ieee_style()

    def run(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build health indices for all flywheel datasets.

        Args:
            datasets: Loaded tabular datasets.

        Returns:
            HI construction results.
        """
        results: Dict[str, Any] = {}
        for item in datasets:
            name = self._slug(item["name"])
            df = item["data"].copy()
            hi_df, meta = self.build_hi(df)
            out_path = self.project_root / "processed_data" / f"{name}_health_index.csv"
            hi_df.to_csv(out_path, index=False)
            figures = self.plot_hi(hi_df, name)
            results[name] = {"metadata": meta, "processed_path": str(out_path), "figures": figures}
        return results

    def build_hi(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Construct HI(t) from degradation-sensitive variables.

        Args:
            df: Input flywheel data.

        Returns:
            Tuple of processed HI DataFrame and metadata.
        """
        time_col = self._time_column(df)
        feature_cols = [
            c
            for c in ["current", "temperature", "friction_torque", "speed_rpm"]
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not feature_cols:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != time_col]
        work = df[[time_col] + feature_cols].copy() if time_col else df[feature_cols].copy()
        features = work[feature_cols].astype(float).interpolate().bfill().ffill()
        oriented = features.copy()
        for col in oriented.columns:
            values = oriented[col].to_numpy()
            if np.nanstd(values) <= 1e-12:
                continue
            corr = np.corrcoef(np.arange(len(oriented)), values)[0, 1]
            if np.isfinite(corr) and corr < 0:
                oriented[col] = -oriented[col]
        normalized = pd.DataFrame(
            MinMaxScaler().fit_transform(oriented),
            columns=[f"{c}_norm" for c in feature_cols],
            index=df.index,
        )
        pca = PCA(n_components=1, random_state=42)
        pca_hi = pca.fit_transform(normalized).ravel()
        pca_hi = MinMaxScaler().fit_transform(pca_hi.reshape(-1, 1)).ravel()
        if np.corrcoef(np.arange(len(pca_hi)), pca_hi)[0, 1] < 0:
            pca_hi = 1.0 - pca_hi
        hi_raw = pca_hi
        window = max(3, int(len(df) * 0.08))
        hi_smooth = pd.Series(hi_raw).rolling(window, min_periods=1, center=True).mean().to_numpy()
        hi_derivative = np.gradient(hi_smooth)
        result = pd.DataFrame()
        result[time_col or "sample_index"] = df[time_col].to_numpy() if time_col else np.arange(len(df))
        for col in normalized.columns:
            result[col] = normalized[col].to_numpy()
        result["HI_raw"] = hi_raw
        result["HI_smooth"] = hi_smooth
        result["HI_derivative"] = hi_derivative
        meta = {
            "time_column": time_col or "sample_index",
            "feature_columns": feature_cols,
            "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
            "autoencoder_interface": "reserved",
        }
        return result, meta

    def plot_hi(self, hi_df: pd.DataFrame, name: str) -> List[str]:
        """Plot HI curve, degradation trend, smoothing, and derivative.

        Args:
            hi_df: Health-index DataFrame.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        x_col = hi_df.columns[0]
        x = hi_df[x_col]
        paths: List[str] = []
        plt.figure(figsize=(6.6, 3.2))
        plt.plot(x, hi_df["HI_raw"], color="#A0AEC0", linewidth=0.9, label="Raw HI")
        plt.plot(x, hi_df["HI_smooth"], color="#D62728", linewidth=1.4, label="Smoothed HI")
        plt.xlabel(x_col)
        plt.ylabel("Health Index")
        plt.title("Health Index Curve")
        plt.legend()
        paths.append(save_figure(self.fig_dir / f"{name}_health_index_curve.png", self.dpi))

        fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
        axes[0].plot(x, hi_df["HI_smooth"], color="#2B6CB0", linewidth=1.3)
        axes[0].set_title("Degradation Trend")
        axes[0].set_ylabel("Smoothed HI")
        axes[1].plot(x, hi_df["HI_derivative"], color="#C05621", linewidth=1.0)
        axes[1].axhline(0, color="#4A5568", linewidth=0.7)
        axes[1].set_title("Derivative Change of HI")
        axes[1].set_xlabel(x_col)
        axes[1].set_ylabel("dHI/dt")
        paths.append(save_figure(self.fig_dir / f"{name}_hi_trend_and_derivative.png", self.dpi))
        return paths

    @staticmethod
    def _time_column(df: pd.DataFrame) -> Optional[str]:
        """Infer a time column from standard candidates.

        Args:
            df: Input DataFrame.

        Returns:
            Time column name if available.
        """
        for candidate in ["day", "time", "timestamp", "date", "cycle"]:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _slug(text: str) -> str:
        """Convert text into a filesystem-safe slug.

        Args:
            text: Raw text.

        Returns:
            Sanitized slug.
        """
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)
