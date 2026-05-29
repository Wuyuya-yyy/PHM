"""Exploratory data analysis for satellite reaction wheel degradation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import seaborn as sns
from scipy.fft import rfft, rfftfreq

from utils.plot_utils import save_figure, set_ieee_style


class FlywheelEDA:
    """Generate publication-quality EDA figures and statistics."""

    def __init__(self, project_root: Path, dpi: int, logger: logging.Logger) -> None:
        """Initialize EDA analysis.

        Args:
            project_root: PHM project root.
            dpi: Figure resolution.
            logger: Project logger.
        """
        self.project_root = project_root
        self.dpi = dpi
        self.logger = logger
        self.fig_dir = project_root / "figures" / "eda"
        set_ieee_style()

    def run(self, datasets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run EDA for all tabular flywheel datasets.

        Args:
            datasets: Loaded tabular datasets.

        Returns:
            Dictionary with EDA statistics and figure paths.
        """
        results: Dict[str, Any] = {}
        for item in datasets:
            name = self._slug(item["name"])
            df = item["data"].copy()
            numeric = df.select_dtypes(include=[np.number])
            if numeric.empty:
                continue
            time_col = self._time_column(df)
            figures: List[str] = []
            stats = {
                "missing_values": df.isna().sum().astype(int).to_dict(),
                "outlier_counts_z3": self._outlier_counts(numeric),
                "correlation": self._nonconstant_numeric(numeric).corr().round(6).fillna(0).to_dict(),
            }
            figures += self._plot_missing(df, name)
            figures += self._plot_trends(df, numeric, time_col, name)
            figures += self._plot_rolling(df, numeric, time_col, name)
            figures += self._plot_fft(df, numeric, time_col, name)
            figures += self._plot_wavelet(df, numeric, time_col, name)
            figures += self._plot_corr(numeric, name)
            figures += self._plot_pairwise(numeric, name)
            results[name] = {"statistics": stats, "figures": figures}
        return results

    @staticmethod
    def _slug(text: str) -> str:
        """Convert text to a filesystem-friendly slug.

        Args:
            text: Input text.

        Returns:
            Sanitized slug.
        """
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)

    @staticmethod
    def _time_column(df: pd.DataFrame) -> Optional[str]:
        """Infer a time or cycle column.

        Args:
            df: DataFrame.

        Returns:
            Column name if found.
        """
        for candidate in ["day", "time", "timestamp", "date", "cycle"]:
            if candidate in df.columns:
                return candidate
        return None

    @staticmethod
    def _outlier_counts(numeric: pd.DataFrame) -> Dict[str, int]:
        """Count z-score outliers for numeric columns.

        Args:
            numeric: Numeric DataFrame.

        Returns:
            Outlier count per column.
        """
        std = numeric.std(ddof=0).replace(0, np.nan)
        z = (numeric - numeric.mean()) / std
        return z.abs().gt(3.0).sum().astype(int).to_dict()

    @staticmethod
    def _nonconstant_numeric(numeric: pd.DataFrame) -> pd.DataFrame:
        """Remove constant columns before correlation-style analysis.

        Args:
            numeric: Numeric DataFrame.

        Returns:
            Numeric DataFrame containing only nonconstant columns.
        """
        nonconstant = numeric.loc[:, numeric.nunique(dropna=True) > 1]
        return nonconstant if not nonconstant.empty else numeric.iloc[:, :0]

    def _plot_missing(self, df: pd.DataFrame, name: str) -> List[str]:
        """Plot missing-value counts.

        Args:
            df: DataFrame.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        plt.figure(figsize=(6.2, 3.0))
        df.isna().sum().plot(kind="bar", color="#4C78A8")
        plt.title("Missing Value Profile")
        plt.ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        return [save_figure(self.fig_dir / f"{name}_missing_values.png", self.dpi)]

    def _plot_trends(
        self, df: pd.DataFrame, numeric: pd.DataFrame, time_col: Optional[str], name: str
    ) -> List[str]:
        """Plot multivariate time trends and key variable changes.

        Args:
            df: Original DataFrame.
            numeric: Numeric DataFrame.
            time_col: Time column name.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        x = df[time_col] if time_col else np.arange(len(df))
        paths: List[str] = []
        cols = [c for c in numeric.columns if c != time_col]
        n = len(cols)
        fig, axes = plt.subplots(max(n, 1), 1, figsize=(7.2, max(2.2, 1.7 * max(n, 1))), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, col in zip(axes, cols):
            ax.plot(x, df[col], linewidth=1.1, color="#1F77B4")
            ax.set_ylabel(col)
            ax.set_title(f"{col} Trend")
        axes[-1].set_xlabel(time_col or "Sample Index")
        paths.append(save_figure(self.fig_dir / f"{name}_time_series_trends.png", self.dpi))
        for target, title in [
            ("current", "Current Change Analysis"),
            ("temperature", "Temperature Change Analysis"),
            ("speed_rpm", "Speed Change Analysis"),
            ("friction_torque", "Friction Torque Analysis"),
        ]:
            if target in df.columns:
                plt.figure(figsize=(6.4, 3.0))
                plt.plot(x, df[target], linewidth=1.2, color="#2F855A")
                plt.xlabel(time_col or "Sample Index")
                plt.ylabel(target)
                plt.title(title)
                paths.append(save_figure(self.fig_dir / f"{name}_{target}_analysis.png", self.dpi))
        return paths

    def _plot_rolling(
        self, df: pd.DataFrame, numeric: pd.DataFrame, time_col: Optional[str], name: str
    ) -> List[str]:
        """Plot rolling mean and variance for degradation-sensitive features.

        Args:
            df: Original DataFrame.
            numeric: Numeric DataFrame.
            time_col: Time column name.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        x = df[time_col] if time_col else np.arange(len(df))
        cols = [c for c in ["current", "temperature", "speed_rpm", "friction_torque"] if c in numeric.columns]
        if not cols:
            cols = [c for c in numeric.columns if c != time_col][:4]
        window = max(3, int(len(df) * 0.08))
        paths: List[str] = []
        for col in cols:
            series = df[col].astype(float)
            mean = series.rolling(window, min_periods=2).mean()
            var = series.rolling(window, min_periods=2).var()
            fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)
            axes[0].plot(x, series, color="#A0AEC0", linewidth=0.8, label="Raw")
            axes[0].plot(x, mean, color="#D62728", linewidth=1.2, label="Rolling Mean")
            axes[0].legend()
            axes[0].set_title(f"Rolling Mean of {col}")
            axes[1].plot(x, var, color="#805AD5", linewidth=1.2)
            axes[1].set_title(f"Rolling Variance of {col}")
            axes[1].set_xlabel(time_col or "Sample Index")
            paths.append(save_figure(self.fig_dir / f"{name}_{col}_rolling_statistics.png", self.dpi))
        return paths

    def _plot_fft(
        self, df: pd.DataFrame, numeric: pd.DataFrame, time_col: Optional[str], name: str
    ) -> List[str]:
        """Plot FFT spectra for major degradation features.

        Args:
            df: Original DataFrame.
            numeric: Numeric DataFrame.
            time_col: Time column name.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        paths: List[str] = []
        cols = [c for c in ["current", "temperature", "friction_torque"] if c in numeric.columns]
        if not cols:
            cols = [c for c in numeric.columns if c != time_col][:3]
        for col in cols:
            y = df[col].astype(float).interpolate().bfill().ffill().to_numpy()
            if len(y) < 4:
                continue
            y = y - np.nanmean(y)
            yf = np.abs(rfft(y))
            xf = rfftfreq(len(y), d=1.0)
            plt.figure(figsize=(6.4, 3.0))
            plt.plot(xf[1:], yf[1:], color="#C05621", linewidth=1.1)
            plt.xlabel("Normalized Frequency")
            plt.ylabel("Amplitude")
            plt.title(f"FFT Spectrum of {col}")
            paths.append(save_figure(self.fig_dir / f"{name}_{col}_fft_spectrum.png", self.dpi))
        return paths

    def _plot_wavelet(
        self, df: pd.DataFrame, numeric: pd.DataFrame, time_col: Optional[str], name: str
    ) -> List[str]:
        """Plot continuous wavelet scalograms for selected features.

        Args:
            df: Original DataFrame.
            numeric: Numeric DataFrame.
            time_col: Time column name.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        paths: List[str] = []
        cols = [c for c in ["current", "temperature", "friction_torque"] if c in numeric.columns][:2]
        for col in cols:
            y = df[col].astype(float).interpolate().bfill().ffill().to_numpy()
            if len(y) < 8:
                continue
            scales = np.arange(1, min(64, len(y) // 2))
            coeff, _ = pywt.cwt(y - np.mean(y), scales, "morl")
            plt.figure(figsize=(6.6, 3.2))
            plt.imshow(np.abs(coeff), aspect="auto", cmap="viridis", origin="lower")
            plt.colorbar(label="Magnitude")
            plt.xlabel("Sample Index")
            plt.ylabel("Scale")
            plt.title(f"Wavelet Scalogram of {col}")
            paths.append(save_figure(self.fig_dir / f"{name}_{col}_wavelet_scalogram.png", self.dpi))
        return paths

    def _plot_corr(self, numeric: pd.DataFrame, name: str) -> List[str]:
        """Plot a correlation heatmap.

        Args:
            numeric: Numeric DataFrame.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        corr_source = self._nonconstant_numeric(numeric)
        if corr_source.shape[1] < 2:
            return []
        plt.figure(figsize=(5.8, 4.8))
        sns.heatmap(corr_source.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
        plt.title("Correlation Heatmap")
        return [save_figure(self.fig_dir / f"{name}_correlation_heatmap.png", self.dpi)]

    def _plot_pairwise(self, numeric: pd.DataFrame, name: str) -> List[str]:
        """Plot selected multivariate relationships.

        Args:
            numeric: Numeric DataFrame.
            name: Dataset slug.

        Returns:
            Saved figure paths.
        """
        numeric = self._nonconstant_numeric(numeric)
        cols = [c for c in ["current", "temperature", "speed_rpm", "friction_torque"] if c in numeric.columns]
        if len(cols) < 2:
            return []
        sample = numeric[cols].dropna()
        if len(sample) > 500:
            sample = sample.sample(500, random_state=42)
        grid = sns.pairplot(sample, corner=True, diag_kind="hist", plot_kws={"s": 14, "alpha": 0.75})
        grid.fig.suptitle("Multivariate Relationship Analysis", y=1.02)
        return [save_figure(self.fig_dir / f"{name}_multivariate_relationships.png", self.dpi)]
