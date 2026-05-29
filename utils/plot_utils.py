"""Plotting utilities for IEEE-style PHM figures."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


def set_ieee_style() -> None:
    """Set a clean publication-oriented plotting style."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 120,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.linewidth": 0.8,
        }
    )


def save_figure(path: Path, dpi: int = 320, close: bool = True) -> str:
    """Save the current Matplotlib figure.

    Args:
        path: Target image path.
        dpi: Output resolution.
        close: Whether to close the active figure.

    Returns:
        String path of the saved figure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    if close:
        plt.close()
    return str(path)


def safe_title(text: str, suffix: Optional[str] = None) -> str:
    """Build a concise English plot title.

    Args:
        text: Base title.
        suffix: Optional suffix.

    Returns:
        Formatted title string.
    """
    return f"{text} - {suffix}" if suffix else text
