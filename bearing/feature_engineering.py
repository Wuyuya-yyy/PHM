"""XJTU-SY bearing feature extraction and latent-feature preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def time_domain_features(signal: np.ndarray) -> dict[str, float]:
    """Compute classical time-domain bearing degradation features."""
    x = np.asarray(signal, dtype=float).ravel()
    abs_x = np.abs(x)
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(abs_x))
    mean_abs = float(np.mean(abs_x))
    root_amp = float(np.mean(np.sqrt(abs_x)) ** 2)
    return {
        "rms": rms,
        "peak": peak,
        "mean_abs": mean_abs,
        "std": float(np.std(x)),
        "skewness": float(skew(x, bias=False)) if len(x) > 2 else 0.0,
        "kurtosis": float(kurtosis(x, fisher=False, bias=False)) if len(x) > 3 else 0.0,
        "crest_factor": peak / (rms + 1e-12),
        "impulse_factor": peak / (mean_abs + 1e-12),
        "shape_factor": rms / (mean_abs + 1e-12),
        "margin_factor": peak / (root_amp + 1e-12),
    }


def frequency_domain_features(signal: np.ndarray, fs: float = 25600.0, bands: int = 8) -> dict[str, float]:
    """Compute spectrum energy, centroid, entropy, and band energies."""
    x = np.asarray(signal, dtype=float).ravel()
    spec = np.abs(rfft(x - np.mean(x))) ** 2
    freq = rfftfreq(len(x), d=1.0 / fs)
    prob = spec / (np.sum(spec) + 1e-12)
    out = {
        "spectral_energy": float(np.sum(spec)),
        "spectral_centroid": float(np.sum(freq * prob)),
        "spectral_entropy": float(-np.sum(prob * np.log(prob + 1e-12))),
    }
    chunks = np.array_split(spec, bands)
    total = np.sum(spec) + 1e-12
    for idx, chunk in enumerate(chunks):
        out[f"band_energy_{idx + 1}"] = float(np.sum(chunk) / total)
    return out


def time_frequency_features(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> dict[str, float]:
    """Compute wavelet sub-band energy ratios."""
    x = np.asarray(signal, dtype=float).ravel()
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, level=min(level, max_level))
    energies = np.asarray([np.sum(c**2) for c in coeffs], dtype=float)
    total = float(np.sum(energies) + 1e-12)
    return {f"wavelet_energy_{idx}": float(value / total) for idx, value in enumerate(energies)}


def quality_metrics(series: np.ndarray) -> dict[str, float]:
    """Evaluate monotonicity, trendability, and robustness of a degradation feature."""
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)
    monotonicity = abs(np.sum(dy > 0) - np.sum(dy < 0)) / max(1, len(dy))
    t = np.arange(len(y), dtype=float)
    trendability = abs(np.corrcoef(t, y)[0, 1]) if len(y) > 2 and np.std(y) > 0 else 0.0
    smooth = pd.Series(y).rolling(5, center=True, min_periods=1).mean().to_numpy()
    robustness = float(np.exp(-np.mean(np.abs(y - smooth) / (np.abs(smooth) + 1e-12))))
    return {
        "monotonicity": float(monotonicity),
        "trendability": float(trendability),
        "robustness": robustness,
    }


def construct_pca_hi(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Build PCA-HI from bearing features."""
    numeric = feature_df.select_dtypes(include=[np.number]).drop(columns=["sample_id"], errors="ignore")
    z = StandardScaler().fit_transform(numeric)
    hi = PCA(n_components=1, random_state=42).fit_transform(z).ravel()
    if len(hi) > 2 and np.corrcoef(np.arange(len(hi)), hi)[0, 1] < 0:
        hi = -hi
    out = feature_df.copy()
    out["PCA_HI"] = MinMaxScaler().fit_transform(hi.reshape(-1, 1)).ravel()
    return out


def autoencoder_hi_interface(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Reserve the AutoEncoder-HI interface with a reproducible PCA fallback."""
    out = construct_pca_hi(feature_df)
    out["AE_HI"] = out["PCA_HI"]
    return out


def extract_bearing_file(path: Path, fs: float = 25600.0) -> dict[str, float]:
    """Extract features from one XJTU-SY bearing CSV/TXT file."""
    df = pd.read_csv(path, header=None)
    x = df.iloc[:, 0].to_numpy(dtype=float)
    tail = path.stem.split("_")[-1]
    feats: dict[str, float] = {"sample_id": float(tail) if tail.isdigit() else 0.0}
    feats.update(time_domain_features(x))
    feats.update(frequency_domain_features(x, fs=fs))
    feats.update(time_frequency_features(x))
    return feats


def process_xjtu_sy(root: Path, out_dir: Path, fs: float = 25600.0) -> dict[str, Any]:
    """Process XJTU-SY files when present; otherwise emit schema and interface files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".txt"}]) if root.exists() else []
    if not files:
        schema = {
            "status": "XJTU-SY dataset not found yet",
            "expected_output": "Bearing Latent Features",
            "feature_groups": ["time_domain", "frequency_domain", "time_frequency"],
            "time_domain_features": [
                "rms",
                "peak",
                "mean_abs",
                "std",
                "skewness",
                "kurtosis",
                "crest_factor",
                "impulse_factor",
                "shape_factor",
                "margin_factor",
            ],
            "frequency_domain_features": ["spectral_energy", "spectral_centroid", "spectral_entropy", "band_energy_*"],
            "time_frequency_features": ["wavelet_energy_*"],
            "metrics": ["monotonicity", "trendability", "robustness"],
            "downstream": ["bearing_encoder", "transfer_learning", "shared_latent_space"],
        }
        (out_dir / "bearing_latent_feature_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return schema

    rows = [extract_bearing_file(path, fs=fs) for path in files]
    feature_df = pd.DataFrame(rows).sort_values("sample_id").reset_index(drop=True)
    hi_df = autoencoder_hi_interface(feature_df)
    numeric = hi_df.select_dtypes(include=[np.number]).drop(columns=["sample_id"], errors="ignore")
    latent = PCA(n_components=min(8, numeric.shape[1]), random_state=42).fit_transform(StandardScaler().fit_transform(numeric))
    latent_df = pd.DataFrame(latent, columns=[f"bearing_latent_{idx + 1}" for idx in range(latent.shape[1])])
    latent_df.insert(0, "sample_id", hi_df.get("sample_id", pd.Series(np.arange(len(hi_df)))))

    feature_df.to_csv(out_dir / "xjtu_sy_bearing_features.csv", index=False)
    hi_df.to_csv(out_dir / "xjtu_sy_bearing_hi.csv", index=False)
    latent_df.to_csv(out_dir / "bearing_latent_features.csv", index=False)
    return {
        "status": "processed",
        "n_files": len(files),
        "feature_quality": {col: quality_metrics(hi_df[col].to_numpy()) for col in ["PCA_HI", "AE_HI"]},
        "latent_shape": list(latent_df.shape),
    }
