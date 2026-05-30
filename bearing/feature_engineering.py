"""XJTU-SY bearing feature extraction and latent-feature preparation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.fft import rfft, rfftfreq
from scipy.signal import stft
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
        "peak_to_peak": float(np.ptp(x)),
        "mean_abs": mean_abs,
        "mean": float(np.mean(x)),
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
        "dominant_frequency": float(freq[int(np.argmax(spec[1:])) + 1]) if len(spec) > 1 else 0.0,
    }
    chunks = np.array_split(spec, bands)
    total = np.sum(spec) + 1e-12
    for idx, chunk in enumerate(chunks):
        out[f"band_energy_{idx + 1}"] = float(np.sum(chunk) / total)
    return out


def time_frequency_features(signal: np.ndarray, wavelet: str = "db4", level: int = 3) -> dict[str, float]:
    """Compute wavelet-packet and STFT sub-band energy ratios."""
    x = np.asarray(signal, dtype=float).ravel()
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    wp_level = min(level, max_level)
    packet = pywt.WaveletPacket(data=x, wavelet=wavelet, mode="symmetric", maxlevel=wp_level)
    nodes = packet.get_level(wp_level, order="freq")
    energies = np.asarray([np.sum(node.data**2) for node in nodes], dtype=float)
    total = float(np.sum(energies) + 1e-12)
    out = {f"wavelet_packet_energy_{idx + 1}": float(value / total) for idx, value in enumerate(energies)}
    prob = energies / total
    out["wavelet_packet_entropy"] = float(-np.sum(prob * np.log(prob + 1e-12)))
    _, _, zxx = stft(x - np.mean(x), nperseg=min(1024, len(x)), noverlap=min(512, max(0, len(x) // 4)))
    power = np.abs(zxx) ** 2
    stft_band_power = np.asarray([np.sum(chunk) for chunk in np.array_split(power, 4, axis=0)], dtype=float)
    stft_total = float(np.sum(stft_band_power) + 1e-12)
    for idx in range(len(stft_band_power)):
        out[f"stft_band_energy_{idx + 1}"] = float(stft_band_power[idx] / stft_total)
    out["stft_total_energy"] = float(np.sum(stft_band_power))
    return out


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
    numeric = _feature_matrix(feature_df)
    if len(numeric) < 2 or numeric.shape[1] == 0:
        out = feature_df.copy()
        out["PCA_HI"] = np.linspace(0.0, 1.0, len(feature_df)) if len(feature_df) else []
        return out
    z = StandardScaler().fit_transform(numeric)
    pca = PCA(n_components=1, random_state=42)
    hi = pca.fit_transform(z).ravel()
    if len(hi) > 2 and np.corrcoef(np.arange(len(hi)), hi)[0, 1] < 0:
        hi = -hi
    out = feature_df.copy()
    out["PCA_HI"] = MinMaxScaler().fit_transform(hi.reshape(-1, 1)).ravel()
    out.attrs["pca_explained_variance_ratio"] = float(pca.explained_variance_ratio_[0])
    return out


def autoencoder_hi_interface(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Reserve the AutoEncoder-HI interface with a reproducible PCA fallback."""
    out = construct_pca_hi(feature_df)
    out["AE_HI"] = out["PCA_HI"]
    return out


def _sample_number(path: Path) -> int:
    match = re.search(r"\d+", path.stem)
    return int(match.group(0)) if match else 0


def _parse_xjtu_path(path: Path, root: Path) -> dict[str, Any]:
    rel = path.relative_to(root)
    parts = rel.parts
    condition = parts[0] if len(parts) >= 3 else "unknown_condition"
    bearing_id = parts[1] if len(parts) >= 3 else path.parent.name
    sample_id = _sample_number(path)
    return {
        "condition": condition,
        "bearing_id": bearing_id,
        "sample_id": sample_id,
        "sample_key": f"{condition}/{bearing_id}/{sample_id:05d}",
        "relative_path": rel.as_posix(),
    }


def _prefix_features(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _channel_feature_set(signal: np.ndarray, prefix: str, fs: float) -> dict[str, float]:
    out: dict[str, float] = {}
    out.update(_prefix_features(prefix, time_domain_features(signal)))
    out.update(_prefix_features(prefix, frequency_domain_features(signal, fs=fs)))
    out.update(_prefix_features(prefix, time_frequency_features(signal)))
    return out


def _feature_matrix(feature_df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = {"sample_id", "sample_key", "condition", "bearing_id", "relative_path", "PCA_HI", "AE_HI", "stage"}
    numeric = feature_df.select_dtypes(include=[np.number]).drop(columns=list(drop_cols), errors="ignore")
    return numeric.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _bearing_sort_key(path: Path) -> tuple[str, str, int]:
    return (path.parent.parent.name, path.parent.name, _sample_number(path))


def extract_bearing_file(path: Path, fs: float = 25600.0) -> dict[str, float]:
    """Extract features from one XJTU-SY bearing CSV/TXT file."""
    return extract_bearing_file_with_root(path, path.parents[2], fs=fs)


def extract_bearing_file_with_root(path: Path, root: Path, fs: float = 25600.0) -> dict[str, Any]:
    """Extract horizontal, vertical, and resultant vibration features from one file."""
    df = pd.read_csv(path)
    numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if numeric.shape[1] < 2:
        numeric = pd.read_csv(path, header=None).apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    horizontal = numeric.iloc[:, 0].dropna().to_numpy(dtype=float)
    vertical = numeric.iloc[:, 1].dropna().to_numpy(dtype=float) if numeric.shape[1] > 1 else horizontal
    n = min(len(horizontal), len(vertical))
    horizontal = horizontal[:n]
    vertical = vertical[:n]
    resultant = np.sqrt(horizontal**2 + vertical**2)
    feats: dict[str, Any] = _parse_xjtu_path(path, root)
    feats.update(_channel_feature_set(horizontal, "horizontal", fs))
    feats.update(_channel_feature_set(vertical, "vertical", fs))
    feats.update(_channel_feature_set(resultant, "resultant", fs))
    return feats


def _construct_group_hi(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    quality: dict[str, Any] = {}
    variance_ratios = []
    for (condition, bearing_id), group in feature_df.groupby(["condition", "bearing_id"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        hi_group = autoencoder_hi_interface(group)
        hi_values = hi_group["PCA_HI"].to_numpy()
        hi_group["HI_smooth"] = pd.Series(hi_values).rolling(9, center=True, min_periods=1).mean().to_numpy()
        q1 = float(np.quantile(hi_group["HI_smooth"], 1 / 3))
        q2 = float(np.quantile(hi_group["HI_smooth"], 2 / 3))
        hi_group["stage"] = np.select(
            [hi_group["HI_smooth"] < q1, hi_group["HI_smooth"] < q2],
            ["Healthy", "Slow Degradation"],
            default="Accelerated Degradation",
        )
        frames.append(hi_group)
        key = f"{condition}/{bearing_id}"
        quality[key] = quality_metrics(hi_group["HI_smooth"].to_numpy())
        quality[key]["n_samples"] = int(len(hi_group))
        variance_ratios.append(float(hi_group.attrs.get("pca_explained_variance_ratio", 0.0)))
    hi_df = pd.concat(frames, ignore_index=True) if frames else feature_df.copy()
    summary = {
        "per_bearing": quality,
        "mean_pca_explained_variance_ratio": float(np.mean(variance_ratios)) if variance_ratios else 0.0,
    }
    return hi_df, summary


def _evaluate_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Score every extracted feature for degradation representation quality."""
    feature_cols = list(_feature_matrix(feature_df).columns)
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        group_scores = []
        directions = []
        for (condition, bearing_id), group in feature_df.groupby(["condition", "bearing_id"], sort=True):
            series = group.sort_values("sample_id")[col].to_numpy(dtype=float)
            metrics = quality_metrics(series)
            direction = 0.0
            if len(series) > 2 and np.std(series) > 0:
                direction = float(np.corrcoef(np.arange(len(series), dtype=float), series)[0, 1])
            group_scores.append(metrics)
            directions.append(direction)
        if not group_scores:
            continue
        monotonicity = float(np.mean([item["monotonicity"] for item in group_scores]))
        trendability = float(np.mean([item["trendability"] for item in group_scores]))
        robustness = float(np.mean([item["robustness"] for item in group_scores]))
        transferability = float(1.0 - np.std([item["trendability"] for item in group_scores]))
        transferability = float(np.clip(transferability, 0.0, 1.0))
        overall = 0.35 * monotonicity + 0.30 * trendability + 0.20 * robustness + 0.15 * transferability
        rows.append(
            {
                "feature": col,
                "monotonicity": monotonicity,
                "trendability": trendability,
                "stability": robustness,
                "transferability": transferability,
                "degradation_direction": "increasing" if np.nanmean(directions) >= 0 else "decreasing",
                "overall_score": float(overall),
            }
        )
    return pd.DataFrame(rows).sort_values("overall_score", ascending=False).reset_index(drop=True)


def _build_latent_features(hi_df: pd.DataFrame, latent_dim: int = 8) -> pd.DataFrame:
    numeric = _feature_matrix(hi_df)
    if len(numeric) < 2 or numeric.shape[1] == 0:
        latent = np.zeros((len(hi_df), 1))
    else:
        n_components = min(latent_dim, numeric.shape[1], len(numeric))
        latent = PCA(n_components=n_components, random_state=42).fit_transform(StandardScaler().fit_transform(numeric))
    latent_df = pd.DataFrame(latent, columns=[f"bearing_latent_{idx + 1}" for idx in range(latent.shape[1])])
    meta_cols = [col for col in ["sample_key", "condition", "bearing_id", "sample_id", "PCA_HI", "HI_smooth", "stage"] if col in hi_df]
    latent_df = pd.concat([hi_df[meta_cols].reset_index(drop=True), latent_df], axis=1)
    return latent_df


def _plot_bearing_hi(hi_df: pd.DataFrame, fig_dir: Path, dpi: int = 320) -> list[str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for (condition, bearing_id), group in hi_df.groupby(["condition", "bearing_id"], sort=True):
        group = group.sort_values("sample_id")
        plt.figure(figsize=(8, 4))
        plt.plot(group["sample_id"], group["PCA_HI"], color="#8A4F7D", alpha=0.35, linewidth=1.0, label="PCA_HI")
        plt.plot(group["sample_id"], group["HI_smooth"], color="#276FBF", linewidth=1.8, label="HI_smooth")
        for stage, color in [
            ("Healthy", "#4C956C"),
            ("Slow Degradation", "#F2C14E"),
            ("Accelerated Degradation", "#D1495B"),
        ]:
            subset = group[group["stage"] == stage]
            if not subset.empty:
                plt.scatter(subset["sample_id"], subset["HI_smooth"], s=6, color=color, label=stage, alpha=0.75)
        plt.title(f"XJTU-SY {condition} {bearing_id} Health Indicator")
        plt.xlabel("Sample index")
        plt.ylabel("Normalized HI")
        plt.ylim(-0.05, 1.05)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=7, ncol=2)
        out_path = fig_dir / f"{condition}_{bearing_id}_bearing_hi.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        paths.append(str(out_path))
    return paths


def _plot_feature_quality(quality_df: pd.DataFrame, fig_dir: Path, dpi: int = 320, top_n: int = 20) -> str | None:
    """Plot the top degradation-sensitive features."""
    if quality_df.empty:
        return None
    fig_dir.mkdir(parents=True, exist_ok=True)
    top = quality_df.head(top_n).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["overall_score"], color="#2A9D8F")
    plt.xlabel("Overall degradation representation score")
    plt.ylabel("Feature")
    plt.title("Top XJTU-SY Bearing Feature Quality Scores")
    plt.grid(axis="x", alpha=0.25)
    out_path = fig_dir / "bearing_feature_quality_top20.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return str(out_path)


def process_xjtu_sy(root: Path, out_dir: Path, fs: float = 25600.0) -> dict[str, Any]:
    """Process XJTU-SY files when present; otherwise emit schema and interface files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".txt"}],
        key=_bearing_sort_key,
    ) if root.exists() else []
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
            "time_frequency_features": ["wavelet_packet_energy_*", "stft_band_energy_*"],
            "metrics": ["monotonicity", "trendability", "stability", "transferability"],
            "downstream": ["bearing_encoder", "transfer_learning", "shared_latent_space"],
        }
        (out_dir / "bearing_latent_feature_schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return schema

    rows = [extract_bearing_file_with_root(path, root, fs=fs) for path in files]
    feature_df = pd.DataFrame(rows).sort_values(["condition", "bearing_id", "sample_id"]).reset_index(drop=True)
    feature_quality_df = _evaluate_features(feature_df)
    hi_df, quality_summary = _construct_group_hi(feature_df)
    latent_df = _build_latent_features(hi_df)
    bearing_out = out_dir / "bearing"
    bearing_out.mkdir(parents=True, exist_ok=True)
    feature_path = bearing_out / "xjtu_sy_bearing_features.csv"
    feature_quality_path = bearing_out / "xjtu_sy_feature_quality.csv"
    hi_path = bearing_out / "xjtu_sy_bearing_hi.csv"
    latent_path = bearing_out / "bearing_latent_features.csv"
    summary_path = bearing_out / "bearing_feature_summary.json"
    feature_df.to_csv(feature_path, index=False)
    feature_quality_df.to_csv(feature_quality_path, index=False)
    hi_df.to_csv(hi_path, index=False)
    latent_df.to_csv(latent_path, index=False)
    figures = _plot_bearing_hi(hi_df, out_dir.parent / "figures" / "bearing")
    quality_figure = _plot_feature_quality(feature_quality_df, out_dir.parent / "figures" / "bearing")
    if quality_figure:
        figures.append(quality_figure)
    summary = {
        "status": "processed",
        "n_files": len(files),
        "n_conditions": int(feature_df["condition"].nunique()),
        "n_bearings": int(feature_df[["condition", "bearing_id"]].drop_duplicates().shape[0]),
        "feature_groups": ["time_domain", "frequency_domain", "wavelet_packet", "stft", "trend/stage"],
        "channels": ["horizontal", "vertical", "resultant"],
        "feature_path": str(feature_path),
        "feature_quality_path": str(feature_quality_path),
        "hi_path": str(hi_path),
        "latent_path": str(latent_path),
        "figures": figures,
        "feature_quality": quality_summary,
        "latent_shape": list(latent_df.shape),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        **summary,
        "summary_path": str(summary_path),
    }
