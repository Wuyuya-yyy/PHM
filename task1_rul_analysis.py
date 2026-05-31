"""Complete Task 1 health-stage and RUL analysis for the reaction wheel data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _load_config_source(project_root: Path) -> Path | None:
    config_path = project_root / "configs" / "default_config.yaml"
    if not config_path.exists():
        return None
    try:
        source = None
        for line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("source_data_dir:"):
                source_text = stripped.split(":", 1)[1].strip().strip("\"'")
                source = Path(source_text)
                break
        return source if source.exists() else None
    except Exception:
        return None


def _set_plot_style() -> None:
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
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _save_figure(path: Path, dpi: int = 320) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=dpi)
    plt.close()
    return str(path)


def _resolve_source_dir(project_root: Path, source_data_dir: str | None) -> Path:
    candidates: list[Path] = []
    if source_data_dir:
        candidates.append(Path(source_data_dir))
    config_source = _load_config_source(project_root)
    if config_source:
        candidates.append(config_source)
    candidates.extend(
        [
            project_root / "raw_data",
            project_root.parent / "2026杭电数模校赛题目",
        ]
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        if _find_attachment(candidate, "3500") and _find_attachment(candidate, "1800"):
            return candidate
    raise FileNotFoundError(
        "Cannot locate attachment CSV files. Pass --source-data-dir with the contest data directory."
    )


def _find_attachment(root: Path, token: str) -> Path | None:
    matches = sorted(
        p
        for p in root.rglob("*.csv")
        if token in p.name and "reaction_wheel" in p.name.lower()
    )
    return matches[0] if matches else None


def _read_attachments(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    a1_path = _find_attachment(source_dir, "3500")
    a2_path = _find_attachment(source_dir, "1800")
    if a1_path is None or a2_path is None:
        raise FileNotFoundError(f"Missing attachment CSV files under {source_dir}")
    return pd.read_csv(a1_path), pd.read_csv(a2_path), {
        "attachment1": str(a1_path),
        "attachment2": str(a2_path),
    }


def _exp_curve(t: np.ndarray, y0: float, amplitude: float, rate: float, t0: float, scale: float) -> np.ndarray:
    tau = (t - t0) / scale
    return y0 + amplitude * (np.exp(rate * tau) - 1.0)


def _fit_exponential_rul(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    current_time: float,
    label: str,
) -> dict[str, Any]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]
    t0 = float(t[0])
    scale = float(max(t[-1] - t[0], 1.0))
    y_span = float(max(y[-1] - y[0], np.nanmax(y) - np.nanmin(y), 1e-3))

    def model(tt: np.ndarray, y0: float, amplitude: float, rate: float) -> np.ndarray:
        return _exp_curve(tt, y0, amplitude, rate, t0, scale)

    try:
        popt, _ = curve_fit(
            model,
            t,
            y,
            p0=[float(y[0]), y_span, 1.0],
            bounds=([-np.inf, 1e-10, 1e-10], [np.inf, np.inf, 20.0]),
            maxfev=50000,
        )
        pred = model(t, *popt)
        y0, amplitude, rate = map(float, popt)
        crossing_arg = (threshold - y0) / amplitude + 1.0
        if threshold <= float(y[-1]):
            eol_time = current_time
        elif crossing_arg <= 1.0 or rate <= 0.0:
            eol_time = float("inf")
        else:
            eol_time = t0 + scale * float(np.log(crossing_arg) / rate)
        status = "ok"
    except Exception as exc:
        pred = np.repeat(float(np.nanmean(y)), len(y))
        y0, amplitude, rate = float(y[0]), 0.0, 0.0
        eol_time = float("nan")
        status = f"fit_failed: {exc}"

    rul = float(max(0.0, eol_time - current_time)) if np.isfinite(eol_time) else eol_time
    return {
        "method": label,
        "threshold": float(threshold),
        "current_value": float(y[-1]),
        "current_time": float(current_time),
        "predicted_eol_day": float(eol_time),
        "rul_days": rul,
        "parameters": {
            "y0": float(y0),
            "amplitude": float(amplitude),
            "rate": float(rate),
            "t0": t0,
            "time_scale": scale,
        },
        "metrics": _metrics(y, pred),
        "status": status,
    }


def _linear_recent_rul(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    current_time: float,
    window_days: float,
    label: str,
) -> dict[str, Any]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(values, dtype=float)
    mask = t >= (current_time - window_days)
    if mask.sum() < 5:
        mask = np.arange(len(t)) >= max(0, len(t) - 10)
    slope, intercept = np.polyfit(t[mask], y[mask], deg=1)
    pred = slope * t + intercept
    if slope <= 0:
        eol_time = float("inf")
    else:
        eol_time = float((threshold - intercept) / slope)
    rul = float(max(0.0, eol_time - current_time)) if np.isfinite(eol_time) else eol_time
    return {
        "method": label,
        "threshold": float(threshold),
        "current_value": float(y[-1]),
        "current_time": float(current_time),
        "predicted_eol_day": float(eol_time),
        "rul_days": rul,
        "parameters": {
            "slope_per_day": float(slope),
            "intercept": float(intercept),
            "window_days": float(window_days),
        },
        "metrics": _metrics(y[mask], pred[mask]),
        "status": "ok",
    }


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _build_common_hi(a1: pd.DataFrame, a2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common_cols = [
        col
        for col in ["current", "temperature", "speed_rpm"]
        if col in a1.columns and col in a2.columns and a1[col].nunique(dropna=True) > 1
    ]
    if not common_cols:
        raise ValueError("No nonconstant common degradation features found for calibrated HI.")

    a1_oriented = a1[common_cols].astype(float).copy()
    a2_oriented = a2[common_cols].astype(float).copy()
    directions: dict[str, str] = {}
    for col in common_cols:
        corr = np.corrcoef(a1["day"].to_numpy(dtype=float), a1_oriented[col].to_numpy(dtype=float))[0, 1]
        if np.isfinite(corr) and corr < 0:
            a1_oriented[col] = -a1_oriented[col]
            a2_oriented[col] = -a2_oriented[col]
            directions[col] = "decreasing_flipped"
        else:
            directions[col] = "increasing"

    feature_scaler = MinMaxScaler()
    a1_norm = feature_scaler.fit_transform(a1_oriented)
    a2_norm = feature_scaler.transform(a2_oriented)
    pca = PCA(n_components=1, random_state=42)
    a1_score = pca.fit_transform(a1_norm).ravel()
    a2_score = pca.transform(a2_norm).ravel()
    score_scaler = MinMaxScaler()
    a1_hi = score_scaler.fit_transform(a1_score.reshape(-1, 1)).ravel()
    a2_hi = score_scaler.transform(a2_score.reshape(-1, 1)).ravel()
    if np.corrcoef(a1["day"].to_numpy(dtype=float), a1_hi)[0, 1] < 0:
        a1_hi = 1.0 - a1_hi
        a2_hi = 1.0 - a2_hi

    a1_out = pd.DataFrame({"day": a1["day"], "HI_common_raw": a1_hi})
    a2_out = pd.DataFrame({"day": a2["day"], "HI_common_raw": a2_hi})
    a1_window = max(3, min(9, len(a1_out) // 10))
    a2_window = 61
    a1_out["HI_common_smooth"] = a1_out["HI_common_raw"].rolling(a1_window, min_periods=1).mean()
    a2_out["HI_common_smooth"] = a2_out["HI_common_raw"].rolling(a2_window, min_periods=1).mean()
    meta = {
        "feature_columns": common_cols,
        "directions": directions,
        "pca_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "threshold_definition": "Attachment 1 full-life endpoint in A1-calibrated common-feature HI space.",
        "failure_threshold": float(a1_out["HI_common_raw"].iloc[-1]),
    }
    return a1_out, a2_out, meta


def _stage_label(current_index: int, boundaries: list[int]) -> str:
    if current_index < boundaries[0]:
        return "Healthy"
    if current_index < boundaries[1]:
        return "Slow Degradation"
    return "Accelerated Degradation"


def _make_prediction_frame(
    a2: pd.DataFrame,
    current_result: dict[str, Any],
    theoretical_result: dict[str, Any],
    hi_result: dict[str, Any],
    horizon_day: int,
) -> pd.DataFrame:
    days = np.arange(int(a2["day"].min()), horizon_day + 1, dtype=float)
    out = pd.DataFrame({"day": days.astype(int)})
    for col, result in [
        ("current_exp", current_result),
        ("current_theoretical_exp", theoretical_result),
        ("common_hi_exp", hi_result),
    ]:
        p = result["parameters"]
        out[col] = _exp_curve(
            days,
            p["y0"],
            p["amplitude"],
            p["rate"],
            p["t0"],
            p["time_scale"],
        )
    return out


def _plot_current_rul(
    a2: pd.DataFrame,
    pred: pd.DataFrame,
    current_threshold: float,
    theoretical_threshold: float,
    current_result: dict[str, Any],
    theoretical_result: dict[str, Any],
    path: Path,
    dpi: int,
) -> str:
    _set_plot_style()
    plt.figure(figsize=(7.2, 4.1))
    plt.plot(a2["day"], a2["current"], color="#4A5568", linewidth=0.8, alpha=0.65, label="Observed current")
    plt.plot(a2["day"], a2["current_theoretical"], color="#2B6CB0", linewidth=1.0, label="Theoretical current")
    plt.plot(pred["day"], pred["current_exp"], color="#D62728", linewidth=1.2, label="Measured-current exponential extrapolation")
    plt.plot(pred["day"], pred["current_theoretical_exp"], color="#2F855A", linewidth=1.2, label="Theoretical-current exponential extrapolation")
    plt.axhline(current_threshold, color="#D62728", linestyle="--", linewidth=0.9, label="Measured current failure threshold")
    plt.axhline(theoretical_threshold, color="#2F855A", linestyle="--", linewidth=0.9, label="Theoretical current failure threshold")
    for result, color in [(current_result, "#D62728"), (theoretical_result, "#2F855A")]:
        if np.isfinite(result["predicted_eol_day"]):
            plt.axvline(result["predicted_eol_day"], color=color, linestyle=":", linewidth=0.9)
    plt.axvline(float(a2["day"].iloc[-1]), color="#111111", linestyle="-.", linewidth=0.9, label="Current day")
    plt.xlabel("Operating day")
    plt.ylabel("Current / A")
    plt.title("Attachment 2 RUL Extrapolation by Current Degradation")
    plt.legend(fontsize=7, ncol=2)
    return _save_figure(path, dpi=dpi)


def _plot_hi_rul(
    a1_hi: pd.DataFrame,
    a2_hi: pd.DataFrame,
    pred: pd.DataFrame,
    threshold: float,
    hi_result: dict[str, Any],
    linear_result: dict[str, Any],
    path: Path,
    dpi: int,
) -> str:
    _set_plot_style()
    plt.figure(figsize=(7.2, 4.1))
    plt.plot(a1_hi["day"], a1_hi["HI_common_raw"], color="#A0AEC0", linewidth=0.8, label="Attachment 1 calibrated HI")
    plt.plot(a2_hi["day"], a2_hi["HI_common_raw"], color="#4A5568", linewidth=0.6, alpha=0.45, label="Attachment 2 calibrated HI raw")
    plt.plot(a2_hi["day"], a2_hi["HI_common_smooth"], color="#2B6CB0", linewidth=1.1, label="Attachment 2 calibrated HI smooth")
    plt.plot(pred["day"], pred["common_hi_exp"], color="#D62728", linewidth=1.2, label="HI exponential extrapolation")
    plt.axhline(threshold, color="#D62728", linestyle="--", linewidth=0.9, label="Failure threshold")
    for result, color, label in [(hi_result, "#D62728", "HI exp EOL"), (linear_result, "#2F855A", "Recent-linear EOL")]:
        if np.isfinite(result["predicted_eol_day"]):
            plt.axvline(result["predicted_eol_day"], color=color, linestyle=":", linewidth=0.9, label=label)
    plt.axvline(float(a2_hi["day"].iloc[-1]), color="#111111", linestyle="-.", linewidth=0.9, label="Current day")
    plt.xlabel("Operating day")
    plt.ylabel("A1-calibrated common HI")
    plt.title("Attachment 2 RUL Extrapolation by Comparable Health Index")
    plt.legend(fontsize=7, ncol=2)
    return _save_figure(path, dpi=dpi)


def _write_report(summary: dict[str, Any], path: Path) -> None:
    methods = summary["rul_methods"]
    lines = [
        "# Task 1 Completion Report: Reaction Wheel Degradation and RUL",
        "",
        "## 1. Completion Status",
        "",
        "Task 1 is completed as a self-contained flywheel PHM baseline: degradation indicators are identified, multiple degradation models are compared, full-life stages are segmented, and Attachment 2 is assigned a current health stage with RUL estimates.",
        "",
        "## 2. Effective Degradation Indicators",
        "",
        "- Current is the primary degradation fingerprint because it tracks the torque needed to overcome increasing bearing friction.",
        "- Temperature provides a coupled thermal degradation signal.",
        "- Friction torque is available in Attachment 1 and directly supports the physical interpretation.",
        "- The constructed HI fuses common current-temperature degradation evidence into a single monotonic health state.",
        "",
        "## 3. Stage Assessment",
        "",
        f"- Recommended segmentation method: `{summary['stage_assessment']['recommended_method']}`.",
        f"- Attachment 2 Bayesian-style boundaries: `{summary['stage_assessment']['boundaries']}`.",
        f"- Current operating day: `{summary['stage_assessment']['current_day']}`.",
        f"- Current health stage: **{summary['stage_assessment']['current_stage']}**.",
        "",
        "## 4. Failure Threshold Definition",
        "",
        "The failure state is anchored by the full-life endpoint of Attachment 1. This gives a data-driven end-of-life reference rather than using the maximum value inside Attachment 2, which is only a truncated in-orbit monitoring window.",
        "",
        f"- Measured current threshold: `{summary['thresholds']['measured_current_failure']:.6f}` A.",
        f"- Theoretical current threshold: `{summary['thresholds']['theoretical_current_failure']:.6f}` A.",
        f"- Comparable HI threshold: `{summary['thresholds']['common_hi_failure']:.6f}`.",
        "",
        "## 5. Attachment 2 RUL Results",
        "",
        "| Method | Predicted EOL Day | RUL / days | In-sample R2 |",
        "|---|---:|---:|---:|",
    ]
    for item in methods:
        eol = item["predicted_eol_day"]
        rul = item["rul_days"]
        r2 = item["metrics"].get("R2", float("nan"))
        eol_text = f"{eol:.1f}" if np.isfinite(eol) else "inf"
        rul_text = f"{rul:.1f}" if np.isfinite(rul) else "inf"
        lines.append(f"| {item['method']} | {eol_text} | {rul_text} | {r2:.4f} |")
    lines.extend(
        [
            "",
            f"Recommended RUL: **{summary['recommended_result']['rul_days']:.1f} days**, corresponding to predicted EOL day **{summary['recommended_result']['predicted_eol_day']:.1f}**. The recommended result uses the A1-calibrated common HI exponential model because it transfers the full-life failure threshold from Attachment 1 while using Attachment 2's observed degradation trend.",
            "",
            "## 6. Generated Files",
            "",
            f"- Summary JSON: `{summary['outputs']['summary_json']}`",
            f"- Prediction CSV: `{summary['outputs']['prediction_csv']}`",
            f"- Current RUL figure: `{summary['outputs']['current_rul_figure']}`",
            f"- HI RUL figure: `{summary['outputs']['hi_rul_figure']}`",
            "",
            "## 7. Assumptions and Limits",
            "",
            "- The failure endpoint is defined by Attachment 1's full-life endpoint.",
            "- Attachment 2 is treated as a truncated monitoring sequence of the same 1 Nms-class reaction wheel.",
            "- The RUL is a Task-1 no-transfer baseline; bearing-to-flywheel transfer learning is left for Task 3.",
            "- The LSTM result in the existing project is an in-sample one-step baseline and is not used as the main long-horizon RUL estimator here.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run(source_data_dir: str | None = None) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    source_dir = _resolve_source_dir(project_root, source_data_dir)
    a1, a2, source_paths = _read_attachments(source_dir)
    stage_path = project_root / "results" / "attachment2_reaction_wheel_1800d_data_stage_boundaries.json"
    boundaries = json.loads(stage_path.read_text(encoding="utf-8"))["bayesian"]
    current_day = float(a2["day"].iloc[-1])
    current_stage = _stage_label(len(a2) - 1, boundaries)

    a1_common_hi, a2_common_hi, common_hi_meta = _build_common_hi(a1, a2)
    current_threshold = float(a1["current"].iloc[-1])
    theoretical_threshold = float(a1["current_theoretical"].iloc[-1])
    common_hi_threshold = float(common_hi_meta["failure_threshold"])

    current_exp = _fit_exponential_rul(
        a2["day"].to_numpy(),
        a2["current"].to_numpy(),
        current_threshold,
        current_day,
        "Measured current exponential",
    )
    theoretical_exp = _fit_exponential_rul(
        a2["day"].to_numpy(),
        a2["current_theoretical"].to_numpy(),
        theoretical_threshold,
        current_day,
        "Theoretical current exponential",
    )
    common_hi_exp = _fit_exponential_rul(
        a2_common_hi["day"].to_numpy(),
        a2_common_hi["HI_common_smooth"].to_numpy(),
        common_hi_threshold,
        current_day,
        "A1-calibrated common HI exponential",
    )
    common_hi_linear = _linear_recent_rul(
        a2_common_hi["day"].to_numpy(),
        a2_common_hi["HI_common_smooth"].to_numpy(),
        common_hi_threshold,
        current_day,
        180.0,
        "A1-calibrated common HI recent-linear",
    )

    recommended = common_hi_exp
    finite_eol = [
        item["predicted_eol_day"]
        for item in [current_exp, theoretical_exp, common_hi_exp, common_hi_linear]
        if np.isfinite(item["predicted_eol_day"])
    ]
    horizon_day = int(max(finite_eol + [current_day + 365]))
    horizon_day = max(horizon_day + 120, int(current_day + 365))
    pred_frame = _make_prediction_frame(a2, current_exp, theoretical_exp, common_hi_exp, horizon_day)

    results_dir = project_root / "results"
    figures_dir = project_root / "figures" / "rul"
    reports_dir = project_root / "reports"
    prediction_csv = results_dir / "task1_rul_predictions.csv"
    pred_frame.to_csv(prediction_csv, index=False)

    current_fig = _plot_current_rul(
        a2,
        pred_frame,
        current_threshold,
        theoretical_threshold,
        current_exp,
        theoretical_exp,
        figures_dir / "attachment2_current_rul_extrapolation.png",
        dpi=320,
    )
    hi_fig = _plot_hi_rul(
        a1_common_hi,
        a2_common_hi,
        pred_frame,
        common_hi_threshold,
        common_hi_exp,
        common_hi_linear,
        figures_dir / "attachment2_common_hi_rul_extrapolation.png",
        dpi=320,
    )

    summary: dict[str, Any] = {
        "task": "Task 1 reaction wheel degradation modeling and health assessment",
        "source_data_dir": str(source_dir),
        "source_files": source_paths,
        "stage_assessment": {
            "recommended_method": "bayesian",
            "boundaries": boundaries,
            "current_day": current_day,
            "current_index": int(len(a2) - 1),
            "current_stage": current_stage,
        },
        "thresholds": {
            "definition": "Attachment 1 full-life endpoint is used as failure reference.",
            "measured_current_failure": current_threshold,
            "theoretical_current_failure": theoretical_threshold,
            "common_hi_failure": common_hi_threshold,
        },
        "common_hi": common_hi_meta,
        "rul_methods": [current_exp, theoretical_exp, common_hi_exp, common_hi_linear],
        "recommended_result": recommended,
        "outputs": {
            "summary_json": str(results_dir / "task1_rul_summary.json"),
            "prediction_csv": str(prediction_csv),
            "current_rul_figure": current_fig,
            "hi_rul_figure": hi_fig,
            "report": str(reports_dir / "task1_completion_report.md"),
        },
    }
    summary_json = results_dir / "task1_rul_summary.json"
    summary_json.write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(_jsonable(summary), reports_dir / "task1_completion_report.md")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete Task 1 RUL analysis.")
    parser.add_argument("--source-data-dir", default=None, help="Directory containing attachment 1/2 CSV files.")
    args = parser.parse_args()
    summary = run(args.source_data_dir)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
