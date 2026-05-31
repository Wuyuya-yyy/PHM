"""Bearing degradation-model comparison for Task 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _exp_model(x: np.ndarray, c: float, a: float, b: float) -> np.ndarray:
    return c + a * (np.exp(b * x) - 1.0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _fit_exponential(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        popt, _ = curve_fit(_exp_model, x, y, p0=[float(y[0]), max(float(y[-1] - y[0]), 1e-3), 1.0], maxfev=20000)
        pred = _exp_model(x, *popt)
        return pred, {"parameters": [float(v) for v in popt], "metrics": _metrics(y, pred), "status": "ok"}
    except Exception as exc:
        pred = np.repeat(float(np.mean(y)), len(y))
        return pred, {"parameters": [], "metrics": _metrics(y, pred), "status": f"failed: {exc}"}


def _fit_wiener(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    increments = np.diff(y)
    drift = float(np.mean(increments)) if len(increments) else 0.0
    diffusion = float(np.std(increments, ddof=1)) if len(increments) > 1 else 0.0
    pred = y[0] + drift * np.arange(len(y))
    return pred, {"drift": drift, "diffusion": diffusion, "metrics": _metrics(y, pred), "status": "ok"}


def _fit_rf(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    features = np.column_stack([x, x**2, np.sqrt(np.maximum(x, 0.0))])
    model = RandomForestRegressor(n_estimators=120, max_depth=5, random_state=42, min_samples_leaf=3)
    model.fit(features, y)
    pred = model.predict(features)
    return pred, {"metrics": _metrics(y, pred), "status": "ok", "model": "RandomForestRegressor"}


def _stage_boundaries(group: pd.DataFrame) -> tuple[int, int]:
    labels = group["stage"].to_numpy()
    slow = np.where(labels == "Slow Degradation")[0]
    accel = np.where(labels == "Accelerated Degradation")[0]
    b1 = int(slow[0]) if len(slow) else max(1, len(group) // 3)
    b2 = int(accel[0]) if len(accel) else max(b1 + 1, 2 * len(group) // 3)
    return b1, b2


def _plot_representative(
    group: pd.DataFrame,
    preds: dict[str, np.ndarray],
    out_dir: Path,
    dpi: int,
) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    condition = str(group["condition"].iloc[0])
    bearing_id = str(group["bearing_id"].iloc[0])
    x = np.linspace(0, 1, len(group))
    y = group["HI_smooth"].to_numpy(dtype=float)
    b1, b2 = _stage_boundaries(group)
    plt.figure(figsize=(7.2, 4.1))
    plt.plot(x, y, color="#2D3748", linewidth=1.2, label="Observed Bearing HI")
    colors = {"exponential": "#D1495B", "wiener": "#2A9D8F", "random_forest": "#2B6CB0"}
    for name, pred in preds.items():
        plt.plot(x, pred, linewidth=1.1, label=name, color=colors[name])
    plt.axvline(x[b1], color="#F2C14E", linestyle="--", linewidth=0.9, label="stage boundary")
    plt.axvline(x[b2], color="#F2C14E", linestyle="--", linewidth=0.9)
    plt.xlabel("Normalized life progress")
    plt.ylabel("Bearing HI")
    plt.title(f"Task 2 Bearing Degradation Model Comparison: {condition} {bearing_id}")
    plt.legend(fontsize=8)
    path = out_dir / f"{condition}_{bearing_id}_degradation_model_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
    return str(path)


def run_bearing_degradation_modeling(project_root: Path, dpi: int = 320) -> dict[str, Any]:
    hi_path = project_root / "results" / "bearing" / "xjtu_sy_bearing_hi.csv"
    hi = pd.read_csv(hi_path)
    rows: list[dict[str, Any]] = []
    figures: list[str] = []
    representative_keys = {("35Hz12kN", "Bearing1_1"), ("37.5Hz11kN", "Bearing2_1"), ("40Hz10kN", "Bearing3_1")}

    for (condition, bearing_id), group in hi.groupby(["condition", "bearing_id"], sort=True):
        group = group.sort_values("sample_id").reset_index(drop=True)
        x = np.linspace(0, 1, len(group))
        y = group["HI_smooth"].to_numpy(dtype=float)
        exp_pred, exp_info = _fit_exponential(x, y)
        wiener_pred, wiener_info = _fit_wiener(x, y)
        rf_pred, rf_info = _fit_rf(x, y)
        b1, b2 = _stage_boundaries(group)
        model_infos = {"exponential": exp_info, "wiener": wiener_info, "random_forest": rf_info}
        best = min(model_infos, key=lambda k: model_infos[k]["metrics"]["RMSE"])
        rows.append(
            {
                "condition": condition,
                "bearing_id": bearing_id,
                "n_samples": int(len(group)),
                "healthy_to_slow_index": b1,
                "slow_to_accelerated_index": b2,
                "best_model": best,
                "exp_RMSE": exp_info["metrics"]["RMSE"],
                "exp_R2": exp_info["metrics"]["R2"],
                "wiener_RMSE": wiener_info["metrics"]["RMSE"],
                "wiener_R2": wiener_info["metrics"]["R2"],
                "rf_RMSE": rf_info["metrics"]["RMSE"],
                "rf_R2": rf_info["metrics"]["R2"],
            }
        )
        if (condition, bearing_id) in representative_keys:
            figures.append(
                _plot_representative(
                    group,
                    {"exponential": exp_pred, "wiener": wiener_pred, "random_forest": rf_pred},
                    project_root / "figures" / "bearing_models",
                    dpi,
                )
            )

    model_df = pd.DataFrame(rows)
    out_dir = project_root / "results" / "bearing_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "bearing_degradation_model_comparison.csv"
    model_df.to_csv(model_path, index=False)
    summary = {
        "status": "processed",
        "n_bearings": int(len(model_df)),
        "model_comparison_path": str(model_path),
        "figures": figures,
        "mean_metrics": {
            "exponential_RMSE": float(model_df["exp_RMSE"].mean()),
            "wiener_RMSE": float(model_df["wiener_RMSE"].mean()),
            "random_forest_RMSE": float(model_df["rf_RMSE"].mean()),
            "exponential_R2": float(model_df["exp_R2"].mean()),
            "wiener_R2": float(model_df["wiener_R2"].mean()),
            "random_forest_R2": float(model_df["rf_R2"].mean()),
        },
        "best_model_counts": model_df["best_model"].value_counts().to_dict(),
    }
    summary_path = out_dir / "bearing_degradation_model_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(run_bearing_degradation_modeling(root), ensure_ascii=False, indent=2))
