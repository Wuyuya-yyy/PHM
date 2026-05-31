"""Task 3 transfer learning and Task 4 health-management reporting.

This module builds on the latest Task-1 RUL baseline produced by
``task1_rul_analysis.py``.  It does not replace Task 1; it calibrates the
Task-1 flywheel RUL with degradation knowledge learned from XJTU-SY bearings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


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


def _resample_curve(values: np.ndarray, size: int = 200) -> np.ndarray:
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, size)
    curve = np.interp(x_new, x_old, values)
    return (curve - np.nanmin(curve)) / (np.nanmax(curve) - np.nanmin(curve) + 1e-12)


def _bearing_reference(bearing_hi: pd.DataFrame, feature_quality: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    curves: list[np.ndarray] = []
    for (condition, bearing_id), group in bearing_hi.groupby(["condition", "bearing_id"], sort=True):
        group = group.sort_values("sample_id")
        hi = group["HI_smooth"].to_numpy(dtype=float)
        n = len(hi)
        late = hi[int(0.7 * n) :]
        slope = np.polyfit(np.linspace(0, 1, n), hi, deg=1)[0] if n > 2 else 0.0
        curves.append(_resample_curve(hi))
        rows.append(
            {
                "condition": condition,
                "bearing_id": bearing_id,
                "life_samples": int(n),
                "late_hi_mean": float(np.mean(late)),
                "final_hi": float(hi[-1]),
                "normalized_slope": float(slope),
                "physical_comparability": (
                    "outer-ring comparable"
                    if bearing_id in {"Bearing1_1", "Bearing3_1", "Bearing3_3"}
                    else "general bearing degradation"
                ),
            }
        )
    table = pd.DataFrame(rows)
    top = feature_quality.head(10)
    return {
        "per_bearing": table,
        "mean_curve": np.mean(curves, axis=0),
        "median_late_hi": float(table["late_hi_mean"].median()),
        "median_final_hi": float(table["final_hi"].median()),
        "median_normalized_slope": float(table["normalized_slope"].median()),
        "top_feature_transferability": float(top["transferability"].mean()),
        "top_feature_trendability": float(top["trendability"].mean()),
        "top_feature_score": float(top["overall_score"].mean()),
    }


def _load_task1_baseline(project_root: Path) -> dict[str, Any]:
    task1_path = project_root / "results" / "task1_rul_summary.json"
    if not task1_path.exists():
        from task1_rul_analysis import run

        run()
    return json.loads(task1_path.read_text(encoding="utf-8"))


def _transfer_calibrate(task1: dict[str, Any], bearing_ref: dict[str, Any]) -> dict[str, Any]:
    baseline = task1["recommended_result"]
    base_rul = float(baseline["rul_days"])
    base_hi = float(baseline["current_value"])
    bearing_late = float(bearing_ref["median_late_hi"])
    severity_scale = float(np.clip(base_hi / (bearing_late + 1e-12), 0.85, 1.20))
    transfer_confidence = float(
        np.clip(
            0.45 * bearing_ref["top_feature_transferability"]
            + 0.35 * bearing_ref["top_feature_trendability"]
            + 0.20 * bearing_ref["top_feature_score"],
            0.0,
            1.0,
        )
    )
    transferred_rul = base_rul / severity_scale
    interval_width = 0.30 * (1.0 - transfer_confidence) + 0.12
    return {
        "baseline_method": baseline["method"],
        "baseline_rul_days": base_rul,
        "baseline_eol_day": float(baseline["predicted_eol_day"]),
        "baseline_current_value": base_hi,
        "bearing_late_hi_reference": bearing_late,
        "severity_scale": severity_scale,
        "transfer_confidence": transfer_confidence,
        "transferred_rul_days": float(transferred_rul),
        "transferred_eol_day": float(task1["stage_assessment"]["current_day"] + transferred_rul),
        "transferred_rul_interval": [
            float(max(0.0, transferred_rul * (1.0 - interval_width))),
            float(transferred_rul * (1.0 + interval_width)),
        ],
    }


def _warning_level(stage: str, hi: float, rul_days: float) -> dict[str, str]:
    if stage == "Accelerated Degradation" and (hi >= 0.85 or rul_days <= 365):
        return {
            "level": "Level 3 - Critical",
            "recommended_action": "限制高负载机动，准备冗余轮切换或任务降级方案，并提高遥测采样频率。",
        }
    if stage == "Accelerated Degradation" or hi >= 0.70 or rul_days <= 730:
        return {
            "level": "Level 2 - Warning",
            "recommended_action": "进入重点监测，缩短健康评估周期，限制连续长时间高转速工作。",
        }
    if stage == "Slow Degradation" or hi >= 0.50:
        return {
            "level": "Level 1 - Attention",
            "recommended_action": "保持常规运行，但将该飞轮纳入趋势跟踪清单。",
        }
    return {"level": "Level 0 - Normal", "recommended_action": "维持常规监测。"}


def _plot_domain_similarity(
    project_root: Path,
    bearing_mean_curve: np.ndarray,
    task1_predictions: pd.DataFrame,
    dpi: int,
) -> tuple[str, dict[str, float]]:
    common_hi = task1_predictions["common_hi_exp"].to_numpy(dtype=float)
    common_hi = common_hi[np.isfinite(common_hi)]
    flywheel_curve = _resample_curve(common_hi[: max(3, min(len(common_hi), 2409))])
    bearing_curve = bearing_mean_curve
    similarity = float(cosine_similarity(flywheel_curve.reshape(1, -1), bearing_curve.reshape(1, -1))[0, 0])
    rmse = float(np.sqrt(np.mean((flywheel_curve - bearing_curve) ** 2)))
    corr = float(np.corrcoef(flywheel_curve, bearing_curve)[0, 1])

    _set_plot_style()
    plt.figure(figsize=(6.4, 4.0))
    x = np.linspace(0, 1, len(flywheel_curve))
    plt.plot(x, flywheel_curve, color="#2B6CB0", linewidth=1.8, label="Flywheel common-HI trajectory")
    plt.plot(x, bearing_curve, color="#D1495B", linewidth=1.8, label="XJTU-SY bearing mean HI trajectory")
    plt.xlabel("Normalized life progress")
    plt.ylabel("Normalized HI")
    plt.title("Task 3 Cross-domain Degradation Trajectory Similarity")
    plt.legend()
    path = project_root / "figures" / "transfer_health" / "flywheel_bearing_domain_similarity.png"
    fig_path = _save_figure(path, dpi=dpi)
    return fig_path, {"cosine_similarity": similarity, "trajectory_rmse": rmse, "trend_correlation": corr}


def _plot_rul_comparison(
    project_root: Path,
    task1_predictions: pd.DataFrame,
    task1: dict[str, Any],
    transfer: dict[str, Any],
    dpi: int,
) -> str:
    _set_plot_style()
    current_day = float(task1["stage_assessment"]["current_day"])
    threshold = float(task1["thresholds"]["common_hi_failure"])
    base_eol = float(transfer["baseline_eol_day"])
    transfer_eol = float(transfer["transferred_eol_day"])

    plt.figure(figsize=(7.2, 4.1))
    plt.plot(task1_predictions["day"], task1_predictions["common_hi_exp"], color="#2B6CB0", linewidth=1.3, label="Task-1 common-HI extrapolation")
    plt.axhline(threshold, color="#4A5568", linestyle="--", linewidth=0.9, label="Failure threshold")
    plt.axvline(current_day, color="#111111", linestyle="-.", linewidth=0.9, label="Current day")
    plt.axvline(base_eol, color="#2B6CB0", linestyle=":", linewidth=1.1, label=f"No-transfer EOL {base_eol:.1f}")
    plt.axvline(transfer_eol, color="#D1495B", linestyle=":", linewidth=1.1, label=f"Bearing-transfer EOL {transfer_eol:.1f}")
    low, high = transfer["transferred_rul_interval"]
    plt.axvspan(current_day + low, current_day + high, color="#D1495B", alpha=0.16, label="Transfer RUL interval")
    plt.xlabel("Operating day")
    plt.ylabel("A1-calibrated common HI")
    plt.title("Task 3 RUL Comparison: Task-1 Baseline vs Bearing Transfer")
    plt.legend(fontsize=7, ncol=2)
    path = project_root / "figures" / "transfer_health" / "attachment2_transfer_rul_comparison.png"
    return _save_figure(path, dpi=dpi)


def _write_health_report(project_root: Path, result: dict[str, Any], feature_quality: pd.DataFrame) -> str:
    report_path = project_root / "reports" / "health_management_report_cn.md"
    transfer = result["bearing_transfer"]
    warning = result["warning"]
    lines = [
        "# 任务三与任务四报告：跨领域迁移学习与飞轮健康管理",
        "",
        "## 1. 当前健康状态",
        "",
        f"附件 2 当前观测至第 `{result['current_day']:.0f}` 天，任务一判定当前阶段为 `{result['current_stage']}`。",
        f"任务一 A1-calibrated common HI 当前值为 `{transfer['baseline_current_value']:.4f}`，无迁移推荐 RUL 为 `{transfer['baseline_rul_days']:.1f}` 天。",
        f"结合轴承迁移校准后，RUL 为 `{transfer['transferred_rul_days']:.1f}` 天，区间为 `[{transfer['transferred_rul_interval'][0]:.1f}, {transfer['transferred_rul_interval'][1]:.1f}]` 天。",
        f"当前预警等级为：`{warning['level']}`。",
        "",
        "## 2. 迁移学习合理性",
        "",
        "滚动轴承和反作用轮轴承具有相同的关键退化机理：润滑状态恶化导致摩擦增强，进而表现为振动能量、电流或温度等观测量上升。",
        f"归一化寿命 HI 轨迹相似度为 `{result['domain_similarity']['cosine_similarity']:.4f}`，趋势相关系数为 `{result['domain_similarity']['trend_correlation']:.4f}`，说明两类退化过程在健康状态空间中具有较强同向性。",
        "",
        "## 3. 迁移策略",
        "",
        "本阶段采用可解释的严重度校准迁移：保留队友任务一给出的 A1-calibrated common HI RUL 作为无迁移基线，再利用 XJTU-SY 轴承后期退化 HI 分布和高质量退化特征，对附件 2 的 RUL 进行保守校准。",
        "该策略迁移的是归一化退化状态和后期严重度，而不是直接迁移分钟级轴承寿命，因此避免了轴承加速寿命试验和飞轮在轨天级数据之间的时间尺度冲突。",
        "",
        "## 4. RUL 对比",
        "",
        "| 方法 | EOL day | RUL / days |",
        "|---|---:|---:|",
        f"| 任务一无迁移基线 | {transfer['baseline_eol_day']:.1f} | {transfer['baseline_rul_days']:.1f} |",
        f"| 轴承迁移校准 | {transfer['transferred_eol_day']:.1f} | {transfer['transferred_rul_days']:.1f} |",
        "",
        f"迁移校准因子 severity_scale = `{transfer['severity_scale']:.4f}`，迁移置信度 = `{transfer['transfer_confidence']:.4f}`。",
        "",
        "## 5. 预警机制与建议",
        "",
        "| 等级 | 判据 | 含义 |",
        "|---|---|---|",
        "| Level 0 - Normal | 健康期且 RUL > 730 天 | 正常运行 |",
        "| Level 1 - Attention | 缓慢退化或 HI 升高 | 趋势关注 |",
        "| Level 2 - Warning | 加速退化或 RUL <= 730 天 | 重点监测 |",
        "| Level 3 - Critical | 加速退化且 HI 高或 RUL <= 365 天 | 高风险 |",
        "",
        f"当前建议：{warning['recommended_action']}",
        "",
        "## 6. 高质量迁移特征 Top 5",
        "",
        "| 特征 | 单调性 | 趋势性 | 稳定性 | 可迁移性 | 综合评分 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in feature_quality.head(5).iterrows():
        lines.append(
            f"| {row['feature']} | {row['monotonicity']:.4f} | {row['trendability']:.4f} | {row['stability']:.4f} | {row['transferability']:.4f} | {row['overall_score']:.4f} |"
        )
    lines += [
        "",
        "## 7. 不确定性与局限性",
        "",
        "- XJTU-SY 是地面加速寿命试验，飞轮附件 2 是在轨监测数据，两者工作环境和采样模态不同。",
        "- 当前迁移方法强调可解释性和稳健性，尚不是端到端深度域适配模型。",
        "- 附件 2 缺少真实失效终点，因此 RUL 是模型外推估计。",
        "- 预警等级应作为工程辅助决策，不能替代姿控系统安全规则。",
        "",
        "## 8. 图像",
        "",
    ]
    for fig in result["figures"]:
        rel = Path(fig).relative_to(project_root)
        lines.append(f"![{rel.stem}](../{rel.as_posix()})")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def generate_transfer_health_management(project_root: Path, dpi: int = 320) -> dict[str, Any]:
    """Generate Task-3 transfer analysis and Task-4 health-management outputs."""
    task1 = _load_task1_baseline(project_root)
    bearing_hi = pd.read_csv(project_root / "results" / "bearing" / "xjtu_sy_bearing_hi.csv")
    bearing_latent = pd.read_csv(project_root / "results" / "bearing" / "bearing_latent_features.csv")
    feature_quality = pd.read_csv(project_root / "results" / "bearing" / "xjtu_sy_feature_quality.csv")
    task1_predictions = pd.read_csv(project_root / "results" / "task1_rul_predictions.csv")

    bearing_ref = _bearing_reference(bearing_hi, feature_quality)
    transfer = _transfer_calibrate(task1, bearing_ref)
    stage = task1["stage_assessment"]["current_stage"]
    warning = _warning_level(stage, transfer["baseline_current_value"], transfer["transferred_rul_days"])
    sim_fig, similarity = _plot_domain_similarity(project_root, bearing_ref["mean_curve"], task1_predictions, dpi)
    rul_fig = _plot_rul_comparison(project_root, task1_predictions, task1, transfer, dpi)

    result = {
        "status": "processed",
        "task1_baseline_source": str(project_root / "results" / "task1_rul_summary.json"),
        "current_day": float(task1["stage_assessment"]["current_day"]),
        "current_stage": stage,
        "bearing_reference": {
            "median_late_hi": bearing_ref["median_late_hi"],
            "median_final_hi": bearing_ref["median_final_hi"],
            "median_normalized_slope": bearing_ref["median_normalized_slope"],
            "top_feature_transferability": bearing_ref["top_feature_transferability"],
            "top_feature_trendability": bearing_ref["top_feature_trendability"],
            "top_feature_score": bearing_ref["top_feature_score"],
        },
        "bearing_transfer": transfer,
        "domain_similarity": similarity,
        "warning": warning,
        "figures": [rul_fig, sim_fig],
    }
    result_path = project_root / "results" / "transfer_health_management_results.json"
    result_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_health_report(project_root, result, feature_quality)
    result["outputs"] = {"result_json": str(result_path), "report": report_path}
    result_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(_jsonable(generate_transfer_health_management(root)), ensure_ascii=False, indent=2))
