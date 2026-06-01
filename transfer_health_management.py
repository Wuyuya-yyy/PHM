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


def _stage_ratio_factor(task1: dict[str, Any], bearing_hi: pd.DataFrame) -> dict[str, float]:
    """Estimate a transfer factor from the proportion of accelerated degradation."""
    current_index = float(task1["stage_assessment"]["current_index"])
    current_progress = current_index / max(float(task1["stage_assessment"]["current_day"]), 1.0)
    accel_ratios = []
    for _, group in bearing_hi.groupby(["condition", "bearing_id"], sort=True):
        labels = group.sort_values("sample_id")["stage"].to_numpy()
        accel = np.where(labels == "Accelerated Degradation")[0]
        if len(accel):
            accel_start = float(accel[0]) / max(len(labels) - 1, 1)
            accel_ratios.append(1.0 - accel_start)
    bearing_accel_ratio = float(np.median(accel_ratios)) if accel_ratios else 1.0 / 3.0
    # Attachment 2 is already in accelerated degradation; a larger bearing late-stage
    # ratio means the accelerated stage is not necessarily immediately terminal.
    stage_factor = float(np.clip(1.0 + 0.30 * bearing_accel_ratio, 0.90, 1.20))
    return {
        "current_progress_proxy": current_progress,
        "bearing_accelerated_stage_ratio_median": bearing_accel_ratio,
        "stage_factor": stage_factor,
    }


def _build_transfer_comparison(task1: dict[str, Any], severity_transfer: dict[str, Any], stage_info: dict[str, float]) -> pd.DataFrame:
    base_rul = float(severity_transfer["baseline_rul_days"])
    severity_rul = float(severity_transfer["transferred_rul_days"])
    stage_rul = base_rul * stage_info["stage_factor"]
    combined_factor = (base_rul / severity_rul + 1.0 / stage_info["stage_factor"]) / 2.0
    combined_rul = base_rul / max(combined_factor, 1e-12)
    current_day = float(task1["stage_assessment"]["current_day"])
    rows = [
        {
            "method": "no_transfer",
            "description": "Task-1 A1-calibrated common HI baseline",
            "transfer_factor": 1.0,
            "rul_days": base_rul,
            "eol_day": current_day + base_rul,
        },
        {
            "method": "severity_transfer",
            "description": "Bearing late-life HI severity calibration",
            "transfer_factor": float(severity_transfer["severity_scale"]),
            "rul_days": severity_rul,
            "eol_day": current_day + severity_rul,
        },
        {
            "method": "stage_ratio_transfer",
            "description": "Bearing accelerated-stage proportion calibration",
            "transfer_factor": float(1.0 / stage_info["stage_factor"]),
            "rul_days": stage_rul,
            "eol_day": current_day + stage_rul,
        },
        {
            "method": "combined_transfer",
            "description": "Average of severity and stage-ratio transfer factors",
            "transfer_factor": float(combined_factor),
            "rul_days": float(combined_rul),
            "eol_day": current_day + combined_rul,
        },
    ]
    return pd.DataFrame(rows)


def _build_sensitivity(comparison: pd.DataFrame) -> pd.DataFrame:
    base = comparison[comparison["method"] == "combined_transfer"].iloc[0]
    base_factor = float(base["transfer_factor"])
    base_rul = float(comparison[comparison["method"] == "no_transfer"]["rul_days"].iloc[0])
    rows = []
    for delta in [-0.10, -0.05, 0.0, 0.05, 0.10]:
        factor = base_factor * (1.0 + delta)
        rows.append(
            {
                "factor_delta": delta,
                "transfer_factor": factor,
                "rul_days": base_rul / factor,
                "rul_change_vs_nominal_days": base_rul / factor - float(base["rul_days"]),
            }
        )
    return pd.DataFrame(rows)


def _observed_similarity(project_root: Path, bearing_ref: dict[str, Any], task1_predictions: pd.DataFrame, dpi: int) -> tuple[dict[str, float], str]:
    observed = task1_predictions[task1_predictions["day"] <= 1800]["common_hi_exp"].to_numpy(dtype=float)
    flywheel_curve = _resample_curve(observed)
    bearing_curve = bearing_ref["mean_curve"]
    similarity = {
        "observed_days": 1800,
        "cosine_similarity": float(cosine_similarity(flywheel_curve.reshape(1, -1), bearing_curve.reshape(1, -1))[0, 0]),
        "trajectory_rmse": float(np.sqrt(np.mean((flywheel_curve - bearing_curve) ** 2))),
        "trend_correlation": float(np.corrcoef(flywheel_curve, bearing_curve)[0, 1]),
    }

    _set_plot_style()
    plt.figure(figsize=(6.4, 4.0))
    x = np.linspace(0, 1, len(flywheel_curve))
    plt.plot(x, flywheel_curve, color="#2B6CB0", linewidth=1.8, label="Flywheel observed 0-1800d")
    plt.plot(x, bearing_curve, color="#D1495B", linewidth=1.8, label="Bearing full-life mean")
    plt.xlabel("Normalized observed progress")
    plt.ylabel("Normalized HI")
    plt.title("Observed-domain Similarity: Flywheel 0-1800d vs Bearing")
    plt.legend()
    path = project_root / "figures" / "transfer_health" / "observed_domain_similarity.png"
    return similarity, _save_figure(path, dpi=dpi)


def _plot_transfer_comparison(project_root: Path, comparison: pd.DataFrame, dpi: int) -> str:
    _set_plot_style()
    plt.figure(figsize=(6.6, 3.8))
    colors = ["#4A5568", "#D1495B", "#2A9D8F", "#2B6CB0"]
    plt.bar(comparison["method"], comparison["rul_days"], color=colors)
    plt.ylabel("RUL / days")
    plt.title("Task 3 RUL Comparison Across Transfer Strategies")
    plt.xticks(rotation=18, ha="right")
    for idx, value in enumerate(comparison["rul_days"]):
        plt.text(idx, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    path = project_root / "figures" / "transfer_health" / "task3_transfer_comparison.png"
    plt.tight_layout()
    return _save_figure(path, dpi=dpi)


def _plot_sensitivity(project_root: Path, sensitivity: pd.DataFrame, dpi: int) -> str:
    _set_plot_style()
    plt.figure(figsize=(6.4, 3.8))
    plt.plot(100 * sensitivity["factor_delta"], sensitivity["rul_days"], marker="o", color="#D1495B", linewidth=1.4)
    plt.xlabel("Transfer coefficient perturbation / %")
    plt.ylabel("Combined-transfer RUL / days")
    plt.title("Task 3 Sensitivity to Transfer Coefficient")
    path = project_root / "figures" / "transfer_health" / "task3_sensitivity.png"
    plt.tight_layout()
    return _save_figure(path, dpi=dpi)


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
        f"结合增强后的综合迁移校准后，推荐 RUL 为 `{result['recommended_rul']['rul_days']:.1f}` 天，区间为 `[{result['recommended_rul']['rul_interval'][0]:.1f}, {result['recommended_rul']['rul_interval'][1]:.1f}]` 天。",
        f"当前预警等级为：`{warning['level']}`。",
        "",
        "## 2. 迁移学习合理性",
        "",
        "滚动轴承和反作用轮轴承具有相同的关键退化机理：润滑状态恶化导致摩擦增强，进而表现为振动能量、电流或温度等观测量上升。",
        f"只使用附件 2 已观测 0-1800 天计算的跨域相似度为 `{result['observed_similarity']['cosine_similarity']:.4f}`，趋势相关系数为 `{result['observed_similarity']['trend_correlation']:.4f}`。",
        f"扩展预测轨迹与轴承全寿命均值轨迹的相似度为 `{result['domain_similarity']['cosine_similarity']:.4f}`，说明两类退化过程在健康状态空间中具有较强同向性。",
        "",
        "## 3. 迁移策略",
        "",
        "本阶段采用物理一致性约束下的退化严重度迁移校准，不是 DANN/CORAL/MMD 深度迁移训练。具体做法是：保留队友任务一给出的 A1-calibrated common HI RUL 作为无迁移基线，再利用 XJTU-SY 轴承后期退化 HI 分布、阶段比例和高质量退化特征，对附件 2 的 RUL 进行校准。",
        "该策略迁移的是归一化退化状态和后期严重度，而不是直接迁移分钟级轴承寿命，因此避免了轴承加速寿命试验和飞轮在轨天级数据之间的时间尺度冲突。",
        "",
        "## 4. RUL 对比",
        "",
        "| 方法 | EOL day | RUL / days |",
        "|---|---:|---:|",
    ]
    for row in result["transfer_comparison"]:
        lines.append(f"| {row['method']} | {row['eol_day']:.1f} | {row['rul_days']:.1f} |")
    lines += [
        "",
        f"严重度迁移因子 severity_scale = `{transfer['severity_scale']:.4f}`，迁移置信度 = `{transfer['transfer_confidence']:.4f}`。",
        "",
        "## 5. 敏感性分析",
        "",
        "对综合迁移系数进行 `-10%`、`-5%`、`0`、`+5%`、`+10%` 扰动，检查推荐 RUL 对迁移系数的敏感程度。结果写入 `results/task3_transfer_sensitivity.csv`。",
        "",
        "## 6. 预警机制与建议",
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
        "## 7. 高质量迁移特征 Top 5",
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
        "## 8. 不确定性与局限性",
        "",
        "- XJTU-SY 是地面加速寿命试验，飞轮附件 2 是在轨监测数据，两者工作环境和采样模态不同。",
        "- 当前迁移方法强调可解释性和稳健性，尚不是端到端深度域适配模型。",
        "- 附件 2 缺少真实失效终点，因此 RUL 是模型外推估计。",
        "- 预警等级应作为工程辅助决策，不能替代姿控系统安全规则。",
        "",
        "## 9. 图像",
        "",
    ]
    for fig in result["figures"]:
        rel = Path(fig).relative_to(project_root)
        lines.append(f"![{rel.stem}](../{rel.as_posix()})")
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def _write_task3_report(project_root: Path, result: dict[str, Any], report_path: Path) -> str:
    lines = [
        "# 问题三增强实验报告：跨领域迁移学习",
        "",
        "## 1. 实验目标",
        "",
        "本实验将问题二中从 XJTU-SY 滚动轴承学习到的退化规律迁移到反作用轮，用于改进问题一的附件 2 RUL 预测结果。",
        "当前方法是物理一致性约束下的退化严重度迁移校准，不是 DANN/CORAL/MMD 深度迁移训练。",
        "",
        "## 2. 为什么可以迁移",
        "",
        "滚动轴承与反作用轮轴承都存在润滑退化、摩擦增强、退化观测量上升这一共同物理链条。轴承侧表现为振动能量和时频特征变化，飞轮侧表现为电流、温度和 common HI 上升。",
        f"仅使用附件 2 已观测 0-1800 天时，飞轮 HI 与轴承 HI 的归一化轨迹相似度为 `{result['observed_similarity']['cosine_similarity']:.4f}`，趋势相关系数为 `{result['observed_similarity']['trend_correlation']:.4f}`。",
        "",
        "## 3. 迁移方法对比",
        "",
        "| 方法 | 说明 | RUL / 天 | EOL day |",
        "|---|---|---:|---:|",
    ]
    for row in result["transfer_comparison"]:
        lines.append(f"| {row['method']} | {row['description']} | {row['rul_days']:.1f} | {row['eol_day']:.1f} |")
    lines += [
        "",
        f"推荐采用 `{result['recommended_rul']['method']}`，RUL 为 `{result['recommended_rul']['rul_days']:.1f}` 天，区间为 `[{result['recommended_rul']['rul_interval'][0]:.1f}, {result['recommended_rul']['rul_interval'][1]:.1f}]` 天。",
        "",
        "## 4. 敏感性分析",
        "",
        "| 迁移系数扰动 | 迁移系数 | RUL / 天 | 相对名义 RUL 变化 / 天 |",
        "|---:|---:|---:|---:|",
    ]
    for row in result["transfer_sensitivity"]:
        lines.append(
            f"| {100 * row['factor_delta']:.0f}% | {row['transfer_factor']:.4f} | {row['rul_days']:.1f} | {row['rul_change_vs_nominal_days']:.1f} |"
        )
    lines += [
        "",
        "敏感性分析表明，推荐 RUL 会随迁移系数变化而单调变化，但在 ±10% 扰动内仍处于同一量级，说明迁移结论对小范围参数扰动具有一定稳定性。",
        "",
        "## 5. 输出文件",
        "",
        "- `results/task3_observed_similarity.json`",
        "- `results/task3_transfer_comparison.csv`",
        "- `results/task3_transfer_sensitivity.csv`",
        "- `results/task3_transfer_summary.json`",
        "- `figures/transfer_health/observed_domain_similarity.png`",
        "- `figures/transfer_health/task3_transfer_comparison.png`",
        "- `figures/transfer_health/task3_sensitivity.png`",
        "",
        "## 6. 图像",
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
    stage_info = _stage_ratio_factor(task1, bearing_hi)
    comparison = _build_transfer_comparison(task1, transfer, stage_info)
    sensitivity = _build_sensitivity(comparison)
    combined = comparison[comparison["method"] == "combined_transfer"].iloc[0]
    recommended_interval = [
        float(sensitivity["rul_days"].min()),
        float(sensitivity["rul_days"].max()),
    ]
    stage = task1["stage_assessment"]["current_stage"]
    warning = _warning_level(stage, transfer["baseline_current_value"], float(combined["rul_days"]))
    sim_fig, similarity = _plot_domain_similarity(project_root, bearing_ref["mean_curve"], task1_predictions, dpi)
    observed_similarity, observed_fig = _observed_similarity(project_root, bearing_ref, task1_predictions, dpi)
    rul_fig = _plot_rul_comparison(project_root, task1_predictions, task1, transfer, dpi)
    comparison_fig = _plot_transfer_comparison(project_root, comparison, dpi)
    sensitivity_fig = _plot_sensitivity(project_root, sensitivity, dpi)

    comparison_path = project_root / "results" / "task3_transfer_comparison.csv"
    sensitivity_path = project_root / "results" / "task3_transfer_sensitivity.csv"
    observed_similarity_path = project_root / "results" / "task3_observed_similarity.json"
    comparison.to_csv(comparison_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    observed_similarity_path.write_text(json.dumps(_jsonable(observed_similarity), ensure_ascii=False, indent=2), encoding="utf-8")

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
        "stage_ratio_transfer": stage_info,
        "transfer_comparison": comparison.to_dict(orient="records"),
        "transfer_sensitivity": sensitivity.to_dict(orient="records"),
        "recommended_rul": {
            "method": "combined_transfer",
            "rul_days": float(combined["rul_days"]),
            "eol_day": float(combined["eol_day"]),
            "rul_interval": recommended_interval,
        },
        "observed_similarity": observed_similarity,
        "domain_similarity": similarity,
        "warning": warning,
        "figures": [rul_fig, sim_fig, observed_fig, comparison_fig, sensitivity_fig],
        "task3_outputs": {
            "observed_similarity_json": str(observed_similarity_path),
            "transfer_comparison_csv": str(comparison_path),
            "transfer_sensitivity_csv": str(sensitivity_path),
            "transfer_summary_json": str(project_root / "results" / "task3_transfer_summary.json"),
        },
    }
    result_path = project_root / "results" / "transfer_health_management_results.json"
    result_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = _write_health_report(project_root, result, feature_quality)
    task3_summary_path = project_root / "results" / "task3_transfer_summary.json"
    task3_report_path = project_root / "reports" / "task3_transfer_report_cn.md"
    _write_task3_report(project_root, result, task3_report_path)
    result["outputs"] = {"result_json": str(result_path), "report": report_path, "task3_report": str(task3_report_path)}
    result_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    task3_summary_path.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(_jsonable(generate_transfer_health_management(root)), ensure_ascii=False, indent=2))
