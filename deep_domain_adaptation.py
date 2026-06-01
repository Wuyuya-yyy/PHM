"""Deep cross-domain adaptation experiments for bearing-to-flywheel PHM."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from encoders.bearing_encoder import BearingEncoder
from encoders.flywheel_encoder import FlywheelEncoder
from transfer_learning.adaptation import CORALLoss, DANNAdapter, MMDLoss
from utils.plot_utils import save_figure, set_ieee_style


@dataclass(frozen=True)
class DeepAdaptationConfig:
    """Training configuration for compact reproducible experiments."""

    latent_dim: int = 8
    hidden_dim: int = 32
    batch_size: int = 64
    epochs: int = 260
    learning_rate: float = 1e-3
    train_progress_limit: float = 0.70
    random_seed: int = 2026
    align_weight: float = 0.20
    recon_weight: float = 0.05
    domain_weight: float = 0.12
    max_bearing_rows: int = 6000


class CrossDomainPHMNet(nn.Module):
    """Two encoders, two decoders, and one shared degradation head."""

    def __init__(self, flywheel_dim: int, bearing_dim: int, cfg: DeepAdaptationConfig) -> None:
        super().__init__()
        self.flywheel_encoder = FlywheelEncoder(
            input_dim=flywheel_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=(64, cfg.hidden_dim),
            dropout=0.05,
        )
        self.bearing_encoder = BearingEncoder(
            input_dim=bearing_dim,
            latent_dim=cfg.latent_dim,
            hidden_dims=(96, 48),
            dropout=0.08,
        )
        self.flywheel_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, flywheel_dim),
        )
        self.bearing_decoder = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, bearing_dim),
        )
        self.progress_head = nn.Sequential(
            nn.Linear(cfg.latent_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def encode_flywheel(self, x: torch.Tensor) -> torch.Tensor:
        return self.flywheel_encoder(x)

    def encode_bearing(self, x: torch.Tensor) -> torch.Tensor:
        return self.bearing_encoder(x)

    def predict_progress(self, z: torch.Tensor) -> torch.Tensor:
        return self.progress_head(z).squeeze(-1)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_inputs(project_root: Path, cfg: DeepAdaptationConfig) -> dict[str, Any]:
    task1 = json.loads((project_root / "results" / "task1_rul_summary.json").read_text(encoding="utf-8"))
    a1_raw = pd.read_csv(task1["source_files"]["attachment1"])
    a2_raw = pd.read_csv(task1["source_files"]["attachment2"])
    a1_hi = pd.read_csv(project_root / "processed_data" / "attachment1_reaction_wheel_3500d_data_health_index.csv")
    a2_hi = pd.read_csv(project_root / "processed_data" / "attachment2_reaction_wheel_1800d_data_health_index.csv")
    bearing = pd.read_csv(project_root / "results" / "bearing" / "xjtu_sy_bearing_hi.csv")
    quality = pd.read_csv(project_root / "results" / "bearing" / "xjtu_sy_feature_quality.csv")

    a1_common, a2_common = _build_common_hi(a1_raw, a2_raw)
    a1 = a1_hi.merge(a1_common, on="day", how="left")
    a2 = a2_hi.merge(a2_common, on="day", how="left")
    a1["HI_common_derivative"] = a1["HI_common_smooth"].diff().fillna(0.0)
    a2["HI_common_derivative"] = a2["HI_common_smooth"].diff().fillna(0.0)

    flywheel_cols = [
        "current_norm",
        "temperature_norm",
        "speed_rpm_norm",
        "HI_common_raw",
        "HI_common_smooth",
        "HI_common_derivative",
    ]
    for df in (a1, a2):
        for col in flywheel_cols:
            if col not in df.columns:
                df[col] = 0.0
        df[flywheel_cols] = df[flywheel_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    top_features = quality.sort_values("overall_score", ascending=False)["feature"].tolist()
    bearing_cols = [col for col in top_features if col in bearing.columns][:28]
    for extra in ["PCA_HI", "HI_smooth"]:
        if extra in bearing.columns and extra not in bearing_cols:
            bearing_cols.append(extra)

    bearing = bearing.replace([np.inf, -np.inf], np.nan)
    bearing[bearing_cols] = bearing[bearing_cols].fillna(bearing[bearing_cols].median(numeric_only=True))
    bearing["life_progress"] = bearing.groupby(["condition", "bearing_id"])["sample_id"].transform(
        lambda s: s / max(float(s.max()), 1.0)
    )
    if len(bearing) > cfg.max_bearing_rows:
        bearing = bearing.sample(cfg.max_bearing_rows, random_state=cfg.random_seed).sort_index()

    a1_max_day = float(a1["day"].max())
    a1["life_progress"] = a1["day"] / a1_max_day
    train_mask = a1["life_progress"] <= cfg.train_progress_limit
    flywheel_scaler = StandardScaler().fit(a1.loc[train_mask, flywheel_cols])
    bearing_scaler = StandardScaler().fit(bearing[bearing_cols])

    return {
        "a1": a1,
        "a2": a2,
        "bearing": bearing,
        "a1_max_day": a1_max_day,
        "flywheel_cols": flywheel_cols,
        "bearing_cols": bearing_cols,
        "x_f_train": flywheel_scaler.transform(a1.loc[train_mask, flywheel_cols]).astype(np.float32),
        "y_f_train": a1.loc[train_mask, "life_progress"].to_numpy(np.float32),
        "x_f_test": flywheel_scaler.transform(a1.loc[~train_mask, flywheel_cols]).astype(np.float32),
        "test_days": a1.loc[~train_mask, "day"].to_numpy(np.float32),
        "x_f_a2": flywheel_scaler.transform(a2[flywheel_cols]).astype(np.float32),
        "a2_days": a2["day"].to_numpy(np.float32),
        "x_b": bearing_scaler.transform(bearing[bearing_cols]).astype(np.float32),
        "y_b": bearing["life_progress"].to_numpy(np.float32),
    }


def _build_common_hi(a1: pd.DataFrame, a2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_cols = [
        col
        for col in ["current", "temperature", "speed_rpm"]
        if col in a1.columns and col in a2.columns and a1[col].nunique(dropna=True) > 1
    ]
    a1_oriented = a1[common_cols].astype(float).copy()
    a2_oriented = a2[common_cols].astype(float).copy()
    for col in common_cols:
        corr = np.corrcoef(a1["day"].to_numpy(dtype=float), a1_oriented[col].to_numpy(dtype=float))[0, 1]
        if np.isfinite(corr) and corr < 0:
            a1_oriented[col] = -a1_oriented[col]
            a2_oriented[col] = -a2_oriented[col]
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
    a1_out["HI_common_smooth"] = a1_out["HI_common_raw"].rolling(a1_window, min_periods=1).mean()
    a2_out["HI_common_smooth"] = a2_out["HI_common_raw"].rolling(61, min_periods=1).mean()
    return a1_out, a2_out


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, generator=generator)


def _distribution_metrics(zf: np.ndarray, zb: np.ndarray) -> dict[str, float]:
    n = min(len(zf), len(zb), 2000)
    zf_t = torch.tensor(zf[:n], dtype=torch.float32)
    zb_t = torch.tensor(zb[:n], dtype=torch.float32)
    return {
        "centroid_distance": float(np.linalg.norm(zf.mean(axis=0) - zb.mean(axis=0))),
        "coral_distance": float(CORALLoss()(zf_t, zb_t).detach().cpu()),
        "mmd_distance": float(MMDLoss()(zf_t, zb_t).detach().cpu()),
    }


def _evaluate_rul(model: CrossDomainPHMNet, data: dict[str, Any], method: str) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        x_test = torch.tensor(data["x_f_test"], dtype=torch.float32)
        pred_progress = model.predict_progress(model.encode_flywheel(x_test)).cpu().numpy()
        x_a2 = torch.tensor(data["x_f_a2"], dtype=torch.float32)
        pred_a2 = model.predict_progress(model.encode_flywheel(x_a2)).cpu().numpy()

    days = data["test_days"].astype(float)
    true_rul = data["a1_max_day"] - days
    pred_progress = np.clip(pred_progress, 0.05, 1.20)
    pred_eol = np.clip(days / pred_progress, days, 8000.0)
    pred_rul = pred_eol - days
    sample_idx = np.linspace(0, len(days) - 1, min(10, len(days))).astype(int)
    point_rows = [
        {
            "method": method,
            "day": float(days[idx]),
            "true_rul_days": float(true_rul[idx]),
            "predicted_rul_days": float(pred_rul[idx]),
            "absolute_error_days": float(abs(pred_rul[idx] - true_rul[idx])),
            "predicted_progress": float(pred_progress[idx]),
        }
        for idx in sample_idx
    ]
    current_day = float(data["a2_days"][-1])
    a2_progress = float(np.clip(pred_a2[-1], 0.05, 1.20))
    a2_eol = float(np.clip(current_day / a2_progress, current_day, 8000.0))
    return {
        "method": method,
        "test_mae_days": float(mean_absolute_error(true_rul, pred_rul)),
        "test_rmse_days": float(np.sqrt(mean_squared_error(true_rul, pred_rul))),
        "test_mape": float(np.mean(np.abs(pred_rul - true_rul) / np.maximum(true_rul, 1.0))),
        "a2_predicted_progress": a2_progress,
        "a2_predicted_eol_day": a2_eol,
        "a2_predicted_rul_days": float(a2_eol - current_day),
        "point_predictions": point_rows,
    }


def _train_one(method: str, data: dict[str, Any], cfg: DeepAdaptationConfig) -> dict[str, Any]:
    model = CrossDomainPHMNet(data["x_f_train"].shape[1], data["x_b"].shape[1], cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    f_loader = _loader(data["x_f_train"], data["y_f_train"], cfg.batch_size, True, cfg.random_seed)
    b_loader = _loader(data["x_b"], data["y_b"], cfg.batch_size, True, cfg.random_seed + 7)
    coral = CORALLoss()
    mmd = MMDLoss()
    dann = DANNAdapter(cfg.latent_dim, lambda_=1.0) if method == "dann" else None
    if dann is not None:
        optimizer = torch.optim.Adam(list(model.parameters()) + list(dann.parameters()), lr=cfg.learning_rate, weight_decay=1e-4)

    losses: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        if dann is not None:
            dann.train()
        totals = {"loss": 0.0, "supervised": 0.0, "alignment": 0.0, "domain": 0.0, "reconstruction": 0.0}
        steps = 0
        for (xf, yf), (xb, yb) in zip(f_loader, b_loader):
            optimizer.zero_grad()
            zf = model.encode_flywheel(xf)
            zb = model.encode_bearing(xb)
            supervised = F.mse_loss(model.predict_progress(zf), yf) + F.mse_loss(model.predict_progress(zb), yb)
            reconstruction = F.mse_loss(model.flywheel_decoder(zf), xf) + F.mse_loss(model.bearing_decoder(zb), xb)
            alignment = torch.tensor(0.0)
            domain = torch.tensor(0.0)
            if method == "coral":
                alignment = coral(zf, zb)
            elif method == "mmd":
                alignment = mmd(zf, zb)
            elif method == "dann" and dann is not None:
                logits = dann(torch.cat([zf, zb], dim=0))
                labels = torch.cat([torch.zeros(len(zf), dtype=torch.long), torch.ones(len(zb), dtype=torch.long)])
                domain = F.cross_entropy(logits, labels)
            elif method == "autoencoder_joint":
                alignment = mmd(zf, zb)
            recon_weight = cfg.recon_weight * (3.0 if method == "autoencoder_joint" else 1.0)
            loss = supervised + recon_weight * reconstruction + cfg.align_weight * alignment + cfg.domain_weight * domain
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            totals["supervised"] += float(supervised.detach())
            totals["alignment"] += float(alignment.detach())
            totals["domain"] += float(domain.detach())
            totals["reconstruction"] += float(reconstruction.detach())
            steps += 1
        if epoch in {0, cfg.epochs // 2, cfg.epochs - 1}:
            losses.append({k: v / max(steps, 1) for k, v in totals.items()} | {"epoch": float(epoch + 1)})

    model.eval()
    with torch.no_grad():
        zf = model.encode_flywheel(torch.tensor(data["x_f_train"], dtype=torch.float32)).cpu().numpy()
        zb = model.encode_bearing(torch.tensor(data["x_b"], dtype=torch.float32)).cpu().numpy()
    metrics = _evaluate_rul(model, data, method)
    metrics["distribution"] = _distribution_metrics(zf, zb)
    metrics["training_losses"] = losses
    metrics["flywheel_latent"] = zf
    metrics["bearing_latent"] = zb
    return metrics


def _plot_metrics(project_root: Path, summary: pd.DataFrame, dpi: int) -> list[str]:
    set_ieee_style()
    fig_dir = project_root / "figures" / "deep_transfer"
    figures: list[str] = []
    plt.figure(figsize=(7.0, 3.2))
    colors = ["#6C757D", "#2B6CB0", "#2F855A", "#C05621", "#6B46C1"]
    plt.bar(summary["method"], summary["test_mae_days"], color=colors[: len(summary)])
    plt.ylabel("A1 held-out RUL MAE / days")
    plt.xlabel("Deep transfer method")
    plt.xticks(rotation=18, ha="right")
    plt.title("Deep Domain Adaptation: RUL Error Comparison")
    figures.append(save_figure(fig_dir / "deep_transfer_rul_error_comparison.png", dpi=dpi))

    plt.figure(figsize=(7.0, 3.2))
    plt.plot(summary["method"], summary["centroid_distance"], marker="o", label="Centroid")
    plt.plot(summary["method"], summary["mmd_distance"], marker="s", label="MMD")
    plt.ylabel("Latent distribution distance")
    plt.xlabel("Deep transfer method")
    plt.xticks(rotation=18, ha="right")
    plt.title("Latent Alignment Distance After Training")
    plt.legend()
    figures.append(save_figure(fig_dir / "deep_transfer_latent_distance.png", dpi=dpi))
    return figures


def _plot_latent(project_root: Path, method_results: list[dict[str, Any]], dpi: int) -> str:
    set_ieee_style()
    chosen = min(method_results, key=lambda item: item["test_mae_days"])
    zf = chosen["flywheel_latent"]
    zb = chosen["bearing_latent"]
    n_b = min(len(zb), 2000)
    z = np.vstack([zf, zb[:n_b]])
    labels = np.array(["flywheel"] * len(zf) + ["bearing"] * n_b)
    coords = PCA(n_components=2, random_state=2026).fit_transform(z)
    plt.figure(figsize=(6.0, 4.2))
    for label, color in [("flywheel", "#2B6CB0"), ("bearing", "#D1495B")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=11, alpha=0.70, label=label, color=color)
    plt.xlabel("PCA latent axis 1")
    plt.ylabel("PCA latent axis 2")
    plt.title(f"Aligned Latent Space: {chosen['method']}")
    plt.legend()
    return save_figure(project_root / "figures" / "deep_transfer" / "deep_transfer_best_latent_pca.png", dpi=dpi)


def _write_report(project_root: Path, result: dict[str, Any]) -> str:
    report_path = project_root / "reports" / "deep_transfer_report_cn.md"
    best = result["best_method"]
    lines = [
        "# 深度域适配增强实验报告",
        "",
        "## 1. 实验定位",
        "",
        "本实验补充真正的 PyTorch 深度迁移训练流程，包括 Bearing Encoder、Flywheel Encoder、共享 latent、RUL/退化进度预测头，以及 DANN、CORAL、MMD、AutoEncoder-joint 四类域适配训练策略。",
        "原有任务三推荐结论仍采用可解释的退化严重度迁移校准；本实验作为深度迁移可行性验证和论文增强证据。",
        "",
        "## 2. 训练与验证设置",
        "",
        f"- 飞轮输入特征：`{result['flywheel_features']}`",
        f"- 轴承输入特征数量：`{len(result['bearing_features'])}`",
        f"- latent 维度：`{result['config']['latent_dim']}`",
        f"- 训练轮数：`{result['config']['epochs']}`",
        "- 验证方式：使用附件 1 已知全寿命终点，将前 70% 作为飞轮训练段，后 30% 作为真实 RUL 误差测试段。",
        "",
        "## 3. 迁移前后 RUL 误差对比",
        "",
        "| 方法 | A1测试 MAE/天 | A1测试 RMSE/天 | A2参考 RUL/天 | latent MMD | latent CORAL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["method_summary"]:
        lines.append(
            f"| {row['method']} | {row['test_mae_days']:.1f} | {row['test_rmse_days']:.1f} | "
            f"{row['a2_predicted_rul_days']:.1f} | {row['mmd_distance']:.4f} | {row['coral_distance']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"最佳深度迁移方法为 `{best['method']}`，其附件 1 后 30% 真实 RUL 测试 MAE 为 `{best['test_mae_days']:.1f}` 天。",
            f"该方法对附件 2 当前时刻给出的深度模型参考 RUL 为 `{best['a2_predicted_rul_days']:.1f}` 天。",
            "",
            "## 4. 结论",
            "",
            "与 no_adaptation 相比，若 CORAL/MMD/DANN/AutoEncoder-joint 的测试误差和 latent 分布距离下降，则说明轴承退化表征对飞轮 RUL 建模具有可迁移价值。若个别深度方法误差不降，论文中应如实说明深度迁移受样本量、时间尺度差异和模态差异限制。",
            "",
            "## 5. 输出文件",
            "",
            "- `results/deep_transfer/deep_transfer_summary.json`",
            "- `results/deep_transfer/deep_transfer_method_comparison.csv`",
            "- `results/deep_transfer/deep_transfer_rul_point_predictions.csv`",
            "- `results/deep_transfer/deep_transfer_latent_features.csv`",
            "- `figures/deep_transfer/deep_transfer_rul_error_comparison.png`",
            "- `figures/deep_transfer/deep_transfer_latent_distance.png`",
            "- `figures/deep_transfer/deep_transfer_best_latent_pca.png`",
            "",
            "## 6. 图像",
            "",
            "![deep_transfer_rul_error_comparison](../figures/deep_transfer/deep_transfer_rul_error_comparison.png)",
            "",
            "![deep_transfer_latent_distance](../figures/deep_transfer/deep_transfer_latent_distance.png)",
            "",
            "![deep_transfer_best_latent_pca](../figures/deep_transfer/deep_transfer_best_latent_pca.png)",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)


def run_deep_domain_adaptation(project_root: Path, dpi: int = 320) -> dict[str, Any]:
    """Run trained DANN/CORAL/MMD/AutoEncoder cross-domain experiments."""
    cfg = DeepAdaptationConfig()
    _set_seed(cfg.random_seed)
    data = _load_inputs(project_root, cfg)
    methods = ["no_adaptation", "coral", "mmd", "dann", "autoencoder_joint"]
    method_results = [_train_one(method, data, cfg) for method in methods]

    summary_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    for item in method_results:
        dist = item["distribution"]
        summary_rows.append(
            {
                "method": item["method"],
                "test_mae_days": item["test_mae_days"],
                "test_rmse_days": item["test_rmse_days"],
                "test_mape": item["test_mape"],
                "a2_predicted_progress": item["a2_predicted_progress"],
                "a2_predicted_eol_day": item["a2_predicted_eol_day"],
                "a2_predicted_rul_days": item["a2_predicted_rul_days"],
                "centroid_distance": dist["centroid_distance"],
                "coral_distance": dist["coral_distance"],
                "mmd_distance": dist["mmd_distance"],
            }
        )
        point_rows.extend(item["point_predictions"])
        for idx, row in enumerate(item["flywheel_latent"]):
            latent_rows.append({"method": item["method"], "domain": "flywheel", "sample_index": idx} | {f"z{i+1}": float(v) for i, v in enumerate(row)})
        for idx, row in enumerate(item["bearing_latent"][:2000]):
            latent_rows.append({"method": item["method"], "domain": "bearing", "sample_index": idx} | {f"z{i+1}": float(v) for i, v in enumerate(row)})

    result_dir = project_root / "results" / "deep_transfer"
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("test_mae_days")
    points_df = pd.DataFrame(point_rows)
    latent_df = pd.DataFrame(latent_rows)
    summary_path = result_dir / "deep_transfer_method_comparison.csv"
    points_path = result_dir / "deep_transfer_rul_point_predictions.csv"
    latent_path = result_dir / "deep_transfer_latent_features.csv"
    summary_df.to_csv(summary_path, index=False)
    points_df.to_csv(points_path, index=False)
    latent_df.to_csv(latent_path, index=False)

    figures = _plot_metrics(project_root, summary_df, dpi)
    figures.append(_plot_latent(project_root, method_results, dpi))
    best = summary_df.iloc[0].to_dict()
    result = {
        "status": "processed",
        "method": "trained_deep_domain_adaptation",
        "config": cfg.__dict__,
        "validation_design": {
            "known_life_dataset": "Attachment 1",
            "train_progress_limit": cfg.train_progress_limit,
            "test_progress_range": [cfg.train_progress_limit, 1.0],
            "metric": "RUL prediction error on held-out A1 full-life tail",
        },
        "flywheel_features": data["flywheel_cols"],
        "bearing_features": data["bearing_cols"],
        "method_summary": summary_df.to_dict(orient="records"),
        "best_method": best,
        "outputs": {
            "summary_json": str(result_dir / "deep_transfer_summary.json"),
            "method_comparison_csv": str(summary_path),
            "rul_point_predictions_csv": str(points_path),
            "latent_features_csv": str(latent_path),
            "report": str(project_root / "reports" / "deep_transfer_report_cn.md"),
        },
        "figures": figures,
    }
    (result_dir / "deep_transfer_summary.json").write_text(
        json.dumps(_jsonable(result), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result["outputs"]["report"] = _write_report(project_root, result)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    print(json.dumps(_jsonable(run_deep_domain_adaptation(root)), ensure_ascii=False, indent=2))
