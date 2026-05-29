"""Run the cross-modal shared latent degradation prototype."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from latent_space.cross_modal_prototype import CrossModalSharedLatentPrototype, LatentPrototypeConfig
from physics_informed import PhysicsInformedConstraint
from transfer_learning.adaptation import CORALLoss, DANNAdapter, DomainClassifier, MMDLoss
from utils.io_utils import load_config, write_json


def _as_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def run() -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / "configs" / "default_config.yaml")
    latent_cfg = config.get("latent_space", {})
    prototype_config = LatentPrototypeConfig(
        latent_dim=int(latent_cfg.get("latent_dim", 8)),
        random_seed=int(config["project"].get("random_seed", 42)),
        dpi=int(config["project"].get("dpi", 320)),
        dropout=float(latent_cfg.get("dropout", 0.15)),
        n_clusters=int(latent_cfg.get("n_clusters", 3)),
    )

    prototype = CrossModalSharedLatentPrototype(prototype_config)
    results_dir = project_root / "results" / "latent_space"
    figures_dir = project_root / "figures" / "latent_space"
    reports_dir = project_root / "reports"
    embeddings_dir = results_dir / "embeddings"
    clustering_dir = results_dir / "clustering"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("attachment1_reaction_wheel_3500d_data", project_root / "processed_data" / "attachment1_reaction_wheel_3500d_data_health_index.csv"),
        ("attachment2_reaction_wheel_1800d_data", project_root / "processed_data" / "attachment2_reaction_wheel_1800d_data_health_index.csv"),
    ]

    dataset_results: List[Dict[str, Any]] = []
    for dataset_name, hi_path in datasets:
        feature_frame = prototype.prepare_flywheel_features(hi_path)
        embedding = prototype.encode_flywheel(feature_frame)
        embedding_columns = [f"z_f_{idx + 1}" for idx in range(embedding.shape[1])]
        embedding_frame = pd.DataFrame(embedding, columns=embedding_columns)
        embedding_frame.insert(0, "day", feature_frame["day"].to_numpy())
        embedding_path = embeddings_dir / f"{dataset_name}_flywheel_embedding.csv"
        embedding_frame.to_csv(embedding_path, index=False)

        analysis = prototype.analyze_latent_space(
            embedding=embedding,
            days=feature_frame["day"].to_numpy(),
            dataset_name=dataset_name,
            figure_dir=figures_dir,
        )
        cluster_path = clustering_dir / f"{dataset_name}_latent_clusters.csv"
        pd.DataFrame(
            {
                "day": feature_frame["day"].to_numpy(),
                "cluster": analysis["cluster_labels"],
            }
        ).to_csv(cluster_path, index=False)

        dataset_results.append(
            {
                "dataset": dataset_name,
                "feature_dim": int(feature_frame.shape[1] - 1),
                "embedding_dim": int(embedding.shape[1]),
                "embedding_path": str(embedding_path),
                "cluster_path": str(cluster_path),
                "umap_status": analysis["umap_status"],
                "figures": analysis["figures"],
            }
        )

    bearing_encoder = prototype.build_bearing_encoder_interface(input_dim=32)
    domain_classifier = DomainClassifier(prototype_config.latent_dim)
    dann = DANNAdapter(prototype_config.latent_dim)
    coral = CORALLoss()
    mmd = MMDLoss()
    physics = PhysicsInformedConstraint()

    example_latent = torch.randn(16, prototype_config.latent_dim)
    example_target = torch.randn(16, prototype_config.latent_dim)
    interface_summary = {
        "bearing_encoder": {
            "input_dim": bearing_encoder.input_dim,
            "latent_dim": bearing_encoder.latent_dim,
            "future_inputs": ["RMS", "Kurtosis", "FFT features", "Wavelet features", "STFT features"],
        },
        "domain_classifier": str(domain_classifier),
        "dann_adapter": str(dann),
        "coral_example_loss": float(coral(example_latent, example_target).detach().cpu()),
        "mmd_example_loss": float(mmd(example_latent, example_target).detach().cpu()),
        "physics_constraints": [
            "degradation monotonicity",
            "temporal continuity",
            "physical consistency",
        ],
        "physics_temporal_example_loss": float(physics.temporal_continuity_loss(example_latent).detach().cpu()),
    }

    result = {
        "framework": "Cross-modal Shared Latent Degradation Framework",
        "paper_topic": "基于跨模态共享退化表征的卫星飞轮预测式健康管理研究",
        "latent_dim": prototype_config.latent_dim,
        "datasets": dataset_results,
        "interfaces": interface_summary,
    }
    write_json(json.loads(json.dumps(result, default=_as_jsonable)), results_dir / "latent_prototype_results.json")
    _write_reports(project_root, result)
    return result


def _write_reports(project_root: Path, result: Dict[str, Any]) -> None:
    reports_dir = project_root / "reports"
    latent_report = f"""# Latent Space Report

## 1. Objective

This report documents the current Cross-modal Shared Latent Degradation Prototype. The objective is to establish a Shared Latent Degradation Space in which reaction-wheel telemetry degradation features and future XJTU-SY bearing vibration features can be represented by a unified latent state z(t).

## 2. Current Flywheel Encoder

The implemented Flywheel Encoder is a PyTorch module with the structure:

```text
Linear -> ReLU -> BatchNorm -> Dropout -> Linear -> ReLU -> BatchNorm -> Dropout -> Linear
```

Input features include HI, normalized current/temperature/friction/speed features, rolling statistics, and derivative features. The output is the flywheel latent embedding z_f with latent dimension {result["latent_dim"]}.

## 3. Bearing Encoder Interface

The Bearing Encoder interface has been created for future XJTU-SY data. It accepts bearing degradation descriptors such as RMS, kurtosis, FFT features, wavelet features, and STFT features, and maps them into z_b in the same latent dimension.

## 4. Generated Embeddings and Clustering Results

| Dataset | Feature Dim | Embedding Dim | Embedding File | Cluster File | UMAP |
|---|---:|---:|---|---|---|
"""
    for item in result["datasets"]:
        latent_report += (
            f"| {item['dataset']} | {item['feature_dim']} | {item['embedding_dim']} | "
            f"`{item['embedding_path']}` | `{item['cluster_path']}` | {item['umap_status']} |\n"
        )

    latent_report += """
## 5. Latent Trajectory and Stage Separation

The prototype generates PCA and t-SNE latent trajectory plots, latent cluster plots, and PCA-based stage separation plots. UMAP is generated automatically when `umap-learn` is installed; otherwise it is skipped without blocking the workflow.

## 6. Physics-informed Extension

The physics-informed constraint module reserves monotonicity, temporal continuity, and physical consistency losses. These constraints will support degradation continuity and physically interpretable latent dynamics in the next stage.
"""
    (reports_dir / "latent_space_report.md").write_text(latent_report, encoding="utf-8")

    framework_report = f"""# Multimodal Framework

## 1. Core Theory

The project formalizes PHM as cross-modal degradation representation learning. Flywheel current-temperature telemetry and bearing vibration signals are heterogeneous observations, but both are assumed to reflect the same latent degradation state z(t). The framework therefore maps modality-specific observations into a Shared Latent Degradation Space.

## 2. EEG-fMRI Inspired Analogy

EEG-fMRI fusion provides the theoretical analogy only. EEG is temporally sensitive and reflects local dynamic changes, corresponding to bearing vibration signals that are highly sensitive to incipient faults. fMRI reflects stable global structural evolution, corresponding to flywheel current and temperature signals that show long-term global degradation trends. The project remains a PHM and multimodal degradation representation framework.

## 3. Implemented Prototype Modules

- `encoders/flywheel_encoder.py`: PyTorch Flywheel Encoder for z_f.
- `encoders/bearing_encoder.py`: PyTorch Bearing Encoder interface for z_b.
- `latent_space/cross_modal_prototype.py`: shared latent prototype and trajectory analysis.
- `physics_informed/constraints.py`: monotonicity, continuity, and consistency loss interfaces.
- `transfer_learning/adaptation.py`: DANN, CORAL, MMD, and Domain Classifier interfaces.

## 4. Transfer Learning Route

Future bearing integration will use the flywheel branch as the source-domain degradation reference and XJTU-SY bearing features as the target-domain degradation branch. DANN, CORAL, and MMD are reserved to align z_f and z_b distributions while retaining degradation-stage separability.

## 5. Current Deliverables

- Flywheel latent embedding CSV files.
- Latent clustering CSV files.
- PCA and t-SNE trajectory plots.
- Stage separation plots.
- Transfer-learning and physics-informed interfaces.
"""
    (reports_dir / "multimodal_framework.md").write_text(framework_report, encoding="utf-8")

    model_summary = f"""# Model Summary

## Encoder Summary

- Flywheel Encoder: PyTorch MLP, configurable latent dimension, BatchNorm, Dropout, embedding export.
- Bearing Encoder: PyTorch MLP interface prepared for XJTU-SY RMS, kurtosis, FFT, wavelet, and STFT features.
- Latent dimension: {result["latent_dim"]}.

## Shared Latent Space

The current prototype produces z_f for reaction-wheel data and reserves z_b for bearing data. The shared latent space supports downstream transfer learning, multimodal fusion, and RUL modeling.

## Transfer Interfaces

- DANN: gradient reversal and domain classifier.
- CORAL: covariance alignment.
- MMD: RBF-kernel distribution alignment.
- Domain Classifier: latent-domain discrimination module.

## Physics-informed Interfaces

- Degradation monotonicity.
- Temporal continuity.
- Physical consistency.
"""
    (reports_dir / "model_summary.md").write_text(model_summary, encoding="utf-8")


if __name__ == "__main__":
    output = run()
    print(json.dumps(output, indent=2, ensure_ascii=False))
