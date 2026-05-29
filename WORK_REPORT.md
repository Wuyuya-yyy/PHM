# PHM Project Work Report

Generated on: 2026-05-29  
Project path: `C:\Users\35135\Desktop\PHM`  
Research title: `Satellite Reaction Wheel PHM Based on Cross-modal Shared Latent Degradation Representation`

## 1. Current Work Status

The PHM project has been upgraded from a flywheel-only degradation analysis pipeline to a **Cross-modal Shared Latent Degradation Framework** prototype. The current system now contains two connected research layers:

1. **Reaction Wheel PHM Baseline**
   - Data scanning and profiling.
   - Exploratory degradation analysis.
   - Health Index construction.
   - Degradation stage segmentation.
   - Exponential physical model, Wiener baseline, and LSTM baseline.

2. **Cross-modal Shared Latent Degradation Prototype**
   - Flywheel Encoder.
   - Bearing Encoder interface.
   - Shared Latent Degradation Space prototype.
   - Latent trajectory and clustering analysis.
   - Physics-informed constraint interfaces.
   - Transfer learning interfaces including DANN, CORAL, MMD, and Domain Classifier.

The current stage is not final model training. It is a research prototype that establishes the engineering and theoretical foundation for later XJTU-SY bearing integration, transfer learning, RUL prediction, and multimodal PHM modeling.

## 2. Core Research Innovation

The main innovation of the project is the construction of a **Shared Latent Degradation Space** for heterogeneous degradation signals.

Although reaction-wheel telemetry and bearing vibration signals belong to different modalities, they both reflect the same underlying degradation state:

```text
z(t): latent degradation state
```

The project therefore defines modality-specific encoders:

```text
z_f(t) = E_f(x_f(t))
z_b(t) = E_b(x_b(t))
```

where:

- `x_f(t)` denotes flywheel current, temperature, friction torque, HI, rolling statistics, and derivative features.
- `x_b(t)` denotes future bearing vibration features such as RMS, kurtosis, FFT features, wavelet features, and STFT features.
- `z_f(t)` and `z_b(t)` are mapped into a shared latent degradation representation.

This makes the project different from a conventional single-device PHM model. The framework is designed for **cross-modal representation learning**, **transferable degradation modeling**, and **multimodal health management**.

## 3. EEG-fMRI Inspired Theoretical Motivation

The project uses EEG-fMRI fusion only as a theoretical analogy.

| EEG-fMRI Fusion | PHM Correspondence |
|---|---|
| EEG has high temporal sensitivity | Bearing vibration is sensitive to early local faults |
| fMRI reflects stable global structure | Flywheel current and temperature reflect long-term global degradation |
| Cross-modal neural representation | Shared latent degradation representation |

The project subject remains PHM, not biomedical signal analysis. The analogy supports the theoretical argument that heterogeneous signals can be aligned through a shared latent state when they describe the same underlying system evolution.

## 4. Data Identified

The current flywheel data source is the local contest data directory configured in `configs/default_config.yaml`.

```text
configs.default_config.paths.source_data_dir
```

Detected data:

- Attachment 1: `reaction_wheel_3500d_data.csv`
  - Shape: `176 x 6`
  - Columns: `day`, `current`, `current_theoretical`, `temperature`, `speed_rpm`, `friction_torque`
- Attachment 2: `reaction_wheel_1800d_data.csv`
  - Shape: `1801 x 5`
  - Columns: `day`, `current`, `current_theoretical`, `temperature`, `speed_rpm`

## 5. Phase-1 Flywheel Baseline Results

### 5.1 Health Index Construction

The current flywheel HI is constructed by:

```text
direction-consistent feature normalization -> PCA fusion -> HI_raw -> smoothing -> HI_smooth -> derivative
```

Generated files:

- `processed_data/attachment1_reaction_wheel_3500d_data_health_index.csv`
- `processed_data/attachment2_reaction_wheel_1800d_data_health_index.csv`

### 5.2 Degradation Stage Segmentation

Attachment 1 stage boundaries:

- Threshold method: `[58, 123]`
- Derivative method: `[96, 144]`
- Bayesian-style method: `[57, 116]`

Attachment 2 stage boundaries:

- Threshold method: `[594, 1260]`
- Derivative method: `[978, 1446]`
- Bayesian-style method: `[606, 1219]`

The Bayesian-style segmentation is currently recommended as the main reportable stage division because it is closer to a probabilistic change-point interpretation.

### 5.3 Model Metrics

Attachment 1:

- Exponential physical model: RMSE `0.007396`, MAE `0.005886`, R2 `0.999705`
- Wiener baseline: RMSE `0.168578`, MAE `0.154013`, R2 `0.846646`
- LSTM baseline: RMSE `0.010983`, MAE `0.008450`, R2 `0.999349`

Attachment 2:

- Exponential physical model: RMSE `0.007908`, MAE `0.006293`, R2 `0.999018`
- Wiener baseline: RMSE `0.108083`, MAE `0.099624`, R2 `0.816659`
- LSTM baseline: RMSE `0.009661`, MAE `0.007697`, R2 `0.998535`

## 6. New Cross-modal Latent Prototype

The following new modules have been added:

- `encoders/flywheel_encoder.py`
  - PyTorch Flywheel Encoder.
  - Structure: `Linear -> ReLU -> BatchNorm -> Dropout`.
  - Input: HI, current, temperature, friction torque, rolling statistics, derivative features.
  - Output: flywheel latent embedding `z_f`.

- `encoders/bearing_encoder.py`
  - PyTorch Bearing Encoder interface.
  - Reserved input: RMS, kurtosis, FFT features, wavelet features, STFT features.
  - Output: bearing latent embedding `z_b`.

- `latent_space/cross_modal_prototype.py`
  - Shared Latent Degradation Space prototype.
  - Supports latent embedding generation, PCA analysis, t-SNE analysis, clustering, and stage separation visualization.

- `physics_informed/constraints.py`
  - Reserved physics-informed constraints:
    - degradation monotonicity;
    - temporal continuity;
    - physical consistency.

- `transfer_learning/adaptation.py`
  - Transfer learning interfaces:
    - DANN;
    - Domain Classifier;
    - CORAL;
    - MMD.

- `run_cross_modal_latent.py`
  - One-click execution script for the current latent prototype.

## 7. New Generated Outputs

The cross-modal latent prototype has been executed successfully with:

```powershell
cd C:\Users\35135\Desktop\PHM
& 'D:\Program Files\Python312\python.exe' run_cross_modal_latent.py
```

Generated embedding results:

- `results/latent_space/embeddings/attachment1_reaction_wheel_3500d_data_flywheel_embedding.csv`
- `results/latent_space/embeddings/attachment2_reaction_wheel_1800d_data_flywheel_embedding.csv`

Generated clustering results:

- `results/latent_space/clustering/attachment1_reaction_wheel_3500d_data_latent_clusters.csv`
- `results/latent_space/clustering/attachment2_reaction_wheel_1800d_data_latent_clusters.csv`

Generated structured result:

- `results/latent_space/latent_prototype_results.json`

Generated reports:

- `reports/latent_space_report.md`
- `reports/multimodal_framework.md`
- `reports/model_summary.md`

Generated latent-space figures:

- `figures/latent_space/attachment1_reaction_wheel_3500d_data_latent_trajectory_pca.png`
- `figures/latent_space/attachment1_reaction_wheel_3500d_data_latent_cluster_pca.png`
- `figures/latent_space/attachment1_reaction_wheel_3500d_data_latent_trajectory_tsne.png`
- `figures/latent_space/attachment1_reaction_wheel_3500d_data_latent_cluster_tsne.png`
- `figures/latent_space/attachment1_reaction_wheel_3500d_data_stage_separation_pca.png`
- `figures/latent_space/attachment2_reaction_wheel_1800d_data_latent_trajectory_pca.png`
- `figures/latent_space/attachment2_reaction_wheel_1800d_data_latent_cluster_pca.png`
- `figures/latent_space/attachment2_reaction_wheel_1800d_data_latent_trajectory_tsne.png`
- `figures/latent_space/attachment2_reaction_wheel_1800d_data_latent_cluster_tsne.png`
- `figures/latent_space/attachment2_reaction_wheel_1800d_data_stage_separation_pca.png`

Current latent dimension: `8`.

Feature dimensions:

- Attachment 1 flywheel feature dimension: `19`
- Attachment 2 flywheel feature dimension: `15`

UMAP is currently skipped because `umap-learn` is not installed. PCA and t-SNE have been generated normally.

## 8. Team Collaboration Package

A team collaboration package has also been generated:

```text
Team_Share_Package/
```

It includes:

- `For_Paper_Writer/`
  - paper-ready figures;
  - Markdown reports;
  - `Paper_Figure_Index.md`.
- `For_Bearing_Engineer/`
  - `Bearing_Task_Guide.md`.
- `Shared_Theory/`
  - `Core_Theory.md`.
- `Shared_Figures/`
  - deduplicated key figures.
- `Shared_Results/`
  - metrics, HI data, stage boundaries, and result summaries.
- `Shared_Config/`
  - requirements, config, source entry points, and interface modules.

This package is designed for the paper writer and the XJTU-SY bearing engineer to continue collaboration independently.

## 9. Updated Folder Description

### `encoders/`

New module for modality-specific neural encoders.

- `flywheel_encoder.py`: encodes flywheel telemetry and HI features into `z_f`.
- `bearing_encoder.py`: reserves XJTU-SY bearing feature encoding into `z_b`.

### `latent_space/`

Shared latent degradation modeling module.

- `shared_latent_model.py`: original base API.
- `cross_modal_prototype.py`: new latent prototype with embedding, trajectory analysis, clustering, and plotting.

### `transfer_learning/`

Cross-domain adaptation module.

- `domain_adapter.py`: original adapter interface.
- `adaptation.py`: DANN, CORAL, MMD, and Domain Classifier interfaces.

### `physics_informed/`

Physics-informed learning module.

- `constraints.py`: monotonicity, temporal continuity, and physical consistency losses.

### `figures/latent_space/`

Publication-style latent trajectory, cluster, and stage separation figures.

### `results/latent_space/`

Latent embeddings, clustering outputs, and structured prototype results.

### `reports/`

Now contains both Phase-1 reports and cross-modal framework reports.

## 10. Current Scientific Contribution Summary

The current project contribution can be summarized as follows:

1. A complete reaction-wheel PHM baseline has been established from raw data profiling to degradation modeling.
2. An interpretable HI construction pipeline has been implemented using normalized degradation features and PCA fusion.
3. Three degradation stage segmentation methods have been compared.
4. Physical and data-driven degradation prediction baselines have been generated.
5. A cross-modal Shared Latent Degradation Space prototype has been implemented.
6. Flywheel and bearing encoders have been modularized for future multimodal fusion.
7. DANN, CORAL, and MMD interfaces have been reserved for transfer learning.
8. Physics-informed constraints have been reserved for monotonic and continuous degradation representation.
9. IEEE-style figures and collaboration-oriented reports have been generated automatically.

## 11. Recommended Next Work

Recommended continuation order:

1. Add XJTU-SY bearing data ingestion.
2. Implement bearing feature extraction:
   - RMS;
   - kurtosis;
   - crest factor;
   - spectral centroid;
   - FFT band energy;
   - wavelet packet energy;
   - STFT features.
3. Construct bearing HI with the same field convention as flywheel HI.
4. Generate bearing latent embedding `z_b`.
5. Align `z_f` and `z_b` using:
   - CORAL;
   - MMD;
   - DANN.
6. Add latent alignment losses:
   - domain alignment loss;
   - stage consistency loss;
   - monotonicity loss;
   - temporal continuity loss.
7. Extend the final paper around:
   - Shared Latent Degradation Space;
   - cross-modal PHM;
   - EEG-fMRI inspired heterogeneous representation;
   - flywheel-bearing transfer learning.

## 12. Important Notes

- Use English canonical file names such as `attachment1_*` and `attachment2_*` for future experiments.
- Chinese-named duplicate result files from earlier runs should be ignored in formal reporting.
- The current latent encoder is a prototype and is not yet trained with supervised bearing labels.
- UMAP can be enabled later by installing `umap-learn`.
- The current framework is ready for XJTU-SY bearing feature integration and shared latent alignment.
