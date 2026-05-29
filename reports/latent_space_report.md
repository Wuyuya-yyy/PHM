# Latent Space Report

## 1. Objective

This report documents the current Cross-modal Shared Latent Degradation Prototype. The objective is to establish a Shared Latent Degradation Space in which reaction-wheel telemetry degradation features and future XJTU-SY bearing vibration features can be represented by a unified latent state z(t).

## 2. Current Flywheel Encoder

The implemented Flywheel Encoder is a PyTorch module with the structure:

```text
Linear -> ReLU -> BatchNorm -> Dropout -> Linear -> ReLU -> BatchNorm -> Dropout -> Linear
```

Input features include HI, normalized current/temperature/friction/speed features, rolling statistics, and derivative features. The output is the flywheel latent embedding z_f with latent dimension 8.

## 3. Bearing Encoder Interface

The Bearing Encoder interface has been created for future XJTU-SY data. It accepts bearing degradation descriptors such as RMS, kurtosis, FFT features, wavelet features, and STFT features, and maps them into z_b in the same latent dimension.

## 4. Generated Embeddings and Clustering Results

| Dataset | Feature Dim | Embedding Dim | Embedding File | Cluster File | UMAP |
|---|---:|---:|---|---|---|
| attachment1_reaction_wheel_3500d_data | 19 | 8 | `C:\Users\35135\Desktop\PHM\results\latent_space\embeddings\attachment1_reaction_wheel_3500d_data_flywheel_embedding.csv` | `C:\Users\35135\Desktop\PHM\results\latent_space\clustering\attachment1_reaction_wheel_3500d_data_latent_clusters.csv` | skipped: umap-learn is not installed |
| attachment2_reaction_wheel_1800d_data | 15 | 8 | `C:\Users\35135\Desktop\PHM\results\latent_space\embeddings\attachment2_reaction_wheel_1800d_data_flywheel_embedding.csv` | `C:\Users\35135\Desktop\PHM\results\latent_space\clustering\attachment2_reaction_wheel_1800d_data_latent_clusters.csv` | skipped: umap-learn is not installed |

## 5. Latent Trajectory and Stage Separation

The prototype generates PCA and t-SNE latent trajectory plots, latent cluster plots, and PCA-based stage separation plots. UMAP is generated automatically when `umap-learn` is installed; otherwise it is skipped without blocking the workflow.

## 6. Physics-informed Extension

The physics-informed constraint module reserves monotonicity, temporal continuity, and physical consistency losses. These constraints will support degradation continuity and physically interpretable latent dynamics in the next stage.
