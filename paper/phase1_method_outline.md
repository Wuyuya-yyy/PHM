# Method Outline

## Shared Latent Degradation Space

Let vibration-derived bearing features be denoted as \(x_b(t)\), and flywheel telemetry features be denoted as \(x_f(t)\). The long-term objective is to learn modality encoders

\[
z_b(t)=E_b(x_b(t)), \quad z_f(t)=E_f(x_f(t))
\]

such that both modalities are projected into a shared latent degradation space \(z(t)\), where degradation monotonicity, stage separability, and RUL predictability are preserved.

## Phase-1 Baseline

The current phase implements the flywheel-only baseline:

1. Data profiling and degradation-oriented EDA.
2. Health index construction by Min-Max normalization and PCA fusion.
3. Three-stage degradation segmentation.
4. Exponential physical degradation fitting.
5. Wiener and LSTM baseline prediction.

The reserved modules under `transfer_learning/`, `multimodal/`, `latent_space/`, and `rul_prediction/` provide the engineering entry points for later cross-domain and cross-modal modeling.
