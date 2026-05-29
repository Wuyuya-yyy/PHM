# Bearing Task Guide

## 1. Current PHM Architecture

The current PHM project implements a flywheel-first baseline: data scanning, EDA, health-index construction, degradation stage segmentation, physical degradation modeling, Wiener prediction, LSTM prediction, and automatic report generation. The engineering layout already reserves 	ransfer_learning/, latent_space/, multimodal/, and ul_prediction/ for XJTU-SY bearing extension.

## 2. Current Flywheel HI Definition

The flywheel health indicator is constructed by direction-consistent feature normalization followed by one-dimensional PCA fusion:

`	ext
feature matrix -> Min-Max normalization -> PCA first component -> HI_raw -> smoothing -> HI_smooth
`

Attachment 1 features: current, temperature, friction_torque, speed_rpm.  
Attachment 2 features: current, temperature, speed_rpm.  
The PCA explained variance ratios are 0.999919989516465 and 0.9999479113670362, respectively.

## 3. Current Stage Division

The project compares threshold, derivative, and Bayesian-style change-point segmentation. For collaborative development, use Bayesian-style boundaries as the main reportable result unless later validation shows otherwise:

| Dataset | Healthy to slow degradation | Slow to accelerated degradation |
|---|---:|---:|
| Attachment 1, 3500 d | 57 | 116 |
| Attachment 2, 1800 d | 606 | 1219 |

## 4. Flywheel Feature List and Data Structure

Attachment 1 fields: day, current, current_theoretical, temperature, speed_rpm, friction_torque.  
Attachment 2 fields: day, current, current_theoretical, temperature, speed_rpm.

Naming convention:

- ttachment1_reaction_wheel_3500d_data: 3500-day flywheel data.
- ttachment2_reaction_wheel_1800d_data: 1800-day flywheel data.
- HI_raw, HI_smooth, HI_derivative: canonical HI fields.
- stage_boundaries: degradation stage split points.

## 5. Bearing Follow-up Tasks

The XJTU-SY bearing module should add a bearing feature extraction pipeline aligned with the flywheel HI interface. Recommended vibration features:

- Time domain: RMS, mean, standard deviation, peak-to-peak value, skewness, kurtosis, crest factor, impulse factor, clearance factor.
- Frequency domain: spectral centroid, spectral entropy, band energy, dominant frequency, sideband energy.
- Time-frequency domain: wavelet packet energy, wavelet entropy, STFT band energy.
- Trend features: rolling mean, rolling variance, degradation slope, monotonicity score.

## 6. Recommended Bearing HI Construction

Use the same interface as flywheel:

`	ext
bearing vibration signal -> feature extraction -> direction alignment -> Min-Max/robust normalization -> PCA or AutoEncoder fusion -> HI_smooth -> stage segmentation
`

This keeps flywheel and bearing experiments comparable under one PHM framework.

## 7. Transfer Learning Interface

Use Shared_Config/interface_modules/domain_adapter.py as the current reserved entry point. The next implementation should define:

- source domain: flywheel telemetry features and HI.
- target domain: XJTU-SY bearing vibration features and HI.
- alignment target: monotonic degradation representation, stage labels, and RUL-related latent variables.
- candidate losses: MMD, CORAL, adversarial domain loss, contrastive stage alignment, monotonicity regularization.

## 8. Shared Latent Space Interface

Use Shared_Config/interface_modules/shared_latent_model.py as the reserved entry point. The intended design is:

`	ext
z_f = E_f(x_f), z_b = E_b(x_b), z in shared latent degradation space
`

The latent space should preserve degradation ordering, stage separability, and RUL predictability across heterogeneous flywheel and bearing signals.
