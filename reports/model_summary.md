# Model Summary

## Encoder Summary

- Flywheel Encoder: PyTorch MLP, configurable latent dimension, BatchNorm, Dropout, embedding export.
- Bearing Encoder: PyTorch MLP interface prepared for XJTU-SY RMS, kurtosis, FFT, wavelet, and STFT features.
- Latent dimension: 8.

## Shared Latent Space

The current prototype produces z_f for reaction-wheel data and reserves z_b for bearing data. The shared latent space supports downstream transfer learning, multimodal fusion, and RUL modeling.

## Transfer Interfaces, Not Completed Training

- DANN: gradient reversal and domain classifier.
- CORAL: covariance alignment.
- MMD: RBF-kernel distribution alignment.
- Domain Classifier: latent-domain discrimination module.

These modules are reserved interfaces. The current repository has not completed DANN adversarial training, CORAL covariance-alignment training, MMD minimization training, or deep AutoEncoder cross-domain joint training. The completed task-three result is a physical-consistency constrained degradation severity transfer calibration with RUL comparison and sensitivity analysis.

## Physics-informed Interfaces

- Degradation monotonicity.
- Temporal continuity.
- Physical consistency.
