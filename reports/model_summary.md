# Model Summary

## Encoder Summary

- Flywheel Encoder: PyTorch MLP, configurable latent dimension, BatchNorm, Dropout, embedding export.
- Bearing Encoder: PyTorch MLP interface prepared for XJTU-SY RMS, kurtosis, FFT, wavelet, and STFT features.
- Latent dimension: 8.

## Shared Latent Space

The current prototype produces z_f for reaction-wheel data and reserves z_b for bearing data. The shared latent space supports downstream transfer learning, multimodal fusion, and RUL modeling.

## Transfer Interfaces and Completed Deep Training

- DANN: gradient reversal and domain classifier.
- CORAL: covariance alignment.
- MMD: RBF-kernel distribution alignment.
- Domain Classifier: latent-domain discrimination module.

The enhanced repository now includes trained deep domain-adaptation experiments in `deep_domain_adaptation.py`. It trains no-adaptation, CORAL, MMD, DANN, and AutoEncoder-joint variants, then validates RUL error on the held-out tail of the known Attachment-1 full-life trajectory.

The best trained method in the current run is MMD. The original task-three recommendation still uses interpretable physical-consistency transfer calibration, while the deep experiment provides an additional validation and conservative reference.

## Physics-informed Interfaces

- Degradation monotonicity.
- Temporal continuity.
- Physical consistency.
