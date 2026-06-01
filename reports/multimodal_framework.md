# Multimodal Framework

## 1. Core Theory

The project formalizes PHM as cross-modal degradation representation learning. Flywheel current-temperature telemetry and bearing vibration signals are heterogeneous observations, but both are assumed to reflect the same latent degradation state z(t). The framework therefore maps modality-specific observations into a Shared Latent Degradation Space.

## 2. EEG-fMRI Inspired Analogy

EEG-fMRI fusion provides the theoretical analogy only. EEG is temporally sensitive and reflects local dynamic changes, corresponding to bearing vibration signals that are highly sensitive to incipient faults. fMRI reflects stable global structural evolution, corresponding to flywheel current and temperature signals that show long-term global degradation trends. The project remains a PHM and multimodal degradation representation framework.

## 3. Implemented and Reserved Modules

- `encoders/flywheel_encoder.py`: PyTorch Flywheel Encoder for z_f.
- `encoders/bearing_encoder.py`: PyTorch Bearing Encoder interface for z_b.
- `latent_space/cross_modal_prototype.py`: shared latent prototype and trajectory analysis.
- `physics_informed/constraints.py`: monotonicity, continuity, and consistency loss interfaces.
- `transfer_learning/adaptation.py`: reserved DANN, CORAL, MMD, and Domain Classifier interfaces.

## 4. Transfer Learning Route

The current completed transfer experiment uses physical-consistency constrained degradation severity calibration. It compares flywheel HI and XJTU-SY bearing HI trajectories, then calibrates flywheel RUL with bearing late-life severity and stage-ratio evidence.

The enhanced repository also includes trained deep domain-adaptation experiments. The training flow is bearing features -> Bearing Encoder -> z_b, flywheel features -> Flywheel Encoder -> z_f, domain-adaptation losses -> aligned latent space, followed by RUL error comparison on the held-out tail of the known Attachment-1 full-life trajectory. In the current run, MMD improves the held-out RUL MAE relative to no-adaptation, while DANN and AutoEncoder-joint are kept as reported comparison methods rather than claimed best methods.

## 5. Current Deliverables

- Flywheel latent embedding CSV files.
- Latent clustering CSV files.
- PCA and t-SNE trajectory plots.
- Stage separation plots.
- Physical-consistency transfer calibration results.
- Trained deep-transfer comparison results.
- Transfer-learning and physics-informed interfaces.
