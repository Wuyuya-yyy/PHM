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

DANN, CORAL, MMD, and deep AutoEncoder joint training are reserved interfaces only in the present repository. They have not been used as trained domain-adaptation experiments, and the current paper should not claim adversarial training, covariance-alignment training, MMD minimization training, or real-test-set error reduction from such deep transfer models.

## 5. Current Deliverables

- Flywheel latent embedding CSV files.
- Latent clustering CSV files.
- PCA and t-SNE trajectory plots.
- Stage separation plots.
- Physical-consistency transfer calibration results.
- Reserved transfer-learning and physics-informed interfaces.
