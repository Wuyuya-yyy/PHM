# Core Theory

## Current Unified Theoretical Mainline

The present project is organized around a unified predictive health management framework in which heterogeneous degradation signals are mapped into a shared latent degradation representation. The central assumption is that different electromechanical components, although observed through different sensing modalities, may exhibit a common low-dimensional degradation trajectory governed by monotonic performance decay, stage transition, and residual useful life evolution.

## 1. Cross-modal Shared Degradation Representation

Let the flywheel telemetry vector be denoted by \(x_f(t)\), including current, temperature, speed, and friction torque. Let the bearing vibration-derived feature vector be denoted by \(x_b(t)\), including time-domain, frequency-domain, and time-frequency health features. The objective is to construct encoders

\[
z_f(t)=E_f(x_f(t)), \qquad z_b(t)=E_b(x_b(t)),
\]

where \(z_f(t)\) and \(z_b(t)\) are embedded into a common degradation manifold. This manifold is required to satisfy monotonicity, stage separability, and RUL predictability.

## 2. Shared Latent Degradation Space

The Shared Latent Degradation Space is defined as a domain-invariant representation space in which the latent variable \(z(t)\) summarizes the health state of heterogeneous equipment. A practical objective can be written as

\[
\mathcal{L}=\mathcal{L}_{rec}+\lambda_1\mathcal{L}_{align}+\lambda_2\mathcal{L}_{mono}+\lambda_3\mathcal{L}_{stage}+\lambda_4\mathcal{L}_{rul},
\]

where reconstruction maintains information sufficiency, alignment reduces inter-domain discrepancy, monotonicity constrains degradation direction, stage loss improves regime separability, and RUL loss connects representation learning with prognostic utility.

## 3. Flywheel-Bearing Correspondence

The flywheel data are characterized by slowly varying telemetry variables, whereas the XJTU-SY bearing data are characterized by high-frequency vibration signals. Their observation spaces are different, but their PHM semantics are aligned:

| PHM Concept | Reaction Wheel | XJTU-SY Bearing |
|---|---|---|
| Degradation signal | current, temperature, friction torque | RMS, kurtosis, spectral energy, wavelet energy |
| Health indicator | PCA-fused HI | feature-fused bearing HI |
| Stage structure | healthy, slow degradation, accelerated degradation | normal, incipient fault, severe fault |
| Prediction target | degradation trend / RUL | degradation trend / RUL |

## 4. EEG-fMRI Fusion Analogy

The methodology is analogous to EEG-fMRI fusion: EEG provides high temporal resolution while fMRI provides spatially informative but heterogeneous signals. In PHM, bearing vibration offers rich dynamic fault signatures, while flywheel telemetry provides slowly evolving system-level degradation evidence. The shared latent space plays the role of a cross-modal neural representation, integrating heterogeneous observations into a common state variable.

## 5. Unified Modeling of Heterogeneous Degradation Signals

The first phase builds a flywheel-only baseline through Min-Max normalization, PCA fusion, stage segmentation, exponential physical modeling, Wiener prediction, and LSTM prediction. The next phase will extend the same pipeline to bearing data and enforce cross-domain consistency in the latent representation.

## 6. Overall PHM Framework

The complete PHM framework consists of data acquisition, degradation-oriented feature extraction, health indicator construction, degradation stage segmentation, shared representation learning, transfer learning, and RUL prediction. The current engineering project has implemented the flywheel branch and reserved modular interfaces for the bearing branch.

## 7. Transfer Learning Strategy

Transfer learning will be used to reduce the distribution discrepancy between flywheel and bearing degradation features. Candidate mechanisms include feature-space alignment, adversarial domain adaptation, CORAL/MMD regularization, and stage-aware contrastive learning. The transferable object is not the raw signal waveform, but the latent degradation semantics.

## 8. Multimodal Fusion Roadmap

The future multimodal fusion route is:

1. Establish reliable single-domain HI for flywheel and bearing.
2. Learn modality-specific encoders \(E_f\) and \(E_b\).
3. Align latent variables using domain discrepancy and degradation-stage constraints.
4. Introduce attention-based fusion for current, thermal, torque, and vibration evidence.
5. Couple the shared latent state with probabilistic RUL prediction.

This theoretical line supports an IEEE-style narrative: from interpretable physical degradation evidence to data-driven shared latent representation, and finally to cross-domain PHM generalization.
