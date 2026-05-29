$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$PackageRoot = Join-Path $ProjectRoot "Team_Share_Package"

if (-not ($PackageRoot.StartsWith($ProjectRoot.Path))) {
    throw "Refusing to write outside project root: $PackageRoot"
}

if (Test-Path $PackageRoot) {
    Remove-Item -LiteralPath $PackageRoot -Recurse -Force
}

$dirs = @(
    "For_Paper_Writer",
    "For_Paper_Writer/Figures/01_EDA",
    "For_Paper_Writer/Figures/02_Health_Index",
    "For_Paper_Writer/Figures/03_Stage_Segmentation",
    "For_Paper_Writer/Figures/04_FFT_Analysis",
    "For_Paper_Writer/Figures/05_Wavelet_Analysis",
    "For_Paper_Writer/Figures/06_Correlation_Heatmap",
    "For_Paper_Writer/Figures/07_Wiener_Prediction",
    "For_Paper_Writer/Figures/08_LSTM_Prediction",
    "For_Paper_Writer/Figures/09_Physical_Model",
    "For_Paper_Writer/Markdown_Reports",
    "For_Bearing_Engineer",
    "Shared_Theory",
    "Shared_Figures",
    "Shared_Results",
    "Shared_Config/configs",
    "Shared_Config/source_entrypoints",
    "Shared_Config/interface_modules"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $PackageRoot $dir) | Out-Null
}

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )
    if (Test-Path $Source) {
        New-Item -ItemType Directory -Force -Path (Split-Path $Destination -Parent) | Out-Null
        Copy-Item -LiteralPath $Source -Destination $Destination -Force
    }
}

function Copy-Figure {
    param(
        [string]$Source,
        [string]$RelativeDestination
    )
    $dest = Join-Path $PackageRoot $RelativeDestination
    Copy-IfExists -Source $Source -Destination $dest
}

$phase = Get-Content -Raw (Join-Path $ProjectRoot "results/phase1_results.json") | ConvertFrom-Json
$configText = Get-Content -Raw (Join-Path $ProjectRoot "configs/default_config.yaml")
$dpi = if ($configText -match "dpi:\s*(\d+)") { [int]$Matches[1] } else { 320 }

$figureRows = New-Object System.Collections.Generic.List[object]
$figNo = 1

function Add-FigureRow {
    param(
        [string]$Source,
        [string]$Category,
        [string]$Name,
        [string]$Use,
        [string]$Section,
        [string]$Caption
    )
    if (-not (Test-Path $Source)) { return }
    $fileName = "Fig_{0:D2}_{1}.png" -f $script:figNo, $Name
    $relative = "For_Paper_Writer/Figures/$Category/$fileName"
    Copy-Figure -Source $Source -RelativeDestination $relative
    $script:figureRows.Add([pscustomobject]@{
        No = "Fig. $script:figNo"
        File = $fileName
        Path = $relative
        Name = $Name
        Use = $Use
        Section = $Section
        Caption = $Caption
    })
    $script:figNo += 1
}

function Get-FigPath {
    param([string]$Relative)
    return Join-Path $ProjectRoot $Relative
}

$datasets = @(
    @{ Key = "attachment1_reaction_wheel_3500d_data"; Short = "A1_3500d"; Label = "Attachment 1 3500-day reaction wheel" },
    @{ Key = "attachment2_reaction_wheel_1800d_data"; Short = "A2_1800d"; Label = "Attachment 2 1800-day reaction wheel" }
)

foreach ($d in $datasets) {
    $short = $d.Short
    $label = $d.Label
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_time_series_trends.png") "01_EDA" "$short`_time_series_trends" "Show long-term telemetry degradation tendency." "Data Description / Exploratory Analysis" "$label multivariate time-series trends."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_current_analysis.png") "01_EDA" "$short`_current_analysis" "Support current-driven degradation interpretation." "Exploratory Analysis" "$label current evolution and distribution analysis."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_temperature_analysis.png") "01_EDA" "$short`_temperature_analysis" "Support thermal degradation coupling analysis." "Exploratory Analysis" "$label temperature evolution and distribution analysis."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_current_fft_spectrum.png") "04_FFT_Analysis" "$short`_current_fft_spectrum" "Frequency-domain evidence for current signal variation." "Frequency-Domain Analysis" "$label current FFT spectrum."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_temperature_fft_spectrum.png") "04_FFT_Analysis" "$short`_temperature_fft_spectrum" "Frequency-domain evidence for temperature signal variation." "Frequency-Domain Analysis" "$label temperature FFT spectrum."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_current_wavelet_scalogram.png") "05_Wavelet_Analysis" "$short`_current_wavelet_scalogram" "Time-frequency degradation evidence." "Time-Frequency Analysis" "$label current wavelet scalogram."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_temperature_wavelet_scalogram.png") "05_Wavelet_Analysis" "$short`_temperature_wavelet_scalogram" "Time-frequency thermal degradation evidence." "Time-Frequency Analysis" "$label temperature wavelet scalogram."
    Add-FigureRow (Get-FigPath "figures/eda/$($d.Key)_correlation_heatmap.png") "06_Correlation_Heatmap" "$short`_correlation_heatmap" "Show variable coupling and degradation consistency." "Feature Correlation Analysis" "$label feature correlation heatmap."
    Add-FigureRow (Get-FigPath "figures/hi/$($d.Key)_health_index_curve.png") "02_Health_Index" "$short`_health_index_curve" "Main HI figure for degradation representation." "Health Indicator Construction" "$label health index curve."
    Add-FigureRow (Get-FigPath "figures/hi/$($d.Key)_hi_trend_and_derivative.png") "02_Health_Index" "$short`_hi_trend_and_derivative" "Show HI monotonic trend and degradation rate." "Health Indicator Construction" "$label HI trend and derivative."
    Add-FigureRow (Get-FigPath "figures/stages/$($d.Key)_stage_labels.png") "03_Stage_Segmentation" "$short`_stage_labels" "Main three-stage degradation segmentation result." "Stage Segmentation" "$label stage labels."
    Add-FigureRow (Get-FigPath "figures/stages/$($d.Key)_stage_statistics.png") "03_Stage_Segmentation" "$short`_stage_statistics" "Compare statistics across degradation stages." "Stage Segmentation" "$label stage statistics."
    Add-FigureRow (Get-FigPath "figures/models/$($d.Key)_current_exponential_fit.png") "09_Physical_Model" "$short`_current_exponential_fit" "Physical degradation law fitting result." "Degradation Modeling" "$label exponential physical-model fit."
    Add-FigureRow (Get-FigPath "figures/models/$($d.Key)_current_wiener_baseline.png") "07_Wiener_Prediction" "$short`_current_wiener_baseline" "Probabilistic Wiener baseline prediction." "RUL Prediction Baseline" "$label Wiener-process baseline."
    Add-FigureRow (Get-FigPath "figures/models/$($d.Key)_current_lstm_baseline.png") "08_LSTM_Prediction" "$short`_current_lstm_baseline" "Data-driven LSTM baseline prediction." "RUL Prediction Baseline" "$label LSTM baseline prediction."
}

$mdFiles = @(
    "README.md",
    "WORK_REPORT.md",
    "reports/phase1_report.md",
    "paper/phase1_method_outline.md"
)
foreach ($file in $mdFiles) {
    Copy-IfExists -Source (Join-Path $ProjectRoot $file) -Destination (Join-Path $PackageRoot ("For_Paper_Writer/Markdown_Reports/" + (Split-Path $file -Leaf)))
}

$paperIndex = @"
# Paper Figure Index

All figures in this directory are copied from the current PHM pipeline outputs and renamed with a unified paper-oriented convention. The project configuration uses `dpi=$dpi`, satisfying the paper-export target of dpi >= 300 for direct manuscript insertion.

| Suggested No. | File | Figure Name | Purpose | Recommended Section | Suggested Description |
|---|---|---|---|---|---|
"@
foreach ($row in $figureRows) {
    $paperIndex += "`n| $($row.No) | ``$($row.Path)`` | $($row.Name) | $($row.Use) | $($row.Section) | $($row.Caption) |"
}
$paperIndex += @"

## Usage Notes

- Use `Fig_09` and `Fig_24` as the primary HI curves when introducing the unified degradation representation.
- Use `Fig_11` and `Fig_26` as the primary degradation stage segmentation evidence.
- Use `Fig_14`/`Fig_29` for Wiener baselines and `Fig_15`/`Fig_30` for LSTM baselines.
- Chinese-named duplicate figures from the original project are intentionally excluded; English canonical outputs are used for collaboration.
"@
Set-Content -Path (Join-Path $PackageRoot "For_Paper_Writer/Paper_Figure_Index.md") -Value $paperIndex -Encoding UTF8

$sharedFigures = @(
    @{ Src="figures/hi/attachment1_reaction_wheel_3500d_data_health_index_curve.png"; Dst="Shared_Figures/SF_01_A1_best_HI_curve.png"; Desc="Attachment 1 best HI curve" },
    @{ Src="figures/hi/attachment2_reaction_wheel_1800d_data_health_index_curve.png"; Dst="Shared_Figures/SF_02_A2_best_HI_curve.png"; Desc="Attachment 2 best HI curve" },
    @{ Src="figures/stages/attachment1_reaction_wheel_3500d_data_stage_labels.png"; Dst="Shared_Figures/SF_03_A1_best_stage_segmentation.png"; Desc="Attachment 1 best stage segmentation" },
    @{ Src="figures/stages/attachment2_reaction_wheel_1800d_data_stage_labels.png"; Dst="Shared_Figures/SF_04_A2_best_stage_segmentation.png"; Desc="Attachment 2 best stage segmentation" },
    @{ Src="figures/eda/attachment1_reaction_wheel_3500d_data_time_series_trends.png"; Dst="Shared_Figures/SF_05_A1_key_degradation_trend.png"; Desc="Attachment 1 key degradation trend" },
    @{ Src="figures/eda/attachment2_reaction_wheel_1800d_data_time_series_trends.png"; Dst="Shared_Figures/SF_06_A2_key_degradation_trend.png"; Desc="Attachment 2 key degradation trend" },
    @{ Src="figures/eda/attachment1_reaction_wheel_3500d_data_correlation_heatmap.png"; Dst="Shared_Figures/SF_07_A1_correlation_heatmap.png"; Desc="Attachment 1 correlation heatmap" },
    @{ Src="figures/eda/attachment2_reaction_wheel_1800d_data_correlation_heatmap.png"; Dst="Shared_Figures/SF_08_A2_correlation_heatmap.png"; Desc="Attachment 2 correlation heatmap" },
    @{ Src="figures/eda/attachment1_reaction_wheel_3500d_data_current_fft_spectrum.png"; Dst="Shared_Figures/SF_09_A1_current_fft.png"; Desc="Attachment 1 key FFT" },
    @{ Src="figures/eda/attachment2_reaction_wheel_1800d_data_current_fft_spectrum.png"; Dst="Shared_Figures/SF_10_A2_current_fft.png"; Desc="Attachment 2 key FFT" },
    @{ Src="figures/models/attachment1_reaction_wheel_3500d_data_current_exponential_fit.png"; Dst="Shared_Figures/SF_11_A1_physical_degradation_model.png"; Desc="Attachment 1 physical degradation model" },
    @{ Src="figures/models/attachment2_reaction_wheel_1800d_data_current_exponential_fit.png"; Dst="Shared_Figures/SF_12_A2_physical_degradation_model.png"; Desc="Attachment 2 physical degradation model" },
    @{ Src="figures/models/attachment1_reaction_wheel_3500d_data_current_wiener_baseline.png"; Dst="Shared_Figures/SF_13_A1_wiener_result.png"; Desc="Attachment 1 Wiener result" },
    @{ Src="figures/models/attachment2_reaction_wheel_1800d_data_current_wiener_baseline.png"; Dst="Shared_Figures/SF_14_A2_wiener_result.png"; Desc="Attachment 2 Wiener result" },
    @{ Src="figures/models/attachment1_reaction_wheel_3500d_data_current_lstm_baseline.png"; Dst="Shared_Figures/SF_15_A1_lstm_result.png"; Desc="Attachment 1 LSTM result" },
    @{ Src="figures/models/attachment2_reaction_wheel_1800d_data_current_lstm_baseline.png"; Dst="Shared_Figures/SF_16_A2_lstm_result.png"; Desc="Attachment 2 LSTM result" }
)

$figureDirectory = "# Shared Figures Directory`n`nThe following figures are the deduplicated key visual evidence set for team discussion and manuscript drafting.`n`n| File | Description | Source |`n|---|---|---|"
foreach ($item in $sharedFigures) {
    Copy-Figure -Source (Join-Path $ProjectRoot $item.Src) -RelativeDestination $item.Dst
    $figureDirectory += "`n| `$($item.Dst)` | $($item.Desc) | `$($item.Src)` |"
}
Set-Content -Path (Join-Path $PackageRoot "Shared_Figures/Figure_Directory.md") -Value $figureDirectory -Encoding UTF8

Copy-IfExists -Source (Join-Path $ProjectRoot "requirements.txt") -Destination (Join-Path $PackageRoot "Shared_Config/requirements.txt")
Copy-IfExists -Source (Join-Path $ProjectRoot "configs/default_config.yaml") -Destination (Join-Path $PackageRoot "Shared_Config/configs/default_config.yaml")
foreach ($file in @("main.py","data_scanner.py","eda.py","health_index.py","stage_segmentation.py","degradation_models.py","report_generator.py")) {
    Copy-IfExists -Source (Join-Path $ProjectRoot $file) -Destination (Join-Path $PackageRoot "Shared_Config/source_entrypoints/$file")
}
foreach ($file in @("transfer_learning/domain_adapter.py","latent_space/shared_latent_model.py","multimodal/fusion.py","rul_prediction/rul_estimator.py")) {
    Copy-IfExists -Source (Join-Path $ProjectRoot $file) -Destination (Join-Path $PackageRoot ("Shared_Config/interface_modules/" + (Split-Path $file -Leaf)))
}

foreach ($file in @(
    "results/phase1_results.json",
    "results/data_summary.json",
    "results/attachment1_reaction_wheel_3500d_data_stage_boundaries.json",
    "results/attachment2_reaction_wheel_1800d_data_stage_boundaries.json",
    "results/attachment1_reaction_wheel_3500d_data_model_metrics.json",
    "results/attachment2_reaction_wheel_1800d_data_model_metrics.json",
    "processed_data/attachment1_reaction_wheel_3500d_data_health_index.csv",
    "processed_data/attachment2_reaction_wheel_1800d_data_health_index.csv"
)) {
    Copy-IfExists -Source (Join-Path $ProjectRoot $file) -Destination (Join-Path $PackageRoot ("Shared_Results/" + (Split-Path $file -Leaf)))
}

$a1Metrics = $phase.degradation_models.attachment1_reaction_wheel_3500d_data
$a2Metrics = $phase.degradation_models.attachment2_reaction_wheel_1800d_data
$a1Stage = $phase.stage_segmentation.attachment1_reaction_wheel_3500d_data.boundaries
$a2Stage = $phase.stage_segmentation.attachment2_reaction_wheel_1800d_data.boundaries
$a1Hi = $phase.health_index.attachment1_reaction_wheel_3500d_data.metadata
$a2Hi = $phase.health_index.attachment2_reaction_wheel_1800d_data.metadata

$resultsSummary = @"
# Results Summary

## 1. Best Model Metrics

| Dataset | Model | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Attachment 1, 3500 d | Exponential physical model | $($a1Metrics.exponential.metrics.RMSE) | $($a1Metrics.exponential.metrics.MAE) | $($a1Metrics.exponential.metrics.R2) |
| Attachment 1, 3500 d | Wiener baseline | $($a1Metrics.wiener.metrics.RMSE) | $($a1Metrics.wiener.metrics.MAE) | $($a1Metrics.wiener.metrics.R2) |
| Attachment 1, 3500 d | LSTM baseline | $($a1Metrics.lstm_baseline.metrics.RMSE) | $($a1Metrics.lstm_baseline.metrics.MAE) | $($a1Metrics.lstm_baseline.metrics.R2) |
| Attachment 2, 1800 d | Exponential physical model | $($a2Metrics.exponential.metrics.RMSE) | $($a2Metrics.exponential.metrics.MAE) | $($a2Metrics.exponential.metrics.R2) |
| Attachment 2, 1800 d | Wiener baseline | $($a2Metrics.wiener.metrics.RMSE) | $($a2Metrics.wiener.metrics.MAE) | $($a2Metrics.wiener.metrics.R2) |
| Attachment 2, 1800 d | LSTM baseline | $($a2Metrics.lstm_baseline.metrics.RMSE) | $($a2Metrics.lstm_baseline.metrics.MAE) | $($a2Metrics.lstm_baseline.metrics.R2) |

## 2. Stage Boundaries

| Dataset | Threshold | Derivative | Bayesian-style |
|---|---|---|---|
| Attachment 1, 3500 d | $($a1Stage.threshold -join ", ") | $($a1Stage.derivative -join ", ") | $($a1Stage.bayesian -join ", ") |
| Attachment 2, 1800 d | $($a2Stage.threshold -join ", ") | $($a2Stage.derivative -join ", ") | $($a2Stage.bayesian -join ", ") |

## 3. HI Statistics and Fields

| Dataset | HI Features | PCA Explained Variance | Processed File |
|---|---|---:|---|
| Attachment 1, 3500 d | $($a1Hi.feature_columns -join ", ") | $($a1Hi.pca_explained_variance_ratio) | ``attachment1_reaction_wheel_3500d_data_health_index.csv`` |
| Attachment 2, 1800 d | $($a2Hi.feature_columns -join ", ") | $($a2Hi.pca_explained_variance_ratio) | ``attachment2_reaction_wheel_1800d_data_health_index.csv`` |

Processed HI fields:

- ``day``: operating time index.
- ``*_norm``: direction-consistent Min-Max normalized degradation features.
- ``HI_raw``: one-dimensional PCA fusion result.
- ``HI_smooth``: smoothed health indicator used for trend analysis.
- ``HI_derivative``: degradation rate proxy for stage segmentation.

## 4. Current Experimental Conclusion

The flywheel telemetry exhibits strongly monotonic degradation behavior. Current, temperature, and friction torque are highly coupled with operating time, while speed is constant at 3000 rpm in the current data. The exponential physical model and LSTM baseline both fit the current evolution with high R2, whereas the Wiener process provides a probabilistic baseline with larger error but useful uncertainty-oriented modeling potential.
"@
Set-Content -Path (Join-Path $PackageRoot "Shared_Results/results_summary.md") -Value $resultsSummary -Encoding UTF8

$bearingGuide = @"
# Bearing Task Guide

## 1. Current PHM Architecture

The current PHM project implements a flywheel-first baseline: data scanning, EDA, health-index construction, degradation stage segmentation, physical degradation modeling, Wiener prediction, LSTM prediction, and automatic report generation. The engineering layout already reserves `transfer_learning/`, `latent_space/`, `multimodal/`, and `rul_prediction/` for XJTU-SY bearing extension.

## 2. Current Flywheel HI Definition

The flywheel health indicator is constructed by direction-consistent feature normalization followed by one-dimensional PCA fusion:

```text
feature matrix -> Min-Max normalization -> PCA first component -> HI_raw -> smoothing -> HI_smooth
```

Attachment 1 features: current, temperature, friction_torque, speed_rpm.  
Attachment 2 features: current, temperature, speed_rpm.  
The PCA explained variance ratios are $($a1Hi.pca_explained_variance_ratio) and $($a2Hi.pca_explained_variance_ratio), respectively.

## 3. Current Stage Division

The project compares threshold, derivative, and Bayesian-style change-point segmentation. For collaborative development, use Bayesian-style boundaries as the main reportable result unless later validation shows otherwise:

| Dataset | Healthy to slow degradation | Slow to accelerated degradation |
|---|---:|---:|
| Attachment 1, 3500 d | $($a1Stage.bayesian[0]) | $($a1Stage.bayesian[1]) |
| Attachment 2, 1800 d | $($a2Stage.bayesian[0]) | $($a2Stage.bayesian[1]) |

## 4. Flywheel Feature List and Data Structure

Attachment 1 fields: day, current, current_theoretical, temperature, speed_rpm, friction_torque.  
Attachment 2 fields: day, current, current_theoretical, temperature, speed_rpm.

Naming convention:

- `attachment1_reaction_wheel_3500d_data`: 3500-day flywheel data.
- `attachment2_reaction_wheel_1800d_data`: 1800-day flywheel data.
- `HI_raw`, `HI_smooth`, `HI_derivative`: canonical HI fields.
- `stage_boundaries`: degradation stage split points.

## 5. Bearing Follow-up Tasks

The XJTU-SY bearing module should add a bearing feature extraction pipeline aligned with the flywheel HI interface. Recommended vibration features:

- Time domain: RMS, mean, standard deviation, peak-to-peak value, skewness, kurtosis, crest factor, impulse factor, clearance factor.
- Frequency domain: spectral centroid, spectral entropy, band energy, dominant frequency, sideband energy.
- Time-frequency domain: wavelet packet energy, wavelet entropy, STFT band energy.
- Trend features: rolling mean, rolling variance, degradation slope, monotonicity score.

## 6. Recommended Bearing HI Construction

Use the same interface as flywheel:

```text
bearing vibration signal -> feature extraction -> direction alignment -> Min-Max/robust normalization -> PCA or AutoEncoder fusion -> HI_smooth -> stage segmentation
```

This keeps flywheel and bearing experiments comparable under one PHM framework.

## 7. Transfer Learning Interface

Use `Shared_Config/interface_modules/domain_adapter.py` as the current reserved entry point. The next implementation should define:

- source domain: flywheel telemetry features and HI.
- target domain: XJTU-SY bearing vibration features and HI.
- alignment target: monotonic degradation representation, stage labels, and RUL-related latent variables.
- candidate losses: MMD, CORAL, adversarial domain loss, contrastive stage alignment, monotonicity regularization.

## 8. Shared Latent Space Interface

Use `Shared_Config/interface_modules/shared_latent_model.py` as the reserved entry point. The intended design is:

```text
z_f = E_f(x_f), z_b = E_b(x_b), z in shared latent degradation space
```

The latent space should preserve degradation ordering, stage separability, and RUL predictability across heterogeneous flywheel and bearing signals.
"@
Set-Content -Path (Join-Path $PackageRoot "For_Bearing_Engineer/Bearing_Task_Guide.md") -Value $bearingGuide -Encoding UTF8

$coreTheory = @"
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
"@
Set-Content -Path (Join-Path $PackageRoot "Shared_Theory/Core_Theory.md") -Value $coreTheory -Encoding UTF8

$readme = @"
# Team Collaboration Share Package

This package is generated from the current PHM project for two collaboration roles: the paper writer and the XJTU-SY bearing engineer.

## 1. Current Progress

Phase 1 has completed the reaction-wheel PHM baseline:

- automatic data scan and profiling;
- degradation-oriented EDA;
- HI construction by direction-consistent normalization and PCA fusion;
- three-method stage segmentation;
- exponential physical model, Wiener baseline, and LSTM baseline;
- paper figures, reports, result JSON files, and reserved transfer-learning interfaces.

## 2. Directory Structure

- `For_Paper_Writer/`: paper-ready figures and Markdown reports.
- `For_Bearing_Engineer/`: bearing task guide and interface requirements.
- `Shared_Theory/`: unified theory line for the manuscript.
- `Shared_Figures/`: deduplicated key figures for team discussion.
- `Shared_Results/`: metrics, HI CSV files, stage boundaries, and summaries.
- `Shared_Config/`: requirements, config, source entry points, and reserved interfaces.

## 3. Key Results

- Attachment 1 Bayesian-style stage boundaries: $($a1Stage.bayesian -join ", ").
- Attachment 2 Bayesian-style stage boundaries: $($a2Stage.bayesian -join ", ").
- Attachment 1 best physical-model R2: $($a1Metrics.exponential.metrics.R2).
- Attachment 2 best physical-model R2: $($a2Metrics.exponential.metrics.R2).
- LSTM baselines also achieve high R2 and serve as data-driven comparison models.

## 4. Current Innovation Points

- A unified Shared Latent Degradation Space is proposed for heterogeneous PHM signals.
- The flywheel HI pipeline is built as the first domain branch.
- Bearing vibration features can be integrated through the same HI-stage-RUL abstraction.
- Transfer learning and multimodal fusion interfaces are already reserved.

## 5. Role Responsibilities

Paper writer:

- Use `For_Paper_Writer/Paper_Figure_Index.md` to select figures.
- Use `Shared_Theory/Core_Theory.md` as the theory backbone.
- Use `Shared_Results/results_summary.md` for numerical results.

Bearing engineer:

- Start from `For_Bearing_Engineer/Bearing_Task_Guide.md`.
- Implement XJTU-SY feature extraction with the same HI field convention.
- Connect bearing HI to transfer-learning and latent-space interfaces.

## 6. Next Tasks

1. Decide the main stage segmentation method for the final paper.
2. Add XJTU-SY bearing feature extraction and bearing HI generation.
3. Implement shared latent degradation encoders.
4. Add RUL prediction figures and uncertainty-aware warning results.
"@
Set-Content -Path (Join-Path $PackageRoot "Team_Collaboration_README.md") -Value $readme -Encoding UTF8
Set-Content -Path (Join-Path $PackageRoot "README.md") -Value $readme -Encoding UTF8

$configSummary = @"
# Shared Config Summary

- `requirements.txt`: Python dependency list.
- `configs/default_config.yaml`: canonical project paths, random seed, DPI, feature candidates, and model parameters.
- `source_entrypoints/`: current executable pipeline files.
- `interface_modules/`: reserved transfer learning, shared latent space, multimodal fusion, and RUL interfaces.

Current important parameters:

- random seed: 42
- figure DPI: $dpi
- LSTM sequence length: 12
- max LSTM epochs: 120
- train ratio: 0.75
- anomaly z-threshold: 3.0
"@
Set-Content -Path (Join-Path $PackageRoot "Shared_Config/config_summary.md") -Value $configSummary -Encoding UTF8

Write-Host "Team share package generated at: $PackageRoot"
Write-Host "Paper figure count: $($figureRows.Count)"
Write-Host "Shared key figure count: $($sharedFigures.Count)"
