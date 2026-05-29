# Phase-1 Report: Satellite Reaction Wheel PHM Based on Shared Degradation Representation

## Abstract
This report establishes the first-stage engineering baseline for predictive health management of satellite reaction wheels. The workflow includes automatic data profiling, degradation-oriented exploratory analysis, interpretable health index construction, stage segmentation, and initial physical/data-driven degradation modeling.

## 1. Data Description
Source directory: `C:\Users\35135\Desktop\数模校赛`

- `附件1 reaction_wheel_3500d_data.csv`: shape=[176, 6], columns=['day', 'current', 'current_theoretical', 'temperature', 'speed_rpm', 'friction_torque']
- `附件2 reaction_wheel_1800d_data.csv`: shape=[1801, 5], columns=['day', 'current', 'current_theoretical', 'temperature', 'speed_rpm']

## 2. Exploratory Degradation Analysis
### attachment1_reaction_wheel_3500d_data
Missing values: `{'day': 0, 'current': 0, 'current_theoretical': 0, 'temperature': 0, 'speed_rpm': 0, 'friction_torque': 0}`
Z-score outlier counts: `{'day': 0, 'current': 0, 'current_theoretical': 0, 'temperature': 0, 'speed_rpm': 0, 'friction_torque': 0}`
![Attachment1 Reaction Wheel 3500D Data Missing Values](../figures/eda/attachment1_reaction_wheel_3500d_data_missing_values.png)

![Attachment1 Reaction Wheel 3500D Data Time Series Trends](../figures/eda/attachment1_reaction_wheel_3500d_data_time_series_trends.png)

![Attachment1 Reaction Wheel 3500D Data Current Analysis](../figures/eda/attachment1_reaction_wheel_3500d_data_current_analysis.png)

![Attachment1 Reaction Wheel 3500D Data Temperature Analysis](../figures/eda/attachment1_reaction_wheel_3500d_data_temperature_analysis.png)

![Attachment1 Reaction Wheel 3500D Data Speed Rpm Analysis](../figures/eda/attachment1_reaction_wheel_3500d_data_speed_rpm_analysis.png)

![Attachment1 Reaction Wheel 3500D Data Friction Torque Analysis](../figures/eda/attachment1_reaction_wheel_3500d_data_friction_torque_analysis.png)

![Attachment1 Reaction Wheel 3500D Data Current Rolling Statistics](../figures/eda/attachment1_reaction_wheel_3500d_data_current_rolling_statistics.png)

![Attachment1 Reaction Wheel 3500D Data Temperature Rolling Statistics](../figures/eda/attachment1_reaction_wheel_3500d_data_temperature_rolling_statistics.png)

### attachment2_reaction_wheel_1800d_data
Missing values: `{'day': 0, 'current': 0, 'current_theoretical': 0, 'temperature': 0, 'speed_rpm': 0}`
Z-score outlier counts: `{'day': 0, 'current': 0, 'current_theoretical': 0, 'temperature': 0, 'speed_rpm': 0}`
![Attachment2 Reaction Wheel 1800D Data Missing Values](../figures/eda/attachment2_reaction_wheel_1800d_data_missing_values.png)

![Attachment2 Reaction Wheel 1800D Data Time Series Trends](../figures/eda/attachment2_reaction_wheel_1800d_data_time_series_trends.png)

![Attachment2 Reaction Wheel 1800D Data Current Analysis](../figures/eda/attachment2_reaction_wheel_1800d_data_current_analysis.png)

![Attachment2 Reaction Wheel 1800D Data Temperature Analysis](../figures/eda/attachment2_reaction_wheel_1800d_data_temperature_analysis.png)

![Attachment2 Reaction Wheel 1800D Data Speed Rpm Analysis](../figures/eda/attachment2_reaction_wheel_1800d_data_speed_rpm_analysis.png)

![Attachment2 Reaction Wheel 1800D Data Current Rolling Statistics](../figures/eda/attachment2_reaction_wheel_1800d_data_current_rolling_statistics.png)

![Attachment2 Reaction Wheel 1800D Data Temperature Rolling Statistics](../figures/eda/attachment2_reaction_wheel_1800d_data_temperature_rolling_statistics.png)

![Attachment2 Reaction Wheel 1800D Data Speed Rpm Rolling Statistics](../figures/eda/attachment2_reaction_wheel_1800d_data_speed_rpm_rolling_statistics.png)


## 3. Health Index Construction
The initial health index HI(t) is constructed by direction-consistent Min-Max normalization and one-dimensional PCA fusion. The AutoEncoder interface is reserved for subsequent cross-modal shared latent degradation representation learning.
### attachment1_reaction_wheel_3500d_data
Feature columns: `['current', 'temperature', 'friction_torque', 'speed_rpm']`
PCA explained variance ratio: `0.9999`
![Attachment1 Reaction Wheel 3500D Data Health Index Curve](../figures/hi/attachment1_reaction_wheel_3500d_data_health_index_curve.png)

![Attachment1 Reaction Wheel 3500D Data Hi Trend And Derivative](../figures/hi/attachment1_reaction_wheel_3500d_data_hi_trend_and_derivative.png)

### attachment2_reaction_wheel_1800d_data
Feature columns: `['current', 'temperature', 'speed_rpm']`
PCA explained variance ratio: `0.9999`
![Attachment2 Reaction Wheel 1800D Data Health Index Curve](../figures/hi/attachment2_reaction_wheel_1800d_data_health_index_curve.png)

![Attachment2 Reaction Wheel 1800D Data Hi Trend And Derivative](../figures/hi/attachment2_reaction_wheel_1800d_data_hi_trend_and_derivative.png)


## 4. Degradation Stage Segmentation
Three segmentation strategies are compared: HI threshold segmentation, derivative-based segmentation, and a Gaussian-evidence Bayesian-style change point detector.
### attachment1_reaction_wheel_3500d_data
Stage boundaries: `{'threshold': [58, 123], 'derivative': [96, 144], 'bayesian': [57, 116]}`
![Attachment1 Reaction Wheel 3500D Data Stage Labels](../figures/stages/attachment1_reaction_wheel_3500d_data_stage_labels.png)

![Attachment1 Reaction Wheel 3500D Data Stage Statistics](../figures/stages/attachment1_reaction_wheel_3500d_data_stage_statistics.png)

### attachment2_reaction_wheel_1800d_data
Stage boundaries: `{'threshold': [594, 1260], 'derivative': [978, 1446], 'bayesian': [606, 1219]}`
![Attachment2 Reaction Wheel 1800D Data Stage Labels](../figures/stages/attachment2_reaction_wheel_1800d_data_stage_labels.png)

![Attachment2 Reaction Wheel 1800D Data Stage Statistics](../figures/stages/attachment2_reaction_wheel_1800d_data_stage_statistics.png)


## 5. Initial Degradation Models
The physical model uses `i(t)=i0+a(exp(bt)-1)`. Data-driven baselines include Wiener drift and LSTM prediction.
### attachment1_reaction_wheel_3500d_data
Target variable: `current`
- exponential: metrics=`{'RMSE': 0.007396004225912381, 'MAE': 0.005886225765994979, 'R2': 0.9997048213769248}`
![Attachment1 Reaction Wheel 3500D Data Current Exponential Fit](../figures/models/attachment1_reaction_wheel_3500d_data_current_exponential_fit.png)

- wiener: metrics=`{'RMSE': 0.1685784384860726, 'MAE': 0.1540130681818181, 'R2': 0.8466461601817585}`
![Attachment1 Reaction Wheel 3500D Data Current Wiener Baseline](../figures/models/attachment1_reaction_wheel_3500d_data_current_wiener_baseline.png)

- lstm_baseline: metrics=`{'RMSE': 0.01098281020863567, 'MAE': 0.008450268926674666, 'R2': 0.9993490950732906}`
![Attachment1 Reaction Wheel 3500D Data Current Lstm Baseline](../figures/models/attachment1_reaction_wheel_3500d_data_current_lstm_baseline.png)

### attachment2_reaction_wheel_1800d_data
Target variable: `current`
- exponential: metrics=`{'RMSE': 0.007908420170681019, 'MAE': 0.006292965354412858, 'R2': 0.9990184252318618}`
![Attachment2 Reaction Wheel 1800D Data Current Exponential Fit](../figures/models/attachment2_reaction_wheel_1800d_data_current_exponential_fit.png)

- wiener: metrics=`{'RMSE': 0.10808310532612404, 'MAE': 0.09962422302424583, 'R2': 0.8166590220318031}`
![Attachment2 Reaction Wheel 1800D Data Current Wiener Baseline](../figures/models/attachment2_reaction_wheel_1800d_data_current_wiener_baseline.png)

- lstm_baseline: metrics=`{'RMSE': 0.009661452427013028, 'MAE': 0.00769715382622587, 'R2': 0.9985350299184412}`
![Attachment2 Reaction Wheel 1800D Data Current Lstm Baseline](../figures/models/attachment2_reaction_wheel_1800d_data_current_lstm_baseline.png)


## 6. Preliminary RUL Analysis
At this stage, RUL analysis is based on monotonic HI progression and degradation model extrapolation. A probabilistic RUL module is reserved under `rul_prediction/` for uncertainty-aware prediction after bearing data and cross-domain samples are added.

## 7. Extension Interfaces
The project reserves modular interfaces for XJTU-SY bearing feature extraction, transfer learning, multimodal fusion, shared latent degradation space learning, Transformer-based forecasting, attention mechanisms, and probabilistic warning.

## 8. XJTU-SY Bearing Feature Engineering
Current status: `XJTU-SY dataset not found yet`
Expected output: `Bearing Latent Features`
Feature groups: `['time_domain', 'frequency_domain', 'time_frequency']`
Quality metrics: `['monotonicity', 'trendability', 'robustness']`
The bearing module provides time-domain, frequency-domain, and time-frequency degradation features for the future Bearing Encoder and shared latent degradation space.
