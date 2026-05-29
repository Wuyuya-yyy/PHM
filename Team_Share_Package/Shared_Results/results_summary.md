# Results Summary

## 1. Best Model Metrics

| Dataset | Model | RMSE | MAE | R2 |
|---|---:|---:|---:|---:|
| Attachment 1, 3500 d | Exponential physical model | 0.007396004225912381 | 0.005886225765994979 | 0.9997048213769248 |
| Attachment 1, 3500 d | Wiener baseline | 0.1685784384860726 | 0.1540130681818181 | 0.8466461601817585 |
| Attachment 1, 3500 d | LSTM baseline | 0.01098281020863567 | 0.008450268926674666 | 0.9993490950732906 |
| Attachment 2, 1800 d | Exponential physical model | 0.007908420170681019 | 0.006292965354412858 | 0.9990184252318618 |
| Attachment 2, 1800 d | Wiener baseline | 0.10808310532612404 | 0.09962422302424583 | 0.8166590220318031 |
| Attachment 2, 1800 d | LSTM baseline | 0.009661452427013028 | 0.00769715382622587 | 0.9985350299184412 |

## 2. Stage Boundaries

| Dataset | Threshold | Derivative | Bayesian-style |
|---|---|---|---|
| Attachment 1, 3500 d | 58, 123 | 96, 144 | 57, 116 |
| Attachment 2, 1800 d | 594, 1260 | 978, 1446 | 606, 1219 |

## 3. HI Statistics and Fields

| Dataset | HI Features | PCA Explained Variance | Processed File |
|---|---|---:|---|
| Attachment 1, 3500 d | current, temperature, friction_torque, speed_rpm | 0.999919989516465 | `attachment1_reaction_wheel_3500d_data_health_index.csv` |
| Attachment 2, 1800 d | current, temperature, speed_rpm | 0.9999479113670362 | `attachment2_reaction_wheel_1800d_data_health_index.csv` |

Processed HI fields:

- `day`: operating time index.
- `*_norm`: direction-consistent Min-Max normalized degradation features.
- `HI_raw`: one-dimensional PCA fusion result.
- `HI_smooth`: smoothed health indicator used for trend analysis.
- `HI_derivative`: degradation rate proxy for stage segmentation.

## 4. Current Experimental Conclusion

The flywheel telemetry exhibits strongly monotonic degradation behavior. Current, temperature, and friction torque are highly coupled with operating time, while speed is constant at 3000 rpm in the current data. The exponential physical model and LSTM baseline both fit the current evolution with high R2, whereas the Wiener process provides a probabilistic baseline with larger error but useful uncertainty-oriented modeling potential.
