# Task 1 Completion Report: Reaction Wheel Degradation and RUL

## 1. Completion Status

Task 1 is completed as a self-contained flywheel PHM baseline: degradation indicators are identified, multiple degradation models are compared, full-life stages are segmented, and Attachment 2 is assigned a current health stage with RUL estimates.

## 2. Effective Degradation Indicators

- Current is the primary degradation fingerprint because it tracks the torque needed to overcome increasing bearing friction.
- Temperature provides a coupled thermal degradation signal.
- Friction torque is available in Attachment 1 and directly supports the physical interpretation.
- The constructed HI fuses common current-temperature degradation evidence into a single monotonic health state.

## 3. Stage Assessment

- Recommended segmentation method: `bayesian`.
- Attachment 2 Bayesian-style boundaries: `[606, 1219]`.
- Current operating day: `1800.0`.
- Current health stage: **Accelerated Degradation**.

## 4. Failure Threshold Definition

The failure state is anchored by the full-life endpoint of Attachment 1. This gives a data-driven end-of-life reference rather than using the maximum value inside Attachment 2, which is only a truncated in-orbit monitoring window.

- Measured current threshold: `1.613900` A.
- Theoretical current threshold: `1.607245` A.
- Comparable HI threshold: `1.000000`.

## 5. Attachment 2 RUL Results

| Method | Predicted EOL Day | RUL / days | In-sample R2 |
|---|---:|---:|---:|
| Measured current exponential | 2382.7 | 582.7 | 0.9990 |
| Theoretical current exponential | 2376.7 | 576.7 | 1.0000 |
| A1-calibrated common HI exponential | 2408.6 | 608.6 | 1.0000 |
| A1-calibrated common HI recent-linear | 2611.5 | 811.5 | 0.9999 |

Recommended RUL: **608.6 days**, corresponding to predicted EOL day **2408.6**. The recommended result uses the A1-calibrated common HI exponential model because it transfers the full-life failure threshold from Attachment 1 while using Attachment 2's observed degradation trend.

## 6. Generated Files

- Summary JSON: `E:\PHM\results\task1_rul_summary.json`
- Prediction CSV: `E:\PHM\results\task1_rul_predictions.csv`
- Current RUL figure: `E:\PHM\figures\rul\attachment2_current_rul_extrapolation.png`
- HI RUL figure: `E:\PHM\figures\rul\attachment2_common_hi_rul_extrapolation.png`

## 7. Assumptions and Limits

- The failure endpoint is defined by Attachment 1's full-life endpoint.
- Attachment 2 is treated as a truncated monitoring sequence of the same 1 Nms-class reaction wheel.
- The RUL is a Task-1 no-transfer baseline; bearing-to-flywheel transfer learning is left for Task 3.
- The LSTM result in the existing project is an in-sample one-step baseline and is not used as the main long-horizon RUL estimator here.