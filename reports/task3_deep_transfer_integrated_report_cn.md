# 问题三深度迁移增强综合报告

## 1. 报告目的

本报告汇总当前数模项目中“轴承退化规律迁移到飞轮 RUL 预测”的最新完成状态。当前项目已经包含两条迁移证据链：

1. 可解释迁移校准：基于 XJTU-SY 轴承 HI、阶段比例、后期退化严重度，对附件 2 飞轮 RUL 进行校准。
2. 深度域适配增强实验：基于 PyTorch 训练 Bearing Encoder、Flywheel Encoder 和共享 latent 表征，并比较 DANN、CORAL、MMD、AutoEncoder-joint 等方法的迁移前后 RUL 误差。

正式论文中建议以“物理一致性约束下的退化严重度迁移校准”为最终推荐方法，以深度域适配实验作为增强验证和保守对照。

## 2. 数据与任务关系

本项目使用三类数据：

- 附件 1：反作用轮 3500 天全寿命数据，用于建立飞轮退化基准、失效阈值和 RUL 验证依据。
- 附件 2：反作用轮 0-1800 天在轨截断数据，用于当前健康状态评估和剩余寿命预测。
- XJTU-SY 轴承数据：3 个工况、15 条轴承全寿命试验、共 9216 个 CSV，用于建立轴承退化表征系统。

迁移学习的核心思想不是直接把轴承寿命时间尺度搬到飞轮上，而是迁移“归一化退化状态、后期严重度、阶段演化规律和 latent 退化表征”。

## 3. 轴承退化表征完成情况

问题二中已经完成 XJTU-SY 轴承退化表征系统：

- 时域特征：RMS、Kurtosis、Skewness、Crest Factor、Peak-to-Peak 等。
- 频域特征：FFT energy、Spectral Entropy、Spectral Centroid 等。
- 时频域特征：Wavelet Packet Energy、STFT features。
- 特征评价：单调性、趋势性、稳定性、可迁移性、综合评分。
- 输出 Bearing HI、阶段标签、latent features。

这些输出为问题三迁移提供了轴承侧的源域退化表征。

## 4. 可解释迁移校准结果

可解释迁移校准先使用任务一的 A1-calibrated common HI 作为无迁移基线，再用轴承后期退化严重度和阶段比例对 RUL 进行校准。

仅使用附件 2 已观测 0-1800 天时，飞轮 HI 与轴承 HI 的归一化轨迹相似度为 `0.9931`，趋势相关系数为 `0.9874`，说明两类退化过程在健康状态空间中具有较强同向性。

RUL 对比如下：

| 方法 | 含义 | RUL / 天 | EOL day |
|---|---|---:|---:|
| no_transfer | 任务一无迁移基线 | 608.6 | 2408.6 |
| severity_transfer | 轴承后期严重度迁移 | 683.7 | 2483.7 |
| stage_ratio_transfer | 轴承阶段比例迁移 | 696.4 | 2496.4 |
| combined_transfer | 综合迁移推荐 | 690.0 | 2490.0 |

最终推荐采用 `combined_transfer`，即附件 2 推荐 RUL 为 `690.0` 天，推荐区间为 `[627.3, 766.7]` 天。

## 5. 敏感性分析

对综合迁移系数进行 `-10%`、`-5%`、`0`、`+5%`、`+10%` 扰动后，RUL 仍保持在同一数量级内，结果区间为 `[627.3, 766.7]` 天。说明迁移结论对小范围参数扰动具有一定稳定性。

## 6. 深度域适配实验设计

为回应“是否真正完成 DANN/CORAL/MMD/AutoEncoder 深度训练”的问题，项目新增了 `deep_domain_adaptation.py`，补充真实 PyTorch 训练流程：

```text
轴承特征 -> Bearing Encoder -> z_b
飞轮特征 -> Flywheel Encoder -> z_f
域适配损失 -> 对齐 z_b 和 z_f
共享退化进度预测头 -> RUL 误差验证
```

训练方法包括：

- `no_adaptation`：无域适配基线。
- `CORAL`：协方差对齐训练。
- `MMD`：最大均值差异最小化训练。
- `DANN`：梯度反转和域判别器对抗训练。
- `AutoEncoder-joint`：跨域联合重构与 latent 对齐。

验证方式：使用附件 1 已知全寿命数据，前 70% 作为训练段，后 30% 作为真实 RUL 测试段。这样可以真实比较迁移前后 RUL 误差变化。

## 7. 深度迁移实验结果

| 方法 | A1 测试 MAE / 天 | A1 测试 RMSE / 天 | A2 参考 RUL / 天 | latent MMD | latent CORAL |
|---|---:|---:|---:|---:|---:|
| MMD | 87.2 | 101.6 | 408.5 | 0.3861 | 0.000255 |
| AutoEncoder-joint | 117.1 | 141.7 | 377.4 | 0.4329 | 0.001059 |
| no_adaptation | 154.2 | 174.0 | 119.0 | 0.8747 | 0.020260 |
| DANN | 167.7 | 199.0 | 6200.0 | 1.0138 | 0.029077 |
| CORAL | 200.0 | 213.4 | 4.9 | 0.8233 | 0.004450 |

结果表明：

- MMD 是本次深度域适配实验中表现最好的方法。
- 与 no_adaptation 相比，MMD 将附件 1 后 30% RUL 测试 MAE 从约 `154.2` 天降至约 `87.2` 天。
- MMD 同时降低了 latent MMD 距离，说明轴承和飞轮的共享退化表征确实被拉近。
- DANN 和 CORAL 在当前数据规模与时间尺度差异下表现不稳定，不能写成优于基线。

## 8. 为什么最终推荐仍采用 690.0 天

深度模型给出的附件 2 参考 RUL 为 `408.5` 天，而可解释综合迁移给出的推荐 RUL 为 `690.0` 天。两者差异来自建模目标不同：

- 可解释迁移校准：基于任务一 A1-calibrated common HI、轴承后期严重度和阶段比例，适合作为最终工程建议。
- 深度域适配实验：基于附件 1 后段验证误差最小化，适合作为模型增强验证和保守对照。

因此，论文中建议写成：

> 综合迁移校准给出推荐 RUL 为 690.0 天；深度 MMD 域适配给出更保守的参考 RUL 为 408.5 天。二者共同说明附件 2 已进入加速退化和重点监测阶段，但尚未达到立即失效状态。

## 9. 当前健康管理结论

附件 2 当前观测至第 1800 天，任务一和迁移结果均表明其处于 `Accelerated Degradation` 阶段。

当前建议：

- 预警等级：`Level 2 - Warning`。
- 推荐进入重点监测。
- 缩短健康评估周期。
- 限制连续长时间高转速、高负载运行。
- 将综合迁移 RUL `690.0` 天作为主推荐，将深度 MMD RUL `408.5` 天作为保守检修参考。

## 10. 论文写作建议

论文中不要写成“所有深度迁移方法均有效”。更严谨的写法是：

1. 先建立飞轮 HI 和 RUL 基线。
2. 再建立轴承退化表征系统。
3. 用 HI 相似性和阶段比例说明轴承退化规律可迁移。
4. 用综合迁移校准给出最终 RUL。
5. 用深度 MMD 域适配验证迁移学习的有效性。
6. 如实说明 DANN/CORAL 在当前数据条件下不稳定，这是模型局限性的一部分。

## 11. 关键输出文件

- `results/task3_transfer_summary.json`
- `results/task3_transfer_comparison.csv`
- `results/task3_transfer_sensitivity.csv`
- `results/deep_transfer/deep_transfer_summary.json`
- `results/deep_transfer/deep_transfer_method_comparison.csv`
- `results/deep_transfer/deep_transfer_rul_point_predictions.csv`
- `results/deep_transfer/deep_transfer_latent_features.csv`
- `reports/task3_transfer_report_cn.md`
- `reports/deep_transfer_report_cn.md`
- `figures/transfer_health/task3_transfer_comparison.png`
- `figures/transfer_health/task3_sensitivity.png`
- `figures/deep_transfer/deep_transfer_rul_error_comparison.png`
- `figures/deep_transfer/deep_transfer_latent_distance.png`
- `figures/deep_transfer/deep_transfer_best_latent_pca.png`

## 12. 总结

当前问题三已经从“只有迁移校准结果”升级为“可解释迁移校准 + 深度域适配验证”的完整实验链条。最终建议采用综合迁移 RUL `690.0` 天，深度 MMD 给出的 `408.5` 天作为保守对照。这样既保证论文结果具有可解释性，也能回应深度迁移训练是否真正完成的问题。
