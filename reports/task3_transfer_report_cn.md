# 问题三增强实验报告：跨领域迁移学习

## 1. 实验目标

本实验将问题二中从 XJTU-SY 滚动轴承学习到的退化规律迁移到反作用轮，用于改进问题一的附件 2 RUL 预测结果。
当前方法是物理一致性约束下的退化严重度迁移校准，不是 DANN/CORAL/MMD 深度迁移训练。

## 2. 为什么可以迁移

滚动轴承与反作用轮轴承都存在润滑退化、摩擦增强、退化观测量上升这一共同物理链条。轴承侧表现为振动能量和时频特征变化，飞轮侧表现为电流、温度和 common HI 上升。
仅使用附件 2 已观测 0-1800 天时，飞轮 HI 与轴承 HI 的归一化轨迹相似度为 `0.9931`，趋势相关系数为 `0.9874`。

## 3. 迁移方法对比

| 方法 | 说明 | RUL / 天 | EOL day |
|---|---|---:|---:|
| no_transfer | Task-1 A1-calibrated common HI baseline | 608.6 | 2408.6 |
| severity_transfer | Bearing late-life HI severity calibration | 683.7 | 2483.7 |
| stage_ratio_transfer | Bearing accelerated-stage proportion calibration | 696.4 | 2496.4 |
| combined_transfer | Average of severity and stage-ratio transfer factors | 690.0 | 2490.0 |

推荐采用 `combined_transfer`，RUL 为 `690.0` 天，区间为 `[627.3, 766.7]` 天。

## 4. 敏感性分析

| 迁移系数扰动 | 迁移系数 | RUL / 天 | 相对名义 RUL 变化 / 天 |
|---:|---:|---:|---:|
| -10% | 0.7938 | 766.7 | 76.7 |
| -5% | 0.8379 | 726.3 | 36.3 |
| 0% | 0.8820 | 690.0 | 0.0 |
| 5% | 0.9261 | 657.2 | -32.9 |
| 10% | 0.9702 | 627.3 | -62.7 |

敏感性分析表明，推荐 RUL 会随迁移系数变化而单调变化，但在 ±10% 扰动内仍处于同一量级，说明迁移结论对小范围参数扰动具有一定稳定性。

## 5. 输出文件

- `results/task3_observed_similarity.json`
- `results/task3_transfer_comparison.csv`
- `results/task3_transfer_sensitivity.csv`
- `results/task3_transfer_summary.json`
- `figures/transfer_health/observed_domain_similarity.png`
- `figures/transfer_health/task3_transfer_comparison.png`
- `figures/transfer_health/task3_sensitivity.png`

## 6. 图像

![attachment2_transfer_rul_comparison](../figures/transfer_health/attachment2_transfer_rul_comparison.png)

![flywheel_bearing_domain_similarity](../figures/transfer_health/flywheel_bearing_domain_similarity.png)

![observed_domain_similarity](../figures/transfer_health/observed_domain_similarity.png)

![task3_transfer_comparison](../figures/transfer_health/task3_transfer_comparison.png)

![task3_sensitivity](../figures/transfer_health/task3_sensitivity.png)
