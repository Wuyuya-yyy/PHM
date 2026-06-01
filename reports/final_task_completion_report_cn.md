# 数模 A 题最终完成度报告

## 1. 当前状态

本机项目已覆盖题目四个任务，并已在队友最新上传版本 `1513050 Add task1 RUL analysis outputs` 的基础上重新补齐任务三和任务四。

完整主流程：

```bash
python3 main.py
```

## 2. 任务一：反作用轮退化建模与健康状态初评估

队友最新版本新增了 `task1_rul_analysis.py`，并输出了任务一 RUL 分析结果。

已完成内容：

- 附件 1、附件 2 数据读取。
- 电流、理论电流、温度、转速等退化指标分析。
- 反作用轮 HI 与阶段划分。
- 指数模型、近期线性模型等 RUL 外推。
- 附件 2 当前健康阶段评估。

核心结果：

- 附件 2 当前阶段：`Accelerated Degradation`。
- 任务一推荐无迁移 RUL：`608.6` 天。
- 任务一推荐 EOL day：`2408.6`。

核心文件：

- `task1_rul_analysis.py`
- `results/task1_rul_summary.json`
- `results/task1_rul_predictions.csv`
- `reports/task1_completion_report.md`
- `figures/rul/`

## 3. 任务二：XJTU-SY 轴承退化建模

已完成内容：

- XJTU-SY 全部 9216 个 CSV 数据读取。
- 3 个工况、15 条轴承全寿命试验处理。
- 水平、垂直、合成振动三通道特征。
- 时域特征：RMS、Kurtosis、Skewness、Crest Factor、Peak-to-Peak 等。
- 频域特征：FFT energy、Spectral Entropy、Spectral Centroid 等。
- 时频域特征：Wavelet Packet Energy、STFT features。
- 特征评价：单调性、趋势性、稳定性、可迁移性、综合评分。
- Bearing HI、阶段标签、latent features。
- 基于 Bearing HI 建立并对比指数退化模型、Wiener 漂移模型、随机森林退化映射模型。

核心文件：

- `results/bearing/xjtu_sy_bearing_features.csv`
- `results/bearing/xjtu_sy_feature_quality.csv`
- `results/bearing/xjtu_sy_bearing_hi.csv`
- `results/bearing/bearing_latent_features.csv`
- `results/bearing_models/bearing_degradation_model_comparison.csv`
- `results/bearing_models/bearing_degradation_model_summary.json`
- `reports/bearing_task_summary_cn.md`
- `figures/bearing/`
- `figures/bearing_models/`

轴承退化模型对比结果：

- 指数模型平均 RMSE：约 `0.0937`。
- Wiener 漂移模型平均 RMSE：约 `0.2160`。
- 随机森林退化映射平均 RMSE：约 `0.0159`。
- 15 条轴承退化序列中，随机森林模型均取得最低 RMSE，用作数据驱动对照模型；指数模型保留物理可解释性；Wiener 模型作为随机过程基线。

## 4. 任务三：跨领域迁移学习

已在队友任务一 RUL 基线基础上重新完成。

迁移策略：

1. 读取任务一推荐的 A1-calibrated common HI RUL 作为无迁移基线。
2. 从 XJTU-SY 轴承中学习后期退化 HI 分布和高质量迁移特征。
3. 比较飞轮 common-HI 轨迹与轴承 HI 轨迹的归一化相似性。
4. 对比无迁移、严重度迁移、阶段比例迁移、综合迁移四种 RUL。
5. 对综合迁移系数做 `±5%`、`±10%` 敏感性分析。

在增强版本中，另补充了真正的 PyTorch 深度域适配训练流程：`no_adaptation`、`CORAL`、`MMD`、`DANN`、`AutoEncoder-joint`。验证方式为使用附件 1 已知全寿命数据，前 70% 训练、后 30% 做真实 RUL 误差测试。

核心结果：

- 任务一无迁移 RUL：`608.6` 天。
- 严重度迁移 RUL：`683.7` 天。
- 阶段比例迁移 RUL：`696.4` 天。
- 综合迁移推荐 RUL：`690.0` 天。
- 推荐 RUL 区间：`627.3 - 766.7` 天。
- 已输出迁移 RUL 对比图、跨域相似性图和敏感性分析图。
- 深度域适配实验中，`MMD` 方法取得最佳附件 1 后 30% RUL 测试 MAE，约 `87.2` 天；`no_adaptation` MAE 约 `154.2` 天，说明训练式分布对齐能降低验证误差。
- MMD 对附件 2 给出的深度模型参考 RUL 约 `408.5` 天，作为保守对照，不直接替代综合迁移推荐 RUL。

核心文件：

- `transfer_health_management.py`
- `results/transfer_health_management_results.json`
- `results/task3_observed_similarity.json`
- `results/task3_transfer_comparison.csv`
- `results/task3_transfer_sensitivity.csv`
- `results/task3_transfer_summary.json`
- `results/deep_transfer/deep_transfer_summary.json`
- `results/deep_transfer/deep_transfer_method_comparison.csv`
- `results/deep_transfer/deep_transfer_rul_point_predictions.csv`
- `results/deep_transfer/deep_transfer_latent_features.csv`
- `figures/transfer_health/attachment2_transfer_rul_comparison.png`
- `figures/transfer_health/flywheel_bearing_domain_similarity.png`
- `figures/transfer_health/observed_domain_similarity.png`
- `figures/transfer_health/task3_transfer_comparison.png`
- `figures/transfer_health/task3_sensitivity.png`
- `figures/deep_transfer/deep_transfer_rul_error_comparison.png`
- `figures/deep_transfer/deep_transfer_latent_distance.png`
- `figures/deep_transfer/deep_transfer_best_latent_pca.png`
- `reports/deep_transfer_report_cn.md`

## 5. 任务四：飞轮健康管理报告

已完成内容：

- 四级预警机制。
- 附件 2 当前预警等级。
- 迁移校准 RUL 和不确定区间。
- 工程建议。
- 不确定性和局限性说明。

核心文件：

- `reports/health_management_report_cn.md`
- `reports/phase1_report.md`

## 6. 后续仍需人工处理

当前仓库提供完整实验、代码、图表和 Markdown 报告。若要正式提交竞赛，还需要将内容排版进学校 LaTeX 模板，形成最终论文 PDF。

需要人工整合：

- 摘要。
- 问题重述。
- 模型假设。
- 符号说明。
- 模型建立与求解。
- 结果分析。
- 灵敏度/误差分析。
- 模型优缺点。
- 参考文献。
- 附录代码。

## 7. 结论

从实验和建模角度，四个任务已经完成闭环：

`任务一飞轮 RUL 基线 -> 任务二轴承退化表征 -> 任务三轴承到飞轮迁移 -> 任务四健康管理决策`

现在主要剩余工作是正式论文写作与排版。
