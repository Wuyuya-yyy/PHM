# Phase 2 实验报告：XJTU-SY 轴承退化表征系统

## 1. 任务目标

本阶段目标不是简单训练一个预测模型，而是建立面向后续跨模态共享退化表征学习的轴承退化表征系统。该系统需要从 XJTU-SY 高速轴承振动数据中提取可解释退化特征，构建 Bearing Health Index，完成退化阶段划分，并输出可供迁移学习使用的 latent features。

本阶段覆盖以下核心任务：

1. XJTU-SY 数据读取。
2. 时域特征提取。
3. 频域特征提取。
4. 时频域特征提取。
5. 特征评价。
6. 轴承 Health Index 构建。
7. 退化阶段划分。

## 2. 数据读取与完整性

XJTU-SY 数据集已解压并接入项目，路径为：

`/Users/nightye/Desktop/数模/XJTU-SY/XJTU-SY_Bearing_Datasets`

数据完整性检查结果如下：

| 工况 | CSV 数量 |
|---|---:|
| 35Hz12kN | 616 |
| 37.5Hz11kN | 1566 |
| 40Hz10kN | 7034 |
| 总计 | 9216 |

每个样本文件包含两路振动信号：

- `Horizontal_vibration_signals`
- `Vertical_vibration_signals`

本实验额外构造合成振动信号：

`resultant = sqrt(horizontal^2 + vertical^2)`

因此后续特征同时覆盖水平、垂直、合成三类通道。

## 3. 特征提取方法

### 3.1 时域特征

已提取任务要求中的全部时域特征：

- RMS
- Kurtosis
- Skewness
- Crest Factor
- Peak-to-Peak

同时保留辅助退化特征：

- peak
- mean
- mean absolute value
- standard deviation
- impulse factor
- shape factor
- margin factor

这些特征用于描述振动幅值、冲击性、分布非对称性和故障冲击增强现象。

### 3.2 频域特征

已提取任务要求中的全部频域特征：

- FFT energy
- Spectral Entropy
- Spectral Centroid

同时提取：

- dominant frequency
- 8 个 FFT 频带能量占比

这些特征用于描述故障演化过程中频谱能量迁移、频谱复杂度变化和主导频率成分变化。

### 3.3 时频域特征

已提取任务要求中的全部时频域特征：

- Wavelet Packet Energy
- STFT features

具体实现：

- 使用 Wavelet Packet Decomposition 得到多频带小波包能量占比。
- 使用 STFT 得到时频矩阵，并提取 4 个频带能量占比和 STFT 总能量。

时频特征用于刻画非平稳振动信号中局部冲击和频带能量随寿命演化的变化。

## 4. 特征评价

为满足后续跨模态共享退化表征学习的需要，本阶段对每一个候选特征进行质量评价。评价指标包括：

- monotonicity：单调退化趋势。
- trendability：与寿命时间轴的相关趋势。
- stability：平滑稳定性和抗噪声能力。
- transferability：不同工况、不同轴承之间趋势一致性。
- overall_score：综合退化表征评分。

特征评价结果输出到：

`results/bearing/xjtu_sy_feature_quality.csv`

该表共包含 114 个特征的评价结果。综合评分靠前的特征包括：

| 特征 | 说明 |
|---|---|
| resultant_mean / resultant_mean_abs | 合成振动幅值随退化增强，趋势性较强 |
| resultant_rms | 反映总体振动能量增长 |
| vertical_mean_abs / vertical_rms | 垂直方向振动幅值退化敏感 |
| horizontal_band_energy_4 | 特定频带能量迁移明显 |
| vertical_stft_band_energy_2 | 时频局部能量变化具有退化指示性 |

对应可视化图像：

`figures/bearing/bearing_feature_quality_top20.png`

## 5. Bearing Health Index 构建

本实验按每条轴承退化试验单独构建 HI，避免不同工况和不同轴承之间的幅值尺度差异直接混合。

处理流程为：

`特征矩阵 -> 标准化 -> PCA 一维融合 -> 方向一致化 -> Min-Max 归一化 -> HI_smooth`

其中：

- `PCA_HI`：PCA 融合得到的原始健康指标。
- `AE_HI`：预留 AutoEncoder HI 接口，目前采用 PCA-HI 作为可复现替代。
- `HI_smooth`：平滑后的健康指标，用于阶段划分和后续迁移学习。

Bearing HI 输出到：

`results/bearing/xjtu_sy_bearing_hi.csv`

该表规模为：

- 9216 行
- 123 列

## 6. 退化阶段划分

本阶段将每条轴承寿命序列按照平滑 HI 划分为三类退化阶段：

- `Healthy`
- `Slow Degradation`
- `Accelerated Degradation`

阶段标签已经写入：

`results/bearing/xjtu_sy_bearing_hi.csv`

全体样本阶段统计如下：

| 阶段 | 样本数 |
|---|---:|
| Healthy | 3074 |
| Slow Degradation | 3067 |
| Accelerated Degradation | 3075 |

15 条轴承试验分别生成了 HI 曲线和阶段散点图，位于：

`figures/bearing/`

## 7. Latent Features 输出

为后续跨模态共享退化表征学习，本阶段输出了轴承 latent features：

`results/bearing/bearing_latent_features.csv`

该表规模为：

- 9216 行
- 15 列

字段包括：

- sample_key
- condition
- bearing_id
- sample_id
- PCA_HI
- HI_smooth
- stage
- bearing_latent_1 至 bearing_latent_8

该表可直接作为目标域轴承退化表征 `z_b`，后续可与飞轮遥测 latent 表征 `z_f` 做 MMD、CORAL、对比学习、阶段一致性约束或共享潜变量建模。

## 8. 最终输出清单

| 输出要求 | 完成状态 | 文件 |
|---|---|---|
| Bearing HI | 已完成 | `results/bearing/xjtu_sy_bearing_hi.csv` |
| 特征评价结果 | 已完成 | `results/bearing/xjtu_sy_feature_quality.csv` |
| 阶段标签 | 已完成 | `results/bearing/xjtu_sy_bearing_hi.csv` |
| 可视化图像 | 已完成 | `figures/bearing/` |
| Latent features | 已完成 | `results/bearing/bearing_latent_features.csv` |
| 处理摘要 | 已完成 | `results/bearing/bearing_feature_summary.json` |

## 9. 查验步骤

在终端进入项目目录：

```bash
cd /Users/nightye/Desktop/PHM
```

### 9.1 查验数据是否完整

```bash
find /Users/nightye/Desktop/数模/XJTU-SY/XJTU-SY_Bearing_Datasets -type f -name "*.csv" | wc -l
find /Users/nightye/Desktop/数模/XJTU-SY/XJTU-SY_Bearing_Datasets -type f -name "*.csv" | awk -F'/XJTU-SY_Bearing_Datasets/' '{print $2}' | cut -d/ -f1 | sort | uniq -c
```

应看到总数为 `9216`，三组工况分别为 `616`、`1566`、`7034`。

### 9.2 查验特征、HI、阶段标签和 latent features

```bash
python3 - <<'PY'
import pandas as pd

files = [
    "results/bearing/xjtu_sy_bearing_features.csv",
    "results/bearing/xjtu_sy_feature_quality.csv",
    "results/bearing/xjtu_sy_bearing_hi.csv",
    "results/bearing/bearing_latent_features.csv",
]

for path in files:
    df = pd.read_csv(path)
    print(path, df.shape)

hi = pd.read_csv("results/bearing/xjtu_sy_bearing_hi.csv")
print(hi["stage"].value_counts())
PY
```

应看到：

- 特征表：`9216 x 119`
- 特征评价表：`114 x 7`
- HI 表：`9216 x 123`
- latent features：`9216 x 15`
- 阶段标签包含三类退化状态。

### 9.3 查验必须特征是否存在

```bash
python3 - <<'PY'
import pandas as pd
df = pd.read_csv("results/bearing/xjtu_sy_bearing_features.csv", nrows=1)
checks = [
    "rms",
    "kurtosis",
    "skewness",
    "crest_factor",
    "peak_to_peak",
    "spectral_energy",
    "spectral_entropy",
    "spectral_centroid",
    "wavelet_packet_energy",
    "stft",
]
for key in checks:
    cols = [c for c in df.columns if key in c]
    print(key, len(cols))
PY
```

每一项数量都应大于 0。

### 9.4 查验可视化图像

```bash
find figures/bearing -type f | sort
```

应看到 15 张轴承 HI 图和 1 张特征评价 Top20 图。

### 9.5 重新运行 Phase 2

如需重新生成 Phase 2 结果，可运行：

```bash
python3 - <<'PY'
from pathlib import Path
from bearing.feature_engineering import process_xjtu_sy

process_xjtu_sy(
    Path("/Users/nightye/Desktop/数模/XJTU-SY/XJTU-SY_Bearing_Datasets"),
    Path("/Users/nightye/Desktop/PHM/results"),
)
PY
```

## 10. 自查结论

对照你的 Phase 2 任务要求，本阶段已经完成：

- XJTU-SY 数据读取。
- 全部指定时域特征。
- 全部指定频域特征。
- Wavelet Packet Energy 和 STFT 时频特征。
- 面向单调性、稳定性、可迁移性的特征评价。
- Bearing HI 构建。
- 三阶段退化标签划分。
- 可视化图像。
- 可供迁移学习使用的 latent features。

结论：你的 Phase 2 轴承退化表征系统已经完成，可以进入后续跨模态共享退化表征学习阶段。
