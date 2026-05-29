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

- For_Paper_Writer/: paper-ready figures and Markdown reports.
- For_Bearing_Engineer/: bearing task guide and interface requirements.
- Shared_Theory/: unified theory line for the manuscript.
- Shared_Figures/: deduplicated key figures for team discussion.
- Shared_Results/: metrics, HI CSV files, stage boundaries, and summaries.
- Shared_Config/: requirements, config, source entry points, and reserved interfaces.

## 3. Key Results

- Attachment 1 Bayesian-style stage boundaries: 57, 116.
- Attachment 2 Bayesian-style stage boundaries: 606, 1219.
- Attachment 1 best physical-model R2: 0.9997048213769248.
- Attachment 2 best physical-model R2: 0.9990184252318618.
- LSTM baselines also achieve high R2 and serve as data-driven comparison models.

## 4. Current Innovation Points

- A unified Shared Latent Degradation Space is proposed for heterogeneous PHM signals.
- The flywheel HI pipeline is built as the first domain branch.
- Bearing vibration features can be integrated through the same HI-stage-RUL abstraction.
- Transfer learning and multimodal fusion interfaces are already reserved.

## 5. Role Responsibilities

Paper writer:

- Use For_Paper_Writer/Paper_Figure_Index.md to select figures.
- Use Shared_Theory/Core_Theory.md as the theory backbone.
- Use Shared_Results/results_summary.md for numerical results.

Bearing engineer:

- Start from For_Bearing_Engineer/Bearing_Task_Guide.md.
- Implement XJTU-SY feature extraction with the same HI field convention.
- Connect bearing HI to transfer-learning and latent-space interfaces.

## 6. Next Tasks

1. Decide the main stage segmentation method for the final paper.
2. Add XJTU-SY bearing feature extraction and bearing HI generation.
3. Implement shared latent degradation encoders.
4. Add RUL prediction figures and uncertainty-aware warning results.
