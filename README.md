# Cross-modal Shared Latent Degradation PHM

This repository implements a research prototype for:

```text
Satellite Reaction Wheel PHM Based on Cross-modal Shared Latent Degradation Representation
```

The codebase keeps one canonical reaction-wheel PHM pipeline, one XJTU-SY bearing feature-engineering branch, and one shared latent degradation prototype.

## Scope

- Reaction-wheel data scanning, EDA, HI construction, stage segmentation, and degradation modeling.
- XJTU-SY bearing feature-engineering interface:
  - time-domain features;
  - frequency-domain features;
  - wavelet time-frequency features;
  - PCA-HI / AutoEncoder-HI fallback;
  - feature quality metrics.
- Cross-modal shared latent degradation prototype:
  - PyTorch Flywheel Encoder;
  - PyTorch Bearing Encoder interface;
  - PCA/t-SNE latent trajectory analysis;
  - latent clustering and stage separation;
  - DANN, CORAL, MMD, and Domain Classifier interfaces;
  - physics-informed monotonicity and continuity constraints.
- Team collaboration package for paper writing and bearing-module development.

## Project Structure

```text
PHM/
├── bearing/              # XJTU-SY bearing feature engineering
├── encoders/             # Flywheel and bearing PyTorch encoders
├── latent_space/         # Shared latent degradation prototype
├── transfer_learning/    # DANN, CORAL, MMD interfaces
├── physics_informed/     # Monotonicity and continuity constraints
├── multimodal/           # Multimodal fusion interface
├── rul_prediction/       # RUL prediction interface
├── figures/              # Generated research figures
├── results/              # Generated CSV/JSON outputs
├── reports/              # Generated Markdown reports
└── Team_Share_Package/   # Collaboration package
```

## Run

Use the Python interpreter available on the local machine. On the current Windows workstation:

```powershell
cd C:\Users\35135\Desktop\PHM
& 'D:\Program Files\Python312\python.exe' main.py
& 'D:\Program Files\Python312\python.exe' run_cross_modal_latent.py
```

`main.py` generates the reaction-wheel baseline and bearing feature schema/results.  
`run_cross_modal_latent.py` generates flywheel latent embeddings, clustering results, and latent-space figures.

## Main Outputs

- `results/phase1_results.json`
- `reports/phase1_report.md`
- `results/bearing_latent_feature_schema.json`
- `results/latent_space/latent_prototype_results.json`
- `results/latent_space/embeddings/*_flywheel_embedding.csv`
- `results/latent_space/clustering/*_latent_clusters.csv`
- `figures/latent_space/`
- `reports/latent_space_report.md`
- `reports/multimodal_framework.md`
- `reports/model_summary.md`
- `WORK_REPORT.md`

## Data Configuration

Edit `configs/default_config.yaml` when the local data paths change:

- `paths.source_data_dir`: reaction-wheel contest data.
- `paths.bearing_data_dir`: future XJTU-SY bearing data.

If the XJTU-SY bearing dataset is not present, the bearing module still emits a schema file describing the expected bearing latent features.
