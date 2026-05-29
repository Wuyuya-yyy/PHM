# Shared Config Summary

- equirements.txt: Python dependency list.
- configs/default_config.yaml: canonical project paths, random seed, DPI, feature candidates, and model parameters.
- source_entrypoints/: current executable pipeline files.
- interface_modules/: reserved transfer learning, shared latent space, multimodal fusion, and RUL interfaces.

Current important parameters:

- random seed: 42
- figure DPI: 320
- LSTM sequence length: 12
- max LSTM epochs: 120
- train ratio: 0.75
- anomaly z-threshold: 3.0
