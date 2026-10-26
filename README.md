# Divide et Calibra: Multiclass Local Calibration via Vector Quantization

[![Paper](https://img.shields.io/badge/arXiv-2605.21060-b31b1b.svg)](https://arxiv.org/abs/2605.21060) ![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4169E1)

Official research code for **Divide et Calibra: Multiclass Local Calibration via Vector Quantization** (NeurIPS 2026).

**Authors:** Cesare Barbera, Lorenzo Perini, Giovanni De Toni, Andrea Passerini, and Andrea Pugnana.

**[Paper](https://arxiv.org/abs/2605.21060)** · **[Repository](https://github.com/Cesbar99/Divide-et-Calibra)**

## Overview

A classifier can be well calibrated *on average* while remaining systematically overconfident or underconfident in particular parts of its representation space. Global post-hoc calibration applies one correction everywhere; independently fitting a correction for every small region, however, quickly becomes data inefficient.

**Divide et Calibra** tackles multiclass *local calibration* using a compositional construction:

1. **Partition the representation space.** Vector quantization (VQ) assigns parts of a pretrained model's latent representation to discrete codewords, inducing local regions.
2. **Share statistical strength across regions.** Rather than learn a completely independent calibration map for each region, the method constructs region-dependent parameters from reusable, codeword-indexed factors.
3. **Calibrate multiclass probabilities.** These factors parameterize Dirichlet-style post-hoc corrections that can adapt to heterogeneous local miscalibration.

The goal is to improve **local calibration**—including in relatively sparse regions—while retaining competitive global calibration and predictive performance. See the paper for the method, assumptions, theoretical analysis, and empirical results.

## Repository structure

```text
Divide-et-Calibra/
├── run.py                          # Main experiment entry point
├── environment.yaml                # Conda environment specification
├── src/
│   ├── actions/                    # Pretraining, calibration, quantization, testing
│   ├── algorithms/                 # Base models and training routines
│   ├── calibrator/                 # VQ and calibration implementations
│   │   ├── VQDirCal.py
│   │   ├── VQCAL_trainer.py
│   │   ├── quantisation_head.py
│   │   └── ...
│   ├── competitors/                # Calibration baselines
│   ├── configs/
│   │   ├── config_local.yaml       # Main experiment configuration
│   │   ├── dataset/                # Dataset-specific configurations
│   │   └── models/                 # Model/stage configurations
│   ├── data_sets/                  # Data loading and preparation
│   ├── hp_opt/                     # Hyperparameter optimization
│   └── utils/                      # Shared utilities
└── README.md
```

## Installation

Clone the repository and create the provided Conda environment:

```bash
git clone https://github.com/Cesbar99/Divide-et-Calibra.git
cd Divide-et-Calibra
conda env create -f environment.yaml
conda activate CAL_env
```

The checked-in environment is a **Linux-oriented, version-pinned research environment** (Python 3.10). Depending on your OS, CUDA installation, and package availability, you may need to adjust the environment specification. GPU execution is configured by default; check `device` and `cuda_device` in `src/configs/config_local.yaml`.

## Before running experiments

> [!IMPORTANT]
> **Pretrained classifiers, extracted predictions, and feature representations are not included in this repository.** The code is not plug-and-play immediately after cloning: you must provide the required model outputs or run the applicable pretraining/data-extraction workflow first.

The experiments use separately stored outputs for the **calibration** and **test** splits, including labels, predictions, logits, and (for representation-based calibration or local calibration metrics) latent features. The original workflow expects experiment artifacts under a results/pretraining directory; inspect the paths referenced by the data loaders and experiment actions, and update them for your machine.

In particular:

- Edit `src/configs/config_local.yaml` for your dataset, model, experiment stages, device, and output paths.
- Check the relevant files in `src/configs/dataset/` and `src/configs/models/`.
- Ensure calibration and test outputs are stored separately, with all required arrays available to the loader.
- Keep `return_features=True` when your experiment requires representation-space local calibration metrics. Without appropriate features, these metrics cannot be computed.

**Note:** Some paths and experimental conventions in the code reflect the authors' original research environment and may need adaptation; the commands below are examples of the supported command-line configuration, **not** a guarantee of reproducibility without the prerequisite artifacts.

## Running experiments

The main entry point is `run.py`. It uses **Hydra/OmegaConf-style command-line overrides**, with configurations in `src/configs/`. The principal experiment modes include `pretrain`, `quantize`, `calibrate`, `competition`, and `test`.

### Example: calibration baselines

The following is adapted from the original repository instructions for evaluating competing post-hoc methods on CIFAR-10 **after the required pretrained classifier outputs have been prepared**:

```bash
CUDA_VISIBLE_DEVICES=0 python -u run.py \
  pretrain=False calibrate=False quantize=False replicate=False test=False \
  competition=True exp_name=competition data=cifar10 \
  gamma=10 dataset.batch_size=128 return_features=True \
  similarity_dim=50 models.max_iter=1000 \
  checkpoint.num_classes=10 checkpoint.epochs=9 \
  models.temp_lr=1e-3 n_bins_calibration_metrics=15
```

The baseline methods configured in `src/configs/config_local.yaml` include **temperature scaling (TS), isotonic regression (IR), Platt scaling (PS), Dirichlet calibration (DC)** and other comparison methods. The available methods can be selected via the `methods` setting.

For the datasets highlighted in the original README, the class-count settings are:

| Dataset key | Number of classes |
| --- | ---: |
| `cifar10` | 10 |
| `cifar100` | 100 |
| `tissue` | 8 |

The code also contains configurations for additional image, tabular, and synthetic datasets under `src/configs/dataset/`. Dataset support may require assets and setup beyond the checked-in repository.

### Run the VQ calibration pipeline

VQ-related experiments are controlled through the `quantize` stage, with code in `src/calibrator/` and `src/actions/quantize.py`. Configure the dataset and model-specific settings in `src/configs/` before enabling `quantize=True` or the multi-stage `run_all` workflow. In particular, the quantization stage iterates over codebook and slot settings (`models.kappas` and `models.slots`), along with configured seeds and dataset subsamples.

Since these stages depend on the chosen pretrained architecture, generated features, model configuration, and saved intermediate artifacts, refer to the stage logic in `run.py` and the corresponding `src/actions/` implementation when reproducing an individual experiment.

## Adding a dataset or base model

To extend the experiments to a new dataset or model:

1. Add a dataset configuration under `src/configs/dataset/` (for example, use `cifar10.yaml` as a reference) and set the appropriate number of classes and class priors where needed.
2. Implement the data loader in `src/data_sets/`.
3. Add or register the architecture and its training routine in `src/algorithms/`.
4. Update `models_map` in `src/configs/config_local.yaml` and the relevant model configuration in `src/configs/models/`.
5. If the model requires a custom training workflow, adapt `src/actions/pretrain.py`.
6. Make sure the calibration/test outputs include the latent features required by any local metrics you intend to report.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{barbera2026divide,
  title     = {Divide et Calibra: Multiclass Local Calibration via Vector Quantization},
  author    = {Barbera, Cesare and Perini, Lorenzo and De Toni, Giovanni and Passerini, Andrea and Pugnana, Andrea},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2605.21060}
}
```

## Questions and feedback

For questions about the implementation or reproduction, please [open a GitHub issue](https://github.com/Cesbar99/Divide-et-Calibra/issues). For methodological details, consult the [paper](https://arxiv.org/abs/2605.21060).
