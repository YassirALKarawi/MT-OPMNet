# MT-OPMNet

**Attention-Enhanced Multi-Task Deep Learning for Joint OSNR Estimation and Modulation Format Recognition in Elastic Optical Networks**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

MT-OPMNet is a multi-task deep learning framework for **Optical Performance Monitoring (OPM)** in elastic optical networks. It jointly performs:

- **OSNR Estimation** — accurate regression of Optical Signal-to-Noise Ratio from amplitude histograms.
- **Modulation Format Identification (MFI)** — classification of modulation formats (OOK, DPSK, DQPSK, 8QAM, 16QAM).

The architecture leverages a **shared 1-D CNN backbone** with task-specific heads, enhanced by a **Channel-Aware Attention Module (CAAM)** and trained with **homoscedastic uncertainty weighting** for automatic task balancing.

<p align="center">
  <img src="figures/architecture.png" alt="MT-OPMNet Architecture" width="700">
</p>

## Key Features

| Feature | Description |
|---|---|
| **Multi-Task Learning** | Shared feature extraction with OSNR regression and MFI classification heads |
| **CAAM** | Channel-Aware Attention Module for adaptive feature recalibration |
| **Uncertainty Weighting** | Learnable homoscedastic uncertainty parameters for automatic loss balancing |
| **Focal Loss** | Addresses class imbalance in modulation format classification |
| **AAH Input** | Amplitude Histogram representation of optical signals as network input |
| **Cosine Annealing** | Learning rate scheduling with warm restarts for stable convergence |

## Architecture

```
Input (AAH: 1 x n_bins)
        │
   ┌────▼────┐
   │  Shared  │   Conv1D blocks (3 layers)
   │ Backbone │   BatchNorm + ReLU + Dropout
   └────┬────┘
        │
   ┌────▼────┐
   │  CAAM   │   Channel-Aware Attention Module
   │ Module  │   GlobalAvgPool → FC → ReLU → FC → Sigmoid
   └────┬────┘
        │
   ┌────▼────┐
   │ Global  │
   │ AvgPool │
   └────┬────┘
        │
   ┌────┴────┐
   │         │
┌──▼──┐  ┌──▼──┐
│OSNR │  │ MFI │
│Head │  │Head │
│(Reg)│  │(Cls)│
└─────┘  └─────┘
```

## Project Structure

```
MT-OPMNet/
├── configs/
│   └── default.json          # Training and model hyperparameters
├── src/
│   ├── __init__.py
│   ├── model.py              # MT-OPMNet architecture with CAAM
│   ├── dataset.py            # AAH dataset generation and loading
│   ├── losses.py             # Focal loss and uncertainty-weighted loss
│   ├── trainer.py            # Training loop with early stopping
│   ├── evaluate.py           # Evaluation metrics and reporting
│   └── utils.py              # Configuration loading and helpers
├── figures/                  # Generated plots and architecture diagrams
├── results/                  # Training results and metrics
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── setup.py                  # Package installation
├── CITATION.cff              # Citation metadata
├── CONTRIBUTING.md           # Contribution guidelines
├── LICENSE                   # MIT License
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/YassirALKarawi/MT-OPMNet.git
cd MT-OPMNet

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

### Training

```bash
# Train with default configuration
python main.py --mode train

# Train with custom config
python main.py --mode train --config configs/default.json

# Fast training mode (fewer epochs, for testing)
python main.py --mode train --fast
```

### Evaluation

```bash
# Evaluate a trained model
python main.py --mode eval --checkpoint results/best_model.pt
```

### Full Pipeline (train + evaluate)

```bash
python main.py --mode full
```

## Configuration

All hyperparameters are defined in `configs/default.json`:

```json
{
    "dataset": {
        "n_symbols": 16384,
        "n_bins": 100,
        "n_realisations": 5,
        "seed": 42,
        "train_ratio": 0.70,
        "val_ratio": 0.15
    },
    "model": {
        "n_classes": 5,
        "use_caam": true
    },
    "training": {
        "batch_size": 128,
        "max_epochs": 200,
        "patience": 20,
        "lr": 1e-3,
        "lr_min": 1e-5,
        "focal_gamma": 2.0,
        "use_uncertainty_weighting": true
    }
}
```

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `n_bins` | Number of amplitude histogram bins | 100 |
| `n_classes` | Number of modulation formats | 5 |
| `use_caam` | Enable Channel-Aware Attention Module | `true` |
| `focal_gamma` | Focal loss focusing parameter | 2.0 |
| `use_uncertainty_weighting` | Automatic multi-task loss balancing | `true` |
| `patience` | Early stopping patience (epochs) | 20 |

## Modulation Formats

| Index | Format | Description |
|---|---|---|
| 0 | OOK | On-Off Keying |
| 1 | DPSK | Differential Phase-Shift Keying |
| 2 | DQPSK | Differential Quadrature Phase-Shift Keying |
| 3 | 8QAM | 8-Quadrature Amplitude Modulation |
| 4 | 16QAM | 16-Quadrature Amplitude Modulation |

## Results

The model achieves competitive performance on joint OPM tasks:

| Metric | Value |
|---|---|
| **OSNR MAE** | < 0.5 dB |
| **OSNR RMSE** | < 0.7 dB |
| **MFI Accuracy** | > 99% |
| **MFI F1-Score** | > 0.99 |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{alkarawi2025mtopmnet,
  title     = {MT-OPMNet: Attention-Enhanced Multi-Task Deep Learning for Joint
               OSNR Estimation and Modulation Format Recognition in Elastic
               Optical Networks},
  author    = {Al-Karawi, Yassir Ameen Ahmed and Alhumaima, Raad S. and
               Al-Raweshidy, Hamed},
  journal   = {IEEE Access},
  year      = {2025}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Authors

- **Yassir Ameen Ahmed Al-Karawi** — University of Diyala, Iraq
- **Raad S. Alhumaima** — University of Diyala / Al-Imam Al-Sadiq University, Iraq
- **Hamed Al-Raweshidy** — Brunel University London, United Kingdom
