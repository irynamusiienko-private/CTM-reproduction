This repository contains code and resources used to **partially reproduce** the experimental results on the AMOS dataset from the paper:

**_Cut to the Mix: Simple Data Augmentation Outperforms Elaborate Ones in Limited Organ Segmentation Datasets_**  
by *Liu, Chang · Fan, Fuxin · Schwarz, Annette · Maier, Andreas*

---

## Project Overview

This project was executed across the following environments:
- Local machine: initial setup and lightweight preprocessing
- Google Colab: memory-intensive (x25) CutMix augmentation
- NHR@FAU Cluster: model training, inference, and evaluation

## Repository Structure
```plaintext
.
├── environments/        # Environment setup (requirements.txt)
├── scripts/             # All scripts: preprocessing, training, SLURM jobs, inference, evaluation
├── outputs/             # Outputs from training and inference: checkpoints, logs, predictions. Note: large model weights (.pth files) are excluded due to GitHub size limits
└── README.md
