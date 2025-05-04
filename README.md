This repository contains code and resources used to partially reproduce the experiments results on the AMOS dataset from the paper: 
**Cut to the Mix: Simple Data Augmentation Outperforms Elaborate Ones in Limited Organ Segmentation Datasets**
by Liu, Chang | Fan, Fuxin | Schwarz, Annette | Maier, Andreas | 

## Project Overview

This project was executed across three environments:
- Local machine: initial setup and lightweight preprocessing
- Google Colab: memory-intensive (x25) CutMix augmentation
- NHR@FAU Cluster: model training, inference, and evaluation

The original repository lacked some key reproduction components and structure. Several additional scripts were created or modified to enable full reproducibility.

## Repository Structure
.
├── environments/        # Environment setup (requirements.txt)
├── scripts/             # All scripts: preprocessing, training, SLURM jobs, inference, evaluation
├── outputs/             # Outputs from training and inference: checkpoints, logs, predictions
└── README.md            