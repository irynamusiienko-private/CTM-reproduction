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
```

## Reproduction Steps

---

### 1. Repository & Environment Setup
- Install dependencies listed in `environments/requirements.txt`.

---

### 2. Dataset Preparation
- Download the [AMOS dataset](https://amos22.grand-challenge.org/).
- Format and augment the dataset using:

  - `scripts/amos_augment.py` - Applies CutMix ×10 / ×25 augmentation  
  - `scripts/amos_prepare_nnunet_dataset.py` - Converts data to nnU-Net format
  - After running the above, execute nnU-Net preprocessing:
```bash
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres
nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres
nnUNetv2_plan_and_preprocess -d 3 -c 3d_fullres
```
---

### 3. Model Training
Training was performed using SLURM job scripts:

- `scripts/train_cutmix1.sh` – for original (baseline)
- `scripts/train_cutmix10.sh` – for CutMix ×10
- `scripts/train_cutmix25.sh` – for CutMix ×25
- `scripts/train.py` – Python script for model training (used by the shell scripts)

---

### 4. Inference
Run inference on each trained model (±TTA):

- `scripts/inference_cutmix1.sh`
- `scripts/inference_cutmix10.sh`
- `scripts/inference_cutmix25.sh`

---

### 5. Evaluation
Compute Dice scores (micro/macro) and TTA impact:

- `scripts/eval_cutmix1.sh`
- `scripts/eval_cutmix10.sh`
- `scripts/eval_cutmix25.sh`
- `scripts/eval_cutmix1_tta.sh`
- `scripts/eval_cutmix10_tta.sh`
- `scripts/eval_cutmix25_tta.sh`

---

### 6. Results
- Recreate the results with:

  - `scripts/cutmix_results_table.py` – Compiles all metrics

---
