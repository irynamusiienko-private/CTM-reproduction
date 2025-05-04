#!/bin/bash
#SBATCH --job-name=eval_cutmix10
#SBATCH --output=eval_cutmix10_%j.out
#SBATCH --error=eval_cutmix10_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

source /etc/profile
module purge
module load python/3.10-anaconda
source activate /home/woody/mrrr/mrrr108v/software/private/conda/envs/tda-env

# === Paths ===
REF_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_raw/Dataset001_AMOSOriginal/labelsTs"
PRED_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/outputs/infercutmix10_tta"
DJ_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_preprocessed/Dataset001_AMOSOriginal/dataset.json"
P_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_preprocessed/Dataset001_AMOSOriginal/nnUNetPlans.json"

# === Evaluation ===
echo "Running nnUNetv2 evaluation"
nnUNetv2_evaluate_folder \
  "$REF_DIR" \
  "$PRED_DIR" \
  -djfile "$DJ_DIR" \
  -pfile "$P_DIR"



