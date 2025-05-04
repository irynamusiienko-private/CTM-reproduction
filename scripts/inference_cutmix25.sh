#!/bin/bash
#SBATCH --job-name=infer_cutmix25
#SBATCH --output=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/infer_cutmix25_%j.out
#SBATCH --error=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/infer_cutmix25_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

echo "=== Inference job started at $(date) ==="

source /etc/profile
module purge
module load python/3.10-anaconda
source activate /home/woody/mrrr/mrrr108v/software/private/conda/envs/tda-env

INPUT_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_raw/Dataset001_AMOSOriginal/imagesTs"
OUTPUT_ROOT="/home/woody/mrrr/mrrr108v/tda-reproduction/outputs"
TASK="infercutmix25"
OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}"

export nnUNet_results="/home/woody/mrrr/mrrr108v/tda-reproduction/outputs/cutmix25"

mkdir -p "$OUTPUT_DIR"

echo "Launching nnU-Net inference"
nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -d Dataset003_AMOSCutMix25x \
  -f 0 \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -c 3d_fullres \
  --disable_tta

# === Run Inference WITH TTA ===
TTA_OUTPUT_DIR="${OUTPUT_ROOT}/${TASK}_tta"
mkdir -p "$TTA_OUTPUT_DIR"

nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$TTA_OUTPUT_DIR" \
  -d Dataset003_AMOSCutMix25x \
  -f 0 \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -c 3d_fullres
echo "Inference finished at $(date)"
