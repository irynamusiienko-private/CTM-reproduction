#!/bin/bash
#SBATCH --job-name=infer_cutmix1
#SBATCH --output=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/infer_cutmix1_%j.out
#SBATCH --error=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/infer_cutmix1_%j.err
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

echo "=== Inference job started at $(date) ==="

source /etc/profile
module purge
module load python/3.10-anaconda
source activate /home/woody/mrrr/mrrr108v/software/private/conda/envs/tda-env

INPUT_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_raw/Dataset001_AMOSOriginal/imagesTs_100"
OUTPUT_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/outputs/infer_cutmix1"
MODEL_DIR="/home/woody/mrrr/mrrr108v/tda-reproduction/outputs/cutmix1/Dataset001_AMOSOriginal/nnUNetTrainer__nnUNetPlans__3d_fullres"

mkdir -p "$OUTPUT_DIR"

echo "Checking GPU status..."
nvidia-smi || echo "GPU failed"

echo "Running nnUNetv2_predict with model from: $MODEL_DIR"
nnUNetv2_predict \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  -f 0 \
  -chk checkpoint_best.pth \
  -tr nnUNetTrainer \
  -p nnUNetPlans \
  -c 3d_fullres \
  --verbose \
  "$MODEL_DIR"

echo "Inference completed at $(date)"
