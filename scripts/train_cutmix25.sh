#!/bin/bash
#SBATCH --job-name=train_cutmix25
#SBATCH --output=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/train_cutmix25_%j.out
#SBATCH --error=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/train_cutmix25_%j.err
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=a100
#SBATCH --time=24:00:00

set -e
set -x

source /etc/profile || { echo "Failed to source /etc/profile"; exit 1; }

mkdir -p /home/woody/mrrr/mrrr108v/tda-reproduction/logs

module load python/3.10-anaconda || { echo "Module load failed"; exit 1; }

eval "$(conda shell.bash hook)" || { echo "Conda shell hook failed"; exit 1; }

echo "Activating conda environment..."
conda activate /home/woody/mrrr/mrrr108v/software/private/conda/envs/tda-env || { echo "Conda activate failed"; exit 1; }

# Diagnostics
echo "Using Python at: $(which python)"
python --version || { echo "Python version check failed"; exit 1; }

echo "Checking CUDA availability..."
python -c "import torch; print('CUDA available?', torch.cuda.is_available())" || { echo "Torch.cuda check failed"; exit 1; }

echo "Running nvidia-smi..."
nvidia-smi || echo "Could not run nvidia-smi"
export nnUNet_raw=/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_raw
export nnUNet_preprocessed=/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_preprocessed
export nnUNet_results=/home/woody/mrrr/mrrr108v/tda-reproduction/outputs

# Launch training
echo "Launching nnUNet training"
python /home/woody/mrrr/mrrr108v/tda-reproduction/scripts/train.py \
  --dataset_id 3 \
  --fold 0 \
  --config 3d_fullres \
  --device cuda \
  --preprocessed_dir /home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_preprocessed \
  --results_dir /home/woody/mrrr/mrrr108v/tda-reproduction/outputs/cutmix25 || { echo "Training script failed"; exit 1; }

echo "Training completed at $(date)"

