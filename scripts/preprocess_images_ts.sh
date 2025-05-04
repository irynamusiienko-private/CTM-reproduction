#!/bin/bash
#SBATCH --job-name=preprocess_test
#SBATCH --output=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/preprocess_test_%j.out
#SBATCH --error=/home/woody/mrrr/mrrr108v/tda-reproduction/logs/preprocess_test_%j.err
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00

source /etc/profile
module purge
module load python/3.10-anaconda

source activate tda-env || { echo "Failed to activate conda env"; exit 1; }

export nnUNet_raw=/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_raw
export nnUNet_preprocessed=/home/woody/mrrr/mrrr108v/tda-reproduction/data/nnUNet_preprocessed
export nnUNet_results=/home/woody/mrrr/mrrr108v/tda-reproduction/outputs

echo "Starting preprocessing of test set..."
which nnUNetv2_preprocess || { echo "nnUNetv2_preprocess not found"; exit 1; }

nnUNetv2_preprocess \
  -d 1 \
  -o $nnUNet_preprocessed \
  -c 3d_fullres \
  --verify_dataset_integrity || { echo "Preprocessing failed"; exit 1; }

echo "Preprocessing finished"
