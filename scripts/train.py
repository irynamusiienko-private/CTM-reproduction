import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run nnUNetv2 training on cluster")
    parser.add_argument("--dataset_id", type=int, required=True, help="Dataset ID (e.g. 1)")
    parser.add_argument("--fold", type=int, default=0, help="Cross-validation fold number")
    parser.add_argument("--config", type=str, default="3d_fullres", help="nnUNet configuration")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--preprocessed_dir", type=str, required=True, help="Path to preprocessed data root")
    parser.add_argument("--results_dir", type=str, required=True, help="Path to output (model checkpoints, logs)")
    return parser.parse_args()

def run_training(args):
    # Set nnUNet environment variables
    os.environ["nnUNet_preprocessed"] = args.preprocessed_dir
    os.environ["nnUNet_results"] = args.results_dir

    print("Environment variables set:")
    print(f"  nnUNet_preprocessed = {args.preprocessed_dir}")
    print(f"  nnUNet_results = {args.results_dir}")
    print("Starting training...")

    cmd = [
        "nnUNetv2_train",
        str(args.dataset_id),
        args.config,
        str(args.fold),
        "-device", args.device
    ]

    print("Running command:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    args = parse_args()
    run_training(args)
