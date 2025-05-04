#!/usr/bin/env python3
"""
AMOS Dataset Preparation for nnUNet

This script prepares the AMOS dataset and its CutMix augmentations for nnUNet training.
It converts the datasets into the format required by nnUNet and creates the necessary 
directory structure and metadata files.

Usage:
    python prepare_nnunet_amos.py --original_dir amos_subset 
                                 --cutmix_x10_dir amos_cutmix_x10 
                                 --cutmix_x25_dir amos_cutmix_x25 
                                 --output_dir nnUNet_raw_data
"""

import os
import shutil
import json
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm

def save_json(filename, data):
    """
    Save data as a JSON file.
    
    Args:
        filename (str): Path to save the JSON file
        data (dict): Data to save
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def prepare_amos_dataset(src_dir, task_id, task_name, nnunet_raw):
    """
    Prepare an AMOS dataset for nnUNet.
    
    Args:
        src_dir (str or Path): Source directory containing the dataset
        task_id (int): Task ID for nnUNet (e.g., 1)
        task_name (str): Task name for nnUNet (e.g., AMOSOriginal)
        nnunet_raw (str or Path): nnUNet raw data base directory
    """
    src_dir = Path(src_dir)
    nnunet_raw = Path(nnunet_raw)
    
    # Create target directory
    target_dir = nnunet_raw / f"Dataset{task_id:03d}_{task_name}"
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(target_dir / "imagesTr", exist_ok=True)
    os.makedirs(target_dir / "labelsTr", exist_ok=True)
    os.makedirs(target_dir / "imagesTs", exist_ok=True)
    
    # Process training data
    print(f"Processing {task_name} training data...")
    train_images = sorted(list((src_dir / "imagesTr").glob("*.nii.gz")))
    train_labels = sorted(list((src_dir / "labelsTr").glob("*.nii.gz")))
    
    for i, img_file in enumerate(tqdm(train_images)):
        # Find corresponding label file
        lbl_file = src_dir / "labelsTr" / img_file.name
        
        if not lbl_file.exists():
            print(f"Warning: Label not found for {img_file}")
            continue
        
        # Get case ID (filename without extension)
        case_id = f"amos_{i:03d}"
        
        # nnUNet expects filenames in format: case_0000.nii.gz for images
        dst_img_file = target_dir / "imagesTr" / f"{case_id}_0000.nii.gz"
        
        # nnUNet expects filenames in format: case.nii.gz for labels
        dst_lbl_file = target_dir / "labelsTr" / f"{case_id}.nii.gz"
        
        # Copy the files
        shutil.copy(img_file, dst_img_file)
        shutil.copy(lbl_file, dst_lbl_file)
    
    # Process validation data as test data for nnUNet
    print(f"Processing {task_name} validation data as test data...")
    val_images = sorted(list((src_dir / "imagesVa").glob("*.nii.gz")))
    
    for i, img_file in enumerate(tqdm(val_images)):
        case_id = f"amos_{i:03d}"
        dst_img_file = target_dir / "imagesTs" / f"{case_id}_0000.nii.gz"
        shutil.copy(img_file, dst_img_file)
    
    # Find all unique label values (organs)
    all_labels = set()
    for lbl_file in train_labels:
        lbl_data = nib.load(lbl_file).get_fdata()
        unique_labels = np.unique(lbl_data)
        all_labels.update([int(l) for l in unique_labels if l > 0])
    
    # Define label dictionary for AMOS - with organ names as keys and label IDs as values
    labels = {"background": 0}
    
    # Map of organ names to label IDs
    organ_map = {
        1: "spleen",
        2: "right kidney",
        3: "left kidney",
        4: "gall bladder",
        5: "esophagus",
        6: "liver",
        7: "stomach",
        8: "arota",
        9: "postcava",
        10: "pancreas",
        11: "right adrenal gland",
        12: "left adrenal gland",
        13: "duodenum",
        14: "bladder",
        15: "prostate/uterus"
    }
    
    # Create the labels dictionary with organ names as keys
    for label_id in sorted(list(all_labels)):
        if label_id in organ_map:
            labels[organ_map[label_id]] = label_id
        else:
            labels[f"organ_{label_id}"] = label_id
    
    # Create dataset.json
    dataset_info = {
        "name": task_name,
        "description": f"AMOS dataset with {task_name} augmentation",
        "reference": "AMOS Challenge",
        "licence": "CC-BY-SA 4.0",
        "channel_names": {
            "0": "CT"
        },
        "labels": labels,
        "numTraining": len(train_images),
        "numTest": len(val_images),
        "file_ending": ".nii.gz"
    }
    
    # Save dataset.json
    save_json(str(target_dir / "dataset.json"), dataset_info)
    
    print(f"Dataset {task_name} prepared at: {target_dir}")
    print(f"Training samples: {len(train_images)}")
    print(f"Test samples: {len(val_images)}")
    print(f"Organs: {len(labels) - 1}")
    print("------------------------------------")


def main():
    """Main function to prepare all datasets for nnUNet."""
    parser = argparse.ArgumentParser(description="Prepare AMOS datasets for nnUNet")
    parser.add_argument("--original_dir", type=str, required=True, help="Path to original AMOS subset")
    parser.add_argument("--cutmix_x10_dir", type=str, required=True, help="Path to CutMix x10 augmented dataset")
    parser.add_argument("--cutmix_x25_dir", type=str, required=True, help="Path to CutMix x25 augmented dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for nnUNet raw data")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare original dataset (Task 1)
    prepare_amos_dataset(
        src_dir=args.original_dir,
        task_id=1,
        task_name="AMOSOriginal",
        nnunet_raw=args.output_dir
    )
    
    # Prepare CutMix x10 dataset (Task 2)
    prepare_amos_dataset(
        src_dir=args.cutmix_x10_dir,
        task_id=2,
        task_name="AMOSCutMix10x",
        nnunet_raw=args.output_dir
    )
    
    # Prepare CutMix x25 dataset (Task 3)
    prepare_amos_dataset(
        src_dir=args.cutmix_x25_dir,
        task_id=3,
        task_name="AMOSCutMix25x",
        nnunet_raw=args.output_dir
    )
    
    print("All datasets prepared successfully!")
    print(f"Now you can run nnUNet preprocessing with:")
    print(f"nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres")
    print(f"nnUNetv2_plan_and_preprocess -d 2 -c 3d_fullres")
    print(f"nnUNetv2_plan_and_preprocess -d 3 -c 3d_fullres")


if __name__ == "__main__":
    main()