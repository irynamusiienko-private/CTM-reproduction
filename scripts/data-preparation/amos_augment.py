#!/usr/bin/env python
"""
CutMix Augmentation Script for AMOS Dataset

This script generates augmented data using the CutMix method for medical image segmentation.
It takes images from the AMOS dataset and creates mixed versions by cutting and pasting
patches between different images.

Usage:
    python cutmix_amos.py --data_dir amos_subset --output_dir amos_cutmix_x10 --num_samples 200
"""

import os
import random
import numpy as np
import nibabel as nib
from pathlib import Path
import shutil
import argparse
from tqdm import tqdm

def generate_cutmix_samples(data_dir, output_dir, num_samples=200, seed=42):
    """
    Generate CutMix augmented samples from the AMOS dataset.
    
    Args:
        data_dir (str): Path to the input AMOS dataset directory
        output_dir (str): Path to save the augmented dataset
        num_samples (int): Number of augmented samples to generate
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for the output
    os.makedirs(output_dir / 'imagesTr', exist_ok=True)
    os.makedirs(output_dir / 'labelsTr', exist_ok=True)
    
    # Copy validation and test directories if they exist
    for subdir in ['imagesVa', 'labelsVa', 'imagesTs', 'labelsTs']:
        if (data_dir / subdir).exists():
            if not (output_dir / subdir).exists():
                shutil.copytree(data_dir / subdir, output_dir / subdir)
    
    # Copy dataset.json if it exists
    if (data_dir / 'dataset.json').exists():
        shutil.copy(data_dir / 'dataset.json', output_dir / 'dataset.json')
    
    # Find all training images and labels
    image_files = sorted(list((data_dir / 'imagesTr').glob('*.nii.gz')))
    
    print(f'Found {len(image_files)} training images')
    
    if len(image_files) == 0:
        print(f'No images found in {data_dir}/imagesTr')
        return
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Select two random images
        img1_file = random.choice(image_files)
        img2_file = random.choice(image_files)
        
        # Find corresponding label files (use same name in labelsTr directory)
        lbl1_file = data_dir / 'labelsTr' / img1_file.name
        lbl2_file = data_dir / 'labelsTr' / img2_file.name
        
        # Load images and labels
        img1 = nib.load(img1_file)
        img2 = nib.load(img2_file)
        lbl1 = nib.load(lbl1_file)
        lbl2 = nib.load(lbl2_file)
        
        # Get data arrays
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()
        lbl1_data = lbl1.get_fdata()
        lbl2_data = lbl2.get_fdata()
        
        # If sizes don't match, resize second image to match first
        if img1_data.shape != img2_data.shape:
            # Simple center crop
            z_start = max(0, (img2_data.shape[0] - img1_data.shape[0]) // 2)
            y_start = max(0, (img2_data.shape[1] - img1_data.shape[1]) // 2)
            x_start = max(0, (img2_data.shape[2] - img1_data.shape[2]) // 2)
            
            z_end = min(img2_data.shape[0], z_start + img1_data.shape[0])
            y_end = min(img2_data.shape[1], y_start + img1_data.shape[1])
            x_end = min(img2_data.shape[2], x_start + img1_data.shape[2])
            
            img2_data = img2_data[z_start:z_end, y_start:y_end, x_start:x_end]
            lbl2_data = lbl2_data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # If still not matching, pad or crop to match exactly
            if img1_data.shape != img2_data.shape:
                img2_data = np.pad(img2_data, 
                                  [(0, max(0, img1_data.shape[0] - img2_data.shape[0])),
                                   (0, max(0, img1_data.shape[1] - img2_data.shape[1])),
                                   (0, max(0, img1_data.shape[2] - img2_data.shape[2]))],
                                  mode='constant')
                lbl2_data = np.pad(lbl2_data,
                                  [(0, max(0, img1_data.shape[0] - lbl2_data.shape[0])),
                                   (0, max(0, img1_data.shape[1] - lbl2_data.shape[1])),
                                   (0, max(0, img1_data.shape[2] - lbl2_data.shape[2]))],
                                  mode='constant')
                
                # If needed, crop
                img2_data = img2_data[:img1_data.shape[0], :img1_data.shape[1], :img1_data.shape[2]]
                lbl2_data = lbl2_data[:img1_data.shape[0], :img1_data.shape[1], :img1_data.shape[2]]
        
        # Generate random cutmix parameters
        # Sample beta parameter from beta distribution (as in the CutMix paper)
        lam = np.random.beta(1.0, 1.0)  # Beta parameter
        
        # Calculate patch size based on lambda
        patch_size = [
            int(img1_data.shape[0] * np.sqrt(1 - lam)),
            int(img1_data.shape[1] * np.sqrt(1 - lam)),
            int(img1_data.shape[2] * np.sqrt(1 - lam))
        ]
        
        # Ensure minimum patch size
        patch_size = [max(p, 10) for p in patch_size]
        
        # Random position for the patch
        z_pos = random.randint(0, img1_data.shape[0] - patch_size[0])
        y_pos = random.randint(0, img1_data.shape[1] - patch_size[1])
        x_pos = random.randint(0, img1_data.shape[2] - patch_size[2])
        
        # Create mixed images
        mixed_img = img1_data.copy()
        mixed_lbl = lbl1_data.copy()
        
        # Apply the cutmix
        mixed_img[z_pos:z_pos+patch_size[0], 
                 y_pos:y_pos+patch_size[1], 
                 x_pos:x_pos+patch_size[2]] = img2_data[z_pos:z_pos+patch_size[0], 
                                                      y_pos:y_pos+patch_size[1], 
                                                      x_pos:x_pos+patch_size[2]]
        
        mixed_lbl[z_pos:z_pos+patch_size[0], 
                 y_pos:y_pos+patch_size[1], 
                 x_pos:x_pos+patch_size[2]] = lbl2_data[z_pos:z_pos+patch_size[0], 
                                                      y_pos:y_pos+patch_size[1], 
                                                      x_pos:x_pos+patch_size[2]]
        
        # Save outputs with AMOS naming convention
        output_img_path = output_dir / 'imagesTr' / f'amos_{i+1000:04d}.nii.gz'
        output_lbl_path = output_dir / 'labelsTr' / f'amos_{i+1000:04d}.nii.gz'
        
        # Create and save the nifti files
        new_img = nib.Nifti1Image(mixed_img, img1.affine, img1.header)
        nib.save(new_img, output_img_path)
        
        new_lbl = nib.Nifti1Image(mixed_lbl, lbl1.affine, lbl1.header)
        nib.save(new_lbl, output_lbl_path)
    
    print(f'Generated {num_samples} CutMix samples in {output_dir}')


def generate_anatomix_samples(data_dir, output_dir, num_samples=200, seed=42):
    """
    Generate AnatoMix augmented samples from the AMOS dataset.
    
    Args:
        data_dir (str): Path to the input AMOS dataset directory
        output_dir (str): Path to save the augmented dataset
        num_samples (int): Number of augmented samples to generate
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for the output
    os.makedirs(output_dir / 'imagesTr', exist_ok=True)
    os.makedirs(output_dir / 'labelsTr', exist_ok=True)
    
    # Copy validation and test directories if they exist
    for subdir in ['imagesVa', 'labelsVa', 'imagesTs', 'labelsTs']:
        if (data_dir / subdir).exists():
            if not (output_dir / subdir).exists():
                shutil.copytree(data_dir / subdir, output_dir / subdir)
    
    # Copy dataset.json if it exists
    if (data_dir / 'dataset.json').exists():
        shutil.copy(data_dir / 'dataset.json', output_dir / 'dataset.json')
    
    # Find all training images and labels
    image_files = sorted(list((data_dir / 'imagesTr').glob('*.nii.gz')))
    
    print(f'Found {len(image_files)} training images')
    
    if len(image_files) == 0:
        print(f'No images found in {data_dir}/imagesTr')
        return
    
    # Find all unique organ IDs
    organ_ids = set()
    for img_file in image_files:
        lbl_file = data_dir / 'labelsTr' / img_file.name
        if lbl_file.exists():
            lbl_data = nib.load(lbl_file).get_fdata()
            unique_labels = np.unique(lbl_data)
            unique_labels = unique_labels[unique_labels > 0]  # Exclude background
            organ_ids.update(set(unique_labels))
    
    organ_ids = sorted(list(organ_ids))
    print(f'Found {len(organ_ids)} unique organs: {organ_ids}')
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Select a random base image
        base_img_file = random.choice(image_files)
        base_lbl_file = data_dir / 'labelsTr' / base_img_file.name
        
        # Load the base image and label
        base_img = nib.load(base_img_file)
        base_img_data = base_img.get_fdata()
        base_lbl = nib.load(base_lbl_file)
        base_lbl_data = base_lbl.get_fdata()
        
        # Create a new label (start with zeros)
        new_lbl_data = np.zeros_like(base_lbl_data)
        
        # For each organ class
        for organ_id in organ_ids:
            # Decide whether to use the base image's organ or a different one
            if random.random() < 0.5:  # 50% chance to use the base image
                # Copy organ from base image if it exists
                if np.any(base_lbl_data == organ_id):
                    new_lbl_data[base_lbl_data == organ_id] = organ_id
            else:
                # Find another image with this organ
                donor_candidates = []
                for img_file in image_files:
                    if img_file == base_img_file:
                        continue  # Skip the base image
                        
                    lbl_file = data_dir / 'labelsTr' / img_file.name
                    if not lbl_file.exists():
                        continue
                        
                    lbl_data = nib.load(lbl_file).get_fdata()
                    if np.any(lbl_data == organ_id):
                        donor_candidates.append(img_file)
                
                if donor_candidates:
                    # Select a random donor
                    donor_img_file = random.choice(donor_candidates)
                    donor_lbl_file = data_dir / 'labelsTr' / donor_img_file.name
                    donor_lbl = nib.load(donor_lbl_file)
                    donor_lbl_data = donor_lbl.get_fdata()
                    
                    # Extract organ mask
                    organ_mask = donor_lbl_data == organ_id
                    
                    # Skip if no voxels for this organ
                    if not np.any(organ_mask):
                        continue
                    
                    # Handle different dimensions by resizing if necessary
                    if donor_lbl_data.shape != base_lbl_data.shape:
                        # Get coordinates of organ in donor image
                        donor_z, donor_y, donor_x = np.where(organ_mask)
                        
                        # Skip if empty
                        if len(donor_z) == 0:
                            continue
                        
                        # Calculate scaling factors
                        scale_z = base_lbl_data.shape[0] / donor_lbl_data.shape[0]
                        scale_y = base_lbl_data.shape[1] / donor_lbl_data.shape[1]
                        scale_x = base_lbl_data.shape[2] / donor_lbl_data.shape[2]
                        
                        # Scale coordinates to base image dimensions
                        base_z = np.clip((donor_z * scale_z).astype(int), 0, base_lbl_data.shape[0] - 1)
                        base_y = np.clip((donor_y * scale_y).astype(int), 0, base_lbl_data.shape[1] - 1)
                        base_x = np.clip((donor_x * scale_x).astype(int), 0, base_lbl_data.shape[2] - 1)
                        
                        # Apply random shift
                        shift_z = random.randint(-10, 10)
                        shift_y = random.randint(-10, 10)
                        shift_x = random.randint(-10, 10)
                        
                        # Apply shift with boundary checks
                        z_new = np.clip(base_z + shift_z, 0, base_lbl_data.shape[0] - 1)
                        y_new = np.clip(base_y + shift_y, 0, base_lbl_data.shape[1] - 1)
                        x_new = np.clip(base_x + shift_x, 0, base_lbl_data.shape[2] - 1)
                        
                        # Set the organ in the new label data
                        for z, y, x in zip(z_new, y_new, x_new):
                            new_lbl_data[z, y, x] = organ_id
                    else:
                        # Dimensions match, we can use array operations
                        # Apply a random spatial shift
                        shift_z = random.randint(-10, 10)
                        shift_y = random.randint(-10, 10)
                        shift_x = random.randint(-10, 10)
                        
                        # Get coordinates of the organ
                        z_indices, y_indices, x_indices = np.where(organ_mask)
                        
                        # Create a new mask for the shifted organ (with base image dimensions)
                        new_mask = np.zeros_like(base_lbl_data, dtype=bool)
                        
                        # Apply shift with boundary checks
                        for z, y, x in zip(z_indices, y_indices, x_indices):
                            z_new = min(max(0, z + shift_z), base_lbl_data.shape[0] - 1)
                            y_new = min(max(0, y + shift_y), base_lbl_data.shape[1] - 1)
                            x_new = min(max(0, x + shift_x), base_lbl_data.shape[2] - 1)
                            new_mask[z_new, y_new, x_new] = True
                        
                        # Add to the new label data
                        new_lbl_data[new_mask] = organ_id
        
        # Save outputs
        output_img_path = output_dir / 'imagesTr' / f'amos_{i+1000:04d}.nii.gz'
        output_lbl_path = output_dir / 'labelsTr' / f'amos_{i+1000:04d}.nii.gz'
        
        # Save the base image unchanged
        nib.save(base_img, output_img_path)
        
        # Create and save the new label
        new_lbl = nib.Nifti1Image(new_lbl_data, base_lbl.affine, base_lbl.header)
        nib.save(new_lbl, output_lbl_path)
    
    print(f'Generated {num_samples} AnatoMix samples in {output_dir}')


def generate_carvemix_samples(data_dir, output_dir, num_samples=200, seed=42):
    """
    Generate CarveMix augmented samples from the AMOS dataset.
    
    CarveMix carves out organs from one image and transfers them to another image.
    
    Args:
        data_dir (str): Path to the input AMOS dataset directory
        output_dir (str): Path to save the augmented dataset
        num_samples (int): Number of augmented samples to generate
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for the output
    os.makedirs(output_dir / 'imagesTr', exist_ok=True)
    os.makedirs(output_dir / 'labelsTr', exist_ok=True)
    
    # Copy validation and test directories if they exist
    for subdir in ['imagesVa', 'labelsVa', 'imagesTs', 'labelsTs']:
        if (data_dir / subdir).exists():
            if not (output_dir / subdir).exists():
                shutil.copytree(data_dir / subdir, output_dir / subdir)
    
    # Copy dataset.json if it exists
    if (data_dir / 'dataset.json').exists():
        shutil.copy(data_dir / 'dataset.json', output_dir / 'dataset.json')
    
    # Find all training images and labels
    image_files = sorted(list((data_dir / 'imagesTr').glob('*.nii.gz')))
    
    print(f'Found {len(image_files)} training images')
    
    if len(image_files) == 0:
        print(f'No images found in {data_dir}/imagesTr')
        return
    
    # Find all unique organ IDs
    # First load a sample label to identify unique organs
    if len(image_files) > 0:
        sample_label = nib.load(data_dir / 'labelsTr' / image_files[0].name).get_fdata()
        unique_organs = np.unique(sample_label)
        unique_organs = unique_organs[unique_organs > 0]  # Exclude background
    else:
        unique_organs = []
        
    print(f'Found {len(unique_organs)} unique organs: {unique_organs}')
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Select two random images
        img1_file = random.choice(image_files)
        img2_file = random.choice(image_files)
        
        # Find corresponding label files
        lbl1_file = data_dir / 'labelsTr' / img1_file.name
        lbl2_file = data_dir / 'labelsTr' / img2_file.name
        
        # Load images and labels
        img1 = nib.load(img1_file)
        img2 = nib.load(img2_file)
        lbl1 = nib.load(lbl1_file)
        lbl2 = nib.load(lbl2_file)
        
        # Get data arrays
        img1_data = img1.get_fdata()
        img2_data = img2.get_fdata()
        lbl1_data = lbl1.get_fdata()
        lbl2_data = lbl2.get_fdata()
        
        # If sizes don't match, resize second image to match first
        if img1_data.shape != img2_data.shape:
            # Simple center crop
            z_start = max(0, (img2_data.shape[0] - img1_data.shape[0]) // 2)
            y_start = max(0, (img2_data.shape[1] - img1_data.shape[1]) // 2)
            x_start = max(0, (img2_data.shape[2] - img1_data.shape[2]) // 2)
            
            z_end = min(img2_data.shape[0], z_start + img1_data.shape[0])
            y_end = min(img2_data.shape[1], y_start + img1_data.shape[1])
            x_end = min(img2_data.shape[2], x_start + img1_data.shape[2])
            
            img2_data = img2_data[z_start:z_end, y_start:y_end, x_start:x_end]
            lbl2_data = lbl2_data[z_start:z_end, y_start:y_end, x_start:x_end]
            
            # If still not matching, pad or crop to match exactly
            if img1_data.shape != img2_data.shape:
                img2_data = np.pad(img2_data, 
                                  [(0, max(0, img1_data.shape[0] - img2_data.shape[0])),
                                   (0, max(0, img1_data.shape[1] - img2_data.shape[1])),
                                   (0, max(0, img1_data.shape[2] - img2_data.shape[2]))],
                                  mode='constant')
                lbl2_data = np.pad(lbl2_data,
                                  [(0, max(0, img1_data.shape[0] - lbl2_data.shape[0])),
                                   (0, max(0, img1_data.shape[1] - lbl2_data.shape[1])),
                                   (0, max(0, img1_data.shape[2] - lbl2_data.shape[2]))],
                                  mode='constant')
                
                # If needed, crop
                img2_data = img2_data[:img1_data.shape[0], :img1_data.shape[1], :img1_data.shape[2]]
                lbl2_data = lbl2_data[:img1_data.shape[0], :img1_data.shape[1], :img1_data.shape[2]]
        
        # Get organ masks from second image
        organ_masks = {}
        for organ_id in unique_organs:
            if np.any(lbl2_data == organ_id):
                organ_masks[int(organ_id)] = lbl2_data == organ_id
        
        # Create output arrays (start with first image)
        mixed_img = img1_data.copy()
        mixed_lbl = lbl1_data.copy()
        
        # Randomly select organs to transfer (1-3 organs)
        num_organs_to_transfer = min(random.randint(1, 3), len(organ_masks))
        organs_to_transfer = random.sample(list(organ_masks.keys()), num_organs_to_transfer)
        
        # Transfer each selected organ
        for organ_id in organs_to_transfer:
            mask = organ_masks[organ_id]
            
            # Remove any existing instance of this organ in the target
            if np.any(mixed_lbl == organ_id):
                mixed_lbl[mixed_lbl == organ_id] = 0
            
            # Transfer the organ
            mixed_img[mask] = img2_data[mask]
            mixed_lbl[mask] = organ_id
        
        # Save outputs
        output_img_path = output_dir / 'imagesTr' / f'amos_{i+1000:04d}.nii.gz'
        output_lbl_path = output_dir / 'labelsTr' / f'amos_{i+1000:04d}.nii.gz'
        
        # Create and save the nifti files
        new_img = nib.Nifti1Image(mixed_img, img1.affine, img1.header)
        nib.save(new_img, output_img_path)
        
        new_lbl = nib.Nifti1Image(mixed_lbl, lbl1.affine, lbl1.header)
        nib.save(new_lbl, output_lbl_path)
    
    print(f'Generated {num_samples} CarveMix samples in {output_dir}')


def generate_objectaug_samples(data_dir, output_dir, num_samples=200, seed=42):
    """
    Generate ObjectAug augmented samples from the AMOS dataset.
    
    ObjectAug applies random transformations to individual organs.
    
    Args:
        data_dir (str): Path to the input AMOS dataset directory
        output_dir (str): Path to save the augmented dataset
        num_samples (int): Number of augmented samples to generate
        seed (int): Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for the output
    os.makedirs(output_dir / 'imagesTr', exist_ok=True)
    os.makedirs(output_dir / 'labelsTr', exist_ok=True)
    
    # Copy validation and test directories if they exist
    for subdir in ['imagesVa', 'labelsVa', 'imagesTs', 'labelsTs']:
        if (data_dir / subdir).exists():
            if not (output_dir / subdir).exists():
                shutil.copytree(data_dir / subdir, output_dir / subdir)
    
    # Copy dataset.json if it exists
    if (data_dir / 'dataset.json').exists():
        shutil.copy(data_dir / 'dataset.json', output_dir / 'dataset.json')
    
    # Find all training images and labels
    image_files = sorted(list((data_dir / 'imagesTr').glob('*.nii.gz')))
    
    print(f'Found {len(image_files)} training images')
    
    if len(image_files) == 0:
        print(f'No images found in {data_dir}/imagesTr')
        return
    
    # Generate samples
    for i in tqdm(range(num_samples)):
        # Select a random base image
        base_img_file = random.choice(image_files)
        base_lbl_file = data_dir / 'labelsTr' / base_img_file.name
        
        # Load the base image and label
        base_img = nib.load(base_img_file)
        base_img_data = base_img.get_fdata()
        base_lbl = nib.load(base_lbl_file)
        base_lbl_data = base_lbl.get_fdata()
        
        # Extract unique organ IDs
        unique_organs = np.unique(base_lbl_data)
        unique_organs = unique_organs[unique_organs > 0]  # Exclude background
        
        # Make copies for augmentation
        aug_img_data = base_img_data.copy()
        aug_lbl_data = base_lbl_data.copy()
        
        # Augment each organ with 50% probability
        for organ_id in unique_organs:
            if random.random() < 0.5:
                continue  # Skip this organ
                
            # Get organ mask
            organ_mask = base_lbl_data == organ_id
            
            # Skip if no voxels for this organ
            if not np.any(organ_mask):
                continue
                
            # Get organ bounding box
            z_indices, y_indices, x_indices = np.where(organ_mask)
            
            # Random shift
            shift_z = random.randint(-5, 5)
            shift_y = random.randint(-5, 5)
            shift_x = random.randint(-5, 5)
            
            # Create new mask for shifted organ
            new_mask = np.zeros_like(organ_mask)
            
            # Apply the shift
            z_new = np.clip(z_indices + shift_z, 0, new_mask.shape[0] - 1)
            y_new = np.clip(y_indices + shift_y, 0, new_mask.shape[1] - 1)
            x_new = np.clip(x_indices + shift_x, 0, new_mask.shape[2] - 1)
            
            # Set values in new mask
            for z, y, x in zip(z_new, y_new, x_new):
                new_mask[z, y, x] = True
            
            # Extract organ intensity values
            organ_intensities = base_img_data[organ_mask]
            
            # Update images
            aug_img_data[organ_mask] = 0  # Remove original organ
            aug_lbl_data[organ_mask] = 0  # Remove original organ
            
            # Place organ in new position
            for idx, (z, y, x) in enumerate(zip(z_new, y_new, x_new)):
                if idx < len(organ_intensities):
                    aug_img_data[z, y, x] = organ_intensities[idx]
                    aug_lbl_data[z, y, x] = organ_id
        
        # Save outputs
        output_img_path = output_dir / 'imagesTr' / f'amos_{i+1000:04d}.nii.gz'
        output_lbl_path = output_dir / 'labelsTr' / f'amos_{i+1000:04d}.nii.gz'
        
        # Create and save the nifti files
        new_img = nib.Nifti1Image(aug_img_data, base_img.affine, base_img.header)
        nib.save(new_img, output_img_path)
        
        new_lbl = nib.Nifti1Image(aug_lbl_data, base_lbl.affine, base_lbl.header)
        nib.save(new_lbl, output_lbl_path)
    
    print(f'Generated {num_samples} ObjectAug samples in {output_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate augmented data for AMOS dataset')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input AMOS dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the augmented dataset')
    parser.add_argument('--method', type=str, required=True, choices=['cutmix', 'anatomix', 'carvemix', 'objectaug'],
                        help='Augmentation method to use')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of augmented samples to generate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.method == 'cutmix':
        generate_cutmix_samples(args.data_dir, args.output_dir, args.num_samples, args.seed)
    elif args.method == 'anatomix':
        generate_anatomix_samples(args.data_dir, args.output_dir, args.num_samples, args.seed)
    elif args.method == 'carvemix':
        generate_carvemix_samples(args.data_dir, args.output_dir, args.num_samples, args.seed)
    elif args.method == 'objectaug':
        generate_objectaug_samples(args.data_dir, args.output_dir, args.num_samples, args.seed)


if __name__ == '__main__':
    main()