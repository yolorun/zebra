#!/usr/bin/env python

"""
Visualize 3D TIFF files with napari, particularly suited for Cellpose segmentation results.
Allows visualization of both raw images and segmentation masks.

Dependencies:
- napari
- numpy
- tifffile
- PyQt5 (for the GUI)

Install with:
pip install napari[all] numpy tifffile PyQt5
"""

import argparse
import os
import numpy as np
import tifffile
import napari


def visualize_tiff(image_path=None, mask_path=None):
    """
    Visualize 3D TIFF files with napari.
    
    Parameters:
    -----------
    image_path : str, optional
        Path to the original image TIFF file
    mask_path : str, optional
        Path to the segmentation mask TIFF file
    """
    # Start napari viewer
    viewer = napari.Viewer()
    
    # Load and add image if provided
    if image_path and os.path.exists(image_path):
        print(f"Loading image: {image_path}")
        image = tifffile.imread(image_path)
        print(f"Image shape: {image.shape}")
        
        # Add image with appropriate settings
        viewer.add_image(
            image,
            name='Original Image',
            colormap='gray',
            contrast_limits=[image.min(), image.max() * 0.5]  # Adjust contrast for better visibility
        )
    
    # Load and add mask if provided
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask: {mask_path}")
        mask = tifffile.imread(mask_path)
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {np.unique(mask)}")
        print(f"Number of objects: {len(np.unique(mask)) - 1}")  # Subtract 1 for background
        
        # Add masks as labels
        viewer.add_labels(
            mask,
            name='Segmentation Masks',
            opacity=0.7,
            blending='additive'
        )
    
    # If neither image nor mask provided
    if not image_path and not mask_path:
        print("Error: At least one of --image or --mask must be provided.")
        return
    
    # Start the napari event loop
    napari.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D TIFF files with napari.')
    parser.add_argument('--image', type=str, help='Path to the original image TIFF file')
    parser.add_argument('--mask', type=str, help='Path to the segmentation mask TIFF file')
    
    args = parser.parse_args()
    
    # Default paths if not provided
    default_output_dir = 'exp/cellpose/out'
    default_image = 'assets/default_stack_channel0.tiff'
    default_mask = os.path.join(default_output_dir, 'default_stack_channel0_cp_masks.tif')
    
    # Use provided paths or defaults
    image_path = args.image if args.image else default_image
    mask_path = args.mask if args.mask else default_mask
    
    # Print installation instructions
    print("==== Napari Installation ====")
    print("If napari is not installed, install it with:")
    print("pip install napari[all] numpy tifffile PyQt5")
    print("=============================")
    
    # Run visualization
    visualize_tiff(image_path, mask_path)