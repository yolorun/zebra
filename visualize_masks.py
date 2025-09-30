#!/usr/bin/env python

"""
Visualize 3D zarr mask files with napari.

Dependencies:
- napari
- tensorstore
- numpy

Install with:
pip install napari[all] tensorstore numpy
"""

import argparse
import os
import numpy as np
import tensorstore as ts
import napari


def visualize_zarr(mask_path=None):
    """
    Visualize 3D zarr files with napari.
    
    Parameters:
    -----------
    mask_path : str, optional
        Path to the segmentation mask zarr file
    """
    # Start napari viewer
    viewer = napari.Viewer()
    
    # Load and add mask if provided
    if mask_path and os.path.exists(mask_path):
        print(f"Loading mask: {mask_path}")
        
        # Create a handle to the local dataset.
        ds_masks = ts.open({
            'open': True,
            'driver': 'zarr3',
            'kvstore': f'file://{mask_path}'
        }).result()

        # Read the entire dataset into a NumPy array
        mask = ds_masks.read().result()

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
    else:
        print(f"Error: Mask path not found: {mask_path}")
        return
    
    # Start the napari event loop
    napari.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize 3D zarr files with napari.')
    parser.add_argument('--mask', type=str, help='Path to the segmentation mask zarr file')
    
    args = parser.parse_args()
    
    # Default paths if not provided
    default_mask = '/home/v/proj/zebra/data/segmentation'
    
    # Use provided paths or defaults
    mask_path = args.mask if args.mask else default_mask
    
    # Print installation instructions
    print("==== Napari Installation ====")
    print("If napari is not installed, install it with:")
    print("pip install napari[all] tensorstore numpy")
    print("=============================")
    
    # Run visualization
    visualize_zarr(mask_path)
