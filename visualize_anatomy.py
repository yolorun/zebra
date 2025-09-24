import tensorstore as ts
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
ANATOMY_DATA_PATH = 'file:///home/v/v/zebra/data/anatomy_clahe_ds'
OUTPUT_DIR = '/home/v/v/zebra/analysis_results'

# --- 1. Load Data ---
def load_volume(path):
    """Loads the 3D anatomy volume from the given path."""
    print(f"Attempting to load volume from: {path}")
    try:
        ds_anatomy = ts.open({
            'open': True,
            'driver': 'zarr3',
            'kvstore': path
        }).result()
        print(f"Successfully opened data store. Schema: {ds_anatomy.schema}")
        return ds_anatomy
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 2. Generate and Save Slice Plots ---
def plot_slices(volume_store, output_dir):
    """Extracts and plots slices from the three central planes of the volume."""
    if volume_store is None: return
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}")

    shape = volume_store.shape
    print(f"Volume shape: {shape}")

    # Define the slices to extract (from the middle of each axis)
    # Assuming shape is (Z, Y, X)
    z_slice_idx, y_slice_idx, x_slice_idx = np.array(shape) // 2

    # Extract slices
    axial_slice = volume_store[:, :, x_slice_idx].read().result()
    coronal_slice = volume_store[:, y_slice_idx, :].read().result()
    sagittal_slice = volume_store[z_slice_idx, :, :].read().result()

    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
    fig.suptitle('Central Slices of Anatomy Volume', fontsize=16, color='white')

    # Plot Axial Slice (XY plane)
    axes[0].imshow(axial_slice.T, cmap='gray', origin='lower')
    axes[0].set_title(f'Axial (XY) Slice at X={x_slice_idx}', color='white')
    axes[0].set_xlabel('Y-axis')
    axes[0].set_ylabel('Z-axis')
    axes[0].tick_params(colors='white')

    # Plot Coronal Slice (XZ plane)
    axes[1].imshow(coronal_slice.T, cmap='gray', origin='lower')
    axes[1].set_title(f'Coronal (XZ) Slice at Y={y_slice_idx}', color='white')
    axes[1].set_xlabel('X-axis')
    axes[1].set_ylabel('Z-axis')
    axes[1].tick_params(colors='white')

    # Plot Sagittal Slice (YZ plane)
    axes[2].imshow(sagittal_slice, cmap='gray')
    axes[2].set_title(f'Sagittal (YZ) Slice at Z={z_slice_idx}', color='white')
    axes[2].set_xlabel('X-axis')
    axes[2].set_ylabel('Y-axis')
    axes[2].tick_params(colors='white')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    save_path = os.path.join(output_dir, 'anatomy_slices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close()
    print(f"- Saved slice plot to {save_path}")

# --- Main Execution ---
if __name__ == '__main__':
    anatomy_volume = load_volume(ANATOMY_DATA_PATH)
    if anatomy_volume is not None:
        plot_slices(anatomy_volume, OUTPUT_DIR)
        print("\nVisualization script complete.")
