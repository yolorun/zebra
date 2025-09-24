import tensorstore as ts
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# --- Configuration ---
POSITION_DATA_PATH = 'file:///home/v/v/zebra/data/position_embedding'
OUTPUT_DIR = '/home/v/v/zebra/analysis_results'

# --- 1. Load Data ---
def load_data(path):
    """Loads the position embedding data from the given path."""
    print(f"Attempting to load data from: {path}")
    try:
        ds = ts.open({
            'open': True,
            'driver': 'zarr',
            'kvstore': path
        }).result()
        data = ds.read().result()
        print(f"Successfully loaded data.")
        return data
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 2. Analyze and Print Info ---
def analyze_data(data):
    """Prints basic properties and statistics of the data."""
    if data is None: return
    
    print("\n--- Data Inspection ---")
    print(f"Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")
    
    num_neurons, num_features = data.shape
    print(f"This dataset contains {num_neurons} entries (likely neurons), each with {num_features} features.")

    print("\n--- Sample Data (first 5 rows) ---")
    print(data[:5])

    print("\n--- Statistical Summary ---")
    stats = {
        'Min': np.min(data, axis=0),
        'Max': np.max(data, axis=0),
        'Mean': np.mean(data, axis=0),
        'Std Dev': np.std(data, axis=0)
    }
    
    # Print stats in a readable format
    header = f"{'Feature':>10}" + ''.join([f'{key:>12}' for key in stats.keys()])
    print(header)
    print('-' * len(header))
    for i in range(num_features):
        row = f"{i:<10}" + ''.join([f'{stats[key][i]:>12.3f}' for key in stats.keys()])
        print(row)

# --- 3. Generate Visualization ---
def visualize_positions(data, output_dir):
    """Creates a 3D scatter plot of the neuron positions."""
    if data is None: return
    
    num_neurons, num_features = data.shape
    if num_features < 3:
        print(f"\nSkipping 3D plot: Data has only {num_features} dimensions.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating 3D scatter plot... (this may take a moment for many points)")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use the first 3 dimensions for plotting
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=1, alpha=0.5)
    
    ax.set_title('3D Visualization of Neuron Positions')
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    
    save_path = os.path.join(output_dir, 'position_embedding_3d_scatter.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"- Saved 3D scatter plot to {save_path}")

# --- Main Execution ---
if __name__ == '__main__':
    position_data = load_data(POSITION_DATA_PATH)
    if position_data is not None:
        analyze_data(position_data)
        visualize_positions(position_data, OUTPUT_DIR)
        print("\nAnalysis of position embedding complete.")
