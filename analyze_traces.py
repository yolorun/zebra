import tensorstore as ts
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
LOCAL_TRACES_PATH = 'file:///home/v/v/zebra/data/traces_zip/traces'
OUTPUT_DIR = '/home/v/v/zebra/analysis_results'

# --- 1. Load Data ---
def load_data(path):
    """Loads the traces data from the given path."""
    print(f"Attempting to load data from: {path}")
    try:
        ds_traces = ts.open({
            'open': True,
            'driver': 'zarr3',
            'kvstore': path
        }).result()
        traces = ds_traces.read().result()
        print(f"Successfully loaded data. Shape: {traces.shape}")
        return traces
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None

# --- 2. Calculate Statistics ---
def calculate_stats(traces):
    """Calculates and prints basic statistics of the traces."""
    if traces is None: return
    print("\n--- Basic Statistics ---")
    print(f"Overall Mean Activity: {np.mean(traces):.4f}")
    print(f"Overall Std Dev: {np.std(traces):.4f}")
    print(f"Min Activity: {np.min(traces):.4f}")
    print(f"Max Activity: {np.max(traces):.4f}")
    
    mean_per_neuron = np.mean(traces, axis=0)
    print(f"Mean activity of the most active neuron: {np.max(mean_per_neuron):.4f}")
    print(f"Mean activity of the least active neuron: {np.min(mean_per_neuron):.4f}")

# --- 3. Generate and Save Plots ---
def generate_plots(traces, output_dir):
    """Generates and saves various plots for the traces data."""
    if traces is None: return
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nPlots will be saved to: {output_dir}")

    # Plot 1: Heatmap of a subset of neurons
    plt.figure(figsize=(12, 8))
    num_neurons_to_plot = min(500, traces.shape[1])
    plt.imshow(traces[:, :num_neurons_to_plot].T, aspect='auto', cmap='viridis')
    plt.title(f'Heatmap of Neural Activity (First {num_neurons_to_plot} Neurons)')
    plt.xlabel('Time Step')
    plt.ylabel('Neuron ID')
    plt.colorbar(label='Normalized Activity (dF/F)')
    heatmap_path = os.path.join(output_dir, 'activity_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    print(f"- Saved heatmap to {heatmap_path}")

    # Plot 2: Mean activity over time
    plt.figure(figsize=(12, 6))
    mean_activity_time = np.mean(traces, axis=1)
    plt.plot(mean_activity_time)
    plt.title('Mean Neural Activity Across All Neurons Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Mean dF/F')
    mean_plot_path = os.path.join(output_dir, 'mean_activity_over_time.png')
    plt.savefig(mean_plot_path)
    plt.close()
    print(f"- Saved mean activity plot to {mean_plot_path}")

    # Plot 3: Histogram of activity values
    plt.figure(figsize=(10, 6))
    plt.hist(traces.flatten(), bins=100, log=True)
    plt.title('Distribution of Activity Values (Log Scale)')
    plt.xlabel('Activity Value (dF/F)')
    plt.ylabel('Frequency (Log)')
    hist_path = os.path.join(output_dir, 'activity_distribution.png')
    plt.savefig(hist_path)
    plt.close()
    print(f"- Saved activity distribution histogram to {hist_path}")

    # Plot 4: Traces of a few sample neurons
    plt.figure(figsize=(12, 6))
    num_samples = min(5, traces.shape[1])
    sample_indices = np.random.choice(traces.shape[1], num_samples, replace=False)
    for i in sample_indices:
        plt.plot(traces[:, i], label=f'Neuron {i}')
    plt.title(f'Activity of {num_samples} Sample Neurons')
    plt.xlabel('Time Step')
    plt.ylabel('dF/F')
    plt.legend()
    sample_traces_path = os.path.join(output_dir, 'sample_neuron_traces.png')
    plt.savefig(sample_traces_path)
    plt.close()
    print(f"- Saved sample traces plot to {sample_traces_path}")

# --- Main Execution ---
if __name__ == '__main__':
    traces_data = load_data(LOCAL_TRACES_PATH)
    if traces_data is not None:
        calculate_stats(traces_data)
        generate_plots(traces_data, OUTPUT_DIR)
        print("\nAnalysis complete.")
