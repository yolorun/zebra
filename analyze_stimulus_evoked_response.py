import tensorstore as ts
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import random

# The path to the local stimulus_evoked_response dataset
# The 'file://' prefix is important for tensorstore
local_stimulus_path = 'file:///home/v/v/zebra/data/stimulus_evoked_response'

print(f"Attempting to load data from: {local_stimulus_path}")

# Create a handle to the local dataset.
# The dataset is in zarr3 format.
try:
    ds_stimulus = ts.open({
        'open': True,
        'driver': 'zarr3',
        'kvstore': local_stimulus_path
    }).result()

    # Read the entire dataset into a NumPy array
    stimulus_data = ds_stimulus.read().result()
    print("Successfully loaded the stimulus_evoked_response dataset into a NumPy array.")
    print(f"Shape of the stimulus_evoked_response array: {stimulus_data.shape}")
    print(f"Data type of the array: {stimulus_data.dtype}")

    # Calculate statistics
    stats = {
        'shape': list(stimulus_data.shape),
        'dtype': str(stimulus_data.dtype),
        'min': float(np.min(stimulus_data)),
        'max': float(np.max(stimulus_data)),
        'mean': float(np.mean(stimulus_data)),
        'std_dev': float(np.std(stimulus_data)),
        'median': float(np.median(stimulus_data))
    }

    # Print stats
    print("\nStatistics for the stimulus_evoked_response dataset:")
    print(json.dumps(stats, indent=4))

    # Write stats to a file
    output_path = 'stimulus_evoked_response_stats.json'
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nSuccessfully wrote statistics to {output_path}")

    # Plotting stimulus change for a few random neurons
    print("\nGenerating plots...")
    num_neurons = stimulus_data.shape[1]
    num_to_plot = 3
    if num_neurons > num_to_plot:
        neuron_indices_to_plot = random.sample(range(num_neurons), num_to_plot)
    else:
        neuron_indices_to_plot = range(num_neurons)

    plt.figure(figsize=(15, 7))
    for i, neuron_idx in enumerate(neuron_indices_to_plot):
        plt.plot(stimulus_data[:, neuron_idx], label=f'Neuron {neuron_idx}')

    plt.title('Stimulus Evoked Response for Random Neurons')
    plt.xlabel('Time Step')
    plt.ylabel('Stimulus Value')
    plt.legend()
    plt.grid(True)
    print("Displaying plot. Close the plot window to exit the script.")
    plt.show(block=True)

except Exception as e:
    print(f"An error occurred: {e}")
