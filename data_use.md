# Using the ZapBench Datasets

This document provides a guide on how to download, load, and use the datasets associated with the ZapBench project.

## Introduction

ZapBench is a benchmark for predicting neural activity in the larval zebrafish brain. The project provides a rich collection of datasets, including time-series of neural activity, 3D volumetric data, and annotations.

This guide focuses on providing actionable steps to get started with the data, with a focus on accessing:

*   Neural activity time-series.
*   The 3D position of each neuron.
*   The experimental conditions for each point in time.

## Downloading the Data

The datasets are hosted on Google Cloud Storage. You can use the `gsutil` command-line tool to download them. Make sure you have `gsutil` installed and configured.

To list all available datasets, you can run:

```bash
gsutil ls gs://zapbench-release/volumes/20240930/
```

To download a specific dataset, for example, the `traces` dataset, you can use the `gsutil -m cp -r` command:

```bash
gsutil -m cp -r gs://zapbench-release/volumes/20240930/traces ./
```

This will download the `traces` dataset to your current directory.

## Data Overview

The following are some of the key datasets available:

*   `traces`: Time-series data of neural activity.
*   `segmentation`: The 3D segmentation of neurons, which can be used to get the position of each neuron.
*   `df_over_f`: Aligned and normalized activity volume.

## Installation

Before you can work with the data, you need to install the necessary Python packages. The `zapbench` library provides helpful utilities for accessing the data.

```bash
pip install zapbench tensorstore matplotlib
```

## Loading and Using the Data

Here are some examples of how to load and use the data, based on the provided notebook.

### Loading Time-Series Data

You can load the time-series data using the `tensorstore` library. The following code snippet shows how to open the `traces` dataset and read a slice of it.

```python
import matplotlib.pyplot as plt
import tensorstore as ts

# Create handle to the remote dataset.
ds_traces = ts.open({
    'open': True,
    'driver': 'zarr3',
    'kvstore': 'gs://zapbench-release/volumes/20240930/traces'
}).result()

# Display info about the dataset.
print(ds_traces.schema)
```

### Understanding Experimental Conditions

The experiment is divided into multiple conditions. You can use the `zapbench.data_utils` module to get the time bounds for each condition.

```python
from zapbench import constants
from zapbench import data_utils

# Print the indexing bounds per condition.
for condition_id, condition_name in enumerate(constants.CONDITION_NAMES):
  inclusive_min, exclusive_max = data_utils.get_condition_bounds(condition_id)
  print(f'{condition_name} has bounds [{inclusive_min}, {exclusive_max}).')
```

### Plotting Traces for a Specific Condition

Once you have the bounds for a condition, you can extract and plot the traces for that condition.

```python
condition_name = 'turning'

# Use the bounds to plot the traces of one of the conditions.
inclusive_min, exclusive_max = data_utils.get_condition_bounds(
    constants.CONDITION_NAMES.index(condition_name))
traces_condition = ds_traces[inclusive_min:exclusive_max, :].read().result()

# Plot traces.
fig = plt.figure(figsize=(12, 12))
plt.title(f'traces for {condition_name} condition')
im = plt.imshow(traces_condition.T, aspect="auto")
plt.xlabel('timestep')
plt.ylabel('neuron')
cbar = fig.colorbar(im)
cbar.set_label("normalized activity (df/f)")
plt.show()
```

### Loading Local Data into NumPy Arrays

Once you have downloaded the datasets, you can load them directly from your local filesystem into NumPy arrays. The process is very similar to loading remote data, but you need to change the `kvstore` path to the local file path.

For example, to load the `traces` dataset from `/home/v/v/zebra/data/traces`:

```python
import tensorstore as ts
import numpy as np

# Create a handle to the local dataset.
# Note the 'kvstore' now points to a local file path.
ds_traces_local = ts.open({
    'open': True,
    'driver': 'zarr3',
    'kvstore': 'file:///home/v/v/zebra/data/traces'
}).result()

# Read the entire dataset into a NumPy array
traces_numpy_array = ds_traces_local.read().result()

print(f"Shape of the traces NumPy array: {traces_numpy_array.shape}")
print(f"Data type: {traces_numpy_array.dtype}")

# You can also read a slice of the data
traces_slice = ds_traces_local[0:1000, 0:10].read().result()
print(f"Shape of the traces slice: {traces_slice.shape}")
```

This same logic applies to all the other datasets you have downloaded. For instance, to load the `segmentation` data, you would change the `kvstore` path to `file:///home/v/v/zebra/data/segmentation`.

## Getting Neuron Positions

To get the 3D position of each neuron, you can use the `segmentation` dataset. This dataset is a 3D volume where each voxel is labeled with the ID of the neuron it belongs to. The neuron IDs in the segmentation volume correspond to the indices in the `traces` data.

### Loading Segmentation Data

First, you need to open the `segmentation` dataset using `tensorstore`.

```python
import numpy as np

# Create handle to the remote dataset.
ds_seg = ts.open({
    'open': True,
    'driver': 'zarr3',
    'kvstore': 'gs://zapbench-release/volumes/20240930/segmentation'
}).result()

# Display info about the dataset.
print(ds_seg.schema)
```

### Calculating Neuron Centroids

With the segmentation data loaded, you can find the center of mass (centroid) of the voxels for a given neuron ID. This will give you the neuron's 3D coordinates.

Here is a function that calculates the centroid for a given neuron ID:

```python
def get_neuron_centroid(segmentation_data, neuron_id):
    """Calculates the centroid of a neuron from the segmentation data.

    Args:
        segmentation_data: The tensorstore handle to the segmentation data.
        neuron_id: The ID of the neuron.

    Returns:
        A numpy array with the (x, y, z) coordinates of the centroid.
    """
    # Find the coordinates of all voxels for the given neuron ID.
    # This is a boolean mask of the same shape as the segmentation data.
    neuron_voxels = (segmentation_data.read().result() == neuron_id)
    
    # Get the indices of the True values in the mask.
    indices = np.argwhere(neuron_voxels)
    
    if indices.size == 0:
        return None # Neuron ID not found

    # Calculate the mean of the indices to get the centroid.
    # The indices are in (z, y, x) order, so we reverse them to get (x, y, z).
    centroid = np.mean(indices, axis=0)[::-1]
    
    return centroid

# Example: Get the centroid of neuron with ID 100
neuron_id = 100
centroid = get_neuron_centroid(ds_seg, neuron_id)

if centroid is not None:
    print(f'Centroid of neuron {neuron_id}: {centroid}')
else:
    print(f'Neuron {neuron_id} not found in the segmentation data.')

```

This will give you the 3D position of each neuron, which you can then use for further analysis and visualization. You now have the tools to get the time-series data, the experimental conditions, and the spatial location of each neuron.

