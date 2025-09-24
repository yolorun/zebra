import tensorstore as ts
import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt


# The path to the local traces dataset
# The 'file://' prefix is important for tensorstore
local_traces_path = 'file:///home/v/v/zebra/data/traces_zip/traces'

print(f"Attempting to load data from: {local_traces_path}")

# Create a handle to the local dataset.
# The dataset is in zarr3 format.
ds_traces_local = ts.open({
    'open': True,
    'driver': 'zarr3',
    'kvstore': local_traces_path
}).result()

# Read the entire dataset into a NumPy array
traces = ds_traces_local.read().result()
print(traces.shape)

# time x neurons
# 19146 is bad

print(np.std(traces[5800:6600, 100]))
# (7879, 71721)

43206740/7879

plt.plot(traces[:, 100]); plt.show(block=True);

# Print the shape of the array
print("Successfully loaded the traces into a NumPy array.")
print(f"Shape of the traces array: {traces.shape}")
