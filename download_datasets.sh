#!/bin/bash

# This script downloads all the ZapBench datasets from Google Cloud Storage.

# The destination directory for the datasets.
DEST_DIR="/home/v/proj/zebra/data"

# The base URL for the datasets.
BASE_URL="gs://zapbench-release/volumes/20240930"

# Create the destination directory if it doesn't exist.
echo "Creating destination directory: $DEST_DIR"
mkdir -p "$DEST_DIR"

# List of datasets to download.
DATASETS=(
#    "traces.zip"
#    "traces",
#    "aligned"
#    "aligned_multiscale"
#    "anatomy"
#    "anatomy_clahe"
#    "anatomy_clahe_ds"
#    "anatomy_clahe_ds_multiscale"
#    "anatomy_clahe_ffn"
#    "anatomy_clahe_multiscale"
#    "annotations"
#    "correlation_matrix"
#    "df_over_f"
#    "df_over_f_xt_chunked"
#    "df_over_f_xyz_chunked"
#    "flow_fields"
#    "mask"
#    "position_embedding"
#!    "raw"
   "segmentation"
   "segmentation_xy"
   "segmentation_xy_multiscale"
    # "stimuli_features"
    # "stimuli_raw"
    # "stimulus_evoked_response"
#    "traces_fluroglancer"
#    "traces_rastermap_sorted"
)

# Loop through the datasets and download them.
for dataset in "${DATASETS[@]}"; do
    echo "Downloading $dataset..."
    gsutil -m cp -r "$BASE_URL/$dataset" "$DEST_DIR"
done

echo "All datasets downloaded successfully."
