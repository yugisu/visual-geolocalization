#!/bin/bash
set -e

# This script downloads the train RGB slice of the SSL4EO-S12-1.1 dataset from Hugging Face.
# NOTE: The downloaded slice of the dataset takes ~200GB of disk space.
# NOTE: This dataset is also available in webdataset format. Please inspect the repositories below.

# Dataset GitHub: https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1
# Dataset Hugging Face: https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1
# Paper: "SSL4EO-S12 v1.1: A Multimodal, Multiseasonal Dataset for Pretraining, Updated", Blumenstiel et al, 2026, https://arxiv.org/abs/2503.00168

source $(dirname "$0")/../.env

for var in DATA_ROOT; do
  [ -n "${!var}" ] || { echo "ERROR: $var is not set"; exit 1; }
done

echo "Data root folder: '$DATA_ROOT'"
mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT"

echo "Downloading dataset metadata..."
uvx hf download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --include "train_metadata.parquet" --local-dir ./SSL4EOS12
echo "Downloading dataset..."
uvx hf download embed2scale/SSL4EO-S12-v1.1 --repo-type dataset --include "train/S2RGB/*" --local-dir ./SSL4EOS12

echo "SSL4EO-S12-1.1 dataset has been successfully downloaded to '$DATA_ROOT/SSL4EOS12/'!"