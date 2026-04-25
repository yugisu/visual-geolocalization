#!/bin/bash
set -e

# This script downloads a trimmed version of the DiffusionSat 256x256 pre-train weights from a self-hosted link in Google Drive.
# NOTE: Checkpoint size is ~20GB.

# DiffusionSat GitHub: https://github.com/samar-khanna/DiffusionSat
# Paper: "DiffusionSat: A Generative Foundation Model for Satellite Imagery", Khanna et al, 2024, https://arxiv.org/abs/2312.03606

source $(dirname "$0")/../.env

for var in CHECKPOINTS_ROOT; do
  [ -n "${!var}" ] || { echo "ERROR: $var is not set"; exit 1; }
done

echo "Checkpoints root folder: '$CHECKPOINTS_ROOT'"
mkdir -p "$CHECKPOINTS_ROOT" && cd "$CHECKPOINTS_ROOT"

echo "Downloading DiffusionSat checkpoint from Google Drive..."
uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed

echo "DiffusionSat checkpoint has been successfully downloaded to '$CHECKPOINTS_ROOT/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed/'!"