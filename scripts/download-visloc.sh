#!/bin/bash
set -e

# This script downloads the UAV-VisLoc dataset from a mirror link in Google Drive.
# NOTE: Dataset size is ~17GB. ~34GB of free storage space is required for the download.

# Mirror Google Drive link: https://drive.google.com/file/d/16vbbiV93rdQL2v_66ccrxICtROugkw2c/view

# Dataset GitHub: https://github.com/IntelliSensing/UAV-VisLoc
# Original Google Drive link: https://drive.google.com/file/d/1xYODANyilEMM3CfWh85APwkTHQeLTcCT/view
# Paper: "UAV-VisLoc: A Large-scale Dataset for UAV Visual Localization", Xu et al, 2024, https://arxiv.org/abs/2405.11936

source $(dirname "$0")/../.env

for var in DATA_ROOT; do
  [ -n "${!var}" ] || { echo "ERROR: $var is not set"; exit 1; }
done

unzip_large() {
  echo "Extracting $1..."
  unzip -u "$1" -d "$2" 2>&1 | awk '/(inflating|extracting):/ { if (++n % 25 == 0) { printf "."; fflush() } } END { print "finished!" }'
}

echo "Data root folder: '$DATA_ROOT'"
mkdir -p "$DATA_ROOT" && cd "$DATA_ROOT"

if [ ! -f visloc.zip ]; then
  echo "Downloading full VisLoc dataset from Google Drive..."
  uvx gdown 16vbbiV93rdQL2v_66ccrxICtROugkw2c -O visloc.zip
fi

unzip_large visloc.zip visloc
mv visloc/'satellite_ coordinates_range.csv' visloc/satellite_coordinates_range.csv

echo "VisLoc dataset has been successfully downloaded to '$DATA_ROOT/visloc/'!"