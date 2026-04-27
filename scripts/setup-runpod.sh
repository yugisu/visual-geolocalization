#!/bin/bash
set -e

### IMPORTANT: Machine requirements
# - 30GB of container disk
# - 90GB of pod volume
# - A100 instance is good

### Copy this file to the runpod instance.
# scp -P PORT -i ~/.ssh/id_ed25519-personal setup-runpod.sh root@RUNPOD_IP:~/setup-runpod.sh

### To start a persistent tmux session for training:
# tmux new -s training
### To reconnect to the tmux session after disconnecting:
# tmux attach -t training

# ============================================================
# NOTE: Required private env variables. Should be provided by the instance.
# ============================================================

# GIT_USER_NAME=""
# GIT_USER_EMAIL=""
# GH_TOKEN=""
# WANDB_API_KEY=""
# HF_TOKEN=""

for var in GIT_USER_NAME GIT_USER_EMAIL GH_TOKEN WANDB_API_KEY HF_TOKEN; do
  [ -n "${!var}" ] || { echo "ERROR: $var is not set"; exit 1; }
done

# ============================================================
# System dependencies
# ============================================================

which unzip &>/dev/null || { apt-get update && apt-get install -y unzip gh tmux; }
which uv &>/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
which claude &>/dev/null || curl -fsSL https://claude.ai/install.sh | bash

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# ============================================================
# Git & GitHub auth
# ============================================================

git config --global user.name "$GIT_USER_NAME"
git config --global user.email "$GIT_USER_EMAIL"

# echo "$GH_TOKEN" | gh auth login --with-token # GH_TOKEN in env is being used as identifier
gh auth setup-git

# ============================================================
# Repositories
# ============================================================

cd /root

[ -d SatDiFuser ] || git clone https://github.com/yugisu/SatDiFuser.git
cd SatDiFuser && git checkout research && cd ..
# [ -d dift ] || git clone https://github.com/yugisu/dift.git
# cd dift && git checkout research && cd ..
# [ -d diffusion_hyperfeatures ] || git clone https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures.git
# cd diffusion_hyperfeatures && git checkout research && cd ..

[ -d visual-geolocalization ] || git clone https://github.com/yugisu/visual-geolocalization.git
[ -d visual-geolocalization-docs ] || git clone https://github.com/yugisu/visual-geolocalization-docs.git
[ -d diffusion-autoresearch ] || git clone https://github.com/yugisu/diffusion-autoresearch.git
[ -d diffusion-vpr ] || git clone https://github.com/yugisu/diffusion-vpr.git

# Populate .env files
cat > ./diffusion-vpr/.env <<EOF
VISLOC_ROOT="/workspace/data/visloc"
SECO_ROOT="/workspace/data/seco_100k/seasonal_contrast_100k"
DIFFUSIONSAT_256_CHCKPT="/workspace/checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed"
HF_HOME="/workspace/.hugging_face"
WANDB_API_KEY="$WANDB_API_KEY"
HF_TOKEN="$HF_TOKEN"
EOF
cat > ./visual-geolocalization/.env <<EOF
DATA_ROOT="/workspace/data/"
CHECKPOINTS_ROOT="/workspace/checkpoints/"
HF_HOME="/workspace/.hugging_face"
WANDB_API_KEY="$WANDB_API_KEY"
HF_TOKEN="$HF_TOKEN"
EOF

STATE_DIR="/workspace/state"

# Preserve state
mkdir -p $STATE_DIR/lightning_logs/
[ -e lightning_logs ] || ln -s $STATE_DIR/lightning_logs/ lightning_logs
mkdir -p $STATE_DIR/wandb/
[ -e wandb ] || ln -s $STATE_DIR/wandb/ wandb
mkdir -p $STATE_DIR/checkpoints/
[ -e checkpoints ] || ln -s $STATE_DIR/checkpoints/ checkpoints

# ============================================================
# Data
# ============================================================

unzip_large() {
  echo "extracting $1"
  unzip -u "$1" -d "$2" 2>&1 | awk '/(inflating|extracting):/ { if (++n % 25 == 0) { printf "."; fflush() } } END { print "finished!" }'
}

mkdir -p /workspace/data && cd /workspace/data

# --- VisLoc full dataset ---
if [ ! -f /workspace/data/visloc.zip ]; then
  uvx gdown 16vbbiV93rdQL2v_66ccrxICtROugkw2c -O visloc.zip
fi

unzip_large visloc.zip visloc
mv visloc/'satellite_ coordinates_range.csv' visloc/satellite_coordinates_range.csv

# # --- VisLoc example dataset ---
# uvx gdown 16tY7tPZiNIoyAhknvyXnp0jAfccIcHtL -O visloc_example.zip
# unzip -q -u visloc_example.zip -d visloc_example
# mv visloc_example/'satellite_ coordinates_range.csv' visloc_example/satellite_coordinates_range.csv
# rm -f visloc_example.zip

# # --- SeCo dataset ---
# uvx gdown 1pEcd78S5t_Bk76dNXRCMZuqqwI_ecpfc -O seco_100k.zip
# unzip -q -u seco_100k.zip -d seco_100k
# rm -f seco_100k.zip

# # --- ViLD dataset ---
# wget https://zenodo.org/records/19223815/files/ViLD_dataset.zip?download=1 -O ViLD_dataset.zip
# unzip -q -u ViLD_dataset.zip -d ViLD_dataset
# rm -f ViLD_dataset.zip

# # --- SSL4EO-S12 example dataset ---
# uvx gdown 1sRWcYbaWs-efXza6kw03GlJQdZHq5iRN -O SSL4EO-S12_example.tar.gz
# mkdir -p SSL4EO-S12_example
# tar -xzf SSL4EO-S12_example.tar.gz -C ./SSL4EO-S12_example/
# rm -f SSL4EO-S12_example.tar.gz

# ============================================================
# Checkpoints
# ============================================================

mkdir -p /workspace/checkpoints && cd /workspace/checkpoints

# Trimmed DiffusionSat 256 checkpoint at 150k steps
# [ -d /workspace/checkpoints/finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed ] || uvx gdown --folder 1VG4yV_fD9UhOa30JzsNRdTwG4cdeJlmX -O finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed


echo ""
echo "=== Setup complete ==="
