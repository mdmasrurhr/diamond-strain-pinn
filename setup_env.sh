#!/bin/bash
# One-time setup on a Linux machine or cluster login node, run from this folder (about 5 minutes, about 1.5 GB):
#   1. Miniforge in ~/miniforge3, only if it is not there yet
#   2. conda environment "dspinn": Python 3.12 and the package versions that produced the paper's results
#      (requirements.txt); PyTorch as its CPU-only build, because the jobs run on CPU cores
#   3. records the installed versions in env_versions.txt and runs check_env.py
# Safe to re-run: an existing Miniforge or environment is reused, nothing outside ~/miniforge3 is changed.
set -eo pipefail
cd "$(dirname "$0")"
CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  echo "== installing Miniforge into $CONDA_DIR"
  inst=$(mktemp "$HOME/miniforge_installer_XXXX.sh")
  curl -fsSL -o "$inst" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
  bash "$inst" -b -p "$CONDA_DIR"
  rm -f "$inst"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"
if [ ! -x "$CONDA_DIR/envs/dspinn/bin/python" ]; then
  echo "== creating the conda environment dspinn (Python 3.12)"
  conda create -y -q -n dspinn python=3.12
fi
conda activate dspinn
echo "== installing packages into $CONDA_PREFIX"
python -m pip install -q --upgrade pip
python -m pip install -q --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
python -m pip install -q -r requirements.txt
python -m pip freeze > env_versions.txt
echo "== installed versions written to env_versions.txt"
python check_env.py
