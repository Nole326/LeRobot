#!/bin/bash
set -e
source /isaac-sim/setup_python_env.sh
# Expose only immutable torch/CUDA prebundles, not the simulator's application packages.
export PYTHONPATH="/workspace/course/venv_bc/lib/python3.11/site-packages:/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle"
export PIP_CACHE_DIR=/workspace/course/cache/pip-runtime
export HF_HOME=/workspace/course/cache/huggingface
export XDG_CACHE_HOME=/workspace/course/cache/xdg
export OMP_NUM_THREADS=4
exec /workspace/course/venv_bc/bin/python "$@"
