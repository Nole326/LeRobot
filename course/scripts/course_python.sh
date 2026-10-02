#!/bin/bash
# Use Isaac's binary/runtime with a course-only application dependency overlay.
set -e
source /isaac-sim/setup_python_env.sh
export PYTHONPATH="/workspace/course/leisaac/source/leisaac:/workspace/course/venv_sim/lib/python3.11/site-packages:${PYTHONPATH:-}"
export PIP_CACHE_DIR=/workspace/course/cache/pip-runtime
export LEISAAC_ASSETS_ROOT=/data/course/assets
export HF_HOME=/workspace/course/cache/huggingface
export XDG_CACHE_HOME=/workspace/course/cache/xdg
export OMP_NUM_THREADS=4
export PYTHONEXE=/workspace/course/venv_sim/bin/python
# Keep NVIDIA's EXP_PATH/CARB_APP_PATH/preload and Vulkan setup, not just Python paths.
exec /isaac-sim/python.sh "$@"
