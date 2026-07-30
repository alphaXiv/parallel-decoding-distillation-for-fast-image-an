#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check --no-cache-dir \
  "diffusers==0.34.0" \
  "huggingface_hub>=0.30,<1" \
  "safetensors>=0.4" \
  "torchvision==0.22.1" \
  "scipy>=1.13" \
  "pillow>=10"

NPROC="$(python -c 'import torch; print(torch.cuda.device_count())')"
torchrun --standalone --nproc-per-node="$NPROC" reproduce.py --config config.json
