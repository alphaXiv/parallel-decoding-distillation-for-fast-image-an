#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check --no-cache-dir \
  "diffusers==0.34.0" \
  "datasets==3.6.0" \
  "huggingface_hub>=0.30,<1" \
  "safetensors>=0.4" \
  "torchvision==0.22.1" \
  "scipy>=1.13" \
  "pillow>=10"

python reproduce.py --config config.json
