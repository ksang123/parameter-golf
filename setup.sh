#!/bin/bash
set -euo pipefail

# Parameter Golf setup — run after clone
# Usage: bash setup.sh [--shards N] [--no-fa3]
#   --shards N   number of training shards (default: 80 = 8B tokens)
#   --no-fa3     skip FA3 install (e.g. no Hopper GPU)

TRAIN_SHARDS=80
INSTALL_FA3=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --shards) TRAIN_SHARDS="$2"; shift 2 ;;
        --no-fa3) INSTALL_FA3=false; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Parameter Golf Setup ==="
echo "Train shards: $TRAIN_SHARDS | FA3: $INSTALL_FA3"

# Venv
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip

# Core deps
pip install -r requirements.txt

# FA3 via HF kernels (Hopper-only, uses torch SDPA flash backend)
if $INSTALL_FA3; then
    echo "=== Installing FA3 (kernels-community/flash-attn3) ==="
    # kernels package is already in requirements.txt
    # Pre-download the FA3 kernel so first training run doesn't stall
    python3 -c "from kernels import get_kernel; get_kernel('kernels-community/flash-attn3'); print('FA3 ready')"
fi

# Dataset
echo "=== Downloading FineWeb sp1024 ($TRAIN_SHARDS train shards) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"

echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
