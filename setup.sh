#!/bin/bash
set -euo pipefail

TRAIN_SHARDS=80

while [[ $# -gt 0 ]]; do
    case $1 in
        --shards) TRAIN_SHARDS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== Parameter Golf Setup ==="

# Ensure pip and venv are available
which pip3 >/dev/null 2>&1 || sudo apt-get install -y python3-pip
python3 -m venv --help >/dev/null 2>&1 || sudo apt-get install -y python3-venv

# Create and activate venv
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Install uv, then packages
pip install uv
uv pip install torch numpy sentencepiece huggingface-hub datasets tqdm zstandard

echo "=== Downloading FineWeb sp1024 ($TRAIN_SHARDS train shards) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards "$TRAIN_SHARDS"

source .venv/bin/activate
echo "=== Setup complete ==="
