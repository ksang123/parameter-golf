#!/bin/bash
# 2x1024 ultra-wide shallow model, INT8 quantization
# 15.7M params, ~14.5MB INT8+zlib, ~39ms/step on H100 -> 8.1B tokens
# A100 ~1.6-2x slower -> ~62-78ms/step -> need ~18min
RUN_ID=2x1024_a100 \
ITERATIONS=20000 \
NUM_LAYERS=2 \
MODEL_DIM=1024 \
NUM_HEADS=16 \
NUM_KV_HEADS=8 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=2000 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=1200 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=50 \
LR_SCHEDULE=cosine \
LR_MIN_FRAC=0.05 \
MATRIX_LR=0.08 \
SCALAR_LR=0.08 \
TIED_EMBED_LR=0.06 \
torchrun --standalone --nproc_per_node=8 train_gpt_9x640.py
