#!/bin/bash
# 10x1024 BitNet b1.58: ternary weights, fp16 tied embeddings
# ~69M params, ~15.4MB ternary artifact
# H100 ~193ms/step -> ~3100 steps -> 1.6B tokens in 10min
# A100 ~2x slower -> ~386ms/step -> need ~20min
RUN_ID=bitnet_10x1024 \
ITERATIONS=20000 \
NUM_LAYERS=10 \
MODEL_DIM=1024 \
NUM_HEADS=16 \
NUM_KV_HEADS=4 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=1000 \
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
