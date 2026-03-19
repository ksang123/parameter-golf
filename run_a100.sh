#!/bin/bash
# 9x640 on 8xA100 — clean baseline-derived script, no recurrence
# 26.5M params, cosine LR, INT4 post-training quantization
# ~17min on A100 to match H100 10min token budget
RUN_ID=9x640_a100 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=2000 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=1020 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=50 \
LR_SCHEDULE=cosine \
LR_MIN_FRAC=0.05 \
MATRIX_LR=0.08 \
SCALAR_LR=0.08 \
TIED_EMBED_LR=0.06 \
torchrun --standalone --nproc_per_node=8 train_gpt_9x640.py
