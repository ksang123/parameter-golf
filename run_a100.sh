#!/bin/bash
# 9x640 on 8xA100 — matches H100 token budget (~4.6B tokens)
# 26.5M params, INT4 artifact ~14.5MB
# H100 step time ~68ms -> ~6800 steps in 10min -> 3.6B tokens
# A100 ~1.6x slower -> ~109ms/step -> need ~11min to match
RUN_ID=9x640_a100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=640 \
NUM_HEADS=10 \
NUM_KV_HEADS=5 \
MLP_HIDDEN=1280 \
RECURRENCES=1 \
QAT_ENABLED=0 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=1500 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=720 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=50 \
LR_SCHEDULE=cosine \
LR_MIN_FRAC=0.05 \
MATRIX_LR=0.06 \
SCALAR_LR=0.06 \
TIED_EMBED_LR=0.05 \
GRAD_CHECKPOINT=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
