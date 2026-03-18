#!/bin/bash
# Run on 8xH100 for the 10-minute challenge
RUN_ID=recurrent_dual_unet_8xh100 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=100 \
GRAD_CLIP_NORM=0.3 \
GRAD_CHECKPOINT=0 \
MATRIX_LR=0.04 \
SCALAR_LR=0.04 \
TIED_EMBED_LR=0.035 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
