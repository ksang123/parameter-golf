#!/bin/bash
# Run 2: Tuned baseline — cosine schedule, higher peak LR, fp16 embed
RUN_ID=baseline_cosine \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=9 \
MODEL_DIM=512 \
NUM_HEADS=8 \
NUM_KV_HEADS=4 \
MLP_HIDDEN=1024 \
RECURRENCES=1 \
QAT_ENABLED=0 \
ITERATIONS=20000 \
TRAIN_BATCH_TOKENS=524288 \
TRAIN_SEQ_LEN=1024 \
VAL_LOSS_EVERY=200 \
VAL_BATCH_SIZE=524288 \
MAX_WALLCLOCK_SECONDS=600 \
TRAIN_LOG_EVERY=50 \
LR_WARMUP_STEPS=50 \
LR_SCHEDULE=cosine \
MATRIX_LR=0.06 \
SCALAR_LR=0.06 \
TIED_EMBED_LR=0.05 \
GRAD_CHECKPOINT=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
