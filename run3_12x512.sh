#!/bin/bash
# Run 3: 12 layers x 512 dim, train bf16, INT4 post-training quant
# 22.6M params, ~58ms/step, ~5.4B tokens in 10min
# Expected: better pre-quant loss, INT4 quant hit is the unknown
RUN_ID=12x512_int4 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=12 \
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
WARMDOWN_ITERS=2400 \
MATRIX_LR=0.02 \
SCALAR_LR=0.02 \
TIED_EMBED_LR=0.03 \
GRAD_CHECKPOINT=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
