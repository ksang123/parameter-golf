# Recurring Dual U-Net Transformer

## Architecture

6 unique transformer blocks recurring 4× = 24 effective layers at 768 dim.

### Dual U-Net Skip Connections
Two levels of skip connections operating simultaneously:
- **Local skips**: within each recurrence pass, blocks 0-2 (encoder) connect to blocks 3-5 (decoder). Per-recurrence learned weights (4×3×768).
- **Global skips**: across the full unrolled 24-layer depth, effective layers 0-11 (encoder) connect to layers 12-23 (decoder). Shared learned weights (12×768).

Both skips applied before the block runs. This means the same shared weights serve different structural roles depending on their position in the macro U-Net — block 0 in recurrence 0 acts as a global encoder, while block 0 in recurrence 2 acts as a global decoder receiving skip connections from recurrence 1.

### Adaptive Loop Embeddings
Per-position content-dependent feedback between recurrences via a low-rank projection (768→64→768) of the hidden state. No mean pooling — each position gets its own feedback signal. Gradients flow through recurrences (no detach), scaled by 0.1 for stability.

### Quantization
INT4 QAT (per-group g=32, fake-quantize with STE) on all large matrices from step 0. INT8 on embeddings. FP16 passthrough on scalars/norms/skip weights. LZMA compression for the final artifact.

## Model Shape

| Parameter | Value |
|-----------|-------|
| vocab_size | 1024 |
| model_dim | 768 |
| num_heads | 12 |
| num_kv_heads | 6 |
| head_dim | 64 |
| mlp_hidden | 1792 |
| unique_blocks | 6 |
| recurrences | 4 |
| effective_depth | 24 |
| total_params | 28,056,648 |

## Training

- Optimizer: Muon (matrices) + Adam (scalars, embeddings)
- LR warmup: 100 steps linear ramp
- Gradient clipping: 0.3
- Gradient checkpointing per block
- Wallclock-aware warmdown: 200 steps

## Key Metrics

_(to be filled after H100 run)_

## Files

- `train_gpt.py` — full training script
- `submission.json` — metadata
- `train.log` — training log from 8xH100 run
