# BitNet b1.58 Ternary v2 — 68M Params, val_bpb=1.1770

**Non-record submission.** Explores ternary quantization (1.58-bit weights) as an alternative to the standard int6 stack. Improves on our earlier PR #139 (1.2029 → 1.1770) and documents systematic findings on what works and what breaks for ternary models.

## Results

| Metric | Value |
|---|---|
| Standard val_bpb | 1.1983 |
| Ternary roundtrip val_bpb | 1.1999 |
| **Sliding window val_bpb (stride=64)** | **1.1770** |
| Roundtrip gap | 0.0016 |
| Artifact size | 15,882,274 bytes (15.88 MB) |
| Training steps | 4,591 in 600s (130.7 ms/step) |
| Model params | 68M ternary |
| Peak memory | 23.2 GB per GPU |

Improvement over PR #139: **1.2029 → 1.1770** (−0.026 BPB).

## Architecture

- 12 transformer layers, 768 dim, 12 heads, 6 KV heads (GQA)
- MLP 3.25× (hidden 2496), relu² activation
- BitLinear for all attention + MLP projections (ternary {-1, 0, 1} with per-group absmax STE)
- U-Net skip connections, tied fp16 embeddings
- RoPE base 200,000, logit softcap 30.0
- Muon optimizer (LR=0.04, momentum 0.99) + Adam for embeddings/scalars

## What Worked for Ternary

### Wider MLP (2304 → 2496)
Ternary packing at 1.6 bits/param makes extra parameters almost free in the artifact. Widening the MLP from 3× to 3.25× added 4.4M params but only 0.8MB to the artifact. This gave a direct improvement: 1.2029 → 1.1983 (standard eval).

### Higher Learning Rate (0.04 vs competition's 0.02–0.025)
Ternary STE gradients are inherently noisy — the quantization function snaps continuous weights to just 3 levels, and the STE passes gradients through as if this didn't happen. Higher LR helps the optimizer punch through this noise floor. The competition consensus LR (0.02–0.025) was tuned for int6 models where STE noise is 20× smaller.

### fp16 Scale Simulation During Training
The critical insight from v1: training computes scales in fp32, but serialization stores them in fp16. Different precision → different rounding → different ternary values → 0.05 BPB gap. Training with `.half().float()` on scales simulates fp16 precision during training, closing the gap to **0.0016 BPB**. This is the single most important technique for ternary roundtrip fidelity.

### Longer Warmdown
The final 25% of training (LR decaying linearly to 0) was the most productive phase. In v1, the last 500 steps alone improved BPB by 0.03. At low LR, the continuous shadow weights converge to values that round cleanly to {-1, 0, 1}, reducing quantization error. Ternary benefits from warmdown more than fp16 models because the quantization grid is so coarse.

### Base-3 Packing (1.6 bits/param)
Ternary values {-1, 0, 1} are encoded as base-3: 5 trits per byte (3⁵ = 243 ≤ 255). This achieves 1.6 bits/param — near the information-theoretic minimum of log₂(3) = 1.585 bits. Combined with LZMA compression, 68M ternary params + fp16 scales fit in 15.82MB.

### 68M Parameters in 15.88MB
At 1.6 bits/param, ternary packs 2.4× more parameters than int6 (68M vs 27M) in the same artifact budget. While each ternary param is less expressive, the sheer parameter count provides more model capacity than might be expected.

## What Broke the Network

These techniques caused the model to plateau at val_loss ≈ 2.4 (vs normal convergence to ≈ 2.0):

### XSA (Exclusive Self-Attention)
XSA subtracts the self-value projection from attention output, removing the "self-attention bias" where tokens attend too much to themselves. For int6 models this gives +0.002 BPB. For ternary, it causes a **complete training plateau at 2.4**. Ternary attention weights are too coarse to exhibit self-attention bias in the first place — the attention patterns are already sparse and noisy. Removing the self-value component removes signal, not noise.

### Weight Decay (0.04)
Decoupled weight decay shrinks continuous weights toward zero. For int6, this improves quantization robustness and generalization. For ternary, it causes **training collapse**. Shrinking weights toward zero means more weights quantize to 0 instead of ±1, making the model progressively sparser until it loses capacity. The ternary STE already provides implicit regularization — adding explicit WD on top is destructive.

### Gradient Clipping (0.3–1.0)
The Muon optimizer normalizes gradients via Newton-Schulz orthogonalization. Adding gradient clipping on top double-normalizes the signal. With 68M params, the global gradient norm is naturally large. Clipping at 1.0 caused training to stall after ~500 steps — the model learned basic patterns, then the clipped gradients couldn't push through to learn more.

## What Didn't Help (But Didn't Break)

### SmearGate + BigramHash
SmearGate blends each token's embedding with the previous token's (learned per-dim gate, 768 params). BigramHash adds a hash-table embedding for token pairs. Together they give +0.01 BPB for int6 models. For ternary, they **hurt by 0.02 BPB** regardless of initialization. The ternary weights can't adapt to the modified input distribution — the quantization is too coarse to learn the subtle adjustments needed.

### OrthoInit
Orthogonal weight initialization is required for SmearGate to work in int6 models. For ternary, the carefully constructed orthogonal structure is **destroyed on the first forward pass** when weights snap to {-1, 0, 1}. The STE gradient tries to maintain orthogonality in the continuous shadow weights, but the actual computation uses ternary values that bear no resemblance to an orthogonal matrix. Default Kaiming initialization works equally well.

### Test-Time Training (TTT)
Causal SGD adaptation during eval (evaluate chunk, record scores, train on chunk, repeat). Gives +0.01–0.03 BPB for int6 models. For ternary: **no improvement** in any variant tested:
- Frozen ternary (only update norms/scales/gates): 1.176 vs 1.177 sliding — negligible
- All params (update dequantized weights): no improvement

The dequantized weights (q × scale) sit in a region of weight space the model was never trained to operate in with continuous values. The loss landscape around ternary-structured weights isn't smooth enough for few-step SGD adaptation.

## QAT Insight

Our full-training STE (quantize every forward from step 0) achieves near-zero roundtrip gap but hurts convergence — the model trains slower because every gradient is corrupted by quantization noise. The competition's "Late QAT" approach (enable STE only in the last 15% of training) is likely optimal: full-precision convergence for 85% of training, then STE to close the quantization gap.

This suggests an unexplored direction: **int4 with late QAT**. Int4 gives 50% more params than int6 (32M vs 21M) in the same budget. With late QAT, the quantization gap should be near-zero (our ternary full-STE gap is only 0.0016 — int4 with 16 levels would be even smaller). Nobody in the competition has tried this.

## Negative Results Summary

| Technique | int6 effect | Ternary effect | Root cause |
|---|---|---|---|
| XSA | +0.002 | 💀 plateau | Ternary attention too coarse for self-bias |
| Weight decay | +0.003 | 💀 plateau | Fights STE, causes sparsity collapse |
| Grad clipping | standard | 💀 stalls | Double-normalizes with Muon |
| SmearGate | +0.01 | −0.02 | Can't adapt to modified inputs |
| OrthoInit | enables SmearGate | no effect | Destroyed by quantization |
| EMA/SWA | +0.005 | 💀 broken | Averaging ternary = non-ternary |
| TTT | +0.01–0.03 | no effect | Loss landscape not smooth around ternary |

## Run Command

```bash
bash setup.sh
bash run_8xh100.sh
```

## References

- BitNet b1.58: [arXiv:2310.11453](https://arxiv.org/abs/2310.11453)
- Muon optimizer: [kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/)
- Prior submission: PR #139 (val_bpb=1.2029)
