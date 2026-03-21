#!/usr/bin/env python3
"""Eval script for BitNet ternary models. DDP + compiled + batched TTT.

Usage:
  torchrun --standalone --nproc_per_node=8 eval_bitnet.py --model final_model.ternary.ptz --ttt --ttt-all --ttt-incremental
"""
import argparse, io, math, os, sys, time, lzma, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from train_bitnet import (
    Hyperparameters, GPT, BitLinear, CastedLinear,
    dequantize_state_dict_ternary, restore_low_dim_params_to_fp32,
    build_sentencepiece_luts, load_validation_tokens,
    eval_val_sliding,
)

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def load_compressed(path):
    data = open(path, "rb").read()
    for fn in [lzma.decompress, zlib.decompress]:
        try: return fn(data)
        except Exception: pass
    if HAS_ZSTD: return zstd.ZstdDecompressor().decompress(data)
    raise RuntimeError("Could not decompress")


def make_model(args, device):
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers,
        model_dim=args.model_dim, num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden=args.mlp_hidden or args.mlp_mult * args.model_dim,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init, xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(model)
    return model


def ttt_train_step(base_model, tokens, seq_len, ttt_opt, device, batch_size=32):
    """One epoch of batched SGD over tokens."""
    base_model.train()
    n = tokens.numel() - 1
    # Build batched sequences
    n_seqs = n // seq_len
    if n_seqs == 0:
        return
    usable = n_seqs * seq_len
    all_x = tokens[:usable].reshape(n_seqs, seq_len).to(dtype=torch.int64, device=device)
    all_y = tokens[1:usable + 1].reshape(n_seqs, seq_len).to(dtype=torch.int64, device=device)
    for bi in range(0, n_seqs, batch_size):
        x = all_x[bi:bi + batch_size]
        y = all_y[bi:bi + batch_size]
        ttt_opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = base_model(x, y)
        loss.backward()
        ttt_opt.step()


def ttt_eval_sliding(
    args, base_model, rank, world_size, device, val_tokens,
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
    stride, batch_seqs, ttt_opt, mode="chunk",
):
    """TTT with sliding window eval. mode='chunk' or 'incremental'."""
    seq_len = args.train_seq_len
    total_tokens = val_tokens.numel() - 1
    chunk_size = args.ttt_chunk_tokens
    chunk_starts = list(range(0, total_tokens, chunk_size))

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)

    for ci, cs in enumerate(chunk_starts):
        ce = min(cs + chunk_size, total_tokens)
        chunk_tokens = val_tokens[cs:ce + 1]

        # --- EVAL this chunk (sliding window, sharded across ranks) ---
        window_starts = [ws for ws in range(0, ce - cs, stride)
                         if min(ws + seq_len, ce - cs) - ws >= 1]
        total_windows = len(window_starts)
        my_s = (total_windows * rank) // world_size
        my_e = (total_windows * (rank + 1)) // world_size
        my_windows = window_starts[my_s:my_e]

        base_model.eval()
        with torch.no_grad():
            for bi in range(0, len(my_windows), batch_seqs):
                batch_ws = my_windows[bi:bi + batch_seqs]
                bsz = len(batch_ws)
                x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
                wlens = []
                for i, ws in enumerate(batch_ws):
                    end = min(ws + seq_len, ce - cs)
                    wlen = end - ws
                    wlens.append(wlen)
                    chunk = chunk_tokens[ws:ws + wlen + 1].to(dtype=torch.int64, device=device)
                    x_batch[i, :wlen] = chunk[:-1]
                    y_batch[i, :wlen] = chunk[1:]
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = base_model.forward_logits(x_batch)
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)).float(),
                    y_batch.reshape(-1), reduction="none",
                ).reshape(bsz, seq_len)
                for i, ws in enumerate(batch_ws):
                    wlen = wlens[i]
                    s = 0 if ws == 0 else max(wlen - stride, 0)
                    scored_nll = nll[i, s:wlen].to(torch.float64)
                    loss_sum += scored_nll.sum()
                    token_count += float(wlen - s)
                    tgt = y_batch[i, s:wlen]
                    prev = x_batch[i, s:wlen]
                    tb = base_bytes_lut[tgt].to(torch.float64)
                    tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                    byte_count += tb.sum()

        # --- TRAIN (causal: only after eval) ---
        if mode == "chunk":
            for _epoch in range(args.ttt_epochs):
                ttt_train_step(base_model, chunk_tokens, seq_len, ttt_opt, device)
        elif mode == "incremental":
            # Train 1 epoch on ALL tokens seen so far
            seen = val_tokens[:ce + 1]
            ttt_train_step(base_model, seen, seq_len, ttt_opt, device)

        if rank == 0 and (ci + 1) % max(len(chunk_starts) // 4, 1) == 0:
            print(f"  chunk {ci+1}/{len(chunk_starts)} ({100*(ci+1)//len(chunk_starts)}%)")

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    val_loss = (loss_sum / token_count).item()
    bpt = val_loss / math.log(2.0)
    tpb = token_count.item() / byte_count.item()
    return val_loss, bpt * tpb


def make_ttt_optimizer(base_model, lr, freeze_ternary, freeze_first_n=2):
    if freeze_ternary:
        bl_weights = {id(m.weight) for m in base_model.modules() if isinstance(m, BitLinear)}
        params = [p for p in base_model.parameters() if id(p) not in bl_weights]
    else:
        params = list(base_model.parameters())
    # Freeze first N blocks (stable low-level features don't need adaptation)
    frozen = set()
    for i in range(min(freeze_first_n, len(base_model.blocks))):
        for p in base_model.blocks[i].parameters():
            frozen.add(id(p))
    params = [p for p in params if id(p) not in frozen]
    return torch.optim.SGD(params, lr=lr, momentum=0.9)


def log0(msg, rank):
    if rank == 0: print(msg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.ternary.ptz")
    parser.add_argument("--ttt", action="store_true", help="TTT frozen ternary (chunk)")
    parser.add_argument("--ttt-all", action="store_true", help="TTT all params (chunk)")
    parser.add_argument("--ttt-incremental", action="store_true", help="TTT all params (incremental)")
    parser.add_argument("--ttt-chunk", type=int, default=2_000_000)
    parser.add_argument("--ttt-lr", type=float, default=2e-3)
    parser.add_argument("--ttt-epochs", type=int, default=3)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-seqs", type=int, default=32)
    cli = parser.parse_args()

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()
    args.num_layers = 12
    args.model_dim = 768
    args.num_heads = 12
    args.num_kv_heads = 6
    args.mlp_hidden = 2496
    args.train_seq_len = 2048
    args.rope_base = 200000
    args.xsa_last_n = 0
    args.ttt_lr = cli.ttt_lr
    args.ttt_epochs = cli.ttt_epochs
    args.ttt_chunk_tokens = cli.ttt_chunk

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    luts = build_sentencepiece_luts(sp, args.vocab_size, device)
    log0(f"Val tokens: {val_tokens.numel() - 1:,}", rank)

    # Load weights
    raw = load_compressed(cli.model)
    tern_obj = torch.load(io.BytesIO(raw), weights_only=False)
    base_sd = dequantize_state_dict_ternary(tern_obj)

    # Build and compile model
    base_model = make_model(args, device)
    base_model.load_state_dict(base_sd, strict=True)
    for m in base_model.modules():
        if isinstance(m, BitLinear): m._skip_quantize = True

    # Warmup compile with a dummy forward+backward
    log0("Compiling model...", rank)
    compiled = torch.compile(base_model, dynamic=False, fullgraph=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        dummy_x = torch.zeros(1, args.train_seq_len, dtype=torch.int64, device=device)
        dummy_y = torch.zeros(1, args.train_seq_len, dtype=torch.int64, device=device)
        loss = compiled(dummy_x, dummy_y)
        loss.backward()
    base_model.zero_grad(set_to_none=True)
    log0("Compiled.", rank)

    # Save clean state for resetting between variants
    clean_sd = {k: v.clone() for k, v in base_model.state_dict().items()}

    # --- 1. Sliding window only (skip if running TTT variants) ---
    if not (cli.ttt or cli.ttt_all or cli.ttt_incremental):
        log0("\n=== Sliding Window Eval ===", rank)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        sl, sb = eval_val_sliding(args, base_model, rank, world_size, device, val_tokens, *luts,
                                  stride=cli.stride, batch_seqs=cli.batch_seqs)
        torch.cuda.synchronize()
        log0(f"val_loss:{sl:.6f} val_bpb:{sb:.6f} time:{time.perf_counter()-t0:.1f}s", rank)

    # --- 2. TTT frozen ternary (chunk) ---
    if cli.ttt:
        base_model.load_state_dict(clean_sd)
        opt = make_ttt_optimizer(base_model, cli.ttt_lr, freeze_ternary=True)
        log0("\n=== TTT frozen ternary (chunk) ===", rank)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tl, tb = ttt_eval_sliding(args, base_model, rank, world_size, device, val_tokens, *luts,
                                  stride=cli.stride, batch_seqs=cli.batch_seqs, ttt_opt=opt, mode="chunk")
        torch.cuda.synchronize()
        log0(f"val_loss:{tl:.6f} val_bpb:{tb:.6f} time:{time.perf_counter()-t0:.1f}s", rank)

    # --- 3. TTT all params (chunk) ---
    if cli.ttt_all:
        base_model.load_state_dict(clean_sd)
        opt = make_ttt_optimizer(base_model, cli.ttt_lr, freeze_ternary=False)
        log0("\n=== TTT all params (chunk) ===", rank)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tl, tb = ttt_eval_sliding(args, base_model, rank, world_size, device, val_tokens, *luts,
                                  stride=cli.stride, batch_seqs=cli.batch_seqs, ttt_opt=opt, mode="chunk")
        torch.cuda.synchronize()
        log0(f"val_loss:{tl:.6f} val_bpb:{tb:.6f} time:{time.perf_counter()-t0:.1f}s", rank)

    # --- 4. TTT all params (incremental) ---
    if cli.ttt_incremental:
        base_model.load_state_dict(clean_sd)
        opt = make_ttt_optimizer(base_model, cli.ttt_lr, freeze_ternary=False)
        log0("\n=== TTT all params (incremental) ===", rank)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tl, tb = ttt_eval_sliding(args, base_model, rank, world_size, device, val_tokens, *luts,
                                  stride=cli.stride, batch_seqs=cli.batch_seqs, ttt_opt=opt, mode="incremental")
        torch.cuda.synchronize()
        log0(f"val_loss:{tl:.6f} val_bpb:{tb:.6f} time:{time.perf_counter()-t0:.1f}s", rank)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
