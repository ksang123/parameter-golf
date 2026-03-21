#!/usr/bin/env python3
"""Standalone eval script for BitNet ternary models.
Loads final_model.ternary.ptz and runs sliding window + TTT variants.

Usage:
  python3 eval_bitnet.py [--model final_model.ternary.ptz] [--ttt] [--ttt-all]
"""
import argparse, glob, io, math, os, sys, time, lzma, zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor

# Import everything from train_bitnet
sys.path.insert(0, os.path.dirname(__file__))
from train_bitnet import (
    Hyperparameters, GPT, BitLinear, CastedLinear,
    dequantize_state_dict_ternary, restore_low_dim_params_to_fp32,
    build_sentencepiece_luts, load_validation_tokens,
    eval_val_sliding, ttt_and_eval_sliding,
)

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False


def load_model(path: str, args: Hyperparameters, device: torch.device):
    data = open(path, "rb").read()
    # Try decompressors
    for decompress in [
        lzma.decompress,
        zlib.decompress,
        lambda d: zstd.ZstdDecompressor().decompress(d) if HAS_ZSTD else None,
    ]:
        try:
            raw = decompress(data)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("Could not decompress model file")

    tern_obj = torch.load(io.BytesIO(raw), weights_only=False)
    sd = dequantize_state_dict_ternary(tern_obj)

    model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_hidden=args.mlp_hidden or args.mlp_mult * args.model_dim,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        xsa_last_n=args.xsa_last_n,
    ).to(device).bfloat16()

    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(model)

    model.load_state_dict(sd, strict=True)
    for mod in model.modules():
        if isinstance(mod, BitLinear):
            mod._skip_quantize = True
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_model.ternary.ptz")
    parser.add_argument("--ttt", action="store_true", help="Run TTT (frozen ternary)")
    parser.add_argument("--ttt-all", action="store_true", help="Run TTT (all params)")
    parser.add_argument("--ttt-chunk", type=int, default=2_000_000)
    parser.add_argument("--ttt-lr", type=float, default=3e-4)
    parser.add_argument("--ttt-epochs", type=int, default=3)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-seqs", type=int, default=32)
    cli = parser.parse_args()

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
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    print(f"Val tokens: {val_tokens.numel() - 1:,}")

    model = load_model(cli.model, args, device)
    print(f"Model loaded from {cli.model}")

    # Sliding window eval (skip if running TTT — TTT includes it)
    if not cli.ttt and not cli.ttt_all:
        print("\n=== Sliding Window Eval ===")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        slide_loss, slide_bpb = eval_val_sliding(
            args, model, 0, 1, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=cli.stride, batch_seqs=cli.batch_seqs,
        )
        torch.cuda.synchronize()
        print(f"val_loss:{slide_loss:.6f} val_bpb:{slide_bpb:.6f} time:{time.perf_counter()-t0:.1f}s")

    # TTT frozen ternary
    if cli.ttt:
        pre_sd = {k: v.clone() for k, v in model.state_dict().items()}
        print("\n=== TTT (frozen ternary) ===")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttt_loss, ttt_bpb = ttt_and_eval_sliding(
            args, model, 0, 1, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=cli.stride, batch_seqs=cli.batch_seqs, freeze_ternary=True,
        )
        torch.cuda.synchronize()
        print(f"val_loss:{ttt_loss:.6f} val_bpb:{ttt_bpb:.6f} time:{time.perf_counter()-t0:.1f}s")
        model.load_state_dict(pre_sd)

    # TTT all params
    if cli.ttt_all:
        pre_sd = {k: v.clone() for k, v in model.state_dict().items()}
        print("\n=== TTT (all params) ===")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ttt_loss, ttt_bpb = ttt_and_eval_sliding(
            args, model, 0, 1, device, val_tokens,
            base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
            stride=cli.stride, batch_seqs=cli.batch_seqs, freeze_ternary=False,
        )
        torch.cuda.synchronize()
        print(f"val_loss:{ttt_loss:.6f} val_bpb:{ttt_bpb:.6f} time:{time.perf_counter()-t0:.1f}s")
        model.load_state_dict(pre_sd)


if __name__ == "__main__":
    main()
