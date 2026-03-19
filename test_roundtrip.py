#!/usr/bin/env python3
"""Test ternary roundtrip on a small trained model. Run on GPU."""
import os; os.environ['TORCH_COMPILE_DISABLE'] = '1'
import torch, sys; sys.path.insert(0, '.')
from train_bitnet import BitLinear, GPT, quantize_state_dict_ternary, dequantize_state_dict_ternary

device = 'cuda'
model = GPT(vocab_size=1024, num_layers=4, model_dim=256, num_heads=4,
    num_kv_heads=2, mlp_mult=2, tie_embeddings=True,
    tied_embed_init_std=0.005, logit_softcap=30.0, rope_base=10000.0,
    qk_gain_init=1.5).bfloat16().to(device)

# Train briefly
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(100):
    x = torch.randint(0, 1024, (4, 128), device=device)
    y = torch.randint(0, 1024, (4, 128), device=device)
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 25 == 0: print(f'train step {step} loss={loss.item():.4f}')

# Fixed eval batch
model.eval()
x = torch.randint(0, 1024, (4, 128), device=device)
y = torch.randint(0, 1024, (4, 128), device=device)

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss_pre = model(x, y)
print(f'\nPre-quant loss: {loss_pre.item():.6f}')

# Populate cache
with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        model(x, y)

# Quantize + dequantize roundtrip
tern_obj, stats = quantize_state_dict_ternary(model.state_dict(), model=model)
sd = dequantize_state_dict_ternary(tern_obj, target_dtype=torch.bfloat16)
model.load_state_dict(sd, strict=True)
for mod in model.modules():
    if isinstance(mod, BitLinear):
        mod._skip_quantize = True

with torch.no_grad():
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        loss_post = model(x, y)
print(f'Post-roundtrip loss: {loss_post.item():.6f}')
print(f'Gap: {abs(loss_post.item() - loss_pre.item()):.8f}')
print(f'\n{"PASS" if abs(loss_post.item() - loss_pre.item()) < 0.001 else "FAIL"}: roundtrip gap')
