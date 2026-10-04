#!/usr/bin/env python3
"""DiT の実 GEMM 形状で達成 TFLOPS を測り、理論ピークと比べる。"""
import torch, time

def bench(m, k, n, dtype=torch.bfloat16, iters=200):
    a = torch.randn(m, k, device="cuda", dtype=dtype)
    b = torch.randn(k, n, device="cuda", dtype=dtype)
    for _ in range(20): a @ b
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): a @ b
    torch.cuda.synchronize()
    sec = (time.perf_counter() - t0) / iters
    return 2 * m * k * n / sec / 1e12, sec * 1000

print(f"{'shape (m,k,n)':28} {'TFLOPS':>8} {'ms':>8}  用途")
cases = [
    ((1396, 2048, 2048), "attn 投影 (batch2, seq698)"),
    ((698, 2048, 2048),  "attn 投影 (batch1)"),
    ((1396, 2048, 5888), "MLP up (SwiGLU)"),
    ((1396, 5888, 2048), "MLP down"),
    ((8192, 8192, 8192), "大きい正方行列 (ピーク参考)"),
]
for (m, k, n), note in cases:
    tf, ms = bench(m, k, n)
    print(f"{str((m,k,n)):28} {tf:8.1f} {ms:8.3f}  {note}")
print()
print("device:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
