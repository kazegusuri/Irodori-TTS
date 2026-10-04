#!/usr/bin/env python3
"""compile あり/なしで large-bf16 のカーネル内訳を取る。

使い方: IRODORI_COMPILE_MODE=... python bench/profile_dit2.py [--compile]
CUDA Graphs 経路は replay が 1 カーネルに見えるので、内訳を見る用途では
max-autotune-no-cudagraphs を使う。
"""
import os, sys
from collections import defaultdict
import torch
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texts import TEXTS

COMPILE = "--compile" in sys.argv
TEXT = dict(TEXTS)["t3_long"]
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large"),
    model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="bf16", codec_device="cuda", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=COMPILE, compile_dynamic=False))
req = lambda: SamplingRequest(text=TEXT, no_ref=True, num_steps=40, cfg_scale_text=3.0, seed=0)
for _ in range(2 if COMPILE else 1):
    rt.synthesize(req(), log_fn=None)
torch.cuda.synchronize()

import time
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    torch.cuda.synchronize()
    _t0 = time.perf_counter()
    rt.synthesize(req(), log_fn=None)
    torch.cuda.synchronize()
    _wall = time.perf_counter() - _t0

# 実カーネルのみ (aten:: などの親イベントを除外) で集計する
kernels = [e for e in prof.key_averages()
           if e.device_time_total > 0 and "::" not in e.key and not e.key.startswith("cuda")]
total = sum(e.device_time_total for e in kernels)
print(f"compile={COMPILE} mode={os.environ.get('IRODORI_COMPILE_MODE','default')!r}")
print(f"kernel 合計 GPU 時間: {total/1000:.1f} ms")
print(f"プロファイル中の wall: {_wall*1000:.1f} ms -> GPU busy {100*total/1000/(_wall*1000):.1f}%\n")
print(f"{'kernel':62} {'ms':>8} {'%':>6} {'calls':>7}")
for e in sorted(kernels, key=lambda x: -x.device_time_total)[:18]:
    print(f"{e.key[:62]:62} {e.device_time_total/1000:8.1f} {100*e.device_time_total/total:6.1f} {e.count:7d}")

cats = defaultdict(lambda: [0.0, 0])
for e in kernels:
    k = e.key.lower()
    if "triton_tem" in k or "cutlass_kernel_gemm" in k or "nvjet" in k or "s16816" in k or (
        "gemm" in k and "implicit" not in k
    ):
        c = "GEMM"          # triton_tem_* は inductor の GEMM テンプレート
    elif "flash_attn" in k or "flashattent" in k:
        c = "attention (FA4)"
    elif "implicit_gemm" in k or "fprop" in k or "sin_pow" in k or "snake" in k:
        c = "codec (conv/activation)"
    elif k.startswith("triton_poi") or k.startswith("triton_red") or k.startswith(
        "triton_per"
    ) or k.startswith("triton_for"):
        c = "elementwise (inductor 融合済み)"
    else:
        c = "other"
    cats[c][0] += e.device_time_total
    cats[c][1] += e.count
print()
for c, (v, n) in sorted(cats.items(), key=lambda x: -x[1][0]):
    print(f"{c:36} {v/1000:8.1f} ms  {100*v/total:5.1f}%  ({n} calls)")
print(f"\n異なるカーネル数: {len(kernels)}")
other = [e for e in kernels if e.key.lower() not in ()]
print("other の内訳 (上位):")
for e in sorted(kernels, key=lambda x: -x.device_time_total):
    k = e.key.lower()
    if not ("triton_tem" in k or "flash" in k or "implicit_gemm" in k or "fprop" in k
            or "sin_pow" in k or k.startswith(("triton_poi","triton_red","triton_per","triton_for"))):
        print(f"   {e.key[:58]:58} {e.device_time_total/1000:7.1f} ms {e.count:6d}")
