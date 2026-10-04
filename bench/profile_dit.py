#!/usr/bin/env python3
"""large-bf16 の sample_rf 内訳を kernel 単位で取る。"""
import os, sys, json
from collections import defaultdict
import torch
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texts import TEXTS

TEXT = dict(TEXTS)["t3_long"]
ckpt = download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large")
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=ckpt, model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="bf16", codec_device="cuda", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=False, compile_dynamic=False))
req = lambda: SamplingRequest(text=TEXT, no_ref=True, num_steps=40, cfg_scale_text=3.0, seed=0)
rt.synthesize(req(), log_fn=None)   # warmup
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    rt.synthesize(req(), log_fn=None)
    torch.cuda.synchronize()

evs = [e for e in prof.key_averages() if e.device_time_total > 0]
total = sum(e.device_time_total for e in evs)
print(f"total GPU time: {total/1000:.1f} ms\n")
print(f"{'op':48} {'GPU ms':>9} {'%':>6} {'calls':>7}")
for e in sorted(evs, key=lambda x: -x.device_time_total)[:22]:
    print(f"{e.key[:48]:48} {e.device_time_total/1000:9.1f} {100*e.device_time_total/total:6.1f} {e.count:7d}")

# カテゴリ集約
cats = defaultdict(float)
for e in evs:
    k = e.key.lower()
    if "gemm" in k or "addmm" in k or "matmul" in k or "linear" in k or "mm" == k or "bmm" in k:
        c = "GEMM (Linear/matmul)"
    elif "attention" in k or "sdpa" in k or "flash" in k or "cute" in k:
        c = "attention"
    elif any(t in k for t in ("mul","add","silu","sigmoid","rsqrt","norm","cat","copy","slice","index","where","to_","sum","mean","stack","chunk","clone","zero","fill","pad","transpose","reshape","view","contiguous","neg","cos","sin","div","sub","pow","sqrt","exp")):
        c = "elementwise / norm / reshape 等"
    else:
        c = "other"
    cats[c] += e.device_time_total
print()
for c, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"{c:36} {v/1000:8.1f} ms  {100*v/total:5.1f}%")

# GEMM の形状を見る
print("\n=== 主要 GEMM 形状 (上位) ===")
shaped = [e for e in prof.key_averages(group_by_input_shape=True)
          if e.device_time_total > 0 and ("mm" in e.key.lower() or "linear" in e.key.lower())]
for e in sorted(shaped, key=lambda x: -x.device_time_total)[:12]:
    print(f"{e.key[:26]:26} {str(e.input_shapes)[:72]:72} {e.device_time_total/1000:8.1f} ms {e.count:6d}")
