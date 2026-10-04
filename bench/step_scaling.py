#!/usr/bin/env python3
"""sample_rf をステップ数に対して測り、固定費と 1 ステップあたりの費用を分離する。"""
import os, sys, time
import torch
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texts import TEXTS

COMPILE = "--compile" in sys.argv
rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large"),
    model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="bf16", codec_device="cuda", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=COMPILE, compile_dynamic=False))
TEXT = dict(TEXTS)["t3_long"]
def run(n):
    r = rt.synthesize(SamplingRequest(text=TEXT, no_ref=True, num_steps=n,
                                      cfg_scale_text=3.0, seed=0), log_fn=None)
    return dict(r.stage_timings)["sample_rf"] * 1000

for n in (1, 40):          # warmup (compile 済ませる)
    run(n); run(n)
print(f"{'steps':>6} {'sample_rf':>11} {'per step':>10}")
res = {}
for n in (1, 2, 4, 8, 16, 40):
    ts = [run(n) for _ in range(3)]
    res[n] = min(ts)
    print(f"{n:6d} {res[n]:10.1f}ms {res[n]/n:9.2f}ms")
# 1 次近似で固定費を出す
slope = (res[40] - res[4]) / 36
fixed = res[40] - slope * 40
print(f"\n1 ステップあたり: {slope:.2f} ms")
print(f"ステップ数に依存しない固定費: {fixed:.1f} ms  (40 steps 中 {100*fixed/res[40]:.1f}%)")
