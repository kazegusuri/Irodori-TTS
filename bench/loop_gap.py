#!/usr/bin/env python3
"""40 ステップのうち、モデル呼び出し内の GPU 時間と呼び出し間の隙間を分離する。

forward_with_encoded_conditions を CUDA event で挟み、
sample_rf 全体の時間と比較する。
"""
import os, sys
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
req = lambda: SamplingRequest(text=TEXT, no_ref=True, num_steps=40,
                              cfg_scale_text=3.0, seed=0)
for _ in range(3):
    rt.synthesize(req(), log_fn=None)       # warmup / compile

inner = rt.model.forward_with_encoded_conditions
events = []

def timed(*a, **kw):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    out = inner(*a, **kw)
    e.record()
    events.append((s, e))
    return out

rt.model.forward_with_encoded_conditions = timed
res = rt.synthesize(req(), log_fn=None)
torch.cuda.synchronize()
rt.model.forward_with_encoded_conditions = inner

sample_ms = dict(res.stage_timings)["sample_rf"] * 1000
in_call = sum(s.elapsed_time(e) for s, e in events)
span = events[0][0].elapsed_time(events[-1][1])
print(f"モデル呼び出し回数: {len(events)}")
print(f"sample_rf 全体:            {sample_ms:8.1f} ms")
print(f"  最初の呼び出し開始〜最後の終了: {span:8.1f} ms")
print(f"  呼び出し内の GPU 時間 合計:    {in_call:8.1f} ms  ({100*in_call/sample_ms:.1f}% of sample_rf)")
print(f"  呼び出し間の隙間:              {span - in_call:8.1f} ms  ({100*(span-in_call)/sample_ms:.1f}%)")
print(f"  ループ外 (エンコード等):        {sample_ms - span:8.1f} ms  ({100*(sample_ms-span)/sample_ms:.1f}%)")
print(f"\n1 呼び出しあたり: 内側 {in_call/len(events):.2f} ms / 隙間 {(span-in_call)/len(events):.2f} ms")
