#!/usr/bin/env python3
"""compile 済みモデル呼び出しを連続実行し、純粋な GPU 時間を測る。

sample_rf で観測した 1 呼び出し 18.7 ms が本当に GPU 計算なのか、
呼び出しごとのホスト側オーバーヘッドを含むのかを判別する。
"""
import os, sys
import torch
from irodori_tts.inference_runtime import (
    InferenceRuntime, RuntimeKey, SamplingRequest, download_hf_checkpoint,
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from texts import TEXTS

rt = InferenceRuntime.from_key(RuntimeKey(
    checkpoint=download_hf_checkpoint("Aratako/Irodori-TTS-v4-Large"),
    model_device="cuda", codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
    model_precision="bf16", codec_device="cuda", codec_precision="fp32",
    codec_deterministic_encode=True, codec_deterministic_decode=True,
    compile_model=True, compile_dynamic=False))
TEXT = dict(TEXTS)["t3_long"]
req = lambda: SamplingRequest(text=TEXT, no_ref=True, num_steps=40,
                              cfg_scale_text=3.0, seed=0)
for _ in range(3):
    rt.synthesize(req(), log_fn=None)

# CFG 有効時 (batch 2) と無効時 (batch 1) の引数をそれぞれ捕まえる
inner = rt.model.forward_with_encoded_conditions
captured = {}
def grab(*a, **kw):
    b = kw["x_t"].shape[0]
    captured.setdefault(b, dict(kw))
    return inner(*a, **kw)
rt.model.forward_with_encoded_conditions = grab
rt.synthesize(req(), log_fn=None)
rt.model.forward_with_encoded_conditions = inner
torch.cuda.synchronize()

torch.set_grad_enabled(False)
for b, kw in sorted(captured.items()):
    with torch.inference_mode():
        for _ in range(3):
            inner(**kw)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    N = 20
    s.record()
    with torch.inference_mode():
        for _ in range(N):
            inner(**kw)
    e.record()
    torch.cuda.synchronize()
    print(f"batch={b}: 連続 {N} 回の GPU 時間 {s.elapsed_time(e):7.1f} ms "
          f"-> 1 回 {s.elapsed_time(e)/N:6.2f} ms")
print("\nsample_rf 側の実測は 1 呼び出し 18.70 ms (batch2/batch1 が各 20 回の平均)")
