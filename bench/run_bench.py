#!/usr/bin/env python3
"""Irodori-TTS のモデル間速度/品質比較ベンチ。

同一テキスト・同一シードで各モデル構成を回し、生成時間と RTF を記録する。
音声は outputs/<text_id>/<config>.wav に保存するので、品質は人手で聴き比べる。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from irodori_tts.inference_runtime import (
    InferenceRuntime,
    RuntimeKey,
    SamplingRequest,
    download_hf_checkpoint,
    save_wav,
)

from texts import TEXTS

BENCH_DIR = Path(__file__).resolve().parent

# steps=None はチェックポイント既定 (RF=40, MeanFlow=4) を使う
CONFIGS = [
    {"id": "small-v4.1-bf16", "repo": "Aratako/Irodori-TTS-v4.1-Small", "precision": "bf16"},
    {"id": "small-v4.1-fp32", "repo": "Aratako/Irodori-TTS-v4.1-Small", "precision": "fp32"},
    {"id": "small-mf-bf16", "repo": "Aratako/Irodori-TTS-v4.1-Small-MF", "precision": "bf16",
     "steps": None},
    {"id": "small-mf-fp32", "repo": "Aratako/Irodori-TTS-v4.1-Small-MF", "precision": "fp32",
     "steps": None},
    {"id": "large-bf16", "repo": "Aratako/Irodori-TTS-v4-Large", "precision": "bf16"},
    {"id": "large-fp32", "repo": "Aratako/Irodori-TTS-v4-Large", "precision": "fp32"},
    {"id": "large-int8wo",
     "repo": "Aratako/Irodori-TTS-v4-Large-Quantized/int8-weight-only", "precision": "bf16"},
    {"id": "large-int8dyn",
     "repo": "Aratako/Irodori-TTS-v4-Large-Quantized/int8-dynamic", "precision": "bf16"},
    {"id": "large-int4wo",
     "repo": "Aratako/Irodori-TTS-v4-Large-Quantized/int4-weight-only", "precision": "bf16"},
    {"id": "large-fp8wo",
     "repo": "Aratako/Irodori-TTS-v4-Large-Quantized/float8-weight-only", "precision": "bf16"},
    {"id": "large-fp8dyn",
     "repo": "Aratako/Irodori-TTS-v4-Large-Quantized/float8-dynamic", "precision": "bf16"},
]

# compile 有効時は形状違いの再コンパイルを計測から外すため、
# 長さの異なる 2 文でウォームアップする
WARMUP_TEXTS = [
    "ウォームアップ用の短い文章です。",
    "こちらはウォームアップ用のやや長い文章で、"
    "異なる系列長でのコンパイルを事前に済ませるために読み上げます。",
]
def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def build_request(
    text: str, *, seed: int, num_steps: int | None, context_kv_cache: bool = True
) -> SamplingRequest:
    return SamplingRequest(
        text=text,
        no_ref=True,
        context_kv_cache=context_kv_cache,
        num_steps=num_steps,
        cfg_scale_text=3.0,
        cfg_scale_caption=3.0,
        cfg_scale_speaker=5.0,
        seed=seed,
    )


def run_config(
    config: dict,
    *,
    seed: int,
    num_steps: int | None,
    repeats: int,
    compile_model: bool,
    compile_dynamic: bool,
    context_kv_cache: bool,
    out_root: Path,
) -> list[dict]:
    config_id = str(config["id"])
    repo_id = str(config["repo"])
    precision = str(config["precision"])
    # config に steps キーがあればそれを優先 (MeanFlow はチェックポイント既定の 4 steps)
    steps = config["steps"] if "steps" in config else num_steps
    suffix = "-compile" if compile_model else ""
    run_id = f"{config_id}{suffix}"

    print(f"\n==== {run_id} ({repo_id}, {precision}, steps={steps}) ====", flush=True)
    t0 = time.perf_counter()
    checkpoint_path = download_hf_checkpoint(repo_id)
    download_sec = time.perf_counter() - t0

    t0 = time.perf_counter()
    runtime = InferenceRuntime.from_key(
        RuntimeKey(
            checkpoint=checkpoint_path,
            model_device="cuda",
            codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
            model_precision=precision,
            codec_device="cuda",
            codec_precision="fp32",
            codec_deterministic_encode=True,
            codec_deterministic_decode=True,
            compile_model=compile_model,
            compile_dynamic=compile_dynamic,
        )
    )
    load_sec = time.perf_counter() - t0
    print(f"[load] download={download_sec:.1f}s init={load_sec:.1f}s", flush=True)

    # ウォームアップ (カーネル初期化と torch.compile を計測から外す)
    t0 = time.perf_counter()
    for warm in WARMUP_TEXTS:
        runtime.synthesize(build_request(warm, seed=seed, num_steps=steps, context_kv_cache=context_kv_cache), log_fn=None)
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - t0
    print(f"[warmup] {warmup_sec:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()

    rows: list[dict] = []
    for text_id, text in TEXTS:
        out_dir = out_root / text_id
        out_dir.mkdir(parents=True, exist_ok=True)
        times = []
        result = None
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = runtime.synthesize(
                build_request(text, seed=seed, num_steps=steps, context_kv_cache=context_kv_cache), log_fn=None
            )
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        assert result is not None
        save_wav(out_dir / f"{run_id}.wav", result.audio, result.sample_rate)

        audio_sec = result.audio.shape[-1] / float(result.sample_rate)
        best = min(times)
        row = {
            "config": run_id,
            "base_config": config_id,
            "repo": repo_id,
            "precision": precision,
            "compile": compile_model,
            "compile_dynamic": compile_dynamic if compile_model else False,
            "num_steps": steps,
            "text_id": text_id,
            "chars": len(text),
            "audio_sec": round(audio_sec, 3),
            "gen_sec_best": round(best, 3),
            "gen_sec_mean": round(sum(times) / len(times), 3),
            "gen_sec_first": round(times[0], 3),
            "rtf": round(best / audio_sec, 4),
            "xrealtime": round(audio_sec / best, 2),
            "stage_timings": {k: round(v, 4) for k, v in result.stage_timings},
            "seed": result.used_seed,
        }
        rows.append(row)
        print(
            f"[{run_id}] {text_id}: gen={best:.2f}s audio={audio_sec:.2f}s "
            f"RTF={row['rtf']:.3f} ({row['xrealtime']}x realtime)",
            flush=True,
        )

    peak = gpu_mem_mb()
    for row in rows:
        row["peak_vram_mb"] = round(peak, 1)
        row["load_sec"] = round(load_sec, 2)
        row["warmup_sec"] = round(warmup_sec, 2)
    print(f"[{run_id}] peak VRAM (torch allocated): {peak:.0f} MiB", flush=True)

    runtime.unload()
    del runtime
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=None, help="走らせる config id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=40, help="RF チェックポイントの steps")
    parser.add_argument("--compile", action="store_true", help="torch.compile を有効化")
    parser.add_argument(
        "--no-context-kv-cache",
        action="store_true",
        help="text/speaker の context K/V 事前計算を無効化",
    )
    parser.add_argument(
        "--compile-dynamic",
        action="store_true",
        help="torch.compile を dynamic=True で行い、系列長ごとの再コンパイルを避ける",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out-root", default=str(BENCH_DIR / "outputs"))
    parser.add_argument("--results", default=str(BENCH_DIR / "results.jsonl"))
    args = parser.parse_args()

    selected = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        selected = [c for c in CONFIGS if c["id"] in wanted]
        missing = wanted - {c["id"] for c in selected}
        if missing:
            raise SystemExit(f"unknown config id: {sorted(missing)}")

    out_root = Path(args.out_root)
    results_path = Path(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a", encoding="utf-8") as fh:
        for config in selected:
            config_id = config["id"]
            try:
                rows = run_config(
                    config,
                    seed=args.seed,
                    num_steps=args.num_steps,
                    repeats=args.repeats,
                    compile_model=bool(args.compile),
                    compile_dynamic=bool(args.compile_dynamic),
                    context_kv_cache=not bool(args.no_context_kv_cache),
                    out_root=out_root,
                )
            except Exception as exc:  # 1 構成の失敗で全体を止めない
                print(f"[{config_id}] FAILED: {type(exc).__name__}: {exc}", flush=True)
                fh.write(
                    json.dumps(
                        {
                            "config": config_id,
                            "compile": bool(args.compile),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    + "\n"
                )
                fh.flush()
                torch.cuda.empty_cache()
                continue
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()


if __name__ == "__main__":
    main()
