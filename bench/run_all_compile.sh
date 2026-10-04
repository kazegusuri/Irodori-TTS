#!/usr/bin/env bash
# compile あり全構成スイープ。CUDA エラーで 1 構成が落ちても他に波及しないよう
# 構成ごとに別プロセスで実行する。
set -u
cd "$(dirname "$0")/.."
CONFIGS="small-v4.1-bf16 small-v4.1-fp32 small-mf-bf16 small-mf-fp32 large-bf16 large-fp32 large-int8wo large-int8dyn large-int4wo large-fp8wo large-fp8dyn"
for c in $CONFIGS; do
  echo "######## $c"
  PYTHONPATH=.:bench uv run --no-sync python bench/run_bench.py \
    --compile --configs "$c" --results bench/results_compile.jsonl 2>&1 \
    | tr '\r' '\n' | grep -a -E "^\[|====|FAILED"
  echo "######## $c exit=$?"
done
