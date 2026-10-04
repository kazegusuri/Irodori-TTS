# Irodori-TTS v4 モデル比較ベンチ

RTX 5090 (32GB, sm_120) 上で v4.1-Small / v4.1-Small-MF (MeanFlow) / v4-Large /
v4-Large 量子化 5 種を、eager と torch.compile の両方で比較。

## 条件
- `--no-ref`（リファレンス音声なし・テキストのみ）、caption なし
- RF チェックポイントは 40 steps / text CFG 3.0、MeanFlow はチェックポイント既定の 4 steps
  （MeanFlow では CFG は蒸留時に織り込み済みで推論時には使われない）
- seed 0 固定、codec は常に fp32、model precision のみ変更
- 各テキスト 3 回生成してベスト値。別途ウォームアップ 2 文（長さ違い）を計測から除外
- 比較テキストは `texts.py` の 6 パターン

## 実行方法
```bash
cd ..          # irodori-tts リポジトリルート
PYTHONPATH=.:bench uv run --no-sync python bench/run_bench.py                 # eager 全構成
PYTHONPATH=.:bench uv run --no-sync python bench/run_bench.py --configs large-bf16
bench/run_all_compile.sh                                                      # compile 全構成
```

`--compile-dynamic` は **使わないこと**。v4.1-Small bf16 の 2 文目で
`CUDA error: an illegal memory access was encountered` が出てプロセスごと落ちる。
静的 compile（`--compile` のみ）は全構成で安定した。CUDA エラーはコンテキストを
壊してプロセス内リトライが効かないため、compile スイープは
`run_all_compile.sh` が構成ごとにプロセスを分けている。

## 聴き比べ
`compare.html` をブラウザで開く。テキストごとにプレイヤー1つ＋モデル切り替えボタンで、
再生位置を保ったまま A/B できる。生 wav は `outputs/<text_id>/<config>.wav`。
compile 版は音声が同一なのでページには載せていない（wav は残してある）。

## 結果（生成時間 秒 / 3 回ベスト、eager → compile）

| config | t1_short | t2_medium | t3_long | t4_numeric | t5_kanji | t6_emotion | 合計 | peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| small-mf-bf16 | 0.06→0.04 | 0.09→0.07 | 0.18→0.17 | 0.11→0.10 | 0.10→0.09 | 0.09→0.08 | **0.62→0.54** | 5.2 GB |
| small-mf-fp32 | 0.08→0.07 | 0.12→0.12 | 0.23→0.23 | 0.15→0.14 | 0.13→0.13 | 0.12→0.11 | 0.83→0.80 | 6.7 GB |
| small-v4.1-bf16 | 0.26→0.14 | 0.29→0.28 | 0.45→0.44 | 0.32→0.31 | 0.30→0.29 | 0.28→0.28 | 1.90→1.75 | 5.2 GB |
| small-v4.1-fp32 | 0.34→0.31 | 0.51→0.51 | 0.92→0.92 | 0.62→0.61 | 0.56→0.55 | 0.49→0.48 | 3.44→3.39 | 6.6 GB |
| large-bf16 | 0.54→0.35 | 0.69→0.67 | 1.46→1.43 | 0.89→0.85 | 0.70→0.67 | 0.61→0.58 | **4.88→4.54** | 10.1 GB |
| large-fp32 | 0.96→0.88 | 1.43→1.41 | 3.04→3.03 | 1.82→1.81 | 1.53→1.51 | 1.36→1.34 | 10.15→9.99 | 16.3 GB |
| large-int8wo | 0.67→11.69※ | 0.85→2.40 | 1.65→3.18 | 1.05→2.59 | 0.86→2.40 | 0.76→2.30 | 5.83→24.56 | 7.5 GB |
| large-fp8wo | 0.89→0.46 | 1.11→1.07 | 1.88→1.83 | 1.30→1.25 | 1.12→1.06 | 1.02→0.97 | 7.32→6.64 | 7.5 GB |
| large-int4wo | 0.97→0.90 | 1.78→1.75 | 4.36→4.32 | 2.44→2.42 | 1.94→1.94 | 1.46→1.45 | 12.95→12.77 | 6.6 GB |
| large-fp8dyn | 1.89→0.41 | 1.93→1.79 | 2.01→1.87 | 1.94→1.80 | 1.93→1.78 | 1.91→1.78 | 11.61→9.43 | 7.5 GB |
| large-int8dyn | 5.46→✗ | 5.50→✗ | 5.81→✗ | 5.56→✗ | 5.53→✗ | 5.50→✗ | 33.36→✗ | 7.5 GB |

- ✗ = compile 時に `RuntimeError: self.size(0) needs to be greater than 16, but got 1`
  （torchao の INT8 dynamic カーネルがバッチ 1 を受け付けない）
- ※ = 3 回とも再コンパイルが走っており計測として信用できない。
  他のテキストの 2.3〜3.2s が int8wo + compile の実力

## 所見

### MeanFlow が圧勝
4 steps で済むため、小さい方のモデルで **RF の 3〜4 倍速**。長文で 156 倍速（RTF 0.006）、
compile 込みなら 0.17s で 27 秒の音声が出る。VRAM も RF 版と同じ。
速度を求めるならこれが本命で、量子化で削る話とは桁が違う。
ただし公開されているのは Small のみ（`Aratako/Irodori-TTS-v4.1-Small-MF`）。
Large の MF が欲しければ `configs/train_v4_small_meanflow.yaml` を元に自前で蒸留する必要がある。

### torch.compile はほぼ効かない
長文では軒並み 1〜3% 改善どまり。効くのは最短文（t1）だけで、large-bf16 が 0.54→0.35s、
fp8dyn が 1.89→0.41s。要するにカーネル起動オーバーヘッドが支配的な領域しか救えず、
計算が詰まっている長文では利得がない。
一方コストは大きく、ウォームアップに 52〜237 秒かかるうえ、
系列長ごとに再コンパイルが走る（初回生成が 37〜55 秒になることがある）。
**常駐サーバで同じ長さを繰り返し投げる用途以外では割に合わない。**

### 量子化は compile でも救われなかった
当初「compile なしでは量子化に不利」と見ていたが、compile しても bf16 を超えたものはゼロ。
int8wo に至っては compile で 3 倍遅くなり、int8dyn は compile 自体が通らない。
5090 で量子化を使う理由は VRAM 削減のみ（int4wo で 6.6GB、bf16 比 -3.5GB）。

### 実運用の推奨
- 速度優先・Small で足りる → **small-mf-bf16**（compile なしで十分、0.04〜0.18s）
- 品質優先（漢字の読みは Large が良いという実聴評価）→ **large-bf16**
  （fp32 比 2.1 倍速・VRAM -6GB、compile は不要）
- VRAM を 7GB 以下に抑えたい → large-int4wo か large-fp8wo

## FlashAttention と compile mode の検証（Large 高速化、精度維持前提）

### FlashAttention は 4 のみ 5090 で動く
- **FA3 は不可**。公式 README の要件が「H100 / H800」で、sm_90a 専用。
  PyPI の `flash-attn-3` は 1KB の中身なしスタブ（0.0.0、yanked 済みの placeholder）で、
  本物は GitHub の `hopper/` からビルドする必要がある。
  リポジトリ側も `attention.py:51` の probe で
  "Blackwell with an SM90-only build" を想定して黙って SDPA に落とす作りになっている。
- **FA4 は動く**。CuTeDSL 実装で `flash_fwd_sm120` / `flash_bwd_sm120` を同梱しており、
  sm_120 に実際にディスパッチされる。純 Python wheel なのでコンパイル不要:

```bash
uv pip install --prerelease=allow flash-attn-4     # 4.0.0b33
```

  ただし FA4 のモジュール名と引数順は FA3 と異なるため、本リポジトリの
  FA3 専用統合には **パッチが必要**（`irodori_tts/attention.py` に適用済み）:
  - import 元が top-level `flash_attn_interface` ではなく `flash_attn.cute`
  - varlen のキーワード名は FA3 と同じだが、第4引数が `qv` なので
    `cu_seqlens_*` / `max_seqlen_*` をキーワードで渡さないと誤爆する

  有効化の確認は `python -c "from irodori_tts import attention as A; print(A._FA_IMPL)"`、
  無効化は `IRODORI_ATTENTION_BACKEND=sdpa`。

### compile mode
`irodori_tts/inference_runtime.py` は `mode` を渡していなかったので
`IRODORI_COMPILE_MODE` で上書きできるようにした（パッチ適用済み）。

- `max-autotune` / `reduce-overhead` は**どちらも失敗する**:
  `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
  by a subsequent run`（`model.py:910` の `_rope_freqs` がキャッシュしたテンソルを
  再利用するため CUDA Graphs と非互換）
- `max-autotune-no-cudagraphs` は動作する

### 結果（large-bf16, 生成時間 秒）

| 構成 | t1_short | t3_long |
|---|---:|---:|
| eager + SDPA（基準） | 0.54 | 1.456 |
| eager + FA4 | 0.54 | 1.400 |
| compile(default) + SDPA | 0.35 | 1.425 |
| compile(max-autotune-no-cudagraphs) + FA4 | 0.52 | **1.380** |

t3_long で基準比 **5%** 改善。t1_short の compile 値は毎回再コンパイルが走るため不安定で、
信頼できるのは t3_long 側。

### 重要: ビット一致は最初から成立しない
同じ seed でも attention カーネルを変えると出力波形は変わる。
FA4 と SDPA の large-bf16 出力を比較すると t3_long で **RMS 差 0.107**（長さは同一）。
bf16 の微小差が 40 ステップの Euler 積分で増幅され、別の（等価に妥当な）サンプルに着地する。
したがって「精度を落とさない」は波形 diff では検証できず、聴感か統計的評価で見るしかない。
TensorRT 化でも同じことが起きる。

### 結論
bf16 のまま Large を速くする余地は小さい。attention を最速カーネルに替えて 3〜4%、
autotune を足して計 5%。DiT は GEMM で詰まっており、すでにハードウェアの素の性能に近い。
TensorRT の主な武器（fusion・起動オーバーヘッド削減・低精度化）のうち、
低精度化を除外すると残りは同じ 5〜15% のレンジで、工数に見合わない可能性が高い。
構造的に速くするなら MeanFlow（ステップ数削減）の方向しかない。

## torch / CUDA 更新の検証（cu128 → cu132）

ドライバは 595.91.07（CUDA 13.2 対応）。**ドライバ更新は不要**で、wheel 側だけで CUDA 13.2 に上げられる。
PyTorch が配布している中では **cu132 + torch 2.14.1** が最新（cu134 の torch wheel は未提供、
cu128 は torch 2.11.0 止まり）。

### GEMM 単体効率（DiT の実形状）

| 形状 (m,k,n) | 2.10+cu128 | 2.14.1+cu132 | 改善 |
|---|---:|---:|---:|
| (1396, 2048, 2048) | 127.8 | 166.1 | +30% |
| (698, 2048, 2048) | 91.1 | 128.2 | +41% |
| (1396, 2048, 5888) | 186.4 | 235.9 | +27% |
| (1396, 5888, 2048) | 130.5 | 195.7 | +50% |
| (8192³) 参考 | 242.1 | 233.9 | -3% |

大きい正方行列は変わらず、**DiT が使う細長い形状だけ** 27〜50% 改善。
cu128 が sm_120 に最適でないカーネルを選んでいた。

### サンプラー実時間（sample_rf / sample_meanflow）

| config / text | 2.10+cu128 | 2.14.1+cu132 | 改善 |
|---|---:|---:|---:|
| large-bf16 / t3_long | 1297.6 ms | **1096.4 ms** | **15.5%** |
| large-bf16 / t2_medium | 619.2 ms | 556.8 ms | 10.1% |
| large-bf16 / t1_short | 489.3 ms | 477.3 ms | 2.5% |
| small-v4.1-bf16 / t3_long | 308.2 ms | 308.1 ms | 0.0% |
| small-mf-bf16 / t3_long | 35.0 ms | 38.1 ms | -8.9% |

GEMM bound な Large の長文ほど効き、起動オーバーヘッド bound な Small / MeanFlow には効かない
（MF は僅かに悪化）。**この更新は Large 専用の施策**。

### 環境構築で踏んだ問題（すべて要対処）

1. **torchaudio が 2.11.0 で開発終了**。cu132 向け x86_64 wheel は存在しない。
   CPU ビルド（`torchaudio==2.11.0+cpu`）なら CUDA バージョン検査を通過して動く
   （検査は `torch.ops._torchaudio.cuda_version()` が None なら skip される）。
2. **`dacvae` → `audiotools` → `torchaudio` と連鎖**しているので、
   torchaudio を単に消すとコーデックが死ぬ。CPU ビルドを入れるのが現状の解。
3. **transformers は 5.12.1 に固定が必要**。5.18.0 だと Small 系が
   `Could not import module 'ModernBertModel'` で落ちる（リポジトリの制約は `<6` なので
   そのままでは 5.18 が入ってしまう）。
4. **silentcipher が torchaudio を module scope で import** するため、
   cu132 環境では電子透かしが無効になる。`watermark.py` の except が `ImportError` のみで
   依存側の `RuntimeError` を拾えていなかったので広げた
   （透かしを諦めない場合は torchaudio 問題を先に解く必要がある）。
5. `torchaudio.save` は 2.11 で torchcodec へ委譲するようになり、未インストールだと
   `ImportError`。soundfile フォールバックの except も広げた。

上記に対応するため、ローカルの clone に以下のパッチを入れた（いずれも上流に出す価値あり）:
- `irodori_tts/attention.py` — FA4 (`flash_attn.cute`) 対応、varlen をキーワード呼び出しに
- `irodori_tts/codec.py` / `inference_runtime.py` — torchaudio を optional 化、
  resample は soxr フォールバック、soundfile フォールバックの except 拡大
- `irodori_tts/watermark.py` — 読み込み失敗時の except 拡大
- `irodori_tts/inference_runtime.py` — `IRODORI_COMPILE_MODE` で compile mode を上書き

### cu132 環境の作り方
```bash
uv venv --python 3.10 venv-cu132
uv pip install --python venv-cu132 --index-url https://download.pytorch.org/whl/cu132 \
  "torch==2.14.1+cu132" "torchao==0.18.0+cu132"
uv pip install --python venv-cu132 --index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match \
  numpy soundfile soxr sentencepiece "safetensors>=0.7.0" "transformers==5.12.1" \
  "huggingface-hub>=0.34.0" "peft>=0.18.0" tqdm pyyaml numba llvmlite \
  "dacvae @ git+https://github.com/facebookresearch/dacvae" \
  "silentcipher @ git+https://github.com/SesameAILabs/silentcipher.git@d46d7d0893a583d8968ab3a6626e2289faec9152"
uv pip install --python venv-cu132 --index-url https://download.pytorch.org/whl/cpu "torchaudio==2.11.0+cpu"
```

## Large 高速化の最終結果（組み合わせ + CUDA Graphs）

生成時間（秒、3回ベスト、透かし処理の時間を差し引いて揃えた値）:

| setup | t1_short | t2_medium | t3_long | t4_numeric | t5_kanji | t6_emotion |
|---|---:|---:|---:|---:|---:|---:|
| A: cu128 eager SDPA（基準） | 0.53 | 0.68 | 1.41 | 0.86 | 0.68 | 0.59 |
| B: cu132 eager SDPA | 0.52 | 0.62 | 1.22 | 0.80 | 0.66 | 0.58 |
| C: cu132 + FA4 + max-autotune-no-cudagraphs | 0.42 | 0.49 | **1.12** | 0.77 | 0.66 | 0.61 |
| D: cu132 + SDPA + reduce-overhead (CUDA Graphs) | **0.34** | **0.46** | 1.19 | 0.78 | **0.63** | **0.56** |

基準比の改善率:

| setup | t1_short | t2_medium | t3_long | t4_numeric | t5_kanji | t6_emotion |
|---|---:|---:|---:|---:|---:|---:|
| B | 2.6% | 9.1% | 13.8% | 7.9% | 3.5% | 2.2% |
| C | 21.3% | 28.2% | **20.8%** | 10.9% | 2.8% | -1.9% |
| D | **36.2%** | **32.1%** | 15.5% | 9.7% | 6.7% | 6.4% |

**短文〜中文は D（CUDA Graphs）、長文は C（FA4 + autotune）が勝つ。**
短文はカーネル起動オーバーヘッド bound なので Graphs が効き、
長文は計算 bound なので FA4 と autotune のカーネル選択が効く、という素直な住み分け。
C と D は排他（下記）なので、用途に合わせてどちらかを選ぶ。

### CUDA Graphs を通すのに必要だった修正と、FA4 との排他性

`reduce-overhead` / `max-autotune` は 2 段階で詰まっていた。

1. **`model.py` の `_rope_freqs` が forward 内でバッファに代入していた**
   （3 箇所、`self._freqs_cis_cache = cache`）。CUDA Graph でキャプチャされたテンソルが
   バッファに残り、次の replay で上書きされて
   `accessing tensor output of CUDAGraphs that has been overwritten` になる。
   → `torch.compiler.is_compiling()` が真のときは代入をスキップして毎回再計算する修正を入れた。
2. **FA4 の CuTeDSL カスタム op が inductor の CUDA Graphs 管理と非互換**。
   `These storage data ptrs are not allocated in pool (0, 1)` で落ちる
   （エラー文の "or in a custom op's meta function" がそのまま該当）。
   `TORCHINDUCTOR_CUDAGRAPH_TREES=0` でも `--no-context-kv-cache` でも回避できず、
   `IRODORI_ATTENTION_BACKEND=sdpa` で FA4 を切ると通った。

つまり **FA4 と CUDA Graphs は現状併用できない**。両方を取るには FA4 側に
meta 関数 / aliasing 情報の登録が必要で、これは上流（flash-attn-4）の課題。

compile mode はホット関数（`forward_with_encoded_conditions`、1 リクエスト 40 回呼ばれる）
にのみ適用し、`encode_conditions` と `build_context_kv_cache` は既定モードのままにしている
（出力がステップ間で再利用されるため）。

### 推奨設定

```bash
# 短文中心（対話・チャット応答など）: CUDA Graphs
IRODORI_ATTENTION_BACKEND=sdpa IRODORI_COMPILE_MODE=reduce-overhead  # + --compile-model

# 長文中心（記事読み上げ・ナレーション）: FA4 + autotune
IRODORI_COMPILE_MODE=max-autotune-no-cudagraphs                      # + --compile-model
```

どちらもウォームアップに 100〜120 秒かかり、系列長ごとに再コンパイルが走るので、
常駐サーバ前提。単発 CLI 実行なら compile なし（B の cu132 eager）が妥当。

## FA4 と CUDA Graphs の併用、および 40 ステップループの最適化

### 結論: FA4 + CUDA Graphs は上流修正なしで併用できる

FA4 は CuTeDSL のカスタム op で、出力テンソルを torch が確保しないため inductor の
CUDA Graphs 管理に弾かれていた（`These storage data ptrs are not allocated in pool`）。
**torch 側で custom op として包み直し、fake kernel を登録して出力を clone すれば通る**
（`irodori_tts/attention.py`）。capture 中の clone は graph の private pool に落ちるため、
FA4 内部が確保したテンソルを inductor に見せずに済む。
clone のコストを eager で払わないよう、`torch.compiler.is_compiling()` が真のときだけ
custom op 経路を使う。

### 40 ステップループで見つかった無駄（2 件、どちらも出力はビット不変）

1. **`build_context_attention_plan` が毎ステップ呼ばれていた。**
   コンテキストマスクと潜在長はステップ間で不変（変わるのは `x_t` と `t` だけ）なのに、
   `forward_with_encoded_conditions` の中で毎回構築していた。
   しかもこの関数は `torch.nonzero` を含むため **device sync が発生**する（FA4 経路のとき）。
   → `TextToLatentRFDiT.build_context_plan()` としてメソッド化し、
   `rf.py` が条件バンドルごとに 1 回だけ構築して `context_plan=` で渡すようにした。
   40 回 → 1〜2 回。
2. **CFG 窓の判定で毎ステップ `t.item()` していた**（`rf.py`）。
   これも 40 回の device sync。スケジュールのホスト側コピーを 1 回作って参照するよう変更。

合計で 1 リクエストあたり最大 80 回の sync を削減。eager 単体では約 1% だが、
データ依存 op と sync がグラフ内から消えることで CUDA Graphs 経路が素直になる。
**同一環境で生成音声がビット一致することを確認済み**（t1/t2/t3/t6 で `np.array_equal` が True）。

### KV キャッシュは既に最適化済み

`build_context_kv_cache` は `rf.py:416-437`、つまりループ（`rf.py:487`）の**前**で
条件バンドルごとに 1 回だけ呼ばれる。ループ内では `context_kv_cache[i]` を
ブロックに渡すだけで、`model.py:358-365` が `context_kv is None` のときだけ
`project_context_kv` を実行する作りになっている。
**毎ステップの再投影やロードは起きていない。** ここは手を入れる余地なし。

### 最終結果（生成時間 秒、透かし分を差し引いて条件を揃えた値）

| setup | t1_short | t2_medium | t3_long | t4_numeric | t5_kanji | t6_emotion |
|---|---:|---:|---:|---:|---:|---:|
| A: cu128 eager SDPA（基準） | 0.53 | 0.68 | 1.41 | 0.86 | 0.68 | 0.59 |
| B: cu132 eager SDPA | 0.52 | 0.62 | 1.22 | 0.80 | 0.66 | 0.58 |
| C: cu132 FA4 + autotune-no-cudagraphs | 0.42 | 0.49 | 1.12 | 0.77 | 0.66 | 0.61 |
| D: cu132 SDPA + CUDA Graphs | 0.34 | 0.46 | 1.19 | 0.78 | 0.63 | 0.56 |
| E: cu132 FA4 + CUDA Graphs | 0.31 | 0.44 | 1.11 | 0.72 | 0.61 | 0.55 |
| **F: cu132 FA4 + max-autotune (CUDA Graphs 込み)** | **0.29** | **0.42** | **1.10** | **0.72** | **0.61** | **0.55** |

基準比の改善率:

| setup | t1_short | t2_medium | t3_long | t4_numeric | t5_kanji | t6_emotion |
|---|---:|---:|---:|---:|---:|---:|
| B | 2.6% | 9.1% | 13.8% | 7.9% | 3.5% | 2.2% |
| C | 21.3% | 28.2% | 20.8% | 10.9% | 2.8% | -1.9% |
| D | 36.2% | 32.1% | 15.5% | 9.7% | 6.7% | 6.4% |
| E | 42.6% | 35.7% | 21.6% | 16.8% | 10.1% | 7.7% |
| **F** | **44.8%** | **37.6%** | **21.8%** | **17.0%** | **10.4%** | **8.2%** |

F が全テキストで最良。短文で 45%、長文で 22% の改善。
F のウォームアップは 191 秒（autotune の探索込み）。

### 推奨設定（更新）

```bash
# 常駐サーバ: F
IRODORI_COMPILE_MODE=max-autotune   # + --compile-model、FA4 インストール済み
# 単発 CLI: compile なし (B)
```

## elementwise 最適化の検討 → 不要と判断

「elementwise が 26%」という数字は **eager 時代のプロファイル**で、compile 後は
inductor が融合済みだった。F 構成（cu132 + FA4 + max-autotune）で取り直した
カーネル内訳（large-bf16 / t3_long）:

| 種別 | GPU 時間 | 比率 | 呼び出し回数 |
|---|---:|---:|---:|
| GEMM（inductor の Triton テンプレート） | 464.1 ms | **72.4%** | 12,600 |
| elementwise（inductor 融合済み triton） | 57.8 ms | **9.0%** | 21,390 |
| attention (FA4 sm_120) | 50.9 ms | 7.9% | 988 |
| codec (conv/activation) | 48.9 ms | 7.6% | 56 |
| other (memcpy/memset 等) | 19.5 ms | 3.0% | 8,765 |

21,390 回呼ばれているのに 57.8 ms しかかかっておらず、1 カーネルあたり 2.7 µs。
AdaLN 融合カーネルを手で書いてもここが上限で、仮に半分削れても**全体の 4% 程度**。
GEMM が 72% を占める構造では投資対効果が合わない。

### ついでに確認した「ループに無駄はないか」

1. **固定費 vs ステップ費用**（ステップ数を振って sample_rf を測定）

| steps | sample_rf | 1 ステップあたり |
|---:|---:|---:|
| 1 | 42.7 ms | 42.74 ms |
| 4 | 94.9 ms | 23.73 ms |
| 16 | 319.2 ms | 19.95 ms |
| 40 | 769.3 ms | 19.23 ms |

1 ステップ 18.73 ms、ステップ数に依存しない固定費は **20.0 ms (2.6%)** だけ。
テキストエンコーダ・KV キャッシュ構築・プラン構築はすべてこの 20 ms に収まっている。

2. **呼び出し内 / 呼び出し間の分離**（CUDA event でモデル呼び出しを挟んで計測）

| 区分 | 時間 | 比率 |
|---|---:|---:|
| 呼び出し内の GPU 時間 | 747.9 ms | **97.3%** |
| 呼び出し間の隙間 | 2.0 ms | 0.3% |
| ループ外（エンコード等） | 18.8 ms | 2.5% |

3. **呼び出しあたりのホスト側オーバーヘッドの有無**
   同じ引数で連続 20 回呼んだ純 GPU 時間は batch=1 で 13.37 ms、batch=2 で 23.36 ms、
   平均 18.37 ms。サンプラー実測の 18.70 ms とほぼ一致するので、
   **1 呼び出しあたりのホスト側オーバーヘッドは実質ゼロ**。

なお途中で観測した「GPU busy 67%」はプロファイラ自身のオーバーヘッドによる見かけの値で、
実際は上記の通り 97% が GPU 計算。

### 結論

Large / bf16 / RTX 5090 は **もう GPU 計算律速で、ホスト側の無駄は残っていない**。
残る手段は
- GEMM を速くする = 低精度化（精度を落とす。しかも 5090 では torchao 経路は実測で遅い）
- ステップ数を減らす = MeanFlow 蒸留（Large 版は未公開、学習データが必要）
の 2 つだけ。TensorRT もこの構造を変えないので出番はない。

### 副産物: カスタム op の autograd ガード

FA4 を包んだ custom op には autograd formula を登録していないため、
勾配が有効な状態（学習）で compile すると
`Trying to backward through irodori.fa4_varlen.default but no autograd formula was registered`
で落ちる。`torch.is_grad_enabled()` が偽のときだけ custom op 経路を使うよう修正済み
（推論は `torch.inference_mode()` 下なので影響なし、F の数値も変化なし）。
