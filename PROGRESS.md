# 担当 S(共有専門家の down を ESIMD に)の進み具合

計画書: `~/work/strata-mesh/IMPL-PLAN-2026-11-24-b70-next.md` の「担当 S」。ブランチ `b70-shared-down`(土台 8aa7a5506)。

## 確かめた事実(コードとパックの索引から。型は計測器の表示で確認: gate/up Q4_K、down Q5_0)
- 共有専門家の行列(層 30): gate/up は 2560 入力 × 640 行(Q4_K)、down は 640 入力 × 2560 行。down の型は計測器が表示する(Q5_0 = 型 6 と見ている)
- 本番(`--native`、LFUSE・QDEDUP なし)の `shared_expert_multi` は1呼び出しで7起動: 入力の q8_1 化 → gate → up → SwiGLU+q8_1 化 → down → スカラーゲートの gemv → sigmoid+掛け算
- gate/up は 640 行で、`STRATA_MMVQ_ESIMD=1` の下限(2048 行)に届かないので wide のカーネルのまま(ESIMD は通らない)
- down の今のカーネルは `native_small_mmvq_kernel<Q50Block,4,true>`(T=1)と `native_mmvq_multi_kernel` の 4 行の形(T=2〜4)。128 スレッドの work-group が 4 行を受け持ち、働くのは 40 スレッドだけ(20 ブロック × 半分ずつ)。バリアと SLM を通る

## 到達点(2026-11-24 07:57): 実装・ビルド・窓の中での確かめまで済み。残りは親の3段の計測
- 最終のコミットは `git log` の先頭。exe trainsv `~/wt/b70sdown/build-sycl/strata` = sha256 頭 579f856f8bd90484、計測器 `build-sycl/shexp_bench` = d00e8db8659c885a
- 旗 `STRATA_SHEXP_ESIMD=1`(既定 0)。1スレッド 2 行。1 列は活性を行ごとに展開、2〜4 列はスレッドで1回展開。`STRATA_SHEXP_ESIMD_RPT` は作らなかった
- 実験用の変種(rpt 1/4/8、var 1/2)は `-DSTRATA_Q50E_EXP` を付けたビルドだけに入る

## 測った値(B70、層 30〜47 の 18 層、µs/呼び出し、1グラフに 18 呼び出し、中央値。記録 trainsv `~/wt/b70sdown-logs/`)
| | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| 共有専門家の全体(今) | 40.5 | 37.6 | 42.0 | 47.4 |
| 入力の q8_1 化 | 2.8 | 2.9 | 2.9 | 2.9 |
| gate(Q4_K、wide) | 5.2 | 5.5 | 6.4 | 7.1 |
| up(Q4_K、wide) | 5.6 | 5.8 | 6.8 | 7.7 |
| SwiGLU+q8_1 化 | 3.0 | 2.9 | 3.0 | 3.0 |
| down(Q5_0、今。区間だけを並べたグラフ) | 24.6 | 18.6 | 20.8 | 24.2 |
| スカラーゲート(gemv + sigmoid 掛け算。全体 − ゲートなしの全体) | 2.1 | 2.2 | 2.6 | 3.1 |
| down だけの比較: 今 → ESIMD(最終の既定) | 20.7 → 4.0 | 16.1 → 4.9 | 18.2 → 5.8 | 20.9 → 6.5 |
| 全体(`STRATA_SHEXP_ESIMD=1`) | 21.6 | 24.6 | 27.5 | 30.5 |
| 18 層の合計の短縮(全体の差 × 18) | 0.340 ms | 0.234 ms | 0.261 ms | 0.303 ms |

- ビット一致: 最終のビルドで、down 単体(var -1・0・10)と `shared_expert_multi`(旗 0 と 1)を 18 層 × T=1〜4 × 入力 5 通り × 2 回流して、32 行すべて差 0(`e-final-1.txt`・`e-final-2.txt`)。実験のビルドでは rpt 1/2/4/8 × var 0/10 も差 0(`c-rpt.txt`)。var 1 は 3〜78 個が違い最大の相対誤差 8.1e-6、var 2 は約半分が違い最大 5.8e-3(`a-var.txt`、層 30・31)
- `run1.status` の a-var が「rc 0」なのに計測器は「8 failures」を出している(終了コード 1 のはず)。原因は調べていない

## 確かめていないこと
- 3段での窓の時間・生成の文面(親が実施)。strata 本体は旗の書式の検査(`STRATA_SHEXP_ESIMD=x` で断る)だけ動かした。段のワーカーとしての起動はしていない
- 計測器のスカラーゲートのベクトルは乱数(値は出力の倍率にだけ関わる)

## 作ったもの
- `sycl/src/kernels/cuda/q5_0_mmvq_esimd.dp.cpp`: 640 入力の Q5_0 の dense GEMV(1〜4 列)。1スレッド RPT 行。和の順は元のカーネルに合わせた(式はファイル先頭のコメント)
- `sycl/include/strata/kernels/q5_0_mmvq_esimd.hpp`: 宣言
- `sycl/src/kernels/cuda/shared_expert.dp.cpp`: `STRATA_SHEXP_ESIMD=1`(既定 0)で `shared_expert_multi` の down を上のカーネルへ。1スレッド 2 行
- `sycl/src/kernels/shexp_bench.cpp`: 区間ごとの内訳と、down の元と新のビット比較・時間

## trainsv の置き場
- ソース `~/wt/b70sdown/`、ビルド台本 `~/wt/b70sdown.sh [target ...]`、記録 `~/wt/b70sdown-logs/`
- 窓で流す台本 `~/wt/b70sdown-run1.sh`(窓を待って流す。結果は `~/wt/b70sdown-logs/*.txt`、流した順は `run1.status`)


