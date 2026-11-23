# 担当 S(共有専門家の down を ESIMD に)の進み具合

計画書: `~/work/strata-mesh/IMPL-PLAN-2026-11-24-b70-next.md` の「担当 S」。ブランチ `b70-shared-down`(土台 8aa7a5506)。

## 確かめた事実(コードとパックの索引から。GPU ではまだ測っていない)
- 共有専門家の行列(層 30): gate/up は 2560 入力 × 640 行(Q4_K)、down は 640 入力 × 2560 行。down の型は計測器が表示する(Q5_0 = 型 6 と見ている)
- 本番(`--native`、LFUSE・QDEDUP なし)の `shared_expert_multi` は1呼び出しで7起動: 入力の q8_1 化 → gate → up → SwiGLU+q8_1 化 → down → スカラーゲートの gemv → sigmoid+掛け算
- gate/up は 640 行で、`STRATA_MMVQ_ESIMD=1` の下限(2048 行)に届かないので wide のカーネルのまま(ESIMD は通らない)
- down の今のカーネルは `native_small_mmvq_kernel<Q50Block,4,true>`(T=1)と `native_mmvq_multi_kernel` の 4 行の形(T=2〜4)。128 スレッドの work-group が 4 行を受け持ち、働くのは 40 スレッドだけ(20 ブロック × 半分ずつ)。バリアと SLM を通る

## 作ったもの(コミット済み。GPU での確かめはまだ)
- `sycl/src/kernels/cuda/q5_0_mmvq_esimd.dp.cpp`: 640 入力の Q5_0 の dense GEMV(1〜4 列)。1スレッド RPT 行。和の順は元のカーネルに合わせた(式はファイル先頭のコメント)
- `sycl/include/strata/kernels/q5_0_mmvq_esimd.hpp`: 宣言
- `sycl/src/kernels/cuda/shared_expert.dp.cpp`: `STRATA_SHEXP_ESIMD=1`(既定 0)で `shared_expert_multi` の down を上のカーネルへ。`STRATA_SHEXP_ESIMD_RPT`(既定 4)
- `sycl/src/kernels/shexp_bench.cpp`: 区間ごとの内訳と、down の元と新のビット比較・時間
- 今は実験用の変種(`STRATA_Q50E_EXP`: rpt 1/2/4/8、var 0/1/2、+10 で活性の展開をスレッドで1回)が入っている。窓で選んだら既定の1つだけ残す

## trainsv の置き場
- ソース `~/wt/b70sdown/`、ビルド台本 `~/wt/b70sdown.sh [target ...]`、記録 `~/wt/b70sdown-logs/`
- 窓で流す台本 `~/wt/b70sdown-run1.sh`(窓を待って流す。結果は `~/wt/b70sdown-logs/*.txt`、流した順は `run1.status`)

## 次の手
1. ビルドを通す(`~/wt/b70sdown.sh strata stage_probe shexp_bench`)
2. `~/wt/b70sdown-logs/READY` を置き、窓で run1 を流す: a-phase(内訳)、b-var(式の形のビット比較)、c-rpt(行数と hoist の時間)
3. 一致する var と速い rpt を既定にして変種を外し、作り直して 18 層 × T=1〜4 を確かめ直す

## 測った値
(まだ無い)
