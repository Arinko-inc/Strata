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

## 仕掛けたもの(2026-11-24、親の指示で一旦止まる前)
- trainsv で `~/wt/b70sdown-chain1.sh` を nohup で流してある: `shexp_bench` をビルド → 通れば `~/wt/b70sdown-logs/READY` を置いて `~/wt/b70sdown-run1.sh`(窓を最長 120 分待ち、開いたら人手なしで3本流す。1本 280 秒まで)に移る
- ビルドの結果は `~/wt/b70sdown-logs/chain1.build`(`exit 0 ...` なら成功)。落ちたときは `chain1.errors` に error の行が残り、READY は置かれない(1回目は `fold` の const の simd に select を呼んで落ちた。直して送り直した後のビルドは未確認)
- strata 本体(exe)は新しいソースではまだビルドしていない(`~/wt/b70sdown/build-sycl/strata` は土台 8aa7a5506 のまま)

## 結果の置き場と読み方(`~/wt/b70sdown-logs/`)
- `run1.status`: 流した順と終了コード。rc 1 は「ビットが違う組がある」、rc 124 は 280 秒の打ち切り
- `a-var.txt`(層 30・31、rpt 4、var 0/1/2/10、時間なし): `cmp down ... var N: 0 of ... differ` の var が元のカーネルと一致する式。var 0 = fma + 積を単独で丸める(専門家の down と同じ)、1 = fma なし、2 = xor なし、10 = var 0 + 活性の展開をスレッドで1回。var 0 が一致しなければ一致した var を `kVar` にする。どれも一致しなければ和の順の読み違い(カーネル先頭のコメントの式を元のカーネルと突き合わせる)
- `b-phase.txt`(18 層、T=1〜4): `phase T=...` の行が1呼び出しの区間ごとの µs(whole・量子化・gate・up・swiglu+量子化・down・スカラーゲート)。`window T=...` の行が 18 層の合計 ms と down の ms。**計画書の 0 の判定: down の ms ÷ 2 が 0.05 ms 未満なら実装をやめて内訳を返す**。同じファイルの `time down` の行が元と rpt 4 var 0 の µs
- `c-rpt.txt`(18 層、rpt 1/2/4/8 × var 0/10): `cmp` の行でビット一致、`time down` の行で一番速い組を選ぶ

## 次の手
1. `chain1.build` と `run1.status` を見る。ビルドが落ちていたら直して `~/wt/b70sdown.sh shexp_bench`、READY を置き直して `nohup ~/wt/b70sdown-run1.sh &`
2. 結果から var と rpt(と hoist)を選び、`q5_0_mmvq_esimd.dp.cpp` の `kVar` と `STRATA_Q50E(...)` の既定、`shared_expert.dp.cpp` の `kShexpEsimdRpt` を合わせ、`#define STRATA_Q50E_EXP` を外す
3. `~/wt/b70sdown.sh strata stage_probe shexp_bench` で作り直し、窓(2回目の READY)で `shexp_bench <gguf> 30,...,47`(既定の引数)を流して 18 層 × T=1〜4 の一致と時間を確かめる。exe の sha256 を控える
4. 記録を `~/work/strata-mesh/bench/` に置くのは親の判断
## 測った値
(まだ無い)
