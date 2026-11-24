# 担当 T 2回目(b70-gemv-t34、土台 b70-n15 = 4ebbe0c99): 大きい行列の T=3〜4 のスレッドあたりの計算量を減らす

1回目の記録: `~/work/strata-mesh/bench/2026-11-24-n15/PROGRESS-t4.md`。trainsv の置き場は同じ(`~/wt/b70t4/`、ビルド `~/wt/b70t4.sh`、IGC の出力 `~/wt/b70t4-logs/dump/dump.sh <名前> -DSTRATA_Q4KE_EXP=1 -DFN=q4k_mmvq_esimd_exp` と `sum.sh <名前>`)。

## やったこと(GPU なし、IGC の出力で確認。09:20)
lean なカーネル(テンプレート引数 DOT=2、rpt の符号 +1600。エンジンが使うのは LOOP 2 の `2400 + 行数`、行数 1〜4):
- 8 レーンの内積: 32 幅の dp4a 1 回 + 飛び飛びの領域の加算 2 回(3 命令。元は 16 幅の dp4a 2 回 + mov・add3・add で 8 命令)。整数の和なので順序を変えても同じ値
- 尺度の複製(各値を 2 回)を領域読み `replicate_vs_w_hs` 1 回に。重みの尺度 8 個をまとめて変換・乗算
- 末尾の `if (row0 + r >= n_out) break` をやめた(範囲外の行は row0 の重みで計算し、row0 の値をもう一度書く)。break があるとループが展開されず、部分和が実行時の添字で読む 1 個の大きい変数になっていた。**これを外すと LOOP 2 の lean は 行数 1〜4 × 列数 1〜4 の 16 個すべてが LOCAL_ROUND_ROBIN、退避 0**
- 命令数(2 行 × 4 列、LOOP 2): 840 → 561。4 行 × 4 列は 931(1回目の形では FIRST_FIT で使えなかった)

## 到達点(2026-11-24 09:27。窓 1 回の中で最後まで済んだ。READY は消した)
- 最終の exe: trainsv `~/wt/b70t4/build-sycl/strata` 115,669,072 バイト、sha256 55f429e308cf3958。計測器 `build-sycl/mmvq_esimd_bench` 9dcdecf68045de33
- 生の計測: `~/work/strata-mesh/bench/2026-11-24-b70t34/`(trainsv `~/wt/b70t4-logs/t34/`)。a-lean2・b-lean0・c-base・d-prov は仮の選び方のビルド、e-final-1・2 / f-lean0 / g-env1 / h-rpt4 / i-loop0 は最終のビルド。IGC のカーネル一覧 `igc-kernels-lean4.txt`

## 選び方(`STRATA_MMVQ_ESIMD=1`、2048 行以上。`native_mmvq.dp.cpp` の `mmvq_esimd_rpt`)
- 入力 2560 超(ssm_out・attn_output 6144×2560): 2 列は LOOP 0 の lean 2 行(1602)、ほかは LOOP 2 の lean 3 行(2403)
- 入力 2560 以下: 4 列は LOOP 0 の lean 3 行(1603)。1〜3 列は、8192 行超(qkv・attn_q)が 1602、8192 行以下(attn_gate)が 3 列 2404・1〜2 列 1601
- `STRATA_MMVQ_ESIMD_LEAN=0` で n15(昼)の選び方、`STRATA_MMVQ_ESIMD_LOOP=0` で朝の選び方(どちらも確かめた)。`_RPT` か `_LOOP` を置いた起動は昼のカーネルから選ぶ

## 測った値(層 30〜47、11 回の中央値、µs/呼び出し。前 = f-lean0(n15 の選び方)、後 = e-final-1 の rpt 0)
| 行列 | T=1 | T=2 | T=3 | T=4 |
|---|---:|---:|---:|---:|
| attn_qkv 2560×10240 | 30.1 → 29.4 | 33.7 → 30.3 | 40.3 → 34.7 | 52.2 → 37.4 |
| attn_gate 2560×6144 | 19.6 → 18.9 | 22.8 → 20.7 | 27.5 → 21.8 | 34.5 → 22.0 |
| ssm_out 6144×2560 | 20.1 → 18.2 | 23.9 → 20.4 | 30.9 → 23.1 | 40.1 → 28.7 |
| attn_q 2560×12288 | 36.2 → 34.9 | 40.3 → 37.4 | 43.7 → 39.8 | 55.3 → 42.1 |
| attn_output 6144×2560 | 20.4 → 18.4 | 24.7 → 20.9 | 31.4 → 24.9 | 40.8 → 30.9 |
- 18 層の合計(wide に対して縮む量): T=1 0.671 → 0.730、T=2 0.621 → 0.775、T=3 0.659 → 0.962、T=4 0.516 → 1.139 ms(n15 から T=1 −0.06、T=2 −0.15、T=3 −0.30、**T=4 −0.62 ms**。2 回目 e-final-2 も 0.730 / 0.775 / 0.965 / 1.136)
- ビット一致: 最終のビルドで 11 種 × T=1〜4 × 5 通り(エンジンの選び方・rpt 2・1602・2403・native_mmvq 経由)= 220 通りで差 0(2 回)。`STRATA_MMVQ_ESIMD_LEAN=0`、`STRATA_MMVQ_ESIMD=1`、`LEAN=1 + RPT=4`、`LOOP=0` を環境に置いた回も各 88 通りで差 0。仮のビルドでは lean の LOOP 0/2 × 1〜4 行の全部で差 0

## 確かめていないこと
- strata 本体を起動していない(3段は親)。旗なしの wide の経路はソースを変えていないことだけ
- 1603(LOOP 0 の 3 行 × 4 列)は IGC の割り当てが LOCAL_FIRST_FIT だが、qkv・gate・q の 4 列で一番速かった。割り当ての種類だけでは速さは決まらない(理由は調べていない)
- 小さい行列(2048 行未満)は wide のまま。lean の 1 行(1601)は 640 行の T=1 で 3.96 µs(wide 4.29)だが、T=4 は 7.0(wide 6.2)なので入れていない

## 次にやるなら
- 活性の尺度(fp16 の d を 8 個集める)が列・ブロックごとにスカラーの mov 8 本。gather の読み 1 回にする案(読みが 1 本増える)
- 2048 行未満の行列の 1 列だけ lean の 1 行にする(窓あたり 0.01 ms 程度)

## 仕掛け
- 窓を待つ台本 `~/wt/b70t4-run2.sh`(30 秒おきに窓を見る。結果は `~/wt/b70t4-logs/t34/`、流れた順と rc は `run2.status`)。a-lean2(rpt 2,802,2401〜2404)・b-lean0(LOOP 0 の lean 1601〜1604)・c-base(`STRATA_MMVQ_ESIMD_LEAN=0` で n15 の選び方)・d-prov(仮の選び方)。`b70t4-run2.sh now <名前> --rpt ...` で 1 本だけ流せる
- `native_mmvq.dp.cpp` の選び方(`mmvq_esimd_rpt` の lean の枝)は仮(2 列以上は 4 行、1 列は 2 行)。窓の結果で決めて作り直し、同じ窓で `--rpt 0` の差 0 と µs を取る

## 環境変数
- `STRATA_MMVQ_ESIMD_LEAN=0`: n15(2026-11-24 昼)の選び方とカーネルに戻す。未設定は lean(ただし `STRATA_MMVQ_ESIMD_RPT` か `_LOOP` を置いた起動は今までどおり昼のカーネルから選ぶ)。`=1` と `_RPT=1|2|4` で lean の行数を固定
