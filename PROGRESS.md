# 担当 T 2回目(b70-gemv-t34、土台 b70-n15 = 4ebbe0c99): 大きい行列の T=3〜4 のスレッドあたりの計算量を減らす

1回目の記録: `~/work/strata-mesh/bench/2026-11-24-n15/PROGRESS-t4.md`。trainsv の置き場は同じ(`~/wt/b70t4/`、ビルド `~/wt/b70t4.sh`、IGC の出力 `~/wt/b70t4-logs/dump/dump.sh <名前> -DSTRATA_Q4KE_EXP=1 -DFN=q4k_mmvq_esimd_exp` と `sum.sh <名前>`)。

## やったこと(GPU なし、IGC の出力で確認。09:20)
lean なカーネル(テンプレート引数 DOT=2、rpt の符号 +1600。エンジンが使うのは LOOP 2 の `2400 + 行数`、行数 1〜4):
- 8 レーンの内積: 32 幅の dp4a 1 回 + 飛び飛びの領域の加算 2 回(3 命令。元は 16 幅の dp4a 2 回 + mov・add3・add で 8 命令)。整数の和なので順序を変えても同じ値
- 尺度の複製(各値を 2 回)を領域読み `replicate_vs_w_hs` 1 回に。重みの尺度 8 個をまとめて変換・乗算
- 末尾の `if (row0 + r >= n_out) break` をやめた(範囲外の行は row0 の重みで計算し、row0 の値をもう一度書く)。break があるとループが展開されず、部分和が実行時の添字で読む 1 個の大きい変数になっていた。**これを外すと LOOP 2 の lean は 行数 1〜4 × 列数 1〜4 の 16 個すべてが LOCAL_ROUND_ROBIN、退避 0**
- 命令数(2 行 × 4 列、LOOP 2): 840 → 561。4 行 × 4 列は 931(1回目の形では FIRST_FIT で使えなかった)

## 仕掛け
- 窓を待つ台本 `~/wt/b70t4-run2.sh`(30 秒おきに窓を見る。結果は `~/wt/b70t4-logs/t34/`、流れた順と rc は `run2.status`)。a-lean2(rpt 2,802,2401〜2404)・b-lean0(LOOP 0 の lean 1601〜1604)・c-base(`STRATA_MMVQ_ESIMD_LEAN=0` で n15 の選び方)・d-prov(仮の選び方)。`b70t4-run2.sh now <名前> --rpt ...` で 1 本だけ流せる
- `native_mmvq.dp.cpp` の選び方(`mmvq_esimd_rpt` の lean の枝)は仮(2 列以上は 4 行、1 列は 2 行)。窓の結果で決めて作り直し、同じ窓で `--rpt 0` の差 0 と µs を取る

## 環境変数
- `STRATA_MMVQ_ESIMD_LEAN=0`: n15(2026-11-24 昼)の選び方とカーネルに戻す。未設定は lean(ただし `STRATA_MMVQ_ESIMD_RPT` か `_LOOP` を置いた起動は今までどおり昼のカーネルから選ぶ)。`=1` と `_RPT=1|2|4` で lean の行数を固定
