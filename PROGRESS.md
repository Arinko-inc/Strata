# 担当 T(b70-gemv-t4): STRATA_MMVQ_ESIMD の 4 列の伸び — 進み具合

計画書: `~/work/strata-mesh/IMPL-PLAN-2026-11-24-b70-next.md` の「担当 T」。土台 8aa7a5506。

## trainsv の置き場
- ソース `~/wt/b70t4/`(arinkomac の worktree から rsync)、ビルドの台本 `~/wt/b70t4.sh [target ...]`、結果 `~/wt/b70t4/build-sycl.done`
- 記録 `~/wt/b70t4-logs/`。窓を待つ台本 `~/wt/b70t4-run1.sh`(窓が開くと a-loop1 / b-loop2 / c-t4 を流す。`b70t4-run1.sh now <名前> <計測器の引数>` で1本だけ流せる)
- IGC の出力(GPU を使わないコンパイルだけ): `~/wt/b70t4-logs/dump/dump.sh <名前> [-D...]` で `q4k_mmvq_esimd.dp.cpp` だけを AOT でコンパイルし、`<名前>/` に asm・zeinfo を出す。`sum.sh <名前>` でカーネルごとに1行(レジスタ数・命令数・バイト数・RA の種類・退避の行数)

## 確かめた事実(2026-11-24 05:50、IGC の出力。GPU は使っていない)
- ESIMD のカーネルはどれも numGRF=128 でコンパイルされている。`grf_size<256>` の指定は AOT の ESIMD(VC)には届いていない(options は `-cl-fp32-correctly-rounded-divide-sqrt -vc-codegen` だけ)。前の回の「rpt 101/102 が rpt 1/2 と同じ時間」と合う
- 2 行 × 4 列(元の形 = LOOP 0)は退避なし(spill 0、private memory なし)。レジスタの不足ではない
- 元の形はループの1周に 256 ブロック 4 個分のコードを展開している。カーネルのバイト数は 2行×3列 28.1 KB、1行×4列 23.9 KB、4行×1列 24.7 KB(ここまでは列数・行数に比例して速い)、2行×4列 38.8 KB、4行×2列 37.8 KB、8行×1列 45.4 KB(この3つが、前の回の計測で 48 行の行列でも 1.5〜2 倍遅い)。境目は 28.5〜37.7 KB の間
- RA の種類も同じ境目で変わる(速い側は LOCAL_ROUND_ROBIN、遅い側は LOCAL_FIRST_FIT か GRAPH_COLORING)

## 推定(窓で確かめる)
- 原因はコードの大きさ(EU の命令キャッシュに入り切らない。32 KB 前後)か、RA が round robin でなくなること(レジスタの使い回しによる依存)のどちらか
- LOOP 1(1周1ブロック、4本の部分和を回す)の 2行×4列は 13.9 KB で LOCAL_FIRST_FIT、LOOP 2(1周1ブロック、実行時のオフセット)は 13.0 KB で LOCAL_ROUND_ROBIN。LOOP 1 が速ければコードの大きさ、LOOP 2 だけ速ければ RA

## 実装の状態
- `q4k_mmvq_esimd.dp.cpp`: テンプレート引数 LOOP(0 元の形 / 1 回す / 2 オフセット)。rpt の符号は `行数 + 400 × LOOP`。実験用の変種(VAR 1〜9・GRF 128・dp4a の連鎖・LOOP 2)は `STRATA_Q4KE_EXP` を定義したときだけ入り、そのときは関数名 `q4k_mmvq_esimd_exp`・タグ `Q4kMmvqExpTag` になる。計測器 `mmvq_esimd_bench` だけがこのファイルをもう一度その定義でコンパイルする(CMake)。strata 本体には LOOP 0 と 1 の 行数 1/2/4/8 だけが入る
- `native_mmvq.dp.cpp`: `STRATA_MMVQ_ESIMD_LOOP`(0/1、未設定は形で選ぶ)。**形で選ぶ規則 `mmvq_esimd_loop` / `mmvq_esimd_rpt` は仮(WIP)。窓の計測で決める**
- 計測器: `--rpt` に 401・402・404・408(LOOP 1)、801…(LOOP 2)を渡せる

## 次の手
1. ビルドが通ったら `~/wt/b70t4-logs/READY` を置き、`~/wt/b70t4-run1.sh` を裏で起こして窓を待つ
2. 窓で a-loop1 / b-loop2 / c-t4 の結果を見て、形で選ぶ規則を決めて作り直し、`--rpt 0,2`(エンジンの選び方)で 132 通りの差 0 と µs を取る。`STRATA_MMVQ_ESIMD_LOOP=0` で元の時間に戻ることも見る
3. exe の大きさ(土台 121,054,048 バイト)と sha256 を記録、コミット
