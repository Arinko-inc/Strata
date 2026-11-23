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

## 仕掛けたもの(2026-11-24 06:25)
- trainsv で `~/wt/b70t4-run1.sh` が nohup で窓を待っている(1分おきに `/home/arinko/b/gpu-window` を見る。120 分で出なければ終わる)。`~/wt/b70t4-logs/READY` は置いた(中身 `b70t4-run1.sh 20`)
- 窓が開くと、人手なしで次の3本を順に流す(各 400 秒で打ち切り、窓の残りが 420 秒を切ると飛ばす)。流れた順と終了コードは `~/wt/b70t4-logs/run1.status`(最後の行が `done`)
  1. `a-loop1.txt`: 層 30〜47 × T=1〜4 × `--rpt 1,2,401,402,404,408`(LOOP 0 の 1・2 行と LOOP 1 の 1・2・4・8 行)。ビット比較と µs
  2. `b-loop2.txt`: 同じく `--rpt 2,801,802,804,808`(LOOP 2)
  3. `c-t4.txt`: T=3,4 だけ、`--rpt 1,2,401,402,404,801,802,804`、15 回、時間だけ
- 使う exe: `~/wt/b70t4/build-sycl/mmvq_esimd_bench`(d978b3cec のソース)。同じビルドの `strata` は 112,811,832 バイト・sha256 757d06f60f1a5f2d(土台 121,054,048 バイト。形で選ぶ規則が仮のままなので、本番用ではない)

## 結果の読み方
- `grep "differ" a-loop1.txt b-loop2.txt | grep -v " 0 of "` が空で、末尾が `mmvq_esimd_bench: 0 failures` ならビット一致
- `grep "^time"` の行: `wide`(本番の基準)と各 `rpt N` の µs/呼び出し。`grep "^sum"` は T ごとの 18 層合計で、wide に対して何 ms 縮むか。**いまの本番の選び方に当たるのは rpt 2(小さい行列の 1 列・4 列は rpt 1)**。それと rpt 402 などを比べる
- 原因の切り分け: attn_qkv の T=4 で rpt 402 が rpt 2(約 55 µs)より速く、3 列からの伸びが 2→3 列と同じ程度(+6 µs 前後)ならコードの大きさ。rpt 402 が遅いままで 802 だけ速ければ RA の種類
- ssm_alpha(48 行)の行は、スレッド1本あたりの時間を見るのに使う(行列が小さいので起動の固定分 + 1本の実行時間)

## 次の手
1. 窓の結果から、形(n_in・n_out・列数)ごとに一番速い「行数 × LOOP」を選び、`native_mmvq.dp.cpp` の `mmvq_esimd_loop` / `mmvq_esimd_rpt`(いまは仮: LOOP は既定 0)を決める。LOOP 2 を採るなら `q4k_mmvq_esimd.dp.cpp` の末尾で LOOP 2 を `#ifdef STRATA_Q4KE_EXP` の外に出し、採らない LOOP を中に入れる
2. 作り直して(`~/wt/b70t4.sh strata stage_probe mmvq_esimd_bench`、約 7〜12 分)、2回目の窓で `--rpt 0,2`(エンジンの選び方。`b70t4-run1.sh now d-auto $ALL --rpt 0,2 --trials 11`)の 132 通りの差 0 と µs を取る。`STRATA_MMVQ_ESIMD_LOOP=0` を付けて元の時間(d-auto-1.txt の rpt 0)に戻ることも見る。READY を置き直す(中身に「2回目」)
3. exe の大きさと sha256 を記録、コミット、報告
