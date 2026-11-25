# relay-read-kernels: 中継(llmsv、RX 7900 XT、HIP、層 [18,30))の読込を速くする(担当 R)

計画書: `~/work/strata-mesh/IMPL-PLAN-2026-11-26-relay-read.md`(arinkomac)。

## 到達点(2026-11-26 01:10 JST)
- 計測の道具を作った(`tools/relay_read/`)
  - `stage_sink.cpp`: 偽の末端。中継の hello に中継と同じ寸法で答え、受け取った行をそのまま返す(GPU もモデルも使わない)
  - `stage_probe_bench.cpp`: 主機の代役(`sycl-prefill-2` の `--bench` つき stage_probe の写し)。雑音の行を塊で送り、塊ごとの時間と出力行のハッシュを出す。リンクは fp32 なので、ハッシュは中継の出力行そのもののビット
  - `relay-read-run.sh TAG EXE [ENV=VAL ...]`: llmsv で 偽の末端(127.0.0.1:7853)+ 中継(7852、本番と同じ引数と旗)+ probe を流す。窓(`~/b/gpu-window`)が無いと動かない
  - `relay-read-win.sh QUEUE`: 窓を待って QUEUE の各行を run.sh に渡す
- ビルド: trainsv `~/wt/relay-read-hip.sh`(約 100 秒)。道具は `g++ -O2 -std=c++20 -Iinclude tools/relay_read/X.cpp src/net/stage_link.cpp -o build-tools/X -lpthread`
- llmsv の置き場: `~/wt/relay-read/`(`strata.base` = 土台 c57a85de6 のビルド d79a8c2065870c74、`stage_sink`、`stage_probe_bench`)、台本 `~/wt/relay-read-run.sh`・`relay-read-win.sh`、記録 `~/wt/relay-read-logs/`(`win.log`・`probe-TAG.log`・`worker-TAG.log`)
- 1 回目の窓を待っている(READY を置いた。queue1.txt = 元の exe の 計時なし / STRATA_PREFILL_TIMING=1 / STRATA_REMOTE_TIMING=1)

## 次の手
1. 窓が開いたら `probe-base-n.log`(塊ごとの ms とハッシュ)と `worker-base-t.log`(`strata prefill timing:` の行 = 区間ごとの ms)から内訳の表を作る
2. 一番大きい区間のカーネルを読み、新しいファイルに速い版を書く(旗で選ぶ。既定は切)
3. 同じ台本で元とハッシュを比べる

## 測った値
(まだ無い)
