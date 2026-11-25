# b70-read-kernels: 末端(trainsv、Arc Pro B70、SYCL、層 [30,48))の読込の 1 塊を縮める(担当 E)

2026-11-26 04:34 JST〜、Fable 5.2 low の子 1 本。土台は本番の末端のソース `b70-n15` 2c199c79b(exe 55f429e308cf3958)。
決まりは `~/work/strata-mesh/IMPL-PLAN-2026-11-26-relay-read.md` と同じ(窓は `trainsv:~/b/gpu-window`、ビルド禁止は `~/b/no-build`)。

## 測り方
- trainsv で末端だけを本番と同じ引数・旗で起こし(127.0.0.1:7861)、`stage_probe --bench N --chunk C` が雑音の行を流す。リンクは fp32 なので probe のハッシュは末端の出力行のビット
- 台本: `tools/b70_read/b70-read-run.sh TAG EXE [ENV=VAL ...]`(trainsv では `~/wt/b70-read/tools/b70_read/`)。トークンは実行時に本番の台本から読む(写しを作らない)。窓を待って待ち行列を流すのは `b70-read-win.sh QUEUE`、集計は `pft.py TAG...`、記録は `trainsv:~/wt/b70-read-logs/`(`worker-TAG.log`・`probe-TAG.log`・`win.log`)
- 1 塊の時間はワーカーの「read in」(印なし)。区間は `STRATA_PREFILL_TIMING=1`(印あり。印ありは 3,072 トークンで約 115 ms 長い。「host grouping」と「dequant」は印の費用がほとんど)
- ビルド: `trainsv:~/wt/b70-read.sh [target ...]`(全部で約 340 秒)

## 基準(本番の exe `~/wt/b70t4/build-sycl/strata` と本番の旗。05:15 の窓)
| 塊 | 位置 | read in(ms、印なし) |
|---:|---:|---:|
| 256 | 0 | 164 |
| 1,024 | 0 | 304 |
| 2,048 | 0 | 500 |
| 3,072 | 0 | 701 |
| 3,072 | 27,648 | 797 |
| 3,072 | 55,296 | 866 |
ハッシュ: 256 e2711083b3085d5c、1,024 83d72c23a3cfdba0、2,048 0bd5247c886af611、3,072 88559b61d22b894f、58,368(19 塊)b2b6c4b7b33cbcf8

区間(3,072 トークン、印あり、ms。位置 0 → 55,296): gemm gate/up 167 → 163、qsa attn 114 → 151、gemm down 69 → 80、gdn recurrence 64 → 65、hc read 52 → 51、gdn 36 → 37、qsa proj 27 → 28、combine 25、gdn out proj 21、**qsa select 4 → 105**、qsa topk 2 → 15、kv stage 0 → 10。
位置で伸びる約 165 ms は、採点(qsa select)+101、attention +37、topk +13、K/V の上げ下ろし(kv stage)+10。

## 既存の旗(本番の exe、3,072 トークン。位置 0 / 55,296)
- `STRATA_SH_STREAM=0`: 701 / 864(基準 701〜710 / 866〜870)。差なし。ハッシュ同じ
- `NEOReadDebugKeys=1 EnableDirectSubmissionController=0`: 702 / 871。差なし。両方: 703 / 864
- (この 3 本は自分のビルドと重なった時間に流した)
