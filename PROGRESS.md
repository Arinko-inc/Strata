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

## 採点のカーネル(`STRATA_SCORES_MQ=2`。新規 `sycl/src/prefill/esimd_scores_mq.dp.cpp`・`include/strata/prefill/esimd_scores_mq.hpp`、口は `prefill.cpp` の 2 か所と `CMakeLists.txt` の 1 行)
- 中身: QSA の層ごと・塊ごとに 1 回、塊のクエリと完成済みのブロックのキーを「積が読む順([成分 c][レーン L])」に並べ替えた写しを作る(`scores_mq_prepare`)。採点のカーネルは 1 スレッドが 1〜3 クエリ × 16 ブロックのタイル 4 枚を受け持ち、タイルの 16 個の点数を 1 回の scatter で書く。演算と順序は今の ESIMD の採点(`STRATA_SCORES_ESIMD=2`)と同じ。末尾のブロック(`dead` を読む・+1e9)を含むタイルは元の 1 件ずつのループで計算する
- スクラッチ: クエリ 塊の行数 × 2 KB(3,072 で 6.3 MiB)、キー 文脈の 4 トークンごとに 512 B(8,192 ブロック = 4 MiB 刻みで伸ばす。58K トークンで 8 MiB、262,144 で 32 MiB)。確保できなければ元のカーネルに戻る
- 結果(05:32〜05:54 の窓、印なし。exe は途中の版 `strata.mq2`): **3,072 トークンの 1 塊 位置 0: 700 → 695、位置 27,648: 809 → 773、位置 55,296: 869 → 791 ms。58,368 トークン(19 塊)17,926 → 17,282 ms(−3.6%)。ハッシュは 3,072 の 1 塊と 19 塊とも基準と同じ**
- 採点の区間(印あり、位置 55,296、QSA 4〜5 層の合計): 元 105 ms、既存の変種 `STRATA_SCORES_ESIMD=26` 55、**1 / 2 / 3 クエリ 27 / 29 / 29**、4 クエリ 66(タイル 1 枚 58、8 枚 71)、6 クエリ 131、キーを 16 個レジスタに持ってクエリを順に読む形 59(クエリ 64 件ずつ)・99(256 件)。効いたのは並べ替えた写しと scatter で、1 スレッドのクエリ数は 1〜3 で差が無い。4 クエリ以上はレジスタがあふれて元より遅い。遅い形はコードから外した(最終の exe は 1〜3 クエリだけ)
- 1 組(クエリ, ブロック)あたり 約 0.14 ns(27 ms ÷ 1.96 億組)。積 512 個 + 木の加算で約 68 命令なので、計算の量でほぼ決まっている

## 既存の旗(3,072 トークン。位置 0 / 55,296。基準 700 / 869)
| 旗 | ms | 判定 |
|---|---|---|
| `STRATA_SH_STREAM=0` | 701 / 864 | 差なし(ビルドと重なった時間) |
| `NEOReadDebugKeys=1 EnableDirectSubmissionController=0` | 702 / 871 | 差なし(同上) |
| 上の 2 つとも | 703 / 864 | 差なし(同上) |
| `STRATA_SCORES_ESIMD=26` | 696 / 812 | 速い(−57)。`STRATA_SCORES_MQ=2` のほうが速い |
| `STRATA_I8_GRF=128` | 699 / 876 | 差なし |
| `STRATA_I8_MT=8` | 722 / 882 | 遅い |
| `STRATA_I8_QONCE=1` | 695 / 870 | 差なし(58,368 トークン 17,869 対 17,926) |
| `STRATA_GDN_XE=1` | 724 / 895 | 遅い |
どれもハッシュは基準と同じ。

## ホストの待ち(`STRATA_PREFILL_HOST_TIMING=1`、3,072 トークン、位置 0)
ホストが GPU を待った 483 ms、グループ分けの間 GPU が空いた 2 ms、専門家と次の層の起動 184 ms。グループ分けは律速ではない。

## 残りの区間をどう見るか(3,072 トークン、位置 0 で約 700 ms)
- 専門家(gate/up 167 + down 69 + combine 25。「dequant」51 と「host grouping」137 は印の費用)。塊の大きさを振ると(256 / 1,024 / 2,048 / 3,072、印あり)gate/up 58 / 91 / 128 / 167、down 45 / 54 / 61 / 69 で、行数に依らない分が gate/up 約 48 ms、down 約 43 ms ある(雑音の入力は 256 トークンでも全 512 専門家に行が当たるので、専門家ごとに 1 回の重みの復号が固定で要る)。行数に比例する分は gate/up 0.039 ms/トークン、down 0.009 ms/トークン
- attention 114(位置 0)→ 151(位置 5 万)。1 塊目はセルが少ないぶん短い。1 セル・6 ヘッドで約 200 命令(うち K の int8 の復号が約 64)で、計算の量で決まっている。縮めるには K の復号を塊ごとに 1 回にまとめる(復号済みの K の写し。1 層 58K セルで 119 MiB)か、採点を int8 の積にする(ビットが変わる)
- GDN の再帰 64、hc read 52、GDN の入力の射影 36、QSA の射影 27、GDN の出力の射影 21: 触っていない
- 位置で伸びる分の残り(採点を入れた後): attention +37、topk +13、kv stage +10、採点 +27

## 置き場
- trainsv `~/wt/b70-read/`(ソースの写しと `build-sycl/`)、`~/wt/b70-read.sh`(ビルド)、`~/wt/b70-read-logs/`(記録)
- exe: `~/wt/b70-read/strata.mq1`(bfe32b5ef789946b、最初の版)、`strata.mq2`(途中の版。4・6 クエリとキー保持型も入っている)、`strata.mq3`(最終のソースのビルド。次の窓でハッシュを確かめる)
- 次の窓の待ち行列: `tools/b70_read/queue6.txt`(`chain6.sh` が待っている)
