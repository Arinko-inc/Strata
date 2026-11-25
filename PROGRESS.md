# relay-read-kernels: 中継(llmsv、RX 7900 XT、HIP、層 [18,30))の読込を速くする(担当 R)

計画書: `~/work/strata-mesh/IMPL-PLAN-2026-11-26-relay-read.md`(arinkomac)。2026-11-26 01:00〜01:50 JST、Fable 5.2 low の子 1 本。


## 3 回目(2026-11-26 02:46〜03:25 JST、同じ担当 R)の到達点: Q5_0 の down の F16 カーネルが実機で動き、中継の読みが 1,024 トークン以上の塊で 35〜56% 短くなった
窓は 1 回使った(03:10〜03:31 JST の枠のうち 03:10〜03:24)。自分のプロセスは残していない。

### 直したもの(コミット 8ab71fe48)
- `f16q50::enabled()` が常に false だった。HIP のビルドのホスト側の pass は `__gfx1100__` を定義しないので `STRATA_F16_DEVICE` が 0 になり、旗を付けても MMQ のままだった(2 回目の exe 999962f6f72415ec は旗 4 つを付けても本番と同じハッシュ)。環境変数だけを見る形に直した(機体の判定は呼び元の `d_covered_on` がしている)
- 診断 `STRATA_PF_FUSED_WHY=1`(`native_supported` の各項と、層 LB の `fused_l` の判定の各項を stderr に出す。旗が無ければ何も変わらない)。`moe_fused_iq.cu` と `prefill.cpp` に 1 か所ずつ
- 窓の台本 `tools/relay_read/relay-read-win3.sh`(1 行 = 1 コマンドの待ち行列。残り 150 秒で止める)と `queue3a〜c.txt`

### 親が 3 段で試すための起動の差分
- exe: **`llmsv:/home/arinko/wt/relay-read/strata.r3`(sha256 頭 01552eb50a0a3b0b、コミット 8ab71fe48 のビルド)**。旗なしでは本番と同じハッシュ(256 / 1,024 / 2,048 / 3,072 / 12,288 = 69968e111c5ea128 / 09e92bb1432e3ced / fe8d50cfc003a5a8 / ad680b8c4f428e89 / edce69467b69384b。`r3b-prod-n`)
- 足す旗(本番の `STRATA_PF_GEMM=1 STRATA_PF_PAD=1` に加えて): **`STRATA_PF_FUSED=1 STRATA_PF_FUSED_KQ=1 STRATA_PF_F16=1 STRATA_PF_F16_Q50=1`**(4 つとも要る)
- 任意でもう 1 つ: `STRATA_PREFILL_STREAM_MIN=384`(fused に入る塊の下限を 1,024 から 384 に下げる。下の表。256 まで下げると 128 トークンの塊が遅くなる)
- 入ったことの確かめ: ワーカーのログに `prompt experts on the fused int8 kernels`、起動時の `the prompt path borrows` が 661 → 580 slots(STREAM_MIN を付けると 575)

### 確かめたこと(PROGRESS 2 回目の「確かめていないこと」1〜4)
1. fused の経路に入る(1,024 トークン以上の塊。256 は MMQ のままでハッシュも本番と同じ)
2. リング: `fused_ring` が真になり、借りる枠は 661 → 580(1.83 → 1.60 GiB)。`STRATA_WK_LEND_SINGLE=1 STRATA_WK_REFILL_EARLY=1` と一緒に起動し、5 通りの塊と 12,288(4 塊)を 3 回ずつ流して落ちない・ハッシュが回ごとに同じ
3. MMQ(基準)との差、1,024 トークンの出力行(12 層を通った後、fp32): 相対 L2 2.1e-3(行ごとの最大 5.8e-3)、行の最大値に対する最大の差 0.93%、NaN / inf なし(雑音の入力)。偏りのある入力(`--skew 64`)では相対 L2 1.4e-3(行ごとの最大 4.5e-3)。1 回目に測った `STRATA_GDN_CHUNKED=1` の 2.2e-3 と同じ大きさ
4. 旗なしのハッシュは本番と同じ(上)

### 中継の読み 1 塊(ms。雑音の入力、3 回目の値。exe は strata.r3、基準 = 本番の旗 + PF_GEMM + PAD)
| トークン | ワーカーの wall(印あり)基準 → 新 | probe の壁時計(印なし)基準 → 新 | 偏りのある入力の probe(印なし)基準 → 新 |
|---:|---|---|---|
| 256 | 170 → 170(MMQ のまま) | 182 → 183 | 204 → 188(基準は 2 回目の exe、別の実行) |
| 1,024 | 312 → 140 | 448 → 306 | 529 → 307 |
| 2,048 | 468 → 216 | 700 → 466 | 795 → 468 |
| 3,072 | 583 → 303 | 910 → 633 | 1,023 → 637 |
| 12,288(3,072 × 4) | — | 2,923 → 1,892 | 3,391 → 1,903 |

- probe の壁時計は fp32 の loopback 2 往復(3,072 で約 330 ms)を含む。中継自身の時間はワーカーの wall
- 傾き(ワーカーの wall、1,024 と 3,072 の 2 点): 0.132 → 0.080 ms/トークン
- 偏りのある入力では基準(MMQ)が遅くなり(3,072 の wall 707)、fused は変わらない(305)。MMQ は 16 専門家の最大行数で列を計算するため
- 窓の最初の実行(`r3-ppf-t`、wall 669)は他より約 15% 遅かった。本番を止めた直後の 1〜2 本は使わず、`r3c-ppf1-t`(583、1 回目の 582 と同じ)を基準にした

### 内訳(3,072 トークン、印あり、ms。`worker-r3c-ppf1-t.log` と `worker-r3b-f50-t.log`)
| 区間 | 基準 | 新 |
|---|---:|---:|
| 合計(wall) | 583 | 303 |
| 専門家(基準: 写し 72 + gate/up 101 + down 197 + host grouping 4。新: 印 `gemm gate/up` に gate/up と down の両方) | 374 | 81 |
| wait copy(新: 枠の到着待ち) | 1 | 8(1,024 では 15) |
| hc read | 45 | 48 |
| GDN の再帰 | 35 | 34 |
| GDN の入力の射影 | 29 | 31 |
| QSA attention | 25 | 25 |
| QSA の射影 | 18 | 19 |
| ほか(GDN out 11・hc write 11・ルーター 10・combine 7・gconv 5・emb 5・qsel 2・gather 1) | 56 | 52 |

新の残りは専門家 27%、hc read 16%、GDN の再帰 11%、GDN の入力の射影 10%。

### fused の中の設定(3,072 トークンの wall、印あり。どれも 1 回の実行の 3 回目)
| 旗 | wall | 専門家 | ハッシュ |
|---|---:|---:|---|
| (なし) | 303 | 81 | db51202ead162f16 |
| `STRATA_PF_F16_BN=64` / `=128` | 301 / 300 | 81 / 82 | 同じ |
| `STRATA_PF_BATCH=128` / `=64` | 303 / 300 | 81 / 85 | 同じ |
| `STRATA_PF_OCC=3` / `=4` | 302 / 299 | 82 / 81 | 同じ |
| `STRATA_GDN_CHUNKED=1` | 283 | 81(GDN の再帰 34 → 11) | 違う(a75fac7e4344b9c0。誤差は fused とは別に 1 回目の 2.2e-3) |
BN・BATCH・OCC は差が 1% 以内で、足す理由が無い。

### 小さい塊(`STRATA_PREFILL_STREAM_MIN`。probe の壁時計、印なし、5 回の中央値、ms)
| トークン | 基準 | fused(下限 1,024 のまま) | 下限 512 | 下限 384 | 下限 256 |
|---:|---:|---:|---:|---:|---:|
| 128 | 143 | 143 | 143 | 143 | 218 |
| 256 | 183 | 183 | 183 | 183 | 199 |
| 384 | 281 | 281 | 300 | 219 | 220 |
| 512 | 318 | 318 | 233 | 233 | 233 |
| 768 | 384 | 384 | 271 | 271 | 271 |
- ワーカーの wall(印あり、下限 64): 64 / 128 / 256 / 512 / 768 = 86 / 86 / 94 / 108 / 124(基準 102 / 133 / 170 / 238 / 285)。wall は縮むのに、probe の壁時計は 128 以下で長くなる(64 トークンで 133 → 180)。wall の外で約 65 ms 増えている。原因は見ていない(借りた枠の出し入れが候補)
- 下限のすぐ下の塊が基準より遅くなる(下限 512 の 384: 281 → 300、下限 256 の 128: 143 → 218)。原因は見ていない。下限 384 のときの 257〜383 トークンは測っていない

### down の MMQ が遅い原因の切り分け(`relay_mmq_bench`、`mmqb.log`。fused に替わったので優先度は下がった)
同じ形 2560x768 で Q5_0 は Q4_K の 1.2〜2.0 倍(R=16: 0.106 / 0.087 ms、R=96: 0.480 / 0.238 ms)。1280x2560 でも Q5_0 0.673 / Q4_K 0.310 ms(R=96)。形ではなく型(Q5_0 の MMQ のカーネル)で遅い。K=640 と 768・512 の差は K に比例する程度で、K が 256 の倍数でないことの影響は見えない。

### 確かめていないこと
- 3 段(主機・中継・末端)での読込の時間、KL、答えの文面(親の担当)
- 実際の文の行での時間と誤差。入力は雑音と `--skew 64`(64 個の中心 + 雑音)だけ
- fused の読みの直後の生成(借りた枠 580 の戻し)への影響。probe は読みしか流していない
- `--batch 4` の複数スロットが同時に読むとき、画像つきの読み
- 12,288 を超える長い読み(KV の退避が絡む長さ)
- `STRATA_PREFILL_STREAM_MIN` を下げたときの上の 2 点の原因

### 次の手
1. 親: 3 段で上の exe と旗 4 つを試す(読込の時間・KL・文面)
2. 残りの大きい区間は hc read(48)・GDN の再帰(34。`STRATA_GDN_CHUNKED=1` で 11、誤差 2.2e-3)・GDN の入力の射影(31)
3. 小さい塊で wall の外に増える約 65 ms の原因を調べれば、下限を 256 以下に下げられる可能性がある

### 置き場(3 回目)
- llmsv `~/wt/relay-read/strata.r3`(01552eb50a0a3b0b)。`strata.q50`(999962f6f72415ec、旗が入らない版)と行の保存(`rows-*.f32`)は消した。記録は `~/wt/relay-read-logs/` の `worker-/probe-r3*`(`r3-*` は 2 回目の exe で fused に入っていない。`r3b-`・`r3c-`・`r3d-` が strata.r3)、`mmqb.log`、`win.log`
- trainsv `~/wt/relay-read/build-hip/strata`(同じもの。`TARGETS=strata ~/wt/relay-read-hip.sh`、約 40 秒)

---

## 2 回目(2026-11-26 01:52〜02:05 JST、同じ担当 R)の到達点: 実装まで。実機では 1 回も流していない
親の指示は「専門家の区間(写し 72 + gate/up 101 + down 197 ms)を縮める」。親のセッションの区切りで止めた。2 回目の窓は開かなかった。

### 書いたもの(コミット 6e1a4ad1d と、その次の WIP コミット)
- **`src/prefill/moe_f16_q50.cu`・`.hpp`(新規)**: Q5_0 の down を gfx11 の F16 の matrix core で計算するカーネル。`moe_f16_kq.cu` の Q5_1 の down のカーネルを写し、ブロックの読み方(22 バイト・語の境界に揃わないので `__builtin_memcpy` で読む)と重みの式(d × (q − 16) = q × d + (−16 d))だけを変えた
- `src/prefill/moe_fused_iq.cu` に口を 2 か所(`d_covered_on` と down の起動)、`CMakeLists.txt` に登録
- **旗**: `STRATA_PF_FUSED=1 STRATA_PF_FUSED_KQ=1 STRATA_PF_F16=1 STRATA_PF_F16_Q50=1`(4 つとも要る。どれも既定は切)。これで当社の Q4_K / Q5_0 の層が、上流にすでにある fused の経路(`prefill.cpp` の `fused_l`)に乗るはず。この経路は (a) 専門家を枠から直接読む(`b.blob[]` = `device_slot`。グループのバッファへの写しが無い)、(b) 64 行のタイルごとに仕事を作る(MMQ のように 16 専門家の最大行数に合わせた列を計算しない)、(c) gate/up は既存の `f16kq::gate_up`(Q4_K)
- `tools/relay_read/relay_mmq_bench.cpp`(新規、CMake の `relay_mmq_bench`、EXCLUDE_FROM_ALL): MMQ の 1 グループ(16 専門家)の積を、型(Q4_K / Q5_0)と形(1280x2560・2560x640・2560x768・2560x512・1280x640)と行数(16・48・64・96)を振って測る
- ビルドは通った(trainsv、exe sha256 頭 999962f6f72415ec)。**動作・速さ・出力は未確認**

### 確かめていないこと(次の担当が最初にやる)
1. 旗 4 つで本当に fused の経路に入るか(ワーカーのログに `prompt experts on the fused int8 kernels` が出る)。入る条件は `stream_all`(塊 1,024 トークン以上・リング > STAGE・`m.src` あり)。256 トークンの塊は MMQ のまま
2. `fused_ring()` が真になると読込のバッファとリングの配置が変わる(起動時の「the prompt path borrows N cache slots」の値が変わる)。中継の `STRATA_WK_LEND_SINGLE`・`STRATA_WK_REFILL_EARLY` と一緒に動くかは見ていない
3. 出力は MMQ とビット一致しない(活性は 32 値ごとの int8、重みは F16 に 1 回丸める)。`DUMP=1024` で行を保存して `rows_cmp` で相対 L2 を見る。1e-2 程度なら丸めの差、1 に近ければカーネルの誤り
4. 旗を付けないときに本番と同じハッシュ(3,072 で ad680b8c4f428e89)になること
- 用意した待ち行列: `llmsv:~/wt/relay-read-logs/queue2.txt`(`q-ppf-t`・`q-f50-t`・`q-f50-n`・`q-ppf-n`・`STRATA_PF_F16_BN=64/128`・`STRATA_PF_BATCH=128`)。流し方: 窓の中で `DUMP=1024 REPS=3 ~/wt/relay-read-win.sh ~/wt/relay-read-logs/queue2.txt`、続けて `~/wt/relay-read-mmqb.sh`(`DONE-queue2.txt` を待って `relay_mmq_bench` を流し `mmqb.log` に書く)。READY は消してある

### down の MMQ が遅い原因: 特定できていない
- 分かったこと: 1 ブロックの時間は gate/up・down とも約 0.008 µs × J(J = タイルの列数)で、K に依らない。1 ブロックの積和は down が gate/up の 0.3 倍
- 外した候補: (1) 書き戻しの飛び飛びの書き込み(同じ格子で 1 層分 0.55 ms。`mmq_wb_bench`、実測)、(2) Q5_0 の展開(`load_tiles_q5_0` は 1 値あたり数命令。ソースを読んだだけ)、(3) ホストの待ち(0〜1 ms、実測)、(4) 専門家の並び順による列の無駄(`STRATA_MMQ_GROUP_SORT`・`RESIDENT_SORT_NE` で時間もハッシュも変わらず。ただし雑音の入力は行数がほぼ均等なので、実際の文では別)
- 残った候補(どれも未計測): (a) 32 値ごとの float の後処理(`vec_dot_q8_0_q8_1_mma` は 16x16 のタイルと 32 値ごとに整数 → float の変換と積和を 256 回)。ただしこれは K に比例するはずで、実測と合わない、(b) LDS の大きさ(x のタイル約 35 KB + y のタイル J × 272 B)による同時実行数、(c) `sum[]`(1 スレッド J × 64 / 128 個)のレジスタあふれ、(d) K = 640 が 256 の倍数でないことによる 3 回目の反復の読み越し
- 切り分けの道具は `relay_mmq_bench`。q5_0 2560x640 と q5_0 1280x2560・q4_k 2560x768 を比べれば、型のせいか形のせいかが分かる。2560x768・2560x512 との比較で (d) が分かる

### 専門家の写しを無くす形について
- gfx1100 で枠から直接読めない理由は見つかっていない。MMQ のままやるなら `PROGRESS.md` 下の「次の手 3.」(連続した枠を `expert_bytes` = 枠の間隔で読む)。ただしグループの 16 専門家が連続した枠に並ぶのは全専門家に行が当たるときだけなので、実際の文では使える場面が限られる
- fused の経路は最初から枠を直接読むので、上の Q5_0 のカーネルが動けば写しは無くなる。こちらを先に確かめる

### 置き場(2 回目で増えたもの)
- llmsv `~/wt/relay-read/strata.q50`(999962f6f72415ec。Q5_0 のカーネル入り)、`relay_mmq_bench`、`~/wt/relay-read-mmqb.sh`、`~/wt/relay-read-logs/queue2.txt`。`strata`(d79a8c2065870c74)は 1 回目のまま
- trainsv `~/wt/relay-read/build-hip/`(`TARGETS="strata relay_mmq_bench" ~/wt/relay-read-hip.sh`、約 40 秒)

---
以下は 1 回目(01:00〜01:50 JST)の記録。

## 結論(01:50 JST 時点)
- エンジンのコードは変えていない。変えたのは計測の道具(`tools/relay_read/`)だけ
- **既存の旗 `STRATA_PF_GEMM=1 STRATA_PF_PAD=1` を中継に足すと、3,072 トークンの 1 塊が 656 → 582 ms(−11%)。出力行のハッシュは本番の旗と同じ**(256・1,024・2,048・3,072 の 1 塊と 12,288 = 4 塊の 5 通り、雑音の入力)。dense の FP16 GEMM(GDN の入出力の射影・QSA の射影・ルーター)が hipBLASLt から 128x256 の WMMA に替わる
- 目安(1 トークンあたり 0.22 ms 以下、1,778 トークンで 480 ms 以下)には届かない。残りの 64% は専門家の MMQ(ggml の RDNA3 の int8 WMMA。11-14 から本番で有効)で、ここは縮めていない
- 窓は 1 回使った(16:19〜16:45 UTC)。2 回目は頼んでいない

## 測り方
- llmsv で「偽の末端 `stage_sink`(受けた行をそのまま返す)← 中継(本番と同じ引数・旗、127.0.0.1:7852)← `stage_probe_bench --bench`(主機の代役)」。リンクは fp32 なので、probe のハッシュは中継の出力行そのもののビット
- 入力は正規乱数の行(seed 固定)。**実際の文の行ではない**。全 512 専門家に行が当たる(256 トークンでも gather が 65 ms)。本番ログの実測(下)より短い塊で速く、3,072 では 15% ほど速い
- 区間は `STRATA_PREFILL_TIMING=1` の GPU の印(印ありは印なしより 3,072 で約 30 ms 長い)。3 回目の値。probe の壁時計は fp32 の loopback 2 往復(3,072 で約 330 ms)を含むので、中継自身の時間はワーカーの `wall` を使う
- 実行: `~/wt/relay-read-run.sh TAG EXE [ENV=VAL ...]`(窓が要る)、集計 `~/wt/pft.sh TAG...`、記録 `llmsv:~/wt/relay-read-logs/`(`worker-TAG.log`・`probe-TAG.log`・`win.log`)
- **注意: `base-*`・`dmmq-*`・`wmma-*`・`pf-*`・`side-*`・`both-*`・`pfnp-*`・`wmmahw-*`・`gdnc-*` の記録は、本番の旗のうち `STRATA_MMQ_KQ=1 STRATA_WMMA_GEMM=1 STRATA_HIP_WMMA=1 STRATA_RDNA_MMVQ_WAVE=3 STRATA_RELAY_REPLY_BUFS=4` が抜けた条件**(台本の該当行に TOKEN があり、`grep -v TOKEN` で落としていた)。本番と同じ条件は `prod-*`・`ppf*-*`・`x-*`・`sk-*`・`d-*`

## 内訳(本番の旗、1 塊、ms。`worker-prod-t.log` と `worker-ppf-t.log`)
| 区間 | 256 | 1,024 | 2,048 | 3,072 | 3,072 の割合 | PF_GEMM+PAD の 3,072 |
|---|---:|---:|---:|---:|---:|---:|
| 合計(GPU の印) | 175 | 338 | 511 | 651 | 100% | 577 |
| 専門家 down(H の量子化 + Q5_0 の MMQ) | 32 | 100 | 165 | 197 | 30% | 196 |
| 専門家 gate/up(Q4_K の MMQ + SwiGLU) | 31 | 62 | 86 | 101 | 16% | 100 |
| 専門家の写し(印の名前は dequant。枠 → グループのバッファ) | 65 | 70 | 71 | 72 | 11% | 72 |
| GDN の入力の射影 | 10 | 27 | 44 | 66 | 10% | 29 |
| hc read | 10 | 18 | 32 | 45 | 7% | 45 |
| GDN の再帰 | 2 | 10 | 23 | 33 | 5% | 34 |
| QSA の射影 | 4 | 12 | 22 | 32 | 5% | 18 |
| GDN の出力の射影 | 3 | 13 | 18 | 26 | 4% | 11 |
| QSA の attention | 0 | 3 | 12 | 24 | 4% | 24 |
| ルーター + 共有専門家 | 3 | 7 | 12 | 17 | 3% | 10 |
| hc write | 1 | 4 | 7 | 11 | 2% | 11 |
| ほか(gather・combine・GDN conv・host grouping・embed・select) | 14 | 12 | 19 | 27 | 4% | 27 |

ホストが塊を待つ時間・塊の前後の時間は 0〜1 ms(`host:` の行)。同期の待ちは無い。

## 中継の読み 1 塊の前後(ms)
| トークン | 本番の旗(ワーカーの wall、印あり) | + PF_GEMM + PAD(同) | probe の壁時計(印なし、3 回の中央値)前 → 後 |
|---:|---:|---:|---|
| 256 | 175 | 170 | 189 → 182 |
| 1,024 | 340 | 309 | 478 → 455 |
| 2,048 | 514 | 466 | 749 → 700 |
| 3,072 | 656 | 582 | 980 → 909 |
| 12,288(3,072 × 4) | — | — | 3,208 → 2,921 |

傾きは 0.171 → 0.146 ms/トークン(256 と 3,072 の 2 点)。

本番ログ(`~/b/strata-stage.log` の `chunk T=… read in … ms`、直近 400 塊、実際の文): T=3,072 は平均 766 ms(319 塊)、T 2,048〜2,303 は平均 782 ms(平均 2,203 トークン、27 塊)、T 1,024〜1,279 は 546 ms(4 塊)、T 256 未満は 171 ms(平均 93 トークン、16 塊)。

## 既存の旗を足した結果(本番の旗 + PF_GEMM + PAD に 1 つずつ、3,072 トークン、印あり)
| 旗 | 合計 ms | 変わった区間 | ハッシュ |
|---|---:|---|---|
| (なし) | 577 | — | 本番と同じ |
| `STRATA_GDN_CHUNKED=1` | 555 | GDN の再帰 34 → 11 | 違う。1,024 トークンの出力行で相対 L2 誤差 2.2e-3(行ごとの最大 6.7e-3)、行の最大値に対する最大の差 1.5%(雑音の入力、`rows_cmp`) |
| `STRATA_PA_FAST=1` | 571 | QSA attention 25 → 18 | 違う(誤差は測っていない) |
| `STRATA_MMQ_GROUP_SORT=1` / `STRATA_MMQ_RESIDENT_SORT_NE=1` | 577 / 576 | なし | 同じ |
| `STRATA_HC_UPMIX=1` / `STRATA_SCORES_THREAD=1` | 577 / 576 | なし | 同じ |
| `STRATA_HC_Q8=1` | 起動しない | `blk.18.hc_attn_down.weight is not a bf16 matrix` | — |
| `STRATA_DENSE_MMQ=1` / `STRATA_DEQUANT_SIDE=1`(本番の旗が抜けた条件で) | 変わらず | なし | 同じ |

## 専門家の MMQ について分かったこと(縮めていない)
- gate/up と down は ggml の `mul_mat_q`(RDNA3 の設定: I=64、128 スレッド、stream-k なし)。1 塊で 16 専門家 × 384 グループ
- 1 ブロックあたりの時間は gate/up・down とも約 0.008 µs × J(J = タイルの列数 16・48・64・96)で、K(2,560 と 640)にほぼ依らない。down は 1 ブロックの積和が gate/up の 0.3 倍なのに時間が同じ。積和の量で決まっていない
- 書き戻しの形(64 個 × J 行、行の間隔 10 KB)だけを同じ格子で流すと 1 層分 0.55 ms(`mmq_wb_bench 96 60`、エンジンの down は 1 層 16 ms)。書き戻しの飛び飛びの書き込みは原因ではない
- Q5_0 の展開(`load_tiles_q5_0`)は 1 値あたり数命令で、原因に見えない。原因は特定できていない
- 専門家の写し 72 ms は 1 塊で約 18 GB(6,144 専門家 × 2.97 MB)を写す時間で、約 250 GB/s。帯域の上限に近い。縮めるには写さずに枠から直接読む形(CUDA の `STRATA_MMQ_DIRECT` の HIP 版)が要る

## 次の手(続きの担当へ)
1. 親: 3 段で `STRATA_PF_GEMM=1 STRATA_PF_PAD=1` を中継に足して読込と KL を測る(exe は本番のまま。下の「起動の差分」)
2. MMQ の 1 ブロックの時間が何で決まるかを単体で測る。`mmq_expert_bench`(CUDA 用)を HIP の Q4_K / Q5_0 で動かし、J と K を振る。候補: LDS の大きさによる同時実行数、`sum[]` のレジスタあふれ
3. 写しを無くす: 16 専門家が連続した枠にあるグループは `Product::w` = 枠、`expert_bytes` = 枠の間隔で直接読める(down の最後の行の読み越し 88 バイトを 0 にする必要がある)
4. 実際の文の行で測る(`STRATA_DUMP_MOE_INPUT` か、主機 [0,18) の出力を保存して probe に読ませる)

## 置き場
- ブランチ `relay-read-kernels`(arinkomac `~/work/strata-wt/relay-read`)。push していない
- trainsv: `~/wt/relay-read/`(`build-hip/strata`、`build-tools/`)、ビルド `~/wt/relay-read-hip.sh`(約 100 秒)。道具は `g++ -O2 -std=c++20 -Iinclude tools/relay_read/X.cpp src/net/stage_link.cpp -o build-tools/X -lpthread`、`mmq_wb_bench.cu` はファイルの頭の clang++ の行
- llmsv: `~/wt/relay-read/strata`(土台 c57a85de6 のビルド、sha256 頭 d79a8c2065870c74。`strata.base` は同じものへのリンク)、`stage_sink`・`stage_probe_bench`・`rows_cmp`・`mmq_wb_bench`、台本 `~/wt/relay-read-run.sh`・`relay-read-win.sh`・`pft.sh`
