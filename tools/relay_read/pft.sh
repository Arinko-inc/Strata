#!/bin/bash
# pft.sh TAG...: the 3rd "prefill timing" line of each chunk size in worker-TAG.log (compact), and the probe's wall times
for tag in "$@"; do
  grep -a "prefill timing: [0-9]* tokens" ~/wt/relay-read-logs/worker-$tag.log | awk -v tag=$tag '
    { n[$4]++; if (n[$4] != 3) next; s=$0; sub(/.*prefill timing: /,"",s); gsub(/ \([0-9.]+%\)/,"",s);
       gsub(/GPU timeline/,"tl",s); gsub(/host staging [0-9]+ ms:/,"",s); gsub(/embed\+steps/,"emb",s); gsub(/router\+shared/,"rtr",s); gsub(/host grouping/,"hg",s); gsub(/gemm gate\/up/,"GU",s); gsub(/gemm down/,"DN",s); gsub(/gdn conv\+gates/,"gconv",s); gsub(/gdn recurrence/,"grec",s); gsub(/gdn out proj/,"gout",s); gsub(/hc read/,"hcr",s); gsub(/hc write/,"hcw",s); gsub(/qsa proj/,"qproj",s); gsub(/qsa attn/,"qattn",s); gsub(/qsa select/,"qsel",s); gsub(/qsa indexer/,"qidx",s); gsub(/wait copy/,"wcp",s); gsub(/dequant/,"dq",s); gsub(/kv stage/,"kvs",s); gsub(/ [a-z]+ 0( |$)/," ",s); gsub(/ [a-z]+ 0( |$)/," ",s); print tag": "s }'
  grep "tokens in" ~/wt/relay-read-logs/probe-$tag.log | awk -v tag=$tag '{k=$4; ms[k]=ms[k]" "$7; h[k]=$10} END {for (k in ms) print tag, "probe", k":"ms[k], h[k]}' | sort -t' ' -k3n
done
