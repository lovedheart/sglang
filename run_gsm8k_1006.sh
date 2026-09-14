#!/usr/bin/env bash
# GSM8K run with server-metric delta for accept length / accept rate.
set -uo pipefail
cd /home/lovedheart/sglang-fork-dev

snap() {
  curl -s -m 5 http://127.0.0.1:8070/metrics |
    awk '/^sglang:generation_tokens_total\{is_streaming="false|is_streaming="false"/{if($0 ~ /generation_tokens_total/) g+=$2}
         /^sglang:spec_verify_calls_total/{v+=$2}
         END{printf "%d %d", g, v}'
}

read -r G0 V0 < <(snap)
echo "before: gen_tok=$G0 verify=$V0"

OUT=gsm8k_pass1_t0_full_1006_p4.jsonl
.venv/bin/python gsm8k_eval_1006.py \
  --tokenizer-path /media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-mtpV4 \
  --num-questions 1319 --num-shots 8 --temperature 0 --parallel 4 \
  --max-new-tokens 512 --out "$OUT"
RC=$?

read -r G1 V1 < <(snap)
DT=$((G1 - G0)); DV=$((V1 - V0))
echo "after : gen_tok=$G1 verify=$V1"
if [ "$DV" -gt 0 ]; then
  awk -v g="$DT" -v v="$DV" 'BEGIN{al=g/v; printf "METRIC-DELTA accept_length=%.4f tok/verify  accept_rate=%.4f  (d_tok=%d d_verify=%d)\n", al, (al-1)/3, g, v}'
fi
python3 -c "
import json,sys
rows=[json.loads(l) for l in open('$OUT')]
cd=sum(r['spec_correct_drafts'] or 0 for r in rows); pd=sum(r['spec_proposed_drafts'] or 0 for r in rows)
print(f'REQ-SIDE accept_rate={cd/pd:.4f} correct={cd} proposed={pd}')
"
exit $RC
