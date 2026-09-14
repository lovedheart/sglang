#!/usr/bin/env python3
"""Full AIME 2026 (I+II, 30 problems) via /generate with per-request spec accept stats."""
import json, re, threading, time, urllib.request

GEN = "http://127.0.0.1:8070/generate"
CONC = 5
OUT = "/home/lovedheart/sglang-fork-dev/aime2026_full_stats.json"

import aime2026_eval as A1     # PROBLEMS (I) + ANSWERS
import aime2026_ii_eval as A2  # PROBLEMS (II) + ANSWERS

# (exam, num, stmt, max_tokens)
JOBS = []
for n, s in A1.PROBLEMS:
    JOBS.append(("I", n, s, 32768 if n != 11 else 65536))
for n, s in A2.PROBLEMS:
    JOBS.append(("II", n, s, 65536 if n in (13, 14) else 32768))

SYS = ("You are a competition mathematician. Solve the problem with careful, complete reasoning. "
       "Put your final answer (an integer from 000 to 999) at the very end as \\boxed{ANSWER}.")

from transformers import AutoTokenizer
TOK = AutoTokenizer.from_pretrained(
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-mtpV4",
    trust_remote_code=True)

lock = threading.Lock()
results = {}
import os
if os.path.exists(OUT):
    with open(OUT) as f: results = json.load(f)

def extract(text):
    m = re.findall(r"\\boxed\{\s*0*(\d{1,3})\s*\}", text)
    if m: return int(m[-1])
    nums = re.findall(r"\b(\d{1,3})\b", text[-500:])
    return int(nums[-1]) if nums else None

def run(job):
    exam, num, stmt, mt = job
    key = f"{exam}-{num}"
    prompt = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"
              f"{stmt}\n\nThink step by step, then end with \\boxed{{}} containing the integer answer.<|im_end|>\n"
              f"<|im_start|>assistant\n")
    body = json.dumps({"text": prompt,
                       "sampling_params": {"max_new_tokens": mt, "temperature": 0.6},
                       "no_stop": False}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            data = json.loads(r.read())
        mi = data["meta_info"]
        text = data.get("text", "")
        pred = extract(text)
        ans = (A1 if exam == "I" else A2).ANSWERS[num]
        res = {"exam": exam, "num": num, "pred": pred, "correct": pred == ans, "answer": ans,
               "completion_tokens": mi.get("completion_tokens"),
               "finish": str(mi.get("finish_reason", {}).get("type")),
               "accept_length": mi.get("spec_accept_length"),
               "accept_rate": mi.get("spec_accept_rate"),
               "correct_drafts": mi.get("spec_num_correct_drafts"),
               "proposed_drafts": mi.get("spec_num_proposed_drafts"),
               "verify_ct": mi.get("spec_verify_ct"),
               "histogram": mi.get("spec_correct_drafts_histogram"),
               "decode_throughput": mi.get("decode_throughput"),
               "elapsed_s": round(time.time()-t0,1)}
    except Exception as e:
        res = {"exam": exam, "num": num, "pred": None, "correct": False, "error": repr(e),
               "elapsed_s": round(time.time()-t0,1)}
    with lock:
        results[key] = res
        with open(OUT, "w") as f: json.dump(results, f, indent=1)
        print(f"{key}: pred={res.get('pred')} ok={res.get('correct')} "
              f"AL={res.get('accept_length')} AR={res.get('accept_rate')} "
              f"tok={res.get('completion_tokens')} {res.get('elapsed_s')}s", flush=True)

if __name__ == "__main__":
    todo = [j for j in JOBS if f"{j[0]}-{j[1]}" not in results]
    print(f"running {len(todo)}/{len(JOBS)} jobs, conc {CONC}", flush=True)
    i = 0
    while i < len(todo):
        th = [threading.Thread(target=run, args=j) for j in todo[i:i+CONC]]
        [t.start() for t in th]; [t.join() for t in th]
        i += CONC
    ok = sum(1 for r in results.values() if r.get("correct"))
    print(f"\nTOTAL: {ok}/{len(results)}")
