#!/usr/bin/env python3
"""Re-run the 4 failed AIME 2026 problems with larger max tokens via /generate (per-request spec stats)."""
import json, re, threading, time, urllib.request
import aime2026_eval as A1
import aime2026_ii_eval as A2
from transformers import AutoTokenizer

GEN = "http://127.0.0.1:8070/generate"
TOK = AutoTokenizer.from_pretrained(
    "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnFP8b128-mtpV4",
    trust_remote_code=True)
SYS = ("You are a competition mathematician. Solve the problem with careful, complete reasoning. "
       "Put your final answer (an integer from 000 to 999) at the very end as \\boxed{ANSWER}.")

P1 = dict(A1.PROBLEMS); P2 = dict(A2.PROBLEMS)
JOBS = [
    ("I", 8, P1[8], 32768),
    ("I", 15, P1[15], 131072),
    ("II", 13, P2[13], 65536),
    ("II", 14, P2[14], 65536),
]

def extract(text):
    m = re.findall(r"\\boxed\{\s*0*(\d{1,3})\s*\}", text)
    if m: return int(m[-1])
    nums = re.findall(r"\b(\d{1,3})\b", text[-500:])
    return int(nums[-1]) if nums else None

def run(exam, num, stmt, mt):
    prompt = TOK.apply_chat_template(
        [{"role":"system","content":SYS},{"role":"user","content":stmt+"\n\nThink step by step, then end with \\boxed{} containing the integer answer."}],
        add_generation_prompt=True, tokenize=False)
    body = json.dumps({"text": prompt,
                       "sampling_params": {"max_new_tokens": mt, "temperature": 0.6}}).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(GEN, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=14400) as r:
            data = json.loads(r.read())
        mi = data["meta_info"]
        pred = extract(data.get("text", ""))
        ans = (A1 if exam=="I" else A2).ANSWERS[num]
        res = {"pred": pred, "answer": ans, "correct": pred == ans,
               "completion_tokens": mi.get("completion_tokens"),
               "finish": mi.get("finish_reason",{}).get("type"),
               "accept_length": mi.get("spec_accept_length"),
               "accept_rate": mi.get("spec_accept_rate"),
               "histogram": mi.get("spec_correct_drafts_histogram"),
               "decode_throughput": mi.get("decode_throughput"),
               "elapsed_s": round(time.time()-t0,1)}
    except Exception as e:
        res = {"pred": None, "correct": False, "error": repr(e), "elapsed_s": round(time.time()-t0,1)}
    print(f"{exam}-{num}: {json.dumps(res)}", flush=True)
    return res

if __name__ == "__main__":
    th = [threading.Thread(target=run, args=j) for j in JOBS]
    [t.start() for t in th]; [t.join() for t in th]
