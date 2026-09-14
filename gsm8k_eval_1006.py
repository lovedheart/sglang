"""GSM8K pass@1 eval vs live sglang server: quality + speculative accept rate/length.

temp=0, thinking off, 8-shot. Parallel workers. Per-request spec stats are read
from meta_info (spec_num_correct_drafts / spec_num_proposed_drafts / spec_verify_ct)
and aggregated; server-metric deltas are captured by the wrapper.
"""

import argparse
import ast
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

INVALID = -9999999
_print_lock = threading.Lock()


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    return "".join(get_one_example(lines, i, True) + "\n\n" for i in range(k))


def get_answer_value(s):
    s = s.replace(",", "")
    nums = re.findall(r"\d+", s)
    if not nums:
        return INVALID
    try:
        return ast.literal_eval(nums[-1])
    except SyntaxError:
        return INVALID


def main(args):
    lines = [json.loads(l) for l in open(args.data_path, encoding="utf-8") if l.strip()]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    shots = get_few_shot_examples(lines, args.num_shots)

    prompts, labels = [], []
    for i in range(args.num_questions):
        raw = shots + get_one_example(lines, i, False)
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": raw}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
        )
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(l != INVALID for l in labels), "bad label in dataset head"

    url = f"http://127.0.0.1:{args.port}/generate"
    results = [None] * args.num_questions
    n_done = [0]
    lock = threading.Lock()

    def gen(prompt):
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
            },
            "stop": ["Question", "Assistant:", "<|separator|>"],
            "stream": False,
        }
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, timeout=args.http_timeout)
                if r.status_code == 200:
                    j = r.json()
                    return j["text"], j.get("meta_info", {})
                r.raise_for_status()
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise RuntimeError(f"request failed: {e}") from e
                time.sleep(2 * (attempt + 1))
        return None, {}

    def work(i):
        out, mi = gen(prompts[i])
        pred = get_answer_value(out)
        results[i] = {
            "index": i,
            "label": labels[i],
            "correct_at": 1 if pred == labels[i] else None,
            "attempts": 1,
            "pred_last": pred,
            "completion_tokens": mi.get("completion_tokens"),
            "spec_correct_drafts": mi.get("spec_num_correct_drafts"),
            "spec_proposed_drafts": mi.get("spec_num_proposed_drafts"),
            "spec_verify_ct": mi.get("spec_verify_ct"),
            "question": lines[i]["question"],
            "gold": lines[i]["answer"],
        }
        with lock:
            n_done[0] += 1
            if n_done[0] % 25 == 0 or n_done[0] == args.num_questions:
                ok = sum(1 for r in results if r and r["correct_at"])
                print(
                    f"[{n_done[0]}/{args.num_questions}] pass={ok} "
                    f"acc={ok / n_done[0]:.3f}",
                    flush=True,
                )

    tic = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        list(ex.map(work, range(args.num_questions)))
    latency = time.perf_counter() - tic

    ok = sum(1 for r in results if r["correct_at"])
    invalid = sum(1 for r in results if r["pred_last"] == INVALID)
    cd = sum(r["spec_correct_drafts"] or 0 for r in results)
    pd_ = sum(r["spec_proposed_drafts"] or 0 for r in results)
    vc = sum(r["spec_verify_ct"] or 0 for r in results)
    tok_gen = sum(r["completion_tokens"] or 0 for r in results)
    al = (vc + cd) / vc if vc else 0.0  # accept length: tokens per verify step
    ar = cd / pd_ if pd_ else 0.0  # accept rate: correct / proposed (steps=3)
    al_per_req = sorted(
        (1 + (r["spec_correct_drafts"] or 0) / r["spec_verify_ct"])
        for r in results
        if r["spec_verify_ct"]
    )
    med = al_per_req[len(al_per_req) // 2] if al_per_req else 0.0

    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n=== RESULT ===")
    print(f"Questions          : {len(results)}")
    print(f"pass@1 accuracy    : {ok / len(results):.4f}  ({ok}/{len(results)})")
    print(f"invalid preds      : {invalid}")
    print(f"latency            : {latency:.1f} s  ({len(results) / latency:.2f} req/s)")
    print(f"gen tokens         : {tok_gen}")
    print(f"spec verify calls  : {vc}")
    print(f"spec correct/prop  : {cd}/{pd_}")
    print(f"ACCEPT LENGTH      : {al:.4f} tokens/verify   (median/req {med:.3f})")
    print(f"ACCEPT RATE        : {ar:.4f}   (=(AL-1)/3 = {(al - 1) / 3:.4f})")
    print(f"detail jsonl       : {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="/home/lovedheart/sglang-fork-dev/benchmark/gsm8k/test.jsonl")
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--num-questions", type=int, default=1319)
    p.add_argument("--num-shots", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--port", type=int, default=8070)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--http-timeout", type=float, default=600)
    p.add_argument("--out", default="/home/lovedheart/sglang-fork-dev/gsm8k_pass1_t0_full_1006_p4.jsonl")
    main(p.parse_args())
