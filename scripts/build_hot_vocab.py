# Build a static hot-vocab token map for the NEXTN draft lm_head.
#
# Counts token frequencies over a generic + math/code corpus using the served
# model's tokenizer, keeps the top-N (128-aligned) as the draft vocab, and
# saves an int64 tensor loadable by sglang's load_token_map().
#
# Usage: .venv/bin/python scripts/build_hot_vocab.py [--n 32768] [--out hot32k.pt]
import argparse
import glob
import json
import os
from collections import Counter

import torch
from transformers import AutoTokenizer

MODEL = "/media/lovedheart/models/Qwen3.8-Flash-Next-NVFP4-w4a16-4o6-attnE3M4b128-mtpV4"
FORK = "/home/lovedheart/sglang-fork-dev"


def corpus_texts():
    # 1. GSM8K train: math word problems + chain-of-thought answers.
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    for r in ds:
        yield r["question"] + "\n" + r["answer"]

    # 2. Prior model outputs on GSM8K (actual decode distribution, any domain).
    for p in glob.glob(os.path.join(FORK, "gsm8k_pass1_t0_full_*.jsonl")):
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k in ("completion", "response", "text", "output"):
                    if isinstance(d.get(k), str):
                        yield d[k]

    # 3. HLE test questions (broad academic English).
    try:
        hle = load_dataset("cais/hle", split="test")
        for r in hle:
            q = r.get("question") or ""
            if isinstance(q, str):
                yield q
    except Exception as e:  # cache missing etc.
        print(f"[skip hle] {e}")

    # 4. Fork source + docs: code/markdown prior for the tokenizer.
    n_code = 0
    for pat in ("python/sglang/**/*.py", "docs/**/*.md", "*.md"):
        for p in glob.glob(os.path.join(FORK, pat), recursive=True):
            if n_code >= 4000:
                break
            try:
                with open(p, encoding="utf-8") as f:
                    txt = f.read(200_000)
            except (OSError, UnicodeDecodeError):
                continue
            n_code += 1
            yield txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32768, help="hot set size (mult of 128)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.n % 128 == 0, "N must be 128-aligned for e3m4 block scales"
    out = args.out or f"{FORK}/hot{args.n // 1024}k.pt"

    tok = AutoTokenizer.from_pretrained(MODEL)
    cnt = Counter()
    n_doc = 0
    for text in corpus_texts():
        n_doc += 1
        cnt.update(tok(text)["input_ids"])
    print(f"tokenized {n_doc} docs, {sum(cnt.values())} tokens, {len(cnt)} distinct")

    # Forced tokens: digits, common punctuation/whitespace/math symbols.
    forced = set()
    for c in "0123456789., \n+-=()$%:?!'\"/*^<>[]{}#@_\\`;\t":
        ids = tok.encode(c, add_special_tokens=False)
        if len(ids) == 1:
            forced.add(ids[0])
    # Common number strings seen in CoT answers ("16", "1.50", ...)
    for s in ("10", "100", "1000", "0.", ".5", "1.", "2.", "$"):
        ids = tok.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            forced.add(ids[0])

    ranked = [t for t, _ in cnt.most_common()]
    hot = set(forced)
    for t in ranked:
        if len(hot) >= args.n:
            break
        hot.add(t)
    hot = sorted(hot)[: args.n]
    # Pad to exactly N if the corpus was tiny (unlikely).
    filler = [t for t in range(args.n) if t not in hot][: args.n - len(hot)]
    hot += filler

    ids = torch.tensor(hot, dtype=torch.int64)
    torch.save(ids, out)
    print(f"saved {len(hot)} ids -> {out}")

    # Coverage check on GSM8K *test* (train was in the corpus).
    from datasets import load_dataset

    test = load_dataset("openai/gsm8k", "main", split="test")
    hotset = set(hot)
    tot = hit = 0
    for r in test:
        for pid in tok(r["question"] + "\n" + r["answer"])["input_ids"]:
            tot += 1
            hit += pid in hotset
    print(f"GSM8K-test token coverage: {hit / tot:.4%} ({hit}/{tot})")


if __name__ == "__main__":
    main()
