#!/usr/bin/env python3
"""Detailed live profiling: prefill 4096 x bs4, then decode 512 tokens x bs4.

Sends real concurrent batches (4 in-flight requests) against the running
sglang server on :8070 and captures separate torch-profiler traces per stage.
"""
import json
import os
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8070"
ROOT = os.environ.get(
    "PROFILE_ROOT", "/home/lovedheart/sglang-fork-dev/profiles/bs4_512_20261102"
)


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode()
    try:
        return json.loads(body) if body.strip() else {"success": True}
    except json.JSONDecodeError:
        return {"raw": body[:200]}


def gen(prompt, max_new, seed):
    return post(
        "/generate",
        {
            "text": prompt,
            "sampling_params": {
                "sampling_seed": seed,
                "temperature": 0.0,
                "max_new_tokens": max_new,
                "ignore_eos": True,
            },
            "stream": False,
        },
    )


def num_running():
    with urllib.request.urlopen(URL + "/metrics", timeout=10) as resp:
        for line in resp.read().decode().splitlines():
            if line.startswith("sglang:num_running_reqs{"):
                return int(float(line.rsplit(" ", 1)[1]))
    return -1


def parallel_gen(jobs):
    threads, results = [], [None] * len(jobs)

    def run(i, prompt, mx, seed):
        try:
            results[i] = gen(prompt, mx, seed)
        except Exception as exc:  # noqa: BLE001
            results[i] = {"error": str(exc)}

    for i, (prompt, mx, seed) in enumerate(jobs):
        t = threading.Thread(target=run, args=(i, prompt, mx, seed))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return results


def synthetic_prompt(n_tokens, tag):
    return f"profile_probe_{tag} " + " ".join(["profile"] * n_tokens)


def start_profile(out_dir, profile_id, num_steps):
    return post(
        "/start_profile",
        {
            "output_dir": out_dir,
            "start_step": None,
            "num_steps": num_steps,
            "activities": ["CPU", "GPU"],
            "with_stack": True,
            "record_shapes": False,
            "profile_id": profile_id,
            "profile_by_stage": False,
        },
    )


def wait_for_trace(out_dir, profile_id, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        import glob

        hits = glob.glob(f"{out_dir}/**/*.trace.json.gz", recursive=True)
        hits = [h for h in hits if profile_id in h or profile_id.replace("-", "") in h]
        if hits:
            return hits
        time.sleep(5)
    return []


def capture_prefill():
    out_dir = f"{ROOT}/prefill"
    # Warmup: 3 rounds of 4 concurrent 4096-token prompts (1 output token each).
    for r in range(3):
        jobs = [
            (synthetic_prompt(4096, f"w{r}p{i}"), 1, 1000 + r * 4 + i) for i in range(4)
        ]
        parallel_gen(jobs)
        print(f"[prefill] warmup round {r + 1}/3 done", flush=True)

    print(start_profile(out_dir, "bs4-prefill", 24), flush=True)
    t0 = time.time()
    for r in range(6):
        jobs = [
            (synthetic_prompt(4096, f"c{r}p{i}"), 1, 2000 + r * 4 + i) for i in range(4)
        ]
        parallel_gen(jobs)
        print(f"[prefill] capture round {r + 1} done at {time.time() - t0:.1f}s", flush=True)
    hits = wait_for_trace(out_dir, "bs4-prefill")
    print("[prefill] traces:", hits, flush=True)
    return bool(hits)


def capture_decode():
    out_dir = f"{ROOT}/decode"
    # Warmup: one short 64-token decode round at bs1 to touch decode graphs.
    gen(synthetic_prompt(1, "warmup"), 64, 999)

    jobs = [(synthetic_prompt(1, f"d{i}"), 512, 3000 + i) for i in range(4)]
    threads, results = [], [None] * len(jobs)

    def run(i):
        try:
            results[i] = gen(*jobs[i])
        except Exception as exc:  # noqa: BLE001
            results[i] = {"error": str(exc)}

    for i in range(4):
        t = threading.Thread(target=run, args=(i,))
        t.start()
        threads.append(t)

    # Wait until all 4 requests are in the running set (pure decode phase).
    deadline = time.time() + 120
    while time.time() < deadline and num_running() < 4:
        time.sleep(0.2)
    print(f"[decode] running={num_running()}, arming profiler", flush=True)
    print(start_profile(out_dir, "bs4-decode-512", 100), flush=True)

    for t in threads:
        t.join()
    print("[decode] all 4 requests finished", flush=True)
    hits = wait_for_trace(out_dir, "bs4-decode-512")
    print("[decode] traces:", hits, flush=True)
    return bool(hits)


if __name__ == "__main__":
    import os

    os.makedirs(f"{ROOT}/prefill", exist_ok=True)
    os.makedirs(f"{ROOT}/decode", exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "both"
    ok = True
    if stage in ("both", "prefill"):
        ok &= capture_prefill()
    if stage in ("both", "decode"):
        ok &= capture_decode()
    sys.exit(0 if ok else 1)
