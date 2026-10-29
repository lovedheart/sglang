#!/usr/bin/env python3
"""Detailed live profiling: prefill 4096 x bs8, then decode 512 tokens x bs8.

Generalized from tools/profile_bs4_512.py (same method, batch raised to the
server's max_running_requests). Read-only attach to a running sglang server:
sends real concurrent batches and captures separate torch-profiler traces per
stage. Never touches the server process.

Layout of one measurement pass per stage (order matters):
  1. warmup rounds   -- fill prefix/graph state, not measured
  2. control pass    -- unprofiled wall-clock throughput + spec-decode accept
                        stats from /generate meta_info (the reportable number)
  3. profiled pass   -- /start_profile with num_steps so the window closes on
                        its own; NOT used for throughput claims

Env knobs: PROFILE_ROOT, PORT, BS, PREFILL_TOKENS, DECODE_TOKENS.

Known API traps (do not "fix" them): /start_profile answers plain text, and
/stop_profile with nothing armed returns HTTP 500.
"""
import glob
import json
import os
import sys
import threading
import time
import urllib.request

URL = f"http://127.0.0.1:{os.environ.get('PORT', '8070')}"
ROOT = os.environ.get("PROFILE_ROOT", "/tmp/profile_bs8_512")
BS = int(os.environ.get("BS", "8"))
PREFILL_TOKENS = int(os.environ.get("PREFILL_TOKENS", "4096"))
DECODE_TOKENS = int(os.environ.get("DECODE_TOKENS", "512"))

PREFILL_CAPTURE_STEPS = 32
DECODE_CAPTURE_STEPS = 100
MAX_CAPTURE_ROUNDS = 12
CONTROL_ROUNDS = 2


def post(path, payload, timeout=900):
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
    if n_tokens <= 1:
        return f"profile_probe_{tag}"
    return f"profile_probe_{tag} " + " ".join(["profile"] * n_tokens)


def metrics():
    out = {}
    try:
        with urllib.request.urlopen(URL + "/metrics", timeout=10) as resp:
            for line in resp.read().decode().splitlines():
                if line.startswith("sglang:"):
                    key, val = line.rsplit(" ", 1)
                    out[key.split("{")[0]] = float(val)
    except Exception:  # noqa: BLE001
        pass
    return out


def num_running():
    return int(metrics().get("sglang:num_running_reqs", -1))


def gpu_note():
    """GPU idleness/util via NVML (nvidia-smi binary is unreliable on this box)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        m = pynvml.nvmlDeviceGetMemoryInfo(h)
        return f"gpu_util={u.gpu}% mem_used={m.used / 1e9:.1f}GiB"
    except Exception as exc:  # noqa: BLE001
        return f"nvml_unavailable:{exc}"


def spec_stats(results):
    """Acceptance stats from /generate meta_info (see spec_decoding_metrics.py)."""
    mean_keys = ("spec_accept_length", "spec_accept_rate", "spec_cap_length")
    sum_keys = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
    agg, seen, hist = {}, 0, None
    for r in results:
        mi = (r or {}).get("meta_info") or {}
        if not mi:
            continue
        seen += 1
        for k in mean_keys + sum_keys:
            if k in mi:
                agg.setdefault(k, []).append(mi[k])
        h = mi.get("spec_correct_drafts_histogram")
        if h:
            hist = [a + b for a, b in zip(hist, h)] if hist else list(h)
    summary = {}
    summary.update({k: round(sum(v) / len(v), 4) for k, v in agg.items() if k in mean_keys})
    summary.update({k: sum(v) for k, v in agg.items() if k in sum_keys})
    if "spec_num_proposed_drafts" in summary:
        summary["spec_accept_rate_recomputed"] = round(
            summary["spec_num_correct_drafts"] / summary["spec_num_proposed_drafts"], 4
        )
    if hist:
        summary["correct_drafts_histogram"] = hist
    summary["requests_with_meta"] = seen
    return summary


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


def trace_hits(out_dir, profile_id):
    hits = glob.glob(f"{out_dir}/**/*.trace.json.gz", recursive=True)
    return [h for h in hits if profile_id in h or profile_id.replace("-", "") in h]


def wait_for_trace(out_dir, profile_id, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        hits = trace_hits(out_dir, profile_id)
        if hits:
            return hits
        time.sleep(5)
    return []


def start_round_async(jobs):
    """Fire a batch of /generate calls and return their threads immediately."""
    threads = []
    for prompt, mx, seed in jobs:
        t = threading.Thread(target=gen, args=(prompt, mx, seed), daemon=True)
        t.start()
        threads.append(t)
    return threads


def collect(report, stage, workload, capture_steps, gate):
    """warmup -> unprofiled control -> profiled capture for one stage."""
    out_dir = f"{ROOT}/{stage}"
    os.makedirs(out_dir, exist_ok=True)
    pid = f"bs{BS}-{stage}"

    for r in range(2):
        parallel_gen(workload(r, "w"))
        print(f"[{stage}] warmup round {r + 1}/2 {gpu_note()}", flush=True)

    t0 = time.time()
    results = []
    for r in range(CONTROL_ROUNDS):
        results.extend(parallel_gen(workload(90 + r, "c")))
    wall = time.time() - t0
    ctrl = {"rounds": CONTROL_ROUNDS, "bs": BS, "wall_s": round(wall, 2)}
    ctrl.update(workload.throughput(results, wall, CONTROL_ROUNDS))
    ctrl["spec"] = spec_stats(results)
    print(f"[{stage}] CONTROL {json.dumps(ctrl)} {gpu_note()}", flush=True)
    report.setdefault("control", {})[stage] = ctrl

    # Profiled pass: keep feeding batches until the num_steps window closes on
    # its own and flushes the trace. Sending a fixed batch count would leave the
    # profiler armed forever on a stage whose step count is not 1:1 with
    # requests (prefill is chunked).
    rounds = 0
    if gate is not None:
        # Decode: the full batch must already be running when the window opens,
        # otherwise the capture records a warm-up window at a partial batch.
        threads = start_round_async(workload(200, "p"))
        gate()
        rounds = 1
    print(f"[{stage}] armed at running={num_running()}", flush=True)
    t0 = time.time()
    print(start_profile(out_dir, pid, capture_steps), flush=True)
    if gate is not None:
        for t in threads:
            t.join()
    while rounds < MAX_CAPTURE_ROUNDS and not trace_hits(out_dir, pid):
        parallel_gen(workload(200 + rounds, "p"))
        rounds += 1
    hits = wait_for_trace(out_dir, pid)
    print(
        f"[{stage}] profiled pass: {rounds} round(s) in {time.time() - t0:.1f}s, "
        f"traces: {hits}",
        flush=True,
    )
    report.setdefault("traces", {})[stage] = hits
    report.setdefault("profiled_rounds", {})[stage] = rounds
    return bool(hits)


def make_prefill_workload():
    def workload(round_base, tag):
        return [
            (synthetic_prompt(PREFILL_TOKENS, f"{tag}{round_base}{i}"), 1, 1000 + round_base * BS + i)
            for i in range(BS)
        ]

    def throughput(results, wall, rounds):
        toks = rounds * BS * PREFILL_TOKENS
        return {"prefill_tokens": toks, "tok_per_s": round(toks / wall, 1)}

    workload.throughput = throughput
    return workload


def make_decode_workload():
    def workload(round_base, tag):
        return [(synthetic_prompt(1, f"{tag}{round_base}{i}"), DECODE_TOKENS, 5000 + round_base * BS + i) for i in range(BS)]

    def throughput(results, wall, rounds):
        toks = sum((r or {}).get("meta_info", {}).get("completion_tokens", DECODE_TOKENS) for r in results)
        verify = sum((r or {}).get("meta_info", {}).get("spec_verify_ct", 0) for r in results)
        # verify_ct is per request and every request in the batch advances on the
        # same forward, so scheduler steps per round ~= one request's verify_ct.
        # Each EAGLE verify step is 1 verify + (num_draft_tokens-1) draft forwards.
        steps_per_round = verify / len(results) if verify else max(1.0, toks / max(1, BS))
        return {
            "decode_tokens": toks,
            "tok_per_s": round(toks / wall, 1),
            "verify_steps_per_round": round(steps_per_round, 1),
            "ms_per_verify_step": round(wall * 1000 / (rounds * steps_per_round), 3),
        }

    workload.throughput = throughput
    return workload


def decode_gate():
    deadline = time.time() + 180
    while time.time() < deadline and num_running() < BS:
        time.sleep(0.2)
    # Let the 1-token prefills retire so the window is pure decode.
    time.sleep(1.0)


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(
        f"config URL={URL} BS={BS} prefill={PREFILL_TOKENS} decode={DECODE_TOKENS} "
        f"root={ROOT} {gpu_note()}",
        flush=True,
    )
    report = {}
    ok = True
    if stage in ("both", "prefill"):
        ok &= collect(report, "prefill", make_prefill_workload(), PREFILL_CAPTURE_STEPS, None)
    if stage in ("both", "decode"):
        ok &= collect(report, "decode", make_decode_workload(), DECODE_CAPTURE_STEPS, decode_gate)
    with open(f"{ROOT}/control.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("REPORT " + json.dumps(report), flush=True)
    sys.exit(0 if ok else 1)
