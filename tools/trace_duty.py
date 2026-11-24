#!/usr/bin/env python3
"""GPU duty-cycle + gap census for one torch-profiler chrome trace.

Kernel *shares* answer "what is expensive". They do not answer "how much of the
wall clock the GPU is actually busy", which is the number that separates a
kernel-bound stage from a dispatch-bound one (e.g. eager prefill where the GPU
sits idle between forwards). This tool streams the trace with ijson (traces are
100 MB+ gzipped, json.load would blow up) and reports:

  span / busy / duty   busy = union of kernel+memcpy+memset intervals, so
                       overlapping streams are not double counted
  gap census           idle stretches on the GPU timeline, bucketed + top N
  stream concurrency   max simultaneous in-flight GPU activities
  api-side totals      CPU launch APIs (cudaGraphLaunch / cudaLaunchKernel)

Usage: python tools/trace_duty.py <trace.json.gz> [more traces...]
"""

import gzip
import json
import sys
from collections import Counter, defaultdict

import ijson

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
BUCKETS = (
    (10, "ge10us"),
    (100, "ge100us"),
    (1000, "ge1ms"),
    (10000, "ge10ms"),
    (10**9, "ge100ms"),
)


def stream_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        for ev in ijson.items(fh, "traceEvents.item"):
            yield ev


def analyse(path):
    gpu = []  # (start, end, name, stream)
    api = Counter()
    api_dur = defaultdict(float)
    for ev in stream_events(path):
        cat = ev.get("cat")
        if cat in GPU_CATS:
            # ijson yields Decimal; keep everything float or the sums explode.
            ts, dur = ev.get("ts"), ev.get("dur")
            if ts is None or dur is None:
                continue
            ts, dur = float(ts), float(dur)
            gpu.append(
                (
                    ts,
                    ts + dur,
                    ev.get("name", "?"),
                    ev.get("args", {}).get("stream", -1),
                )
            )
        elif cat == "cuda_runtime":
            name = ev.get("name", "?")
            if name.startswith("cuda") or name.startswith("cuGraph"):
                api[name] += 1
                api_dur[name] += float(ev.get("dur") or 0.0)
    if not gpu:
        return {"path": path, "error": "no GPU events"}

    gpu.sort()
    span = gpu[-1][1] - gpu[0][0]
    busy = 0.0
    cur_s, cur_e = gpu[0][0], gpu[0][1]
    gaps = []
    for s, e, _n, _st in gpu[1:]:
        if s > cur_e:
            gaps.append((s - cur_e, cur_e))
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s

    sweep = []
    for s, _e, _n, _st in gpu:
        sweep.append((s, 1))
    for _s, e, _n, _st in gpu:
        sweep.append((e, -1))
    sweep.sort()
    depth = mx = 0
    for _t, d in sweep:
        depth += d
        mx = max(mx, depth)

    gaps.sort(reverse=True)
    bucket_counts = Counter()
    bucket_time = defaultdict(float)
    for g, _t in gaps:
        # Descending: the first match is the smallest bucket the gap falls into.
        for lim, label in reversed(BUCKETS):
            if g >= lim:
                bucket_counts[label] += 1
                bucket_time[label] += g
                break
        else:
            bucket_counts["lt10us"] += 1
            bucket_time["lt10us"] += g
    labels = ["lt10us"] + [label for _lim, label in BUCKETS]
    return {
        "path": path,
        "gpu_events": len(gpu),
        "span_ms": round(span / 1000, 1),
        "busy_ms": round(busy / 1000, 1),
        "duty_pct": round(100 * busy / span, 1),
        "max_concurrent_streams": mx,
        "gap_buckets_us": {
            label: [
                bucket_counts.get(label, 0),
                round(bucket_time.get(label, 0) / 1000, 1),
            ]
            for label in labels
        },
        "top_gaps_ms": [[round(g / 1000, 2)] for g, _t in gaps[:10]],
        "cpu_launch_apis": {
            k: [api[k], round(api_dur[k] / 1000, 1)]
            for k in sorted(api_dur, key=lambda k: -api_dur[k])[:6]
        },
    }


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(analyse(p), indent=1))
