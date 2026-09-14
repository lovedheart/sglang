#!/usr/bin/env python3
"""AIME 2026 II eval on local sglang, 5 concurrent, with spec-decode accept metrics."""
import json, re, threading, time, urllib.request

BASE = "http://127.0.0.1:8070/v1/chat/completions"
METRICS = "http://127.0.0.1:8070/metrics"
MODEL = "Qwen3.8-Flash-Next"
CONC = 5
MAX_TOKENS = 32768
OUT = "/home/lovedheart/sglang-fork-dev/aime2026_ii_results.json"

ANSWERS = {1:178,2:243,3:503,4:279,5:190,6:50,7:754,8:245,9:669,10:340,11:132,12:223,13:107,14:157,15:393}

PROBLEMS = [
 (1, "Find the sum of the 10th terms of all arithmetic sequences of integers that have first term equal to 4 and include both 24 and 34 as terms."),
 (2, "A grid of 10 squares in a row: vertices at (i,0) and (i,1) for i=0..10, horizontal edges along y=0 and y=1, vertical edges at each x=i, and in each square a diagonal from (i,0) to (i+1,1). A bug moves along the line segments from vertex to vertex, never traversing the same segment twice and never moving from right to left along a horizontal or diagonal segment. Let N be the number of paths the bug can take from the lower left corner A=(0,0) to the upper right corner B=(10,1). Find sqrt(N)."),
 (3, "Let ABCDE be a nonconvex pentagon with internal angles angle A = angle E = 90 degrees and angle B = angle D = 45 degrees. Suppose that DE < AB, AE = 20, BC = 14*sqrt(2), and points B, C, and D lie on the same side of line AE. Suppose further that AB is an integer with AB < 2026 and the area of pentagon ABCDE is an integer multiple of 16. Find the number of possible values of AB."),
 (4, "For each positive integer n, let f(n) be the value of the base-ten numeral n viewed in base b, where b is the least integer greater than the greatest digit in n. For example, if n = 72, then b = 8, and 72 as a numeral in base 8 equals 7*8 + 2 = 58; therefore f(72) = 58. Find the number of positive integers n less than 1000 such that f(n) = n."),
 (5, "An urn contains n marbles. Each marble is either red or blue, and there are at least 7 marbles of each color. When 7 marbles are drawn randomly from the urn without replacement, the probability that exactly 4 of them are red equals the probability that exactly 5 of them are red. Find the sum of the five least values of n for which this is possible."),
 (6, "Find the sum of all real numbers r such that there is at least one point where the circle with radius r centered at (4, 39) is tangent to the parabola with equation 2y = x^2 - 8x + 12."),
 (7, "A standard fair six-sided die is rolled repeatedly. Each time the die reads 1 or 2, Alice gets a coin; each time it reads 3 or 4, Bob gets a coin; and each time it reads 5 or 6, Carol gets a coin. The probability that Alice and Bob each receive at least two coins before Carol receives any coins can be written as m/n, where m and n are relatively prime positive integers. Find 100m + n."),
 (8, "Isosceles triangle ABC has AB = BC. Let I be the incenter of triangle ABC. The perimeters of triangle ABC and triangle AIC are in the ratio 125 : 6, and all the sides of both triangles have integer lengths. Find the minimum possible value of AB."),
 (9, "Let S denote the value of the infinite sum 1/9 + 1/99 + 1/999 + 1/9999 + ... (the k-th term is 1/(10^k - 1)). Find the remainder when the greatest integer less than or equal to 10^100 * S is divided by 1000."),
 (10, "Let ABC be a triangle with D on segment BC such that AD bisects angle BAC. Let w be the circle that passes through A and is tangent to segment BC at D. Let E != A and F != A be the intersections of w with segments AB and AC, respectively. Suppose that AB = 200, AC = 225, and all of AE, AF, BD, and CD are positive integers. Find the greatest possible value of BC."),
 (11, "Find the greatest integer n such that the cubic polynomial x^3 - (n/6)x^2 + (n - 11)x - 400 has roots alpha^2, beta^2, and gamma^2, where alpha, beta, and gamma are complex numbers, and there are exactly seven different possible values for alpha + beta + gamma."),
 (12, "Consider a tetrahedron with two isosceles triangle faces with side lengths 5*sqrt(10), 5*sqrt(10), and 10, and two isosceles triangle faces with side lengths 5*sqrt(10), 5*sqrt(10), and 18. The four vertices of the tetrahedron lie on a sphere with center S, and the four faces of the tetrahedron are tangent to a sphere with center R. The distance RS can be written as m/n, where m and n are relatively prime positive integers. Find m + n."),
 (13, "Call finite sets of integers S and T cousins if S and T have the same number of elements, S and T are disjoint, and the elements of S can be paired with the elements of T so that the elements in each pair differ by exactly 1. For example, {1, 2, 5} and {0, 3, 4} are cousins. Suppose that the set S has exactly 4040 cousins. Find the least number of elements the set S can have."),
 (14, "For integers a and b, let a (op) b = a - b if a is odd and b is even, and a (op) b = a + b otherwise. Find the number of sequences a_1, a_2, ..., a_n of positive integers such that a_1 + a_2 + ... + a_n = 12 and a_1 (op) a_2 (op) ... (op) a_n = 0, where the operations are performed from left to right; that is, a_1 (op) a_2 (op) a_3 means (a_1 (op) a_2) (op) a_3."),
 (15, "Find the number of ordered 7-tuples (a_1, a_2, a_3, a_4, a_5, a_6, a_7) having the following properties: a_k is in {1, 2, 3} for all k; a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 is a multiple of 3; and a_1*a_2*a_4 + a_2*a_3*a_5 + a_3*a_4*a_6 + a_4*a_5*a_7 + a_5*a_6*a_1 + a_6*a_7*a_2 + a_7*a_1*a_3 is a multiple of 3."),
]

SYSTEM = ("You are a competition mathematician. Solve the problem with careful, complete reasoning. "
          "Put your final answer (an integer from 000 to 999) at the very end as \\boxed{ANSWER}.")

lock = threading.Lock()
results = {}

def spec_metrics():
    try:
        with urllib.request.urlopen(METRICS, timeout=5) as r:
            txt = r.read().decode()
        out = {}
        for line in txt.splitlines():
            if line.startswith("#"): continue
            for key in ("sglang:spec_verify_calls_total", "sglang:spec_accept_length", "sglang:spec_accept_rate"):
                if line.startswith(key + "{"):
                    out[key] = float(line.rsplit(" ", 1)[1])
        return out
    except Exception:
        return {}

def extract(text):
    m = re.findall(r"\\boxed\{\s*0*(\d{1,3})\s*\}", text)
    if m: return int(m[-1])
    m = re.findall(r"[Ff]inal [Aa]nswer[:\s]*\$?\$?\\?[a-zA-Z]*\{?\s*(\d{1,3})\b", text)
    if m: return int(m[-1])
    nums = re.findall(r"\b(\d{1,3})\b", text[-500:])
    return int(nums[-1]) if nums else None

def run(num, stmt):
    prompt = f"{stmt}\n\nThink step by step, then end with \\boxed{{}} containing the integer answer."
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
    }).encode()
    t0 = time.time()
    m_start = spec_metrics()
    try:
        req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        msg = ch["message"]
        full = (msg.get("reasoning_content") or "") + "\n" + (msg.get("content") or "")
        u = data.get("usage", {})
        pred = extract(full)
        res = {"problem": num, "pred": pred, "correct": pred == ANSWERS[num],
               "completion_tokens": u.get("completion_tokens"),
               "elapsed_s": round(time.time()-t0, 1), "finish": ch.get("finish_reason"),
               "spec_at_start": m_start, "spec_at_end": spec_metrics()}
    except Exception as e:
        res = {"problem": num, "pred": None, "correct": False, "error": repr(e),
               "elapsed_s": round(time.time()-t0, 1)}
    with lock:
        results[num] = res
        with open(OUT, "w") as f:
            json.dump({str(k): v for k, v in sorted(results.items())}, f, indent=1)
        print(f"P{num}: pred={res.get('pred')} correct={res.get('correct')} "
              f"tokens={res.get('completion_tokens')} {res.get('elapsed_s')}s", flush=True)

if __name__ == "__main__":
    import os
    if os.path.exists(OUT):
        with open(OUT) as f:
            for k, v in json.load(f).items():
                results[int(k)] = v
    todo = [(n, s) for n, s in PROBLEMS if n not in results]
    print(f"Starting: {len(todo)} problems, concurrency {CONC}", flush=True)
    i = 0
    while i < len(todo):
        batch = todo[i:i+CONC]
        th = [threading.Thread(target=run, args=(n, s)) for n, s in batch]
        [t.start() for t in th]; [t.join() for t in th]
        print(f"batch {[n for n,_ in batch]} done, global metrics: {spec_metrics()}", flush=True)
        i += CONC
    score = sum(1 for r in results.values() if r.get("correct"))
    print(f"\nSCORE: {score}/15", flush=True)
    for num in sorted(results):
        r = results[num]
        print(f"P{num:2d}: ans={ANSWERS[num]:3d} pred={str(r.get('pred')):>5} "
              f"{'OK ' if r.get('correct') else 'MISS'} {r.get('completion_tokens')}tok "
              f"{r.get('elapsed_s')}s finish={r.get('finish')}")
