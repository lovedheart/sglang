#!/usr/bin/env python3
"""AIME 2026 I eval against local sglang server, 5 concurrent requests."""
import json, re, threading, time, urllib.request

BASE = "http://127.0.0.1:8070/v1/chat/completions"
MODEL = "Qwen3.8-Flash-Next"
CONC = 5
MAX_TOKENS = 16384
OUT = "/home/lovedheart/sglang-fork-dev/aime2026_results.json"

PROBLEMS = [
 (1, "Patrick started walking at a constant speed along a straight road from his school to the park. One hour after Patrick left, Tanya started running at a constant speed of 2 miles per hour faster than Patrick walked, following the same straight road from the school to the park. One hour after Tanya left, Jose started bicycling at a constant speed of 7 miles per hour faster than Tanya ran, following the same straight road from the school to the park. All three people arrived at the park at the same time. The distance from the school to the park is m/n miles, where m and n are relatively prime positive integers. Find m+n."),
 (2, "Find the number of positive integer palindromes written in base 10, with no zero digits, and whose digits add up to 13. For example, 42124 has these properties. Recall that a palindrome is a number whose representation reads the same from left to right as from right to left."),
 (3, "A hemisphere with radius 200 sits on top of a horizontal circular disk with radius 200, and the hemisphere and disk have the same center. Let T be the region of points P in the disk such that a sphere of radius 42 can be placed on top of the disk at P and lie completely inside the hemisphere. The area of T divided by the area of the disk is p/q, where p and q are relatively prime positive integers. Find p+q."),
 (4, "Find the number of integers less than or equal to 100 that are equal to a+b+ab for some choice of distinct positive integers a and b."),
 (5, "A plane contains points A and B with AB = 1. Point A is rotated in the plane counterclockwise through an acute angle theta around point B to point A'. Then B is rotated in the plane clockwise through angle theta around point A' to point B'. Suppose AB' = 4/3. The value of cos(theta) can be written as m/n, where m and n are relatively prime positive integers. Find m+n."),
 (6, "The product of all positive real numbers x satisfying the equation (x^(2026) / x^20)^(1/log base-2026 of... — precisely: the 20th root of (x raised to the (2026 / log-base-x-of-20))? NO. Use this exact equation: the 20th root of x^(2026 / (20 * log base 2026 of x))... "),
 (7, "Find the number of functions pi mapping the set A = {1, 2, 3, 4, 5, 6} onto A such that for every a in A, pi(pi(pi(pi(pi(pi(a)))))) = a."),
 (8, "Let N be the number of positive integer divisors of 170^17 that leave a remainder of 5 upon division by 12. Find the remainder when N is divided by 1000."),
 (9, "Joanne has a blank fair six-sided die and six stickers each displaying a different integer from 1 to 6. Joanne rolls the die and then places the sticker labeled 1 on the top face of the die. She then rolls the die again, places the sticker labeled 2 on the top face, and continues this process to place the rest of the stickers in order. If the die ever lands with a sticker already on its top face, the new sticker is placed to cover the old sticker. Let p be the conditional probability that at the end of the process exactly one face has been left blank, given that all the even-numbered stickers are visible on faces of the die. Then p can be written as m/n, where m and n are relatively prime positive integers. Find m+n."),
 (10, "Let triangle ABC have side lengths AB = 13, BC = 14, and CA = 15. Triangle A'B'C' is obtained by rotating triangle ABC about its circumcenter so that A'C' is perpendicular to BC, with A' and B not on the same side of line B'C'. Find the integer closest to the area of hexagon AA'CC'BB'."),
 (11, "The integers from 1 to 64 are placed in some order into an 8x8 grid of cells with one number in each cell. Let a_{i,j} be the number placed in the cell in row i and column j, and let M be the sum of the absolute differences between adjacent cells. That is, M = sum over i=1..8, j=1..7 of (|a_{i,j+1} - a_{i,j}| + |a_{j+1,i} - a_{j,i}|). Find the remainder when the maximum possible value of M is divided by 1000."),
 (12, "Triangle ABC lies in plane P with AB = 6, AC = 4, and angle BAC = 90 degrees. Let D be the reflection across BC of the centroid of triangle ABC. Four spheres, all on the same side of P, have radii 1, 2, 3, and r and are tangent to P at points A, B, C, and D, respectively. The four spheres are also each tangent to a second plane T and are all on the same side of T. The value of r can be written as m/n, where m and n are relatively prime positive integers. Find m+n."),
 (13, "For each nonnegative integer r less than 502, define S_r = sum over m >= 0 of C(10000, 502m + r), where C(10000, n) is defined to be 0 when n > 10000. That is, S_r is the sum of all binomial coefficients C(10000, k) for which 0 <= k <= 10000 and k - r is a multiple of 502. Find the number of integers in the list S_0, S_1, ..., S_501 that are multiples of the prime number 503."),
 (14, "In an equiangular pentagon, the sum of the squares of the side lengths equals 308, and the sum of the squares of the diagonal lengths equals 800. The square of the perimeter of the pentagon can be expressed as m*sqrt(n), where m and n are positive integers and n is not divisible by the square of any prime. Find m+n."),
 (15, "Let a, b, and n be positive integers with both a and b greater than or equal to 2 and less than or equal to 2n. Define an a x b cell loop in a 2n x 2n grid of cells to be the 2a + 2b - 4 cells that surround an (a-2) x (b-2) (possibly empty) rectangle of cells in the grid. For example, the following diagram shows a way to partition a 6 x 6 grid of cells into 4 cell loops. Find the number of ways to partition a 10 x 10 grid of cells into 5 cell loops so that every cell of the grid belongs to exactly one cell loop."),
]

# fix problem 6 with clean statement
PROBLEMS[5] = (6, "The product of all positive real numbers x satisfying the equation (x^(log base 2026 of x))^(1/20) = 26x is an integer P. Find the number of positive integer divisors of P.")

SYSTEM = ("You are a competition mathematician. Solve the problem with careful, complete reasoning. "
          "Put your final answer (an integer from 000 to 999) at the very end as \\boxed{ANSWER}.")

lock = threading.Lock()
results = {}

def extract(text):
    m = re.findall(r"\\boxed\{\s*0*(\d{1,3})\s*\}", text)
    if m: return int(m[-1])
    m = re.findall(r"[Ff]inal [Aa]nswer[:\s]*\$?\$?\\?[a-zA-Z]*\{?\s*(\d{1,3})\b", text)
    if m: return int(m[-1])
    nums = re.findall(r"\b(\d{1,3})\b", text[-500:])
    return int(nums[-1]) if nums else None

def run(idx, num, stmt):
    prompt = f"{stmt}\n\nThink step by step, then end with \\boxed{{}} containing the integer answer."
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=7200) as r:
            data = json.loads(r.read())
        ch = data["choices"][0]
        text = ch["message"].get("content") or ""
        reasoning = (ch["message"].get("reasoning_content") or "")
        full = reasoning + "\n" + text
        u = data.get("usage", {})
        pred = extract(full)
        res = {"problem": num, "pred": pred, "correct": pred == ANSWERS[num],
               "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"),
               "reasoning_tokens": (ch["message"].get("reasoning_tokens") if isinstance(ch["message"].get("reasoning_tokens"), int) else u.get("reasoning_tokens")),
               "elapsed_s": round(time.time()-t0, 1), "finish": ch.get("finish_reason"),
               "tail": full[-400:]}
    except Exception as e:
        res = {"problem": num, "pred": None, "correct": False, "error": repr(e), "elapsed_s": round(time.time()-t0, 1)}
    with lock:
        results[num] = res
        with open(OUT, "w") as f:
            json.dump({str(k): v for k, v in sorted(results.items())}, f, indent=1)
        print(f"P{num}: pred={res.get('pred')} correct={res.get('correct')} tokens={res.get('completion_tokens')} {res.get('elapsed_s')}s", flush=True)

ANSWERS = {1:277,2:62,3:79,4:70,5:65,6:441,7:396,8:244,9:29,10:156,11:896,12:161,13:39,14:681,15:83}



if __name__ == "__main__":
    import os
    if os.path.exists(OUT):
        with open(OUT) as f:
            for k, v in json.load(f).items():
                results[int(k)] = v
    todo = [(num, stmt) for num, stmt in PROBLEMS if num not in results]
    print(f"Resuming: {len(todo)} problems left, concurrency {CONC}", flush=True)
    i = 0
    while i < len(todo):
        batch = todo[i:i+CONC]
        threads = []
        for (num, stmt) in batch:
            t = threading.Thread(target=run, args=(num, num, stmt))
            t.start(); threads.append(t)
        for t in threads: t.join()
        i += CONC
    score = sum(1 for r in results.values() if r.get("correct"))
    print(f"\nSCORE: {score}/15", flush=True)
    for num in sorted(results):
        r = results[num]
        print(f"P{num:2d}: ans={ANSWERS[num]:3d} pred={str(r.get('pred')):>5} {'OK ' if r.get('correct') else 'MISS'} {r.get('completion_tokens')}tok {r.get('elapsed_s')}s")
