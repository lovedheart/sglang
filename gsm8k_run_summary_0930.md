# GSM8K full pass@1 temp=0 thinking-off（2026-09-30，rebase 后首个完整回归）

被测：8070 在线服务（PID 982926，启动 14:48:59 = 已含 9e5baab224 性能回滚，HEAD 9e5baab224，NEXTN 3/1/4，nvfp4 KV，hicache ON）。
脚本：Documents/gsm8k_pass4_eval.py，--tries 1 / temp 0 / 8-shot / 512 tok / 并发 4 / thinking off。

## 精度
- **1273/1319 = 96.51%**，invalid 0
- 对比：0923（rebase 前）1274/1319 = 96.59%，0927（上轮 rebase 后）1261/1319 = 95.60%
- → rebase 后精度回到 rebase 前水平（±1 题，噪声范围内）

## 性能/解码
- 1319 题 / 504.8 s = 2.61 题/s，输出 ~148k tok ≈ 293 tok/s（并发 4，max_running 5）
- TTFT：p50 ≤0.2 s，均值 286 ms；TPOT 中位 ~7–8 ms
- MTP：spec_accept_length gauge 3.36（≈ 0917 的 3.44 持平），draft 接受率 ≈ 79%
- verify 调用 62,989 次 / run

明细 gsm8k_pass1_t0_full_0930.jsonl；日志 /tmp/gsm8k_0930.log。
注：本 run 同时验证了 prefill 性能回滚（4752fe889d revert = 9e5baab224）生效——TTFT/TPOT 无异常尾部。

## par6 run (20:42, restarted server WITH --qsa-indexer-dtype fp8_e4m3, hicache OFF)
- pass@1 temp=0 thinking-off parallel=6: **1275/1319 = 96.66%**, 381.6 s (3.46 req/s; +25% vs par4)
- NEXTN 3/1/4: accept length 3.36 tok/verify (78.6% rate) — identical to par4 baseline
- Flip vs par4 (1273): lost 10, gained 12 → net +2, noise-band (±2–3); 98.3% verdict-identical, 22.2% byte-identical (matches 09-30 rerun noise profile)
- Records: gsm8k_pass1_t0_full_0930_p6.{jsonl,log}
