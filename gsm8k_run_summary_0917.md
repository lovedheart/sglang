# GSM8K 300 + 接受率（2026-09-17）

被测：8070 在线服务（sglang-fork-dev，HEAD 2497b6e39c，NEXTN 3/1/4，nvfp4 KV，hicache ON，max_running_requests=4）。
脚本：Documents/gsm8k_pass4_eval.py，thinking off / temp0.7 / top_p0.95 / 512 tok / 并发 4 / pass@4 早停。

## 精度
- pass@4 = **298/300 = 99.33%**，first-try 97.67%，@1/@2/@3/@4 = .9767/.9867/.9900/.9933，invalid 0
- 314 req / 110.5 s
- 与历次持平：09-16 FP8idx+FP4attn 298/300（并列最高），09-14 297/295

## MTP 接受率（run 前后 /metrics delta）
- decode tok delta = (117135-66403)+(7545-6766) = 51511；verify delta = 38146-23154 = 14992
- **3.44 tok/verify**（= 09-16 的 3.55 持平略低）；accept_length ≈ 3.44，draft 接受率 ≈ (3.44-1)/3 ≈ **81%**
- 瞬时 gauge run 前 3.375 / 0.79

明细 gsm8k_pass4_results_20260917.jsonl；日志 /tmp/gsm8k_0917.log。
