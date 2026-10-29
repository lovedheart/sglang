# Strata v0.1.41 源码级机制说明（用于移植进 sglang fork）

仓库：`/home/lovedheart/projects/Strata`（Niko1221/Strata v0.1.41, git fb58e0db）。
模型常量（`include/strata/kernels/cpu/expert.hpp:34-45`）：H=2560（n_embd）、FF=640（expert 中间宽）、NE=512（每层专家数）、K=10（router top-10）、48 层 → 24576 专家；Q2_0 规范 blob = `3*(H*FF*18/64)` = **1,382,400 B/专家**。native pack（IQ2_XS/IQ3_XXS/Q4_K 等 GGUF 原形）blob 逐层不同，由 `expert_layout()`（`include/strata/kernels/cpu/expert_layout.hpp:18-42`，`native_experts.txt` v3/v4 驱动）提供 `blob_bytes(layer)`。

---

## 1. 自适应专家缓存（VRAM expert tier）

### 关键文件
- `include/strata/core/expert_cache.hpp`（245 行，设计注释极详尽）— 槽位存储 + 驻留表
- `src/core/expert_cache.cpp` — open/admit/fill/verify；`read_expert_profile`/`write_expert_profile`/`rank_learned_profile`（L72-210）
- `src/program/generate.cpp`
  - L4283-4400：`--expert-cache auto` 定容
  - L7380-7560：`adapt()` lambda（每 adapt_every 窗运行一次）— **晋升/换出算法就在这一个 lambda 里**
  - L253-300：`resident_stage_swaps()` — **换出专家写回 RAM 的那段**
  - L7384/7547-7550/7796/7906-7918：`--expert-profile-save` 的 heat 累计与序列化
- `src/core/expert_source.cpp` L2745-2900：`reserve_exchanges`/`stage_exchange`/`commit_exchanges`（RAM 侧 exchange 缓冲与提交）
- `src/core/expert_source.cpp` L3136-3138：热度计数递增点

### 核心数据结构
`ExpertCache`（expert_cache.hpp:81-243，字段级）：
- `uint8_t* base_`：VRAM arena 基址（一次 cudaMalloc，或 VMM 分段 `segs_/seg_size_/mapped_segs_`，#533 `--vram-elastic`）
- `std::vector<int32_t> residency_`：`[n_layers*n_expert]` → slot 或 `kNotResident(-1)`。**这就是命中查询表**，`slot_of(l,e)` O(1)。GPU 侧持镜像 `d_res`（`res_put/res_upload` 上传）。
- `slots_(full)/live_slots_(可缩后)/blob_/off_(每槽偏移, 256B 对齐)/next_free_/layer_next_/admitted_`
- 每槽字节 = `expert_layout().blob_bytes(layer)`（native pack 用 `open_sized` 逐层尺寸，uniform Q2_0 用 `blob=max_blob`）

`ExpertDispatch.usage`（expert_source.hpp:418）：`std::vector<float>`，`[n_layers*n_expert]`，**不是 EWMA，是"每次适配周期乘以 --adapt-decay 的衰减计数器"**。

### 槽位数推算（`--expert-cache auto`，generate.cpp:4305-4347）
```
free = device_free_bytes()            // UMA 机型用 MemAvailable-6GiB（expert_cache.cpp:28-70）
reserve = (vram_reserve_mib + prefill_own_mib)*MiB + mtp_bind + pipe_first
slots = (free - reserve) / max_blob   // 再 min(profile.size())
```
`--expert-cache N` 显式；WDDM-HIP ≥16GB 卡自动保底 reserve 2560MiB（#1376）；分配后"清零再读 free"验证，不足则 `shrink_to` 重开更小（WDDM 提交前不 resident 的坑）。

### 热度统计语义
- 计数：每 verify window 每层，dispatch 时对每个被路由 id `usage[l*NE+e] += 1.0f`（expert_source.cpp:3138）。
- 每个适配轮结束：`for v in usage: v *= adapt_decay`（默认 0.7，generate.cpp:7561/11674）。所以一个计数的总贡献收敛于 `1/(1-decay)`，是"近期路由比例"的估计。
- `--adapt-every N`（默认 4）：每 N 个 decode 轮（verify window）跑一次 adapt；`--adapt-swaps N`（默认 96）：每轮最多 N 次换入换出。

### 晋升/换出算法（adapt()，generate.cpp:7437-7470，伪代码）
```
for each layer l:
  cand = [(usage, e) for 未驻留 e if usage[e] >= 2.0]   // 晋升门槛：本周期计数≥2
  vict = [(usage, e) for 驻留 e]
  cand 降序排; vict 升序 partial_sort 取最冷的 nc 个
  for i in 0..min(|cand|,|vict|):
     if cand[i].usage < vict[i].usage + 1.5: break      // 增益阈值 1.5
     swap{gain=cand-vict, l, in=cand[i].e, out=vict[i].e}
全局按 gain 降序，截断到 adapt_swaps
```
无 LRU/无时间衰减栈——纯"衰减计数 + 层内配额 + 增益阈值"。全局计数器版本曾把命中率锁死在 2.97%（头部两层吃光所有槽，R4.2g 注释，expert_cache.hpp:154-173），`--expert-cache-per-layer` 给每层 `[l*q,(l+1)*q)` 私有区间。

### 换出如何写回 RAM（"a swap copies the evicted expert back…"）
`resident_stage_swaps()`（generate.cpp:253-300）：
1. 对每个 swap：`cudaMemcpyAsync(exchange_buffer(q), cache->device_slot(host_res[l,out]), blob_bytes, D2H, home.stream)` —— **先把被逐专家从 VRAM 槽拷回 pinned exchange 缓冲**；
2. `cudaStreamSynchronize` 后 `src.stage_exchange(l,in,out,q)`（把 `out` 的 RAM 视图指到 exchange 缓冲，CPU 立即可算 `out`）；
3. 主循环再把 `in` 的 RAM blob 拷入该槽（H2D，L7506-7520 `adapt_copy_h2d`），`host_res[out]=kNotResident` **立即生效**（Finding 10：先逐后纳，等待时间归零，91.7→94.4 tok/s）；`pending[(in,slot)]` 挂起，`apply_pending` 等 `adapt_ev` event 落地后 `host_res[in]=slot` 并 `src.commit_exchanges()`；
4. `commit_exchanges()`（expert_source.cpp:2825-2849）：`memcpy(complement_host + off[in], exchange_buf, bytes)` —— **exchange 内容写进 RAM complement 中 `in` 的原位置**，RAM 副本始终恰好持有 VRAM 不持有的专家；`STRATA_EXCHANGE_ROTATE=1` 时用 ownership 翻转替代 memcpy（exchange_storage.hpp）。
- 前提：resident RAM 模式（`complement_ready()`——全部专家在 RAM 有副本），否则 swap 直接读文件（`FileExpertSource`）。

### 跨重启保留（#477，非 "--expert-cache-learned" 而是 `--expert-profile-save`）
- `heat[i] += usage[i]`（每轮 adapt 前，generate.cpp:7550/7796；仅 --expert-profile-save 时分配，double 型累计自启动以来全部路由）。
- `rank_learned_profile`（expert_cache.cpp:141-168）：排序键 = **当前驻留者优先 → heat 降序 → 旧 profile 名次（prior_rank）→ 索引**，输出全部 24576 对的排列。
- `write_expert_profile`（expert_cache.cpp:170-210）：`"STRP"` + uint32[5]{version=1,n_layers,n_expert,n_ranked,n_ranked} + uint16 对 (layer,expert)×N + int32 rank 表[n_layers*n_expert]；`.tmp` 原子上 rename。触发：QUIT 时 + 每 `--expert-profile-save-every`（默认 10）分钟请求间隙（generate.cpp:8679）。启动用 `--expert-profile P` 读回并按序 fill_slot 预填。

### 配置开关
`--expert-cache N|auto`、`--expert-cache-per-layer`、`--adapt-every 4`、`--adapt-decay 0.7`、`--adapt-swaps 96`、`--adapt-async`（4 步异步适配线程，generate.cpp:7570+）、`--expert-profile P`、`--expert-profile-save P[-every M]`、`--vram-elastic --vram-segment-mib`、`--vram-reserve-mib`、`STRATA_UMA_HEADROOM_GIB`、`STRATA_EXCHANGE_ROTATE`、`STRATA_DISJOINT_ADAPT`（docs/DISJOINT_EXPERT_CACHE.md）。

---

## 2. PREFILL_CPU_SHARE auto（0.1.41）

### 关键文件
- `src/prefill/prefill.cpp` L141-190（env 语义）、L2971-3060（臂选择+两计时回收）、L3082-3105（选专家）、L3190-3228（CPU 线程）、L3760-3775（合流+记时）
- `include/strata/prefill/prefill.hpp` L168-182（arm_cpu_share）
- docs/DETAILS.md L1385-1420

### 逐层计时：CUDA events（两个独立机制）
1. **c/g 比率（算 share）**：`m.cpu_ev[0]` 在 CPU 线程启动前记（prefill.cpp:3190-3193，compute stream 上），`cpu_ev[1]` 在本层 GPU 专家工作之后、combine 之前记（L3761）。下一次**层路由 sync 后**（host 早已过 combine、GPU 未跑完，L2022-2036）`cudaEventElapsedTime` 得 GPU 侧 `g_ms`；CPU 侧 `c_ms` 是 CPU 线程自己的 `steady_clock` 墙钟（L3227）。每专家均值做 EMA(α=0.25)：
   `cpu_c_ms, cpu_g_ms`；**share 闭式解**：`cpu_share_now = clamp(g/(c+g), 0.05, 0.9)`（L3027）—— 就是"两边同时收工"的解：CPU 做 x·N_expert 个耗时 x·c，GPU 做 (1-x)·N 耗时 (1-x)·g，令相等得 x=g/(c+g)。**闭式，非二分**。
2. **gate（值不值得共享）**：`cpu_wall[4]`，每 eligible 层进入时记 `cpu_wall[2*arm]`、combine 后记 `cpu_wall[2*arm+1]`（L3778-3784）；相邻同 chunk 的两层各跑一个臂，ratio = 共享层每非驻留专家 ms / 不共享层每非驻留专家 ms（L3028-3040），存 `cpu_ratio[5]`，**取中位数 <1 则 cpu_gate=true**。

### "隔层复测"探索策略（prefill.cpp:2990-2996）
```
i = cpu_layers++
if cpu_nratio < 3:  arm = i & 1          // 前 3 个 ratio 未齐：逐层交替两臂
else:               arm = (i % 29 == 0) != cpu_gate   // 稳态：cpu_gate 臂为主，每第 29 层跑一次另一臂
```
`CPU_PROBE=29` 素数 → 探针沿层号漂移，覆盖各层。`cpu_cold`（首个 eligible 层 / 刚扩容 cpu_x 缓冲）不采样。#1282 教训写进注释：RX 7900 GRE 全亏时 gate 会自动关掉共享。

### 按"chunk 内路由 token 数少"分 CPU 的排序（prefill.cpp:3083-3105）
```
for e in 0..511: if cnt[e]>0 且不驻留: nstream++
                 if cnt[e] <= MAXT(8) 且 blob 可稳定读(pinned 或非 transient): cand.push(cnt[e], e)
sort(cand)                       // 按 token 数升序 —— 先给 CPU 最"没人用"的专家
take = round(share * nstream)    // share: 显式 x 或 cpu_arm? cpu_share_now : 0
for cand 前 take 个: blob_stable() 取指针, on_cpu[e]=1, rows_cpu += cnt[e]
```
约束 `cnt[e] ≤ MAXT=8` 因 CPU kernel 每专家一次吃 ≤8 个 activation（`ExpertJobMulti`）。CPU 侧：`std::async` 线程（prefill.cpp:3194-3228）把 `mixed` 的 D2H 副本按 token 量化（Q2_0→ActQ，native→`native_quant_act`），组 `ExpertJobMulti` 交给 `cpu_pool_->run_split_multi_native()`（借用 decode 的同一池，`g_cpu_pool_busy` CAS 占坑，L2206-2209）；输出行写 pinned `cpu_rows`（Dm 尾部行序），combine 前 `cudaMemcpyAsync(Dm+r0c*N, cpu_rows, H2D)`（L3772）。共享只在 **staged（非 stream_all）chunk** 生效：`cpu_maybe = pool_hold.held && !stream_all && ...`；`STRATA_PREFILL_CPU_SHARE_MAX`（auto 默认 1024，可 3072）以下 chunk 才走 routed-后 staging。默认状态：CUDA、单卡、无 --batch 时 unset ≡ `auto`+MAX=1024（g_share_default，arm_cpu_share，L1838-1849）。

### 开关
`STRATA_PREFILL_CPU_SHARE`(=0|auto|0.x)、`STRATA_PREFILL_CPU_SHARE_MAX`、`STRATA_DBG_CPU_GATE=1`；0.1.41 默认开（<1024 token chunk，-20~-35%）。

---

## 3. Prefill 专家流式双缓冲（ring）

### 关键文件
- `src/prefill/prefill.cpp`：L101 `STAGE=8`（小 chunk 的 8 槽双缓冲）、L237-292 ring 定容、L2233-2368（seq 展开 + issuer 线程）、L3462-3520（staged 模式 stage_one/compute）、L3718-3755（stream_all 模式消费循环）
- `src/program/generate.cpp` L5743-5810（`--prefill auto` 的 chunk 二分扫描 plan_lend）、L6405-6415

### Ring 数据结构
每条 ring 槽 = 一个完整 expert blob 的 VRAM 缓冲：`m.stage_dev[0..ring)`（一块大 alloc 切成 `ring_slots(T)*MAXBLOB`，prefill.cpp:1802），配 `cudaEvent_t copied[cap]`（H2D 完成事件）、`used[cap]`+`used_of`（compute 读完的释放事件）、`stage_live[]`。`seq` 是**整 chunk 预展开的 (layer, expert, blob_ptr, job)** 流（L2233-2289，按层 `seq_start[l]` 分段，resident/peer 专家直接跳过不入 ring）——预取因此可以越过当前层。

### 字节自适应（#583，prefill.cpp:241-292）
```
ring_bytes() = (fused_ring?1024:384) * 1382400B          // pinned_share≥0.9；否则 /4 → 96 槽旧值
ring_budget_slots = clamp(ring_bytes()/MAXBLOB(), 0, ring_cap())   // ring_cap: fused Q2_0 1024, 其余 512
ring_slots(T) = T >= stream_all_min() ? max(16, budget_or_pinned) : STAGE(8)
```
blob 大的 pack（Q8_0 5.2MB）自然拿到更少槽——"按字节定容"动机见 L194-206 注释（4 卡 rig：chunk 1024→6144，87→402 tok/s）。`STRATA_RING_BYTES=0` 回退 0.1.39 槽数规则；`STRATA_PREFILL_RING` 强指。

### 层间预取调度
- **独立 copy stream `m.copy`** 与 compute stream `m.cs`；`cudaMemcpyAsync(blob→stage_dev[sl], H2D, copy)` + `cudaEventRecord(copied[sl], copy)`；消费侧 `cudaStreamWaitEvent(cs, copied[sl])` 后才 launch 该专家 kernel —— GPU kernel 与 H2D 重叠。
- **issuer 线程**（L2316-2360，`STRATA_PREFILL_ISSUER`，默认开）：host 端专用线程沿 `seq` 一路跑到 `a_consumed + ring` 水位，把整个 chunk（跨全部 48 层）的 copy 提前排队——下一层的专家在本层 attention 期间就已 DMA。slot 释放靠 `used[]` event（compute 里 kernel 记 `used[slot]`，consumer 线程 `give_back(upto)` 发布 `a_consumed`，issuer `wait_issued(k)` 自旋等待）。pinned blob 直接 DMA；unpacked/页错误 blob 先由 4~32 线程 Stager 拷进 pinned 环（#1353 的 SSD/RAM profile 探测，mincore）。
- **stream_all**（`T ≥ stream_all_min()`，默认 1024，share 开启时=SHARE_MAX）：ring 覆盖全层流；否则 **staged 模式**（L3697-3717）：路由后按 `order`（去重专家序）现 stage，保持 `STAGE=8` 个在飞，resident 专家不占槽。
- **MMQ group gather**（L3583-3620）：native pack 把 MMQ_GROUP 个专家一次 gather kernel 归组，group 的最后一个 copied event 一次 wait、一个 used event 一次释放整组（WDDM 下省每专家 ~10µs 泡）。

### 路由后 staging 与未路由专家
stream_all 模式：`release_to(e)` 把 id 小于当前路由专家、路由未命中的 seq 条目**只归还槽不算 kernel**（L3723-3729）；`seq[k].e == e` 命中才 `wait_issued+compute`。staged 模式未路由专家根本不入队（`job_of` 只为 order 内专家建）。路由前无法知道会用到谁，所以 stream_all 预发全部非驻留专家、边消费边退租——这就是"字节自适应 ring 越大越好"的原因（一个 ring 槽 ≈0.53 tok/s，L5795 注释）。

### `--prefill auto` chunk 选择（generate.cpp:5756-5810）
```
auto_ceiling = max(8192, min(prefill_auto_max, max_context))    # 裸 auto 恒达 8192
biggest_chunk: 在 256-token 网格上二分（bytes_needed(T) 单调）  # ~7 次探针
scan: c = biggest_chunk(t: ring_room(t) ≥ ring_cap 且 fits(t)) # 优先"ring 满"的最大 chunk
      若 0 → probe(kRingMin=16); 结果 ≥ old_rule(0.1.39 定值列表) 
kAutoLendPct = pinned_share≥0.9 ? 90 : 85   # 借给 prompt 路径的 cache 槽上限
```
prompt 路径 buffers 从 cache 尾部 `lend_slots` 借槽（loan），chunk 后 refill。

### 开关
`--prefill N|auto[:MAX]`、`STRATA_RING_BYTES`、`STRATA_PREFILL_RING`、`STRATA_PREFILL_STREAM_MIN`、`STRATA_PREFILL_ISSUER`、`STRATA_PREFILL_GROUP_GATHER`、`STRATA_STAGER_THREADS/SSD/SLEEP`、`--vram-reserve-mib`、`STRATA_OWNED_PRICE=exact`。

---

## 4. Decode 三路分流

### 关键文件
- `src/core/expert_source.cpp` L3106-3290：`expert_pool_dispatch_multi` —— **三路计划的唯一现场**（每层每 window 一次，host 线程 = 被 GPU doorbell 唤醒的池线程）
- `src/core/verify.cpp`：L1411-1480（graph 里 `post()`：等 flag → grouped kernel 两段 → CPU 行合并）；L2144-2196（`raise_flag`/`fetch_dma`/`publish_plan`）；capture/`pre()` 侧 doorbell
- `src/kernels/cpu/pool.cpp` + `include/strata/kernels/cpu/pool.hpp` — 线程模型
- `src/program/generate.cpp` L1556-1559（`pcie_frac_for_gbps`）、L2458-2470（默认与探测）

### 三路计划（伪代码，expert_source.cpp:3140-3235）
```
kind[n]: -1 CPU | 0 VRAM | 1 PCIe(上传 GPU) | 2 peer/remote GPU
distinct = 去重后的路由专家（路由序）; nmiss = 不驻留且无 helper 持有的 distinct 数
m = (nmiss * pcie_num) >> 8                     # pcie_num = round(pcie_frac*256)
miss_rank=0; 遍历 distinct:
   slot = host_res[l,e] ≥ 0 → kind=0, ptr=cache_base+slot_off[slot]      # 路 1: VRAM 命中
   peer/helper 持有 → kind=2
   否则: 若 miss_rank ≥ nmiss-m 且 fetches<staging_cap → kind=1, 记录 pinned blob   # 路 2: 最后 m 个 miss 走 PCIe
        否则 miss_rank++
发布 GpuPlanSink（pinned mapped）: counts[0]=VRAM组, counts[1]=entries, counts[2]=PCIe组,
  ptr[]/start[]=VRAM 组, ptr2[]/start2[]=PCIe 组(staging 槽 q), dst[]/tok[] 行映射
atomic_fence; publish() → flagA++（mapped 内存, GPU 自旋 kernel wait_flag_ge 观察）
fetch() → copy stream 上 n 条 cudaMemcpyAsync(pinned→staging) + cudaLaunchHostFunc(raise flagB)
然后本线程亲自 run_split_multi_native(jobs_multi) 计算 kind=-1 的 CPU 专家   # 路 3: 就地算
```
注意 **计划先发、CPU 后算**：GPU 拿到 flagA 立刻开算 VRAM 命中组，PCIe 组等 flagB（DMA 落地）再 grouped kernel 二段启动，CPU 结果写 mapped `m_ymiss_` 行、GPU 等 flag（`wait_flag_ge`，verify_kernels.cu:970）后 `copy_rows_from_mapped`+`moe_hit_add` 合并。三路由 **一个 CUDA graph 捕获**（每 window size 一图），host 不碰任何 sync API。CPU 池线程与 GPU 的 barrier 全部是：GPU 自旋 mapped flag kernel ↔ host 原子写 flag（`raise_flag` L2150，seq_cst CAS 单调升）；CPU↔pool worker 用 epoch 化的 64 位 packed head CAS（pool.hpp:1-25 注释、pool.cpp:549-560）。

### 数据事实
- 单 doorbell：GPU 每层把 10 个 id 写进 pinned host 内存（`doorbell_publish*`，verify.cpp:1339-1351），池线程自旋读，无 cudaStreamWaitValue。
- CPU 池 worker 数 = 物理核数-1（host 核，`--pool-workers` 覆盖；hybrid 用 cpu_core/cpu_atom PMU 判 P/E，pool.cpp:228-245；一 worker 一物理核防 SMT 抢带宽）。worker：park(spin `_mm_pause`，超时睡眠)→epoch 唤醒→CAS claim→drain。
- staging：`kStagingBlobs` 个 blob 槽；pcie_mode 0=DMA 入 staging / 1=grouped kernel 直读 mapped arena / 2=graph 内 copy kernel（`--pcie-mode dma|kernel|direct`）。Q2_0 默认 kernel+0.2，native 默认 DMA+0.55。

### q*（带宽比闭式解）
**不存在** `q* = BW_pcie/(BW_pcie+BW_ram)` 这类公式。实际是：
- 经验基值 `base`：native pack 0.55、Q2_0 pack 0.2（论文 Finding 9：0.55 只在"CPU 算术受限、RAM 带宽有余"的 i-quant 上成立；Q2_0 的 CPU 已吃满 RAM 带宽，再 DMA 反而抢带宽）。
- 链路修正（generate.cpp:1556-1559）：`pcie_frac = base * min(1, gbps/20.0)`，gbps 为启动时一次 H2D 探测（`probe_pcie_h2d_gbps`，burst 取最优）。仅此一个闭式，且是线性缩放不是排队平衡解。
- 运行时按 miss 动态分配：`m = nmiss*pcie_num/256`，取 distinct miss **尾部**（路由序最后的）m 个——尾部专家服务的 token 少，GPU 算它们浪费最小。

### 开关
`--pcie-frac x`、`--pcie-mode dma|kernel|direct`、`--pool-workers/--pool-tasks/--pool-affinity all|auto|p-cores`、`--host-core first|last`、`STRATA_DEC_BATCH`、`STRATA_VERIFY_DEVICE_PLAN`（全驻留 zero-doorbell graph，verify.cpp:1657+）、`STRATA_DMA_BATCH`、`STRATA_LFUSE`、`--spec-follow/--peer-device/--expert-cache-device1..3`。

---

## 5. 格式 / 场景 / kernel 耦合点（移植视角）

### GGUF/i-quant 格式依赖（4 处，必须重写或替换）
1. **`expert_layout`/blob 尺寸表**（kernels/cpu/expert_layout.hpp）：一切"每槽字节"的唯一来源。SGlang 侧等价物 = 每层 expert 权重的 `n_expert × (w_gate|w_up|w_down)` 字节表；调度逻辑只消费 `blob_bytes(layer)` 与 `blob_offset(l,e)`，把这张表换成 fused-experts tensor 的 stride 即可（**纯接口替换**）。
2. **native pack 直接保留 GGUF 切片**：`native_expert.cpp` 调 ggml `vec_dot` traits、`moe_fused_iq.cu`/`mmq::gather_native` 按 GGUF block layout gather——这两处是格式内核，若目标 fork 已有自己的 fused MoE grouped GEMM（如 sglang triton/cutlass grouped kernel），整层丢弃换挂即可；PCIe 上传的字节数取 fork 自身 expert 尺寸。
3. **Q2_0 repack（18B/64 权重 + fp16 scale、interleave）**：Strata 私有格式，仅 CPU VNNI kernel 需要；sglang 用 fp8/int4 自带 kernel 时**不移植**。
4. **profile.bin/learned-profile**：只存 (layer,expert) 排列，与量化格式无关——**格式无关，可直接照搬**。

### 纯调度逻辑（格式无关，可近乎原样移植）
- ExpertCache residency 表 + admit（无淘汰）/ per-layer 配额 / replace 发布协议（expert_cache.hpp/cpp 全部）。
- adapt() 的 gain-promotion（阈值 2.0/1.5 需按新模型 K、层数重标）、D2H 写回→exchange→commit 三步 swap 协议（resident_stage_swaps + commit_exchanges 的语义等价物）。
- ring + issuer + event 释放协议（prefill 流式）——只要"expert blob 可从 pinned host 线性 DMA"成立即可。
- PREFILL_CPU_SHARE 的 g/(c+g) 闭式、29 层探针 gate、`cnt[e]≤MAXT` 排序分派。
- decode 三路 kind 表 + flagA/flagB doorbell + `m=nmiss*frac/256` 尾部选取。
- CPU 池 epoch/CAS/park 协议（与模型无关，仅与"每 job 等大"有关）。

### "单请求"假设耦合点
1. `ExpertDispatch.usage` 是全模型一份衰减小表，`adapt()` 假定一个会话驱动它——多并发下热度互相污染；fork 需 (layer,expert) 热度做全局 sum 或 per-request 分桶再全局化。
2. `resident_stage_swaps`/`adapt_fence`：适配轮假定"窗口在飞的集合可枚举并 fence"（`--pipeline-windows` 已做，generate.cpp:7500-7512），可推广到 batch scheduler 的 drain 点。
3. doorbell/graph 每 window 一 capture，`T≤MAXT=8`；batched serving 要 per-batch-size 图 + batch-id 进 plan。
4. `g_cpu_pool_busy` 全局互斥（prefill 借池单持有者）；多流并发 prefill 需按 stage/请求分区。
5. strata 实际支持 batch slot（`--batch`/BGEN）与 layer-split，非严格单请求，但热度/自适应都按单会话调参。

### ggml kernel 依赖点
- CPU：Q2_0/i-quant dot（expert.cpp 移植 ggml quants.c、native_expert.cpp 直调 ggml-cpu traits）；
- GPU oracle：测试链（`STRATA_ORACLE` ggml-cuda，CMakeLists.txt:423-426）；
- 运行时 GPU 路径**无 ggml 调用**（自写 kernels/cuda/* + cuBLAS）；
- i-quant 码本/格点表（third_party/ggml 头，CMakeLists.txt:391-392）。
结论：CPU 专家 kernel 是唯一实质 ggml 依赖；替换成 fork 自带 CPU MoE kernel 后，所有调度机制独立可移植。

### 论文佐证（docs/paper/Strata-Paper.pdf）
§3.1 三层放置；§3.2 doorbell 三路图；§3.4 每 4 轮换 ≤96 专家；Finding 9（copy-engine DMA 55%/20% 的由来）、Finding 8（按字节定槽，IQ3_XXS 1.3-2.3MB → 3556 vs 2673 槽，+13%）、Finding 10（先逐后纳）——与本节源码一致；**全文与源码均无 q\* 带宽比公式**。
