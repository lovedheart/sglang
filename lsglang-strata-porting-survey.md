# Lsglang (@4f19944c5) Strata 移植落点调研报告

调研对象：`/home/lovedheart/Lsglang` main @4f19944c5。行号均为当前 on-disk 行号。

## 1. run_moe_core 全貌

### 1.1 四分支 dispatch（`python/sglang/srt/layers/moe/fused_moe_triton/layer.py:1668-1700`）

```python
def run_moe_core(self, dispatch_output: DispatchOutput) -> CombineInput:      # :1668
    # 1. GPU resident layer: original path unchanged
    if self.is_gpu_resident_layer:                                             # :1673
        return self.quant_method.apply(layer=self, dispatch_output=dispatch_output)
    # 2. CPU resident layer
    if torch.cuda.is_current_stream_capturing() or get_is_capture_mode():      # :1681
        lk_result = self._cpu_decode(hidden_states, topk_weights, topk_ids)
    elif self.should_use_gpu_prefill(dispatch_output.hidden_states):           # :1688
        lk_result = self._gpu_prefill(...)
    else:
        lk_result = self._cpu_prefill(...)                                     # :1696
    ...  # routed_scaling_factor 乘回 :1702-1710
    return StandardCombineInput(hidden_states=lk_result)                       # :1711
```

输入 = `DispatchOutput`（Standard 格式为 namedtuple：hidden_states / topk_ids / topk_weights，layer.py 里只消费这三个字段）。输出 = `StandardCombineInput(hidden_states=...)`，回到 `forward_impl`（:1600）的 `self.dispatcher.combine()`（standard dispatcher 下即透传回主路）。**没有任何全局 batch 信息进入 run_moe_core**——它只看 `hidden_states.size(0)`。

### 1.2 `_cpu_decode`（layer.py:2487-2530）

```python
def _cpu_decode(self, hidden_states, topk_weights, topk_ids):
    bs = hidden_states.size(0)
    draft_tokens = getattr(self, "speculative_num_draft_tokens", 1) or 1
    rows_per_token = ...          # spec draft-token 行折叠（bs % draft_tokens == 0 时）
    real_tokens = bs // rows_per_token
    covered = self._pool_gpu is not None and real_tokens <= LK_POOL_MAX_TOKENS   # LK_POOL_MAX_TOKENS=2 (:125)
    if covered:
        self._pool_gpu_launch(...)          # GPU 侧 pool 半区，跑在 FusedMoE.pool_stream 上 (:2423-2485)
    stream_ptr = torch.cuda.current_stream().cuda_stream
    self.lk_moe.cpu_decode(stream_ptr, bs, self.top_k,
        hidden_states.data_ptr(), topk_ids.data_ptr(), topk_weights.data_ptr(),
        FusedMoE.output_gpu.data_ptr(), real_tokens=..., rows_per_token=...)     # :2499-2508
    if covered:
        torch.cuda.current_stream().wait_stream(FusedMoE.pool_stream)           # :2511 唯一显式同步点
        FusedMoE.output_gpu[:real_tokens].add_(FusedMoE.pool_gout[:real_tokens]) # :2512-2525 pool 半区结果并回
    output = FusedMoE.output_gpu[:hidden_states.size(0)]
    if self.check_nan_in_output: torch.nan_to_num(output, nan=0.0, out=output)
    return output.to(hidden_states.dtype)                                        # fp32→bf16 在这里
```

**同步点**：Python 侧没有 `stream.synchronize()`；CPU 引擎拿到 `stream_ptr` 后由 .so 内部（事件/wait）保证 hidden_states 已就绪。Python 层唯一显式同步是 pool 半区的 `wait_stream`。

**静态输出缓冲 `FusedMoE.output_gpu`**（类属性，class-level 单例，layer.py:2363-2374）：

```python
def _initialize_cuda_graph_buffers(self):
    if not hasattr(FusedMoE, 'cuda_graphs'):
        max_batch_size = self.max_num_seqs      # = max_running_requests×(1+draft tokens)，见 _get_max_num_seqs :2349
        FusedMoE.output_gpu = torch.zeros(
            (max_batch_size, self.hidden_size),
            device=current_device, dtype=torch.float32)   # fp32，decode CUDA-graph 安全
    self._init_pool_gpu_half()
```

fp32→bf16 的转回是 `output.to(hidden_states.dtype)`（拷贝出新 tensor，非原地）。`max_num_seqs` 语义（:2349-2361）：spec 时 `max_running_requests*(1+num_draft_tokens 或 dspark_block_size)`。

### 1.3 `_cpu_prefill`（layer.py:2532-2578）

```python
half = self._pool_prefill_half()          # pool 半区可选（:2580-2590, lk_moe.pool_prefill_half.PoolPrefillHalf.probe）
expert_ids_cpu = topk_ids.to(dtype=torch.int32, device='cpu', non_blocking=True)   # :2549
weights_cpu   = topk_weights.to(dtype=torch.float32, device='cpu', non_blocking=True)
hidden_states_cpu = hidden_states.to(device='cpu', non_blocking=True)
output_cpu = torch.empty_like(hidden_states, dtype=torch.float32, device='cpu')
current_stream = torch.cuda.current_stream()
current_stream.synchronize()                    # :2556 ★全量阻塞同步点（每层一次，prefill 路径）
self.lk_moe.cpu_prefill(hidden_states.size(0), k, ids_ptr, wts_ptr, hidden_ptr, output_cpu.data_ptr(), pool_skip)  # :2558-2566
if pool_skip:
    current_stream.wait_event(gpu_ev)
    output_gpu = output_cpu.to(device, dtype=torch.float32, non_blocking=True)
    output_gpu += gpu_part                       # GPU 半区结果加回
    output_gpu = output_gpu.to(hidden_states.dtype)
else:
    output_gpu = output_cpu.to(device, dtype=hidden_states.dtype, non_blocking=True)
```

关键：D2H（ids/weights/hidden）+ `current_stream.synchronize()` + CPU 计算 + H2D，全部发生在 **Python 主线程阻塞**中；且 `output_cpu = torch.empty_like(..., device='cpu')` 是**非 pinned** 的 pageable host tensor，`non_blocking` 实际退化。**注意** `.to('cpu', non_blocking=True)` 的 D2H 源 tensor 若非 pinned，non_blocking 也不真正异步——同步靠的就是 :2556 那次 synchronize。

### 1.4 `_gpu_prefill`（layer.py:2593-2612）

```python
output = torch.empty_like(hidden_states)          # 每层新分配，走 torch caching allocator
self.lk_moe.gpu_prefill(
    hidden_states.data_ptr(), output.data_ptr(),
    topk_ids.data_ptr(), topk_weights.data_ptr(),
    hidden_states.size(0), topk_ids.size(1),
    torch.cuda.current_stream().cuda_stream)      # 引擎自搬权重，在传入 stream 上排队
```

无 Python 侧同步；NaN/Inf 用 mask 清理。`should_use_gpu_prefill`（:1907-1912）：capture 中恒 False，否则 `is_gpu_prefill_layer && size(0) >= min(LK_GPU_PREFILL_MIN_BATCH_SIZE, max_num_batched_tokens)`（:517）。

### 1.5 每层 lk_moe 实例构造链

`model_loader/loader.py:1071-1086 postprocess_weights` → 每个 FusedMoE：`process_weights_after_loading()`（layer.py:1949）→ `_do_process_weights_after_loading()`（:1918，按 quant_method 分发）→ 对应 `_process_*` → `_initialize_cuda_graph_buffers()`；随后 `clean_weights_after_loading()`（:1968）删除 GPU 权重属性（w13_weight 等 `delattr`，:1974-1980）——**CPU 层运行时 layer 对象上不留权重**。

- **权重指针来源**：不是显式 pinned host tensor。`create_weights` 阶段按 `layer.is_gpu_resident_layer` 把 expert 权重 **直接分配在 CPU**（例：fp8.py:1229-1231 `device = "cpu" if isinstance(layer, FusedMoE) and not layer.is_gpu_resident_layer`；同型 gate 在 modelopt_quant.py:2389/2452/2477/2553、mxfp4_*_moe.py 各处）。`loader.py:150-154 device_loading_context` 对非 resident FusedMoE **跳过搬 GPU**，所以 `.data_ptr()` 传给 .so 的是普通（非 pinned）host 指针；NUMA 摆放靠进程启动时 `numa_utils.py:30-33`（LK_NUMA_INTERLEAVE 线程绑核/interleave），即"内存落在哪个 NUMA node"由分配时线程亲和性决定，不是显式 `register_numa_weights`（`register_numa_weights` 在 srt 侧只有 `n_gram_embedding.py:186` 的 embedding 路径调用，MOE 类构造时自行登记）。`_process_wna16` 例外：`.cpu().transpose(1,2).contiguous()` 产生临时 tensor 再取 `data_ptr()`（:2012-2021）——临时对象被 .so 持有引用与否不可见，是隐藏风险点。

- **MOEConfigV2 字段模板**（各 `_process_*` 一致，见 `_process_fp8` layer.py:2117-2141、`_process_bf6_fp16` :2170-2188、`_process_wna16` :2036-2056、`_process_nvfp4` :2216+）：

```python
cfg = lk_moe.MOEConfigV2()
cfg.layer_id / cfg.route_probe / cfg.gpu_pool = _set_pool_identity(cfg)   # :1986-1992
cfg.num_processes / process_id / gpu_id = _get_processes_info()           # :1981-1984 (EP 或 TP 划分)
cfg.has_gate_proj, cfg.expert_num = num_local_experts, cfg.top_k
cfg.hidden_size, cfg.intermediate_size = intermediate_size_per_partition  # TP 切分后
cfg.max_batch_size = server_args.chunked_prefill_size                     # :512
cfg.max_num_seqs = _get_max_num_seqs()
cfg.stride=32, cfg.group_min_len=10, cfg.group_max_len = get_max_num_group_batch_size()  # :1885-1892
cfg.groupN/groupK = _get_quant_params(...)                                # 量化块尺寸
cfg.activation_type(0=silu/1=swigluoai/2=relu2), cfg.swiglu_limit
cfg.use_gpu_prefill
MOE_{BF16|FP8|WNA16|NVFP4|MXFP4}[(_FP16)](cfg, w13_ptr, w2_ptr, scale_ptrs..., 0, 0)  # 尾部两个 0 = global scale 槽
```

> **移植时注意**：run_moe_core 是 CPU/GPU 混合的唯一仲裁点，且四分支判定全是"层属性 + token 数 + capture 标志"，拿不到 forward_mode；Strata 的 prefill 机制若挂在这里，MIXED/TARGET_VERIFY 批会被当成纯 prefill 计 token（见 §2）。另 `_gpu_prefill` 的 `torch.empty_like` 在每层热路径上走 caching allocator——若 memory pool 被 graph capture 接管（prefill piecewise 的 eager 段外），这条分配是安全的，但新增分配点要先确认自己落在 capture 区还是 eager 区。

## 2. 调度器视角的 batch 信息

- **层内可见信息**：`run_moe_core` 只有 `hidden_states/topk_ids/topk_weights`。层对象自带 `self.layer_id`（layer.py:365）、`self.model_type`（main/mtp/dspark，:523 + `get_model_type_from_layer_name` common.py:4947）。**全局 batch token 数**：EAGER prefill 下 `hidden_states.size(0)` 本身就是全局（chunked）token 数（token 维无 TP 切分；DP 下是本 rank 的）。`get_server_args().chunked_prefill_size` 层内可取（:1886 已有先例）。
- **ForwardBatch → MoE 层通路（现状）**：`ForwardContext`（model_executor/forward_context.py:35-43）frozen dataclass，只有 `attn_backend` 一个字段——**forward_mode / 请求边界不下传到 MoE 层**。per-forward 标量走 `ForwardFlags`（runtime_context.py:640-700）：`is_extend_in_batch`（graph runner capture 时强制 False，由 `ForwardBatch` 构造写 sticky 值，forward_batch_info.py:1449 → dp_attention.py:330 `set_is_extend_in_batch`）。piecewise 路径例外：`TcPiecewiseForwardContext.forward_batch` **携带完整 ForwardBatch**（runner_backend_utils/tc_piecewise_cuda_graph/context_manager.py:70），custom op `moe_forward_piecewise_cuda_graph_impl`（layer.py:2617 起）只取 `forward_context.moe_layers[layer_id]`。要往层里送 prefill/decode 区分，合规做法是给 `ForwardFlags` 加一个 graph-visible flag（照 `is_extend_in_batch` 模式），而不是塞 ForwardContext。
- **请求边界**：`extend_seq_lens_cpu / extend_prefix_lens_cpu`（forward_batch_info.py:522-527）在 ForwardBatch 上有 CPU 镜像，但 MoE 层拿不到——除非走 piecewise ctx 或新增 flag。**层内可推断的仅有** topk_ids 形状 [tokens, k]。
- **MIXED（prefill+decode 混合 chunk）**：`ForwardMode.MIXED`（forward_batch_info.py:114）。开关 `is_mixed_chunk`（scheduler.py:1316-1317，`--enable-mixed-chunk`）；mix 时 `ScheduleBatch.mix_with_running`（schedule_batch.py:3016-3087）把 running decode 行 **追加到 extend batch 尾部**：`extend_lens += [1]*running_bs`（:3083），即 MIXED 批里"哪些行是 decode"= 最后 running_bs 行。MIXED 会正常走 `run_moe_core`（无任何 lk 特判），token 语义 = 全部行（prefill 段 + 每 decode 1 token），此时 `size(0)` ≥ 阈值会触发 `_gpu_prefill`，**decode 尾部行也被当成 prefill token 搬权重**。spec 的 TARGET_VERIFY 同理按 extend 计。
- lk 侧现有"混合"概念与此无关：`get_super_chunk_size()`（common.py:4906，LK_GPU_PREFILL_SUB_M，commit 4eac85c26）是 prefill 内部的 super-chunk 切分（scheduler 预算抬升 + attention token-iterator + `super_chunk_backend.py`），不改变 MoE 层内 token 语义。

> **移植时注意**：想做"按请求边界分派 GPU/CPU"的机制，边界信息目前**进不了 MoE 层**；最省事的路径是仿 `is_extend_in_batch` 在 ForwardFlags 加一个 `_GRAPH_VISIBLE` flag（capture 时 runner 强制 False），并在 MIXED 开启时明确 decode 尾行的语义。

## 3. CUDA graph / capture 约束

- 两个判据：`get_is_capture_mode()`（model_executor/runner_utils/capture_mode.py:44，= 模块级 `is_capture_mode`（`model_capture_mode()` ctx 设置，decode/spec graph capture 全程为 True）或 `is_in_breakable_cuda_graph()`）；`torch.cuda.is_current_stream_capturing()`（驱动级实时状态）。run_moe_core:1681 用 `or` 并联——**capture 中或 capture-mode 下 decode 必走 `_cpu_decode`**；`should_use_gpu_prefill`:1908-1911 双判据都返回 False，即 **prefill 的 GPU 分支被 capture 完全排除**（prefill 若被 piecewise/full graph 捕获，capture 段内 CPU 层自动降级为 `_cpu_decode`——阈值判定失效）。
- decode capture 能进 `_cpu_decode` 的前提：所有中间量都是**预分配静态缓冲且地址恒定**——`FusedMoE.output_gpu`（:2368，capture 前由 `_initialize_cuda_graph_buffers` 建好）、`pool_hbuf/pool_gout/pool_stream` 等 pool 类缓冲（:2400-2414）、lk_moe .so 内部按 `max_num_seqs` 预留的 workspace；capture 段内不允许任何新分配，`_gpu_prefill` 的 `torch.empty_like` 若被捕获会炸。
- full-graph prefill backend 专门加了 `lk_moe.pool_capture_pause()/resume()` 包裹 capture（runner_backend/full_cuda_graph_backend.py:52-62 `_park_expert_pool`）——**pool 的 host 权重泵在 capture 期间必须停车**，这是已踩过的坑。
- `_pool_gpu_launch` 用私有 `FusedMoE.pool_stream` + `wait_stream`（:2444-2475），stream 对象预建于类属性；runtime_context 的命名 stream lease 明确注释"创建是 driver call，必须留在 capture 之外，init/warmup 时 lease"（runtime_context.py:994-998）。

**在 prefill 路径加异步预取/双 stream 会破坏什么**：
1. prefill 若启用 piecewise CUDA graph（tc_piecewise）或 full prefill graph，capture 段内出现 `new Stream()`/事件创建/`empty_like` → capture 失败或指针漂移；eager 段可以，但 **capture 与 replay 的行为分支必须一致**（flag 值要同，参照 `capture_return_pooled_hidden_states` 注释 prefill_cuda_graph_runner.py:299）。
2. torch caching allocator 的 graph memory pool（Resources.graph_memory_pool，runtime_context.py:~615 + `torch.cuda.graphs`）假设 capture 内无跨流竞态；双 stream 必须只用 event/wait_stream 定序且预建对象，否则 graph 外多流写同 buffer → replay 时数据竞争。
3. `.so` 引擎内部有自管线程/pool 泵（pool_capture_pause 的存在即证明），外部 stream 与其内部流的顺序约束只能靠 .so 暴露的 stream 参数（cpu_decode/gpu_prefill 的 stream_ptr）传递。

> **移植时注意**：新 stream/event/buffer 一律在层构造（`_initialize_cuda_graph_buffers` 时机）预建并 lease；capture 分支与 eager 分支要用**同一份**预建对象，且给 full-graph backend 的 capture 加 pause/resume 类钩子。

## 4. GPU 侧"热专家槽位缓存"可行性

- **CPU 层权重现状**：非 resident FusedMoE 运行时**没有** GPU w13/w2 tensor（`clean_weights_after_loading` delattr，layer.py:1968-1980；量化方法 `process_weights_after_loading*` 也被 `not layer.is_gpu_resident_layer` 早退，fp8.py:1541 等）。`quant_method.apply` 只对 resident 层生效。**没有现成 per-expert 切片**：w13 是 `[E, N, K]` 整块 Parameter（构造期），CPU 层连这个都删了；专家寻址完全在 .so 内部（slot_table_fill 按 snapshot 地址表 gather，layer.py:2435-2441）。
- **lk_moe 已有 GPU pool 机制**（现成的"热专家槽位"雏形）：`_init_pool_gpu_half`（:2377-2421）——`pool_layer_info()` 返回 `snapshot_addr/weight_addr/slot_weight_bytes`（.so 映射的 GPU 常驻槽位表），`pool_gds()`、`pool_gpu_report(True)`；triton kernels `moe_gateup/moe_down/reduce_*` 直接按 `slot_bytes` 从槽位算权重地址，`pool_slot_tab` 是 device 端 slot→权重索引。GPU 槽容量/泵送全在闭源 .so 里；`LK_POOL_LAYERS`（common.py:5013-5031）选层。**可借的 headroom 不走 torch pool，而走 lk_moe 静态 device arena**：KV 尺寸前专门查询 `lk_moe.get_device_mem_usage()` 并预留 `LK_GPU_RESERVE_GB`（kv_cache_configurator.py:165-193）。若新缓存用 torch 分配，空闲量查询 = `get_available_gpu_memory(device, gpu_id)`（utils/common.py:422；model_runner.py:1166/1249 打印 load 前后差值）——即 `mem_fraction_static` 分完 KV 后 `torch.cuda.mem_get_info` 的剩余值；expert_pack.py:372 的 `free_bytes - reserve` 再算 slot 数是可抄的分配范式（但务必在 pool_configurator 阶段扣减，事后临时 malloc 会挤 KV 池或 OOM）。
- **pinned H2D 工具**：无通用 weight_transfer 类（grep 无此类）。可复用件：① dwdp `DWDPWeightManager`（layers/moe/dwdp/weight_manager.py，整文件 137 行）——双 buffer + `_copy_stream` + 2×(prefetch/consume) event，`wait_prefetch/record_compute_and_prefetch_next`（forward_impl:1607-1609、1635-1636 已挂在 FusedMoE 前向钩子上），prefetch 的是**对端 GPU VA 而非 host pinned**（P2P copy_），但其"prefetch layer+2、WAR 等待、buf_idx 复用"骨架可整体照搬；② expert_pack.py:388-390 的 `pin_memory=True` staging ring + 专用 stream（file→pinned→GPU）是目前仓库里唯一的 host→GPU pinned 流水线，面向 ExpertPack 文件格式。
- 另注意 `get_stream(name)`（runtime_context.py:994）/`get_buffer` lease 是新 stream/buffer 的正规注册口。

> **移植时注意**：GPU 热槽与 lk_moe 自己的 pool/权重泵会在同一块 device 显存上竞争，容量必须从 `LK_GPU_RESERVE_GB`/pool arena 里出而不是 mem_get_info 的残余；host 侧专家切片要么在 `clean_weights_after_loading` 前留视图（现在会被删），要么经 .so 槽位接口取，别绕开它直接 pin 整个 CPU Parameter（TPM 级显存/内存双拷）。

## 5. 计时与一致性测试基础设施

- **per-layer NVTX**：`--enable-layerwise-nvtx-marker`（arg_groups/fields/exec_.py:550）→ model_runner.py:1225 给整个 model 挂 `PytHooks`（utils/nvtx_pytorch_hooks.py:20，forward pre/post hook 打 NVTX+shape）。
- **op 级 NVTX**：`utils/nvtx_utils.py` `profile_range/profile_method` + `SGLANG_ENABLE_NVTX_SCHEDULER / SGLANG_ENABLE_NVTX_OPERATIONS`（nvtx_utils.py:39-40）；`cupti` 无专用 infra（仅 kernels 目录一处无关命中）。profiler 输出目录 `SGLANG_TORCH_PROFILER_DIR`（utils/profile_utils.py:45-48 含 graph-capture 子目录）。
- **KL/一致性**：`python/sglang/test/kl_test_utils.py:101 compare_kl_divergence`（input vs output logprob 的 KL，阈值 `ACC_THRESHOLDS[model]["kl_div"]`）+ `kl_multiturn_utils.py:100 _replay_and_compare_kl` + `kits/kl_divergence_kit.py`、`logprob_test_utils.py`；消费方 e2e 测试如 `test/registered/e2e/models/test_inkling.py`、`dp_attn/test_dp_attention_bcg_kl.py`、`radix_cache/.../test_unified_radix_cache_kl_*.py`。chunk 一致性单测样例：`test/registered/unit/layers/test_logprob_chunk_stitching.py`。这套正是"prefill vs decode logprob 一致"的现成脚本。

> **移植时注意**：GPU-prefill 与 CPU-prefill 数值必有差异（CPU AVX kernel vs GPU 搬权重路径的量化 round 顺序不同），用 kl_div 阈值法验证三条路径互切换是现成模板；层级耗时直接用 layerwise NVTX 即可归因到 MoE 层。

## 6. 改动风险面（测试/CI 覆盖 + 开关规范）

- **layer.py run_moe_core / forward_impl** 被上游 MoE 全域测试间接覆盖（test/ 下无任何 lk/LVLLM 用例——fork 私有路径无 CI 保护）：`test/registered/unit/layers/moe/`（test_fused_moe_native、test_copy_weight_views_before_h2d、test_moe_runner_extensions、test_fused_moe_triton_config、test_aiter_runner、test_hpc_ops_runner_guard、test_flashinfer_a2a_wide_ep）、`test/registered/lora/test_fused_moe_lora_kernel.py`、`test_lora_moe_tp_logprob_diff.py`、quantization 下 `test_nvfp4_moe_backends / test_mxfp4_sm*`、`test/registered/cpu/test_moe.py`、kernels/ops/moe/*。FusedMoE 有 87 个调用方（codegraph blast radius），改动 forward 通路会波及 LoRA（FusedMoEWithLoRA）、DeepEP（ep_moe/layer.py:279 覆写版 run_moe_core，**不会**进 lk 分支）、NPU fuseep。
- **capture/graph**：`test/registered/unit/model_executor/runner/test_decode_cuda_graph_runner.py`、`test_prefill_cuda_graph_runner*.py`、`test_hidden_state_graph_recapture.py`、`test/registered/cuda_graph/{breakable,full_prefill,piecewise}/`。capture 分支语义变化首先在这些用例爆。
- **调度/mixed chunk**：`test/registered/chunked_prefill/`（多为 scripted 框架）；mix 语义看 schedule_batch.py:3016。
- **显存池**：`test/registered/expert_pack/`（slot 预算范式先例）、kv pool sizing 相关单测。
- **开关规范**：上游 SGLANG_* 变量必须进 `environ.py` 的 `Envs`（EnvField 描述符，`envs.X.get()`，测试用 `.override()`；见 env-var-conventions skill：Rule 1-6，禁新增 `get_bool_env_var("SGLANG_*")` 调用点）。但 **LK_* 分支走的是自己的通道**：全部经 `get_bool_env_var/get_int_env_var/get_str_env_var`（utils/common.py:1262/1282/1330，带 FIXME）读 `os.environ`，集中在 common.py:4892-5035；旧 `LVLLM_*` 名在 import 期由 `_LK_ENV_RENAMES` + `migrate_lk_env_vars`（common.py:1297-1320）自动改名。**新增开关应跟 LK_* 惯例（LK_ 前缀 + common.py 集中访问器 + 进 _LK_ENV_RENAMES 映射），不要塞 environ.py 的 Envs**，除非愿意接受 SGLANG_ 前缀与上游规范。

## 查不到 / 无法确认的点

- lk_moe 2.3.3 为纯 .so（`site-packages/lk_moe/_lk_moe_C*.so`，无 pyi/源码）：`cpu_decode/cpu_prefill/gpu_prefill` 内部同步方式、pool 槽位 GPU 容量与泵送策略、`register_numa_weights` 语义——Python 侧不可见，只能按调用面推断（上文已标注推断处）。
- `_process_wna16` 的 `.cpu().contiguous()` 临时 tensor 生命周期由 .so 接管与否：查不到，移植 W4 路径前需实测。
- test/ 下无任何 lk_moe/LK_* 回归用例（grep 为空）——三机制移植后的正确性测试需自建。
