# AGENTS.md — Developing SGLang in this fork

SGLang is a fast serving framework for LLMs/VLMs. This checkout is a **development
fork** of `sgl-project/sglang`: `origin` = upstream, `fork` = `lovedheart/sglang`.
Day-to-day work happens on topic branches (`qsa/...`, `feat/...`, `rebase/...`)
kept current by rebasing onto upstream `main`.

## Repository layout

- `python/sglang/srt/` — core runtime: `managers/` (Scheduler, TokenizerManager),
  `model_executor/` (ModelRunner, attention backends), `models/` (HF definitions),
  `layers/`, `mem_cache/`, `speculative/`, `entrypoints/` (HTTP servers/APIs),
  `environ.py` (where every `SGLANG_*` env var is defined).
- `python/sglang/lang/` — frontend DSL · `python/sglang/kernels/` — JIT + AOT kernels
  · `python/sglang/multimodal_gen/` — diffusion.
- `rust/` (gRPC server, radix tree, renderer, …), `sgl-model-gateway/` (router),
  `3rdparty/amd` (ROCm).
- `test/` — CI test tree; **`test/README.md` is the canonical reference**.
- `benchmark/` — `bench_serving.py`, `bench_one_batch.py`, etc.
- `docs/` — Mintlify docs site; `docs/cookbook/` holds per-model recommended
  deployments (GPU count, parallelism, quantization, key flags).
- `.claude/skills/` and `.claude/rules/` — repo-specific agent guidance; see below.

## Code navigation — codegraph first

- Answer symbol/architecture/flow questions ("where is X defined", "what calls Y",
  "how does a request flow from API to scheduler") with the codegraph tools —
  `codegraph_explore` first; fall back to grep/read only when it returns nothing
  or looks stale.
- **Always check the index state before using any codegraph tool.** The index is
  trustworthy only if it is initialized and current with a clean tree:
  `git status --porcelain` empty and `.codegraph/` built at the current HEAD
  (`codegraph_status` if available; else compare `.codegraph/` timestamp vs
  `git rev-parse HEAD` / recent commits). If anything is dirty — uncommitted
  edits, or HEAD moved since the last index — run `codegraph_sync` (or re-init)
  first and never trust a stale index for the changed files; for those files
  read the working copy directly until the sync completes.
- This repo is indexed (`.codegraph/`, ~700 MB, gitignored — never commit it).
  If the index is missing, (re)initialize it immediately — no need to ask
  permission; re-sync after file moves, large refactors, or branch switches.
- Big-project strategy: pass `path` pointing at the relevant subtree
  (e.g. `python/sglang/srt/`, `rust/`) instead of indexing/exploring the whole
  tree; for cross-module questions check structure first (`codegraph_files`)
  then explore per-directory; use `codegraph_node` to pull specific symbols
  instead of reading whole files.

## Environment (this machine)

- Use the repo venv: `source /home/lovedheart/sglang-fork-dev/.venv/bin/activate`
  (Python 3.12, editable install → source edits take effect immediately, never
  reinstall for code changes). Recreate with
  `uv venv .venv && uv pip install -e "python[all]"`.
- GPU: NVIDIA RTX PRO 6000 Blackwell (**sm120**), CUDA 13 · 16 cores · ~123 GB RAM.
- Model weights live in `/media/lovedheart/models/` (Qwen3.8-Flash-Next quant
  variants, DeepSeek-V4.1-Flash-Pruned, …).
- If flashinfer JIT fails on `~/.cache/sglang` log permissions, export
  `FLASHINFER_WORKSPACE_BASE=/tmp/fi-ws`.
- Launch with `python -m sglang.launch_server --model-path ...`; pick flags from the
  model's cookbook page (`docs/cookbook/...`) — never guess flags or defaults, read
  them from code or the cookbook.
- **Never stop an `sglang serve` / `launch_server` process — it belongs to the user.**
  A long-lived server runs on `:8070` (shared by local tools, e.g. `ocr`, and other
  agents; it sleeps on idle). Do not kill it, do not run `scripts/killall_sglang.sh`,
  no `pkill`/`kill` on sglang PIDs, and never reuse port 8070. If a test needs a
  server, launch your own on a random free port and shut down only the one you started.
- Untracked root artifacts (`gsm8k_*.{jsonl,log,md}`, `*.pdf`, `internal-docs/`,
  `qsa-spec-decode-notes.md`) are local measurement records/notes — do not commit
  or delete them.

## Workflow

- Commits follow Conventional style: `type(scope): subject`
  (e.g. `perf(qsa): ...`, `test(spec): ...`, `fix(mem_cache): ...`).
- Sync upstream: `git fetch origin && git rebase origin/main`; publish with
  `git push fork <branch>` (`-f` only for your own rebased branch). Tag a
  `backup/<branch>` before large rebases.
- Never push to `origin` (upstream) directly; contributions go through PRs against
  `sgl-project/sglang` using `.github/pull_request_template.md`.
- Never commit scratch tooling (`.codegraph/`, local Dockerfiles, result dumps).

## Testing

- Single file: `python3 test/registered/<area>/<file>.py [TestClass.test_method]`
  (the CI launcher appends `-f` = failfast). Suites:
  `python3 test/run_suite.py --hw cpu|cuda --suite <name>`
  (e.g. `base-a-test-cpu`, `base-b-test-1-gpu-small`).
- New CI tests live in `test/registered/` and must call
  `register_cuda_ci(est_time=..., stage=..., runner_config=...)` with **literal**
  values at module level; end the file with exactly `unittest.main()` or
  `sys.exit(pytest.main([__file__]))` — no custom argparse.
- Pick the lightest suite that meets the need; most tests → `base-b-test-1-gpu-small`.
- Before adding a registered test, identify the production change that would make
  it fail; prefer extending an existing fixture over a new file.

## Lint & format

- `pre-commit run --all-files` (or on changed files only). Formatters:
  **ruff-format** + **isort** + ruff (`F401,F821,UP037`); clang-format for
  CUDA/C++; rustfmt + clippy for Rust.
- Repo guards in `scripts/lint/` enforce test registration, no bare
  `pytest.main()`, workflow job names, etc. — CI will fail if you bypass them.

## Code style — read `.claude/rules/*.md` before editing Python

Key rules (full text in the rule files):

- `msgspec.Struct` for new data containers — no new `@dataclass`.
- No defensive `getattr`/`hasattr`; use `isinstance` narrowing or always-set fields.
- `ForwardBatch.init_new` must not mutate the `ScheduleBatch`.
- Functions < ~100 LOC, files < ~2k LOC; orchestration functions read like
  pseudocode; extract init-static values in `__init__`.
- Comments carry facts not visible from the code; a comment-only diff must not
  touch one-line comments.
- Read the skill named in `.claude/rules/modify-component-must-read.md` before
  touching that component: speculative decoding (`speculative-naming`),
  `Scheduler`/`TokenizerManager`/`ModelRunner` (`large-class-style`),
  env vars (`env-var-conventions`), scripted runtime, kernel placement
  (`kernel-organization`).

## Skills — load the matching `.claude/skills/<name>/SKILL.md` first

`write-sglang-test`, `ci-workflow-guide`, `ci-test-audit`,
`add-jit-kernel`, `add-sgl-kernel`, `kernel-organization`,
`env-var-conventions`, `speculative-naming`, `large-class-style`,
`sglang-runtime-context`, `scripted-runtime-notes`, `debug-cuda-crash`,
`debug-distributed-hang`, `sglang-prod-incident-triage`,
`sglang-bisect-ci-regression`, `generate-profile`, `llm-torch-profiler-analysis`,
`babysit-pr-to-pass-ci`, `cookbook-add-model` / `cookbook-migrate-model` /
`cookbook-review-pr`, `compute-mamba-ratio`, `kl-consistency-test`,
`mechanical-refactor-verify`.

## Subagents — delegate actively when it pays off

- Use subagents for work that benefits from its own context: parallel independent
  investigations, long test/build runs, broad codebase exploration, mechanical
  multi-file edits. Don't delegate what a few direct calls can settle.
- **Parallelize aggressively**: launch independent subagents together, not one
  after another. Cap concurrency to what the machine/LLM serving stack supports
  (check the local sglang/vllm server's `max_running_requests` / `max_num_seqs`);
  if you can't probe it, default to 3 concurrent and queue the rest.
- Prompts must be self-contained (goal, file scope, stop condition). When two
  subagents may touch the same file, state each one's scope in the prompt.
- After subagents that edited code, re-check the index (`codegraph_sync`) and
  spot-verify their summaries with targeted reads/tests — never trust blindly.
  If a result looks wrong, follow up with the same agent (≤2 rounds) before
  switching strategy.
- For >5 independent subtasks or multi-stage orchestration, run them as a
  planned batch (todo-tracked) rather than ad-hoc; a 1–2 item hand-off is just
  a normal subagent call.

## Memory — use reme actively

- **Before** starting any non-trivial task here, `reme_search` the topic/branch/file
  names — substantial history exists (QSA + spec decode, lk_moe, engram_nvme,
  docker images `lovedheart/qwen38-flash-next`, venv/env setup). Checkpoints beat
  rediscovery.
- **During** long or multi-stage tasks, checkpoint after each stage with
  `reme_write` to `digest/tasks/<task-name>.md` (branch, HEAD sha, decisions,
  open TODOs). On failure/resume, `reme_read` the checkpoint and continue from the
  breakpoint.
- **After** finishing, record outcomes, environment quirks, and failed approaches
  (what didn't work matters). Repo-specific facts → `digest/tasks/`;
  cross-project conclusions → `digest/wiki/`.
