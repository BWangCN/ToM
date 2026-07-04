# deception-vla

Framework smoke-verification for an **Alpamayo-R1-style reasoning VLA** — reason-then-act
with discrete action tokens and an insulated flow-matching action expert — instantiated on
**Qwen3-VL-4B-Thinking** for a 2-D pixel-world deception task.

> **Scope.** *Verify ≠ train.* Every run in this repo is a smoke/overfit run whose purpose
> is to prove **framework correctness on one consumer GPU** (12 GB). Real training happens
> later on a cluster. `CLAUDE.md` is the spec of record (locked defaults, hard rules,
> acceptance criteria); read its §9 for everything explicitly **unverified** here
> (multi-GPU, throughput, real-data hyperparameters, Stage-C GRPO execution).

## Architecture (one autoregressive pass, reason → act)

```
frames (K=4, 256²) ─┐
task prompt ────────┼─► Qwen3-VL-4B-Thinking  (LoRA r16 on LM proj layers;
                    │                          vision tower frozen;
                    │                          130 trainable action-token rows)
                    │        │ autoregressive output
                    │        ▼
                    │   <think> reasoning … </think>
                    │   <|act_start|> a¹ … a¹⁶ <|act_end|>     (16 tokens = 8×2 chunk)
                    │        │ last-layer hidden states over the assistant span
                    │        ▼   (detach_condition=True → knowledge insulation)
                    └─►  Flow-matching DiT action expert (~55 M) ─► â ∈ ℝ^{8×2}
```

- **Discrete layer**: 130 new tokens (`<|act_0|>`…`<|act_127|>`, `<|act_start|>`,
  `<|act_end|>`), uniform 128-bin quantization over [-1, 1], shared codebook,
  interleaved dx₁,dy₁,…,dx₈,dy₈. Trained via peft `trainable_token_indices`
  (never `modules_to_save` — that would make ~1 B params trainable).
- **Action expert**: rectified-flow DiT (d=512, depth 8, AdaLN-Zero, cross-attention over
  projected VLM hidden states), 10 Euler steps at inference, fp32.
- **Stages**: A (action-modality injection, action-only data) → B (joint SFT with
  reasoning) share one code path and one trainable set (LoRA + token rows + decoder);
  C (GRPO on the VLM token pathway only, decoder frozen — AR1 precedent) is stubbed in
  `src/dvla/rewards.py` and out of scope on a laptop.

## Smoke results (all 13 acceptance criteria of CLAUDE.md §7 pass)

| Check | Result |
|---|---|
| Toy data 200/20, deterministic (seed 0) | ✓ identical hash across regenerations; validator + causal-locality leakage check pass |
| Deception dynamics of the toy arena | feints deceive the 3-step-lagged pursuer 100 %, hesitation 98 %, honest 0 % |
| Quantizer round-trip | ✓ max abs error ≤ 1/128 |
| Collator masks on the real Thinking chat template | ✓ exact boundaries; prompt+completion == native template render |
| Vocab extension / row freeze | ✓ grads confined to LoRA + 130 rows (runtime grad-test); tied lm_head mirroring verified |
| Stage A (overfit 32, action-only) | act CE **0.041** (≤ 0.05), flow MSE **0.016** (≤ 10 % of init 1.42), peak VRAM 11.3 GB, no OOM, bf16 K=4 primary config |
| Stage B (overfit 10 disjoint triplets) | all three losses ↓ > 99 %, constrained greedy decode reproduces GT act tokens **10/10** |
| Constrained decoding demos | ✓ well-formed `<think>…</think>` + exactly 16 act tokens; plots in `reports/gen_demo_*.png` |
| Conditioning perturbation (§7.9) | median(shift/noise) = **10.0** (pass ≥ 3) through the causal pathway; decoder-direct ratio ≈ 0.1 reported alongside (see below) |
| Language canary ×3 checkpoints | ✓ auto-pass (no `<|act_` leakage, no degeneration); verbatim outputs in the report |
| Three-arm ablation harness | ✓ completes; table in `reports/ablation_table.md` (discrete-only vs AR1-style vs π0.5-style, open-loop + 20 closed-loop rollouts/arm) |
| Checkpoint save / resume / merge | ✓ probe-loss continuity across processes; adapter merge keeps generation well-formed |
| Auto-generated report | `reports/smoke_report.md` |

**A finding worth knowing:** with `think_and_act` conditioning, swapping the *reasoning
text alone* (keeping GT act tokens in the conditioning) barely moves the decoder
(ratio ≈ 0.1) — the act tokens are a sufficient statistic for the chunk, so the decoder
rationally reads them. Reasoning→action causality flows through the **token pathway**
(teacher-force a different-decision reasoning → the model *generates different act
tokens* → chunk shifts by ratio 5–10). This is consistent with AR1 putting the RL handle
on the token pathway. All deviations from the locked defaults are quantified in
`reports/smoke_report.md → Deviations`.

## Requirements

- Linux (verified on Ubuntu 24.04 under WSL2; native Linux works the same) with an
  NVIDIA GPU, **≥ 12 GB VRAM**, driver supporting CUDA 12/13 (`nvidia-smi` must work).
- Python ≥ 3.11 (verified 3.12), ~10 GB disk for the venv + ~9 GB for the model cache.
- Network access to Hugging Face on first run (Qwen3-VL-4B-Thinking is Apache-2.0,
  ungated).

## Reproduce

```bash
# 1) environment (uv shown; plain python -m venv works too)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.12
uv pip install -p .venv/bin/python -r requirements.txt

# 2) sanity: GPU + versions
make env

# 3) dataset (deterministic, seed 0) + unit tests
make data
make test

# 4) full smoke (all stages + evals + report)
make smoke
# -> reports/smoke_report.md
```

Individual targets: `make canary_pre_a stage_a canary_post_a stage_b canary_post_b
resume_check stage_b_thinkonly perturb ablation report`.

Timing expectations on a 12 GB laptop GPU: single runs stay under the 30-minute
hard rule; the full `make smoke` took ~85 min wall on an RTX 4080 Laptop (thermally
throttled ~5–10 s/optimizer-step; breakdown in `reports/timings.json`). Faster desktop
GPUs will be well under that.

## Repository layout

```
CLAUDE.md                    # the spec of record — read this first
Makefile                     # env / data / test / stage_* / perturb / ablation / report / smoke
configs/
  laptop.yaml                # verified config (single source of truth for knobs)
  cluster.yaml               # mirror for the cluster; cluster-only fields marked UNTESTED
  canary_prompts.json        # 5 fixed language-canary prompts
src/dvla/
  action_quant.py            # 128-bin uniform quantizer + chunk<->token helpers
  tokenizer_ext.py           # 130-token vocab extension, LoRA wiring, grad-freeze checks
  collator.py                # chat-template sequence building + exact loss masks
  decoder_flow.py            # flow-matching DiT action expert
  model_wrapper.py           # backbone + LoRA + token rows + decoder assembly, save/load
  losses.py                  # 3-component loss, token-normalized, logged separately
  constrained_decode.py      # act-block constrained decoding (logits masked, not overwritten)
  toy_env.py                 # 1×1 arena, 3-step-lagged pursuer, scripted policies, renderer
  canary.py                  # language canary runner
  rewards.py                 # Stage-C GRPO reward INTERFACES (stubs, NotImplementedError)
scripts/
  make_toy_data.py           # deterministic dataset generator (+ slot log for leakage checks)
  validate_jsonl.py          # dataset validator
  train_sft.py               # one code path for Stages A/B (+ resume, canary, criteria)
  run_canary.py              # standalone canary at any checkpoint
  perturb_test.py            # §7.9 conditioning perturbation test
  ablation_eval.py           # §7.11 three-arm ablation (open-loop + closed-loop rollouts)
  eval_smoke.py              # demos, checkpoint round-trip, report assembly
  diag_flow.py               # diagnostic: flow MSE by timestep + sampled-chunk error
  diag_decoder_fit.py        # diagnostic: decoder-only convergence on frozen conditioning
tests/                       # quant round-trip, collator mask boundaries, vocab freeze
reports/                     # smoke evidence: smoke_report.md, ablation table, demos, canaries
```

## Portability notes / gotchas (hard-won)

- **Constrained decoding**: mask logits by *preserving* the allowed tokens' original
  values. Overwriting them with a constant flattens the distribution and greedy decoding
  collapses to the lowest act id.
- **SFT target construction**: encode the prompt with the processor (native chat
  template), tokenize the completion separately, and **concatenate ids**. Re-tokenizing
  the joint text merges BPE tokens across the boundary (e.g. `"\n" + "\n" → "\n\n"`) and
  silently breaks prompt/label alignment — and generation never re-tokenizes anyway.
- **From-scratch action expert**: a ~55 M DiT needs far more updates than a few hundred
  backbone smoke steps. The trainer runs a decoder-only inner loop on cached, *detached*
  conditioning (default 8 × bs 8 per optimizer step) — knowledge insulation is unchanged.
  See `scripts/diag_decoder_fit.py` for the experiment that motivated it.
- **transformers 5.x / Qwen3-VL**: the model requires `mm_token_type_ids` whenever
  `image_grid_thw` is passed (M-RoPE); the Thinking chat template auto-opens `<think>` in
  the generation prompt; `add_special_tokens()` no longer takes
  `replace_additional_special_tokens`.
- **peft ≥ 0.15**: `trainable_token_indices` works with tied embeddings (lm_head mirroring
  verified by `tests/test_vocab_freeze.py`); an explicit gradient-mask fallback is
  implemented in case a peft version rejects the kwarg.
- **WSL2 only**: do **not** set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  ("CUDA driver error: device not ready" inside fused/foreach optimizer ops). Prefer
  `fused=True` AdamW — the foreach path allocates full-parameter transient buffers that
  can push a 12 GB card over the edge.
- **bf16 adapter serialization** rounds trainable weights (~1e-3 relative); checkpoint
  probe comparisons use a relative-OR-absolute tolerance and store both raw numbers.

## For training-side collaborators

Stage C (GRPO) is **out of scope here by design**: the reward interfaces in
`src/dvla/rewards.py` (`reasoning_quality`, `consistency`, `outcome_reward`) are the
contract; the RL trains the VLM token pathway only and keeps the flow decoder frozen
(AR1's released RL pipeline precedent). The §7.9 perturbation ratio is the intended seed
of the consistency reward. `configs/cluster.yaml` mirrors the verified laptop config with
every cluster-only field explicitly marked `# UNTESTED`.
