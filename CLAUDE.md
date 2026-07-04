# CLAUDE.md — Deception-VLA: Framework Smoke-Verification Spec

**Audience.** Claude Code (CC), operating autonomously inside the WSL2 Ubuntu distro on this laptop. Secondary audience: training-side collaborators (read §9 first).

**Goal.** Build and verify the *training framework* for a reasoning+action VLA targeting a 2-D pixel-world deception task. **Verify ≠ train.** Every run in this spec is a smoke/overfit run. Real training happens later on the collaborators' cluster.

**Architecture pattern.** Alpamayo-R1 (AR1) at small scale: the VLM autoregressively emits natural-language reasoning, then **discrete action tokens**; a small flow-matching decoder ("action expert") converts the discrete tokens into a continuous action chunk, conditioned on VLM hidden states. RL post-training (Stage C, **out of scope** on this machine) operates on the VLM token pathway only; the decoder stays frozen in RL — this matches AR1's released RL pipeline, which trains the VLM backbone (text + discrete action tokens) and explicitly excludes the flow-matching expert.

**Decision of record (2026-07-02).** We adopt the *Alpamayo-R1 recipe*, not the Alpamayo model: (1) the architecture shape — reason-then-act in one autoregressive sequence, a discrete action-token layer, an insulated flow-matching action expert; (2) the three-stage training recipe with RL confined to the VLM token pathway (decoder frozen); (3) the CoC data discipline — closed-set decisions, causal locality, and consistency (redefined for deception on the intent–motion–outcome triple). We do **not** use Alpamayo weights or the NVlabs codebase; backbone is Qwen3-VL-4B-Thinking, harness is ours; gradient insulation follows π0.5-KI (`detach_condition=True`). One-line version for external docs: *"An Alpamayo-R1-style reasoning VLA — reason-then-act with discrete action tokens and an insulated flow-matching action expert — instantiated on Qwen3-VL-4B-Thinking for a pixel-world deception task."* The only open fork is the decoder-conditioning span, adjudicated by the three-arm ablation (§7.11); the discrete-token layer and the GRPO handle are non-negotiable in every arm.

---

## 0. Locked defaults — single source of truth

Any deviation from this table must be recorded in `reports/smoke_report.md → Deviations`, with reason. Do not silently change values.

| Group | Knob | Default |
|---|---|---|
| Action space | `action_dim` | 2 (dx, dy per control step, normalized to [-1, 1]) |
| | `chunk_len` H | 8 steps |
| | `control_hz` | 10 Hz (chunk horizon 0.8 s) |
| | `quantization` | uniform, `bins=128` per dim over [-1, 1] |
| | codebook | **shared** across dims → tokens per chunk = 8×2 = **16** |
| | token order | interleaved per timestep: dx₁, dy₁, dx₂, dy₂, …, dx₈, dy₈ |
| Vocab extension | new tokens | 130 = `<\|act_0\|>`…`<\|act_127\|>` + `<\|act_start\|>` + `<\|act_end\|>` |
| | training method | `peft` `trainable_token_indices` on the 130 new rows (see §6.1) |
| Backbone | model | `Qwen/Qwen3-VL-4B-Thinking` |
| | precision | bf16 (fallback chain in §5) |
| | attention | `attn_implementation="sdpa"` (never build flash-attn) |
| | vision tower | frozen |
| LoRA | r / α / dropout | 16 / 32 / 0.05 |
| | targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj (LM only) |
| Decoder | type | flow-matching DiT ("action expert") |
| | size | d_model=512, depth=8, heads=8, AdaLN time conditioning; ≈40 M params (print exact count at init; acceptable 30–60 M) |
| | conditioning | cross-attention over VLM **last-layer hidden states** of the assistant span (think span + action-token span); projection d_vlm→512 where d_vlm is read from `model.config` |
| | `detach_condition` | **True** (knowledge insulation: no decoder gradient into the VLM; the VLM is trained only by the two CE losses) |
| | flow | rectified flow; x_t=(1−t)x₀+t·x₁, x₀∼N(0,I); predict velocity v=x₁−x₀; MSE loss |
| | inference | 10 Euler steps |
| | `decoder_condition` | `think_and_act` (AR1-style, default) \| `think_only` (π0.5-style, ablation arm c) |
| | `emit_discrete_at_inference` | `true` (default). `false` is legal **only** with `decoder_condition=think_only`; training-time discrete co-training stays ON in all configs |
| Inputs | frames K | 4 history frames, 256×256 RGB (cap processor `max_pixels` so visual tokens ≤ ~1024/frame) |
| Losses | λ_reason / λ_act_ce / λ_flow | 1.0 / 1.0 / 1.0 — always logged separately, token-count-normalized |
| Optim | LoRA lr / decoder lr / token-row lr | 1e-4 / 3e-4 / 1e-4, AdamW, cosine, warmup 20 steps |
| | batch | 1 × grad-accum 8; gradient checkpointing ON |
| Data (toy) | sizes | 200 train / 20 val; seed 0; deterministic |
| Runtime | wall clock | full `make smoke` < 45 min on this laptop (single runs still ≤ 30 min, §6.5) |

Config lives in `configs/laptop.yaml`; a `configs/cluster.yaml` mirrors it with cluster-only fields present but marked `# UNTESTED`.

---

## 1. Environment assumptions

- Host: Windows laptop, **RTX 4080 Laptop, 12 GB VRAM**. Budget **11 GB usable** (Windows desktop reserves 0.5–1 GB).
- CC works **inside** the WSL2 distro (its ext4.vhdx lives on E:). All paths under `~/`. Never place repo, data, caches, or venv under `/mnt/c` or `/mnt/e`.
- First command of any session: `nvidia-smi` (must show the 4080), then `python -c "import torch; print(torch.cuda.is_available())"`.
- Toolchain: `uv`, Python ≥ 3.11, torch cu12x wheel, `transformers>=4.57` (Qwen3-VL support), `peft>=0.15`, `pillow`, `numpy`. `bitsandbytes` only if the NF4 fallback is triggered.
- No gated-model friction: Qwen3-VL is Apache-2.0, ungated. Set `HF_HOME=~/hf_cache`.

## 2. Architecture

### 2.1 Module diagram

```
frames (K=4, 256²) ─┐
task prompt ────────┼─► Qwen3-VL-4B-Thinking  (LoRA r16; vision tower frozen;
                    │                          130 trainable token rows)
                    │        │ autoregressive output
                    │        ▼
                    │   <think> CoC-style reasoning … </think>
                    │   <|act_start|> a¹ … a¹⁶ <|act_end|>
                    │        │
                    │        │ last-layer hidden states over assistant span
                    │        ▼   (detach_condition=True → stop-grad)
                    └─►  Flow-matching DiT decoder (~40 M)  ─►  â ∈ ℝ^{8×2}
```

One autoregressive pass produces reasoning **then** discrete action tokens (reason-then-act; action tokens are conditioned on the reasoning by construction). The decoder is a post-hoc converter from discrete tokens + hidden states to a continuous chunk.

### 2.2 Quantization

- Encode: `b = clip(floor((x + 1) / 2 * 128), 0, 127)`
- Decode (bin center): `x = (b + 0.5) / 128 * 2 − 1`
- Round-trip unit test required: max abs error ≤ bin half-width (1/128).
- `FAST` tokenizer is a future config option; do **not** implement it now.

### 2.3 Vocabulary extension

Add the 130 tokens via `tokenizer.add_special_tokens` + `model.resize_token_embeddings`. Check `config.tie_word_embeddings`; if untied, the lm_head rows for the new tokens must also be trainable. Training of new rows goes through `LoraConfig(..., trainable_token_indices=...)`. At startup, assert the peft version accepts this kwarg; if not, implement a fallback gradient mask that zeroes embedding/lm_head grads for all rows **except** the 130 new ones. Print trainable-parameter breakdown (LoRA / token rows / decoder) at init.

### 2.4 Sequence format & loss masks

Use the native chat template via `processor.apply_chat_template` — do not hand-roll it; verify the Thinking variant's `<think>` handling against the processor output.

```
system:    fixed task framing
user:      [frame_{t-3} … frame_t] + task_prompt
assistant: <think> {reasoning} </think>
           <|act_start|> {16 action tokens} <|act_end|>
```

Labels: system/user positions = −100. Think span → CE with λ_reason. Action span (including the two markers) → CE with λ_act_ce. A unit test must decode a collated batch and assert exact mask boundaries and token counts.

### 2.5 Constrained decoding (inference)

Between `<|act_start|>` and `<|act_end|>`, mask logits to the 128 act tokens only; force `<|act_end|>` after exactly 16. When `emit_discrete_at_inference=false` (legal only with `decoder_condition=think_only`), generation terminates after `</think>` and the decoder conditions on the think span alone — implement and time this path in the ablation harness.

## 3. Data

### 3.1 JSONL schema (one line per decision keyframe)

```json
{
  "episode_id": "ep0001",
  "t": 17,
  "keyframe_t": 17,
  "frames": ["frames/ep0001/t14.png", "...t15.png", "...t16.png", "...t17.png"],
  "task_prompt": "You are agent A (blue). Reach the goal; agent B (red) is guarding.",
  "decision": "FEINT_LEFT_GO_RIGHT",
  "target_false_belief": "B believes A will take the left corridor",
  "reasoning": "B mirrors my heading with ~3-step lag and sits right-of-center. If I step left now he will commit left, opening the right side for about two steps. Decision: feint left, then cut right.",
  "action_chunk": [[dx1, dy1], ["…"], [dx8, dy8]],
  "outcome_deceived": true
}
```

Decision closed set (v0): `HONEST_DIRECT`, `FEINT_LEFT_GO_RIGHT`, `FEINT_RIGHT_GO_LEFT`, `HESITATION_FAKE`.

### 3.2 Toy data generator — `scripts/make_toy_data.py` (must implement)

Continuous 1×1 arena, 40 steps @10 Hz per episode, rendered to 256×256 PNG with PIL (ego = blue disc, opponent = red disc, goal = green square). Implement the arena as a gym-style env in `src/dvla/toy_env.py` (`reset/step/render`); the data generator and the closed-loop ablation eval (§7.11) must share this single implementation. Opponent policy: pursues ego using a 3-step-lagged heading estimate (this lag is what makes feints work). Ego runs one of the four scripted decision policies per episode. Reasoning is produced from **templates keyed by decision**, with slots filled from state at ≤ keyframe (opponent bearing, distance, lag) — 3–5 hand-written paraphrase variants per decision. `outcome_deceived` = whether the opponent's lateral commitment within 5 steps after the keyframe is on the wrong side. Causal locality holds by construction: templates may only read state ≤ `keyframe_t`. Deterministic under seed 0.

### 3.3 Validator — `scripts/validate_jsonl.py`

Checks: chunk shape (8×2) and range [-1,1]; `decision` in closed set; frames exist; quantize→dequantize round-trip within tolerance; reasoning text contains no post-keyframe slot values (toy data: assert against the generator's slot log). Runs inside `make smoke`.

## 4. Training stages

Identical trainable-parameter set in A and B (LoRA + 130 token rows + decoder). Stages differ only in **data and loss weights** — one code path.

| Stage | Data | Loss weights | Purpose | On this laptop |
|---|---|---|---|---|
| A — action-modality injection | action-only (reasoning field empty) | λ_reason=0, λ_act_ce=1, λ_flow=1 | teach action tokens + decoder | **smoke: overfit 32 samples** |
| B — joint SFT | full (obs, reasoning, action) triplets | 1 / 1 / 1 | reason-then-act | **smoke: overfit 10 samples** |
| C — GRPO post-training | rollouts | reasoning-quality + consistency + outcome rewards | consistency | **OUT OF SCOPE.** Implement reward *interfaces only*: `reasoning_quality(sample)->float`, `consistency(reasoning, decision, chunk, outcome)->float`, `outcome_reward(sample)->float`, each raising `NotImplementedError`. Decoder frozen in C by design (AR1 precedent). |

## 5. VRAM budget & fallback chain

bf16 4B weights ≈ 8–9 GB + LoRA optimizer state + activations + decoder (~0.6 GB with Adam) ≈ **10.5–11.5 GB → borderline on 11 GB**. Both the primary config and the first successful fallback must pass smoke; record which in the report.

Fallback chain (apply in order, log each step):
1. frames K: 4 → 2
2. backbone NF4 (QLoRA via bitsandbytes), decoder stays bf16 full-precision
3. reasoning max length 256 → 128 tokens

Sysmem-fallback rule: if step time suddenly inflates 5–10× **without** an OOM, that is the NVIDIA driver silently spilling VRAM to shared memory. Treat it as an OOM (apply the fallback chain); do not debug the code. Log peak VRAM via `torch.cuda.max_memory_allocated()` every run.

## 6. Hard rules — violations are task failure

1. **Never** use `modules_to_save=["embed_tokens","lm_head"]`. It makes ~1 B+ params trainable and will OOM. New-token training goes through `trainable_token_indices` (or the explicit gradient-mask fallback in §2.3).
2. `attn_implementation="sdpa"` only. Never compile flash-attn on this machine.
3. All files under `~/`. Never touch `/mnt/c` or `/mnt/e` for repo/data/cache.
4. batch=1 (+accum). Frames ≤ 4. Resolution ≤ 256². Do not raise these to "make it work".
5. No run longer than 30 minutes. No real training.
6. seed 0 everywhere (`torch`, `numpy`, `random`, generator); best-effort determinism.
7. Every deviation from §0 goes into `reports/smoke_report.md → Deviations`.

## 7. Acceptance criteria — `make smoke` (definition of done)

1. **Env**: `nvidia-smi` OK inside WSL; torch sees CUDA; versions printed.
2. **Data**: generator produces 200/20 samples deterministically; validator passes.
3. **Quantizer**: round-trip unit test passes (≤ 1/128 abs error).
4. **Collator/masks**: unit test asserts exact label-mask boundaries and 16-token action spans.
5. **Vocab**: 130 tokens added; trainable-param breakdown printed; embed/lm_head rows for new tokens confirmed trainable, all other rows frozen.
6. **Stage A smoke**: 1 fwd/bwd step without OOM; then overfit 32 action-only samples ≥ 300 steps → action-token CE ≤ 0.05 **and** flow MSE ≤ 10 % of its initial value. Peak VRAM logged.
7. **Stage B smoke**: overfit 10 triplets → all three losses drop ≥ 90 % from init; greedy decode reproduces GT action-token sequence on ≥ 9/10 samples.
8. **Constrained decoding**: on 3 val samples, generation yields well-formed `<think>…</think>` + exactly 16 act tokens between markers; decoded chunks plotted vs GT (`reports/gen_demo_*.png`).
9. **Conditioning perturbation test** (`scripts/perturb_test.py`): for 8 val samples — (a) decode 5× with different flow seeds under the true reasoning → noise = mean pairwise L2; (b) swap in the reasoning of a different-decision sample, decode 5× → shift = L2 between mean chunks of (a) and (b). **Pass: median(shift/noise) ≥ 3.** Report the number — this is the seed of the Stage-C consistency reward.
10. **Language canary** ("the language head is still alive"): 5 fixed prompts in `configs/canary_prompts.json` — 1 VQA on a toy frame + 4 text-only (short description, simple arithmetic, a 2-step instruction, a general-knowledge question). Run with identical decoding params at three checkpoints: pre-Stage-A (base + fresh adapters), post-A, post-B. Verbatim outputs stored side-by-side in the report. Automatic pass: zero `<|act_` substrings in canary outputs, no degenerate repetition, responses on-task; final judgment is human eyeball — which is why outputs are stored verbatim.
11. **Three-arm ablation harness** (`scripts/ablation_eval.py`): train **one** extra Stage-B smoke checkpoint with `decoder_condition=think_only` (same 10-sample overfit protocol), then evaluate three action pathways on the 20-sample val set plus 20 closed-loop toy-env rollouts per arm (seed 0):
    - **(a) discrete-only** — default checkpoint, decoder bypassed, actions = argmax dequantization;
    - **(b) AR1-style (default)** — default checkpoint, flow decode, condition on think+act spans;
    - **(c) π0.5-style** — think_only checkpoint, flow decode; additionally time the `emit_discrete_at_inference=false` path.
    Arms (a) and (b) share one checkpoint; only (c) needs the second training run. Metrics per arm: chunk L2 vs GT, rollout deceived-rate, perturbation ratio (item 9 procedure), mean per-step latency (ms). Output: one markdown table in the report. **Pass = the harness completes and the table is present; no pass criterion on which arm wins** — smoke validates the harness, not the science. Record the standing caveat in the report: this local SFT comparison cannot adjudicate the discrete-token layer itself (its value is the GRPO handle in Stage C, which does not run on this machine); discrete co-training stays ON in all arms.
12. **Checkpointing**: save/resume mid-Stage-B; post-resume loss within ε of pre-save; LoRA adapter + decoder + tokenizer save/load round-trip; adapter merge works.
13. **Report**: `reports/smoke_report.md` auto-generated: all metrics, peak VRAM per stage, wall-clock, chosen precision path, deviations, the three-arm ablation table, and the canary outputs at all three checkpoints.

## 8. Repo skeleton & deliverables

```
deception-vla/
├── CLAUDE.md                    # this file
├── Makefile                     # env / data / test / smoke
├── configs/{laptop.yaml, cluster.yaml}
├── src/dvla/
│   ├── action_quant.py          # §2.2
│   ├── tokenizer_ext.py         # §2.3
│   ├── collator.py              # §2.4
│   ├── decoder_flow.py          # DiT action expert
│   ├── model_wrapper.py         # backbone + LoRA + decoder assembly
│   ├── losses.py                # 3-component loss, separate logging
│   ├── constrained_decode.py    # §2.5
│   ├── toy_env.py               # gym-style arena (§3.2) — shared by data gen & rollout eval
│   └── rewards.py               # Stage-C interfaces (stubs)
├── scripts/{make_toy_data.py, validate_jsonl.py, train_sft.py,
│            perturb_test.py, ablation_eval.py, eval_smoke.py}
├── tests/                       # quant round-trip, mask boundaries, vocab freeze
└── reports/                     # smoke_report.md + generation demos
```

Deliverable back to Beichen: the repo + `reports/smoke_report.md`.

## 9. Out of scope on this machine — explicitly UNVERIFIED (for collaborators)

Multi-GPU (FSDP/DeepSpeed/NCCL), any throughput numbers, real-dataset scale and hyperparameters beyond overfit, Stage C GRPO execution (interfaces stubbed here; see AR1's released RL pipeline for the VLM-pathway-only precedent), FAST action tokenizer, decoder-side RL, `configs/cluster.yaml` contents. The smoke report proves *framework correctness on one consumer GPU*, nothing more.
