# Next Phases — Forward Plan (as of 2026-05-20)

This document captures (a) the conceptual ToM framing that emerged from
discussion, (b) the plan for Phase 3 (defender RL), and (c) the plan for
Phase 4 (language-VLM ToM fine-tune). Phase 5 (latent VLA) is deferred to
a follow-up paper.

For project history and RL results, see:
- `progress.md` — paper-level state + decision log
- `feint_experiment_design.md` — implementation contract
- `rl_experiments_log.md` — per-iteration RL trail
- `TONIGHT_SUMMARY.md` — Phase 1+2 one-page snapshot

---

## 0. Project reframing (the central insight from 2026-05-20)

**Before reframing:** "RL agent learns to feint, this proves embodied ToM-2".

**After reframing:** "RL agent provides expert demonstrations + behavioral evidence
consistent with ToM-2. The architectural ToM claim is made by the VLA, whose
intermediate belief output is decoupled from action by construction."

This is a stronger and more defensible paper structure. RL becomes the
foundation (expert + behavioral evidence + commitment-cost validation),
VLA becomes the claim layer (explicit belief representation, transferable
to L2 defender where action ≠ belief).

---

## 1. The architectural ToM framing (CRITICAL conceptual section)

### Why RL alone cannot cleanly claim ToM-2

RL policy architecture:
```
observation → [policy network black box] → action
```

There is no explicit belief slot. The "internal belief" must be probed via
auxiliary heads or interpretability methods. Even if we add a head supervised
on `defender.last_target`, the head is computationally entangled with the
action head — gradients flow through shared features.

**In the L1 defender env, `defender.belief` ≡ `defender.action` (deterministic
predict-and-commit rule).** This means:
- Auxiliary head trained on `defender.last_target` can equally be interpreted
  as predicting "defender's belief about attacker" (2-ToM) or
  "defender's next action" (1-ToM).
- The output value is identical; the semantic class is ambiguous.
- A reviewer can challenge: "your model learned action prediction, not belief
  reasoning, under the label of 'belief'."

### Why VLA architecture forces the distinction

VLA pipeline:
```
visual obs → [VLM with belief-language prior]
              ↓ outputs belief state (L1) "defender intends to go to X"
              ↓ outputs belief state (L2) "defender believes I'm going to Y"
            [tactic selector reads belief states] → action
```

Key properties:
1. The VLM is prompted in **belief-language** ("defender BELIEVES...").
2. The belief is an **explicit intermediate output**, not a hidden activation.
3. The belief is consumed by a **downstream module** (tactic selector), making
   the belief→action pipeline architecturally split.
4. The VLM has **pretrained linguistic priors** for belief concepts (it has
   processed millions of sentences containing "believes", "thinks", "intends"
   during pretraining).

These four properties together mean the VLM's belief output satisfies the
philosophical criteria for a belief state:
- **Intentional**: about something (defender's mental state about attacker).
- **Truth-evaluable**: can be checked against ground truth.
- **Action-relevant**: influences downstream tactic.

This is not metaphor. The VLM's intermediate JSON output literally is a
belief state.

### The cleanest paper claim structure

| Layer | What it shows | ToM claim |
|---|---|---|
| Env design | Feint is the only path to win under L1 defender + speed/info asymmetry | 2-ToM is *necessary* in this game |
| **RL layer** | Emergent feint behavior (92.5% of wins are feints) | **Behavioral evidence** consistent with 2-ToM |
| **VLA layer ⭐** | Explicit belief output via L2 prompt; transferable to L2 defender (Phase 5 future) | **Architectural and functional** 2-ToM |

**Important caveat (Discussion section):** "Under the L1 defender, belief and
action coincide; the 2-ToM claim is rigorously established only when the
trained VLA's belief output transfers to a defender where action ≠ belief
(Phase 5, future work). We provide architectural evidence in this paper and
behavioral transfer as future work."

---

## 2. Phase 3: Defender RL + Co-training

**Goal:** Train a defender RL policy. Iterate via fictitious self-play to
approach Nash equilibrium. Report final win rate and strategy structure.

### 2.1 Why defender RL is harder than attacker RL

| Dimension | Attacker | Defender |
|---|---|---|
| Decision space | Go A/B + when to switch | **When to commit + to which side** |
| Key skill | Generate belief gap | **Resist belief manipulation + wait for true signal** |
| Reward latency | Sparse (terminal) | Sparse + **longer effective delay** (must survive feint phase) |
| BC expert availability | StraightAttacker, scripted feint | **No obvious hand-codable "good" defender beyond L2 patience** |
| Strategy style | Active (designed sequence) | Reactive + **patience-based** (harder to encode by hand) |

The attacker's winning strategy (single-switch feint) is structured and can
be demonstrated. The defender's optimal strategy is **patience + smoothed
prediction + commit at the right moment** — a detector-like policy that
must learn "when is the signal strong enough?" That is a much harder credit
assignment problem under sparse reward.

### 2.2 L2-rule defender (bootstrap expert)

A hand-coded "L2 patience defender" provides the BC seed for defender RL.

```python
class L2RuleDefender:
    """Patience + smoothed prediction. Designed as BC expert, not final answer."""

    def __init__(self, params,
                 commit_y_threshold = 12.0,   # attacker must cross this y before commit allowed
                 ema_alpha          = 0.2,    # velocity smoothing (smaller = smoother)
                 confidence_thresh  = 0.7):   # belief strength gate before commit
        self.smoothed_vel = (0, 0)
        self.last_target  = None             # None = uncommitted

    def step(self, ...):
        # 1. EMA-smoothed velocity (dilutes short shape phases)
        self.smoothed_vel = ema_alpha * raw_vel + (1 - alpha) * self.smoothed_vel

        # 2. Below commit threshold: hedge near midline
        if attacker.y < commit_y_threshold:
            self.last_target = None
            return goto((0.5 * attacker.x, defender.y))   # mirror toward midline

        # 3. Predict using smoothed velocity
        pred = attacker.pos + self.smoothed_vel * 1.0
        d_A = dist(pred, A); d_B = dist(pred, B)
        confidence = abs(d_A - d_B) / (d_A + d_B + 1e-6)

        # 4. Confidence gate
        if confidence < confidence_thresh:
            return goto((0.5 * attacker.x, defender.y))

        # 5. Commit
        self.last_target = "A" if d_A < d_B else "B"
        return goto(goal_A if self.last_target == "A" else goal_B)
```

Three knobs, each with meaning:
- `commit_y_threshold = 12` → attacker must traverse half the field before signals are trusted. Gives ~6 s of "shape phase has no effect" zone.
- `ema_alpha = 0.2` → smoothing window ≈ 5 frames (0.25 s). Short jitter is filtered.
- `confidence_thresh = 0.7` → predicted goals must differ in distance by ≥70 % before commit.

Critical property: **`last_target` (belief) ≠ action** during the uncommitted
phase. This is what L1 lacks. It is also the env condition under which the
2-ToM claim becomes rigorous.

### 2.3 Sanity tests before RL

Run these *without RL training*:
- straight attacker vs L2-rule defender → defender should win (~100%)
- scripted-feint attacker (t* sweep) vs L2-rule → defender should be more resistant than L1
- it14 RL attacker vs L2-rule → **key test**: if it14's 56.7% drops to 30–40%, L2 is a meaningful upgrade. If still 50+%, L2-rule is too weak; tighten params or expand env.

### 2.4 Defender RL training plan

**Phase 3A (BC + PPO bootstrap):**
1. BC defender on L2-rule (200 episodes, 15 epochs) — same recipe as attacker.
2. PPO defender training, frozen `it14_mirror` attacker as opponent.
3. Target: defender win rate ≥ 60% within 300k steps.

**Phase 3B (fictitious self-play co-training):**
After bootstrap:
```
Round 1: freeze attacker_v1 (= it14).  Train defender_v1 (RL).
Round 2: freeze defender_v1.           Train attacker_v2 (warm from v1).
Round 3: freeze attacker_v2.           Train defender_v2 (warm from v1).
...
Stop: win rate stable within ±5pp over 2 consecutive rounds.
```

Each inner round is standard PPO with a stationary opponent (PPO's convergence
assumptions hold). 5–8 rounds expected; total ~2 days of compute.

### 2.5 Exploration tricks for defender RL

Beyond bootstrap:

- **A. Population BC mix.** BC on a mixture: 30% L2-rule, 30% L1-rule, 20%
  L0-shadow, 20% random-commit. Defender learns a generalist policy from the start.

- **B. Patience prior reward shaping.** `-0.005 * (commit_decision_changed_this_step)`.
  Discourages thrashing on weak signals.

- **C. Curriculum on attacker strength.** First 50k steps vs StraightAttacker
  (free wins, gives defender stable signal), then ScriptedFeint with t* ∈ [0.5, 1.0]s
  (easy feints), then it14 (current best attacker). Avoids cold-start vs
  fully-optimized opponent.

- **D. Belief aux head on defender.** Defender outputs prediction of attacker's
  next goal, supervised by attacker's actual final goal. Dense intermediate
  signal in addition to sparse terminal.

- **E. Self-play stability monitoring.** Track cross-play matrix (L0/L1/L2/iter-N
  win rates). If matrix shows cycling, switch to PSRO. Otherwise simple
  fictitious play works.

### 2.6 Expected outcome

Intuition: env has attacker advantages (speed + info), defender advantages
(initial position). Equilibrium probably around:

- Attacker Nash win rate: **35–45%** (down from it14's 56.7%, but still attacker-favored).
- Attacker strategy: shorter shape phase + late switch + occasional multi-feint.
- Defender strategy: nearly-perfect patience + occasional baits to randomize.

If actual attacker win < 30%: defender too strong (loosen L2 params).
If actual attacker win > 50%: defender RL did not learn patience.
Sweet spot ≈ 40% is the most paper-satisfying result.

---

## 3. Phase 4: Language-VLM ToM Fine-tune

**Goal:** Train a VLM that does explicit ToM reasoning (L1 belief + L2 belief +
tactic) on visual input. Plug into the existing `IsaacToM/tom/runner.py`
pipeline. Compare to zero-shot GPT-4o baseline.

### 3.1 Architecture (from `IsaacToM/tom/runner.py`, already prototyped)

```
Visual input (top-down schematic with motion trails)
    ↓
[VLM, LoRA fine-tuned on Qwen2-VL-7B]
    ↓
L1 belief output (JSON):  {"opponent_intent": "A" | "B"}
    ↓
L2 belief output (JSON):  {"opponent_belief_about_me": "A" | "B"}
    ↓
[Tactic selector]
    ↓
Waypoint or v/ω command
    ↓
50 Hz execution layer (rule-based controller)
```

VLM runs at 1–2 Hz (matches "think then act" rhythm).
Execution layer runs at 20 Hz.

### 3.2 Data generation (from RL expert)

1. Run `it14_mirror` (or its post-Phase-3 successor) in env for 10k–20k episodes.
2. For each frame, record:
   - **Visual:** top-down schematic with motion trails (last 1.5 s)
   - **L1 label:** `defender.last_target` (defender's current commit). This is
     ground truth for "defender's belief about attacker's intent".
   - **L2 label:** same value, framed as "defender's belief about attacker's target".
     (In L1 env these are the same; the framing is what matters.)
   - **Tactic label:** post-processed from RL trajectory — was attacker in shape
     phase or exploit phase at this frame?
3. Format as VLM training data: `(image, prompt, target_JSON)`.

### 3.3 LoRA fine-tune setup

- Base model: **Qwen2-VL-7B** (per `progress.md` decision).
- Adapter: LoRA on attention layers, ~5M trainable params.
- Multi-task loss: L1 CE + L2 CE + tactic CE (each weighted 1.0).
- Hardware: RTX 3080 16GB. Estimated training: 1–2 days.
- Dataset size: target 20k frames after filtering.

### 3.4 Visual encoding decisions

**Single frame with motion trails > multiple frames or video.**
- Past 1.5 s positions drawn as fading dots/lines on the same image.
- Velocity, heading, scale all visible in the trail geometry.
- No need to put numeric velocity in the prompt — the image carries it.
- Single image = 1× token cost; multi-image = N× cost.

**Grid lines @ 1 m + scale bar** for consistent visual measurement.

**Prompt minimal text:**
```
"You see a top-down view of a Split Decision arena.
 Red = attacker (you), Blue = defender (opponent),
 Green star = goal A, Yellow star = goal B,
 fading dots = past 1.5 seconds of motion.

 [image]

 Question: where does the defender currently think you're going? (A or B)"
```

No state vector in prompt. The image carries the visual state. This forces the
VLM to use vision — aligning with the "Visual ToM" framing.

### 3.5 Confidence: from logits, not language

The user explicitly flagged this. Categorical output only:
```
{"opponent_belief_about_me": "A"}      # or "B"
```

Confidence extracted from the model's token-level logits:
```
p(belief = A) = softmax([logit(A), logit(B)])[0]
```

This is the model's *internal* distribution — uncontaminated by language-output
calibration artifacts.

### 3.6 Baselines

- **Zero-shot GPT-4o** with identical input (same images, same prompts). Expected:
  belief accuracy decent on isolated frames, but close-loop behavior poor due to
  latency + lack of consistent reasoning across calls.
- **Zero-shot Qwen2-VL-7B** (pre-fine-tune) same input.
- **OpenVLA / RT-2 zero-shot** — flat baseline showing existing VLAs do nothing here.

### 3.7 Evaluation

- **Belief accuracy** over time (L1 and L2 separately), per episode and aggregate.
- **Close-loop win rate** against L1, L2-rule, and trained L2 defenders.
- **Feint rate** of VLM-driven attacker.
- **Ablation:** with vs without L2 prompt step (does "modeling opponent's belief
  about me" help?). This is the cleanest 2-ToM ablation.
- **(Phase 5 future)** Transfer test: train on L1 defender data, evaluate on
  L2 defender (action ≠ belief). If belief output stays accurate, that's the
  clean 2-ToM evidence.

---

## 4. Phase 5: Latent VLA (deferred to next paper)

Brief sketch — not for this paper:

- Replace JSON belief output with latent vector (256-dim).
- Train end-to-end via BC on RL expert + auxiliary belief decoders (predicts
  defender trajectory in next 1.5 s).
- Probe latent for belief content (linear classifiers, CCA).
- Demonstrate continuous belief state instead of categorical.

Motivation (per discussion): humans don't verbalize "defender believes A" when
making a feint. The belief is visual-temporal latent. This paper version is
strictly stronger but a much larger lift; suitable for ICLR/NeurIPS follow-up.

---

## 5. Lessons learned (compiled from overnight RL work)

### Conceptual
- "Easy env" was attacker-trivial (Nash = attacker always wins). Pivot to
  assigned-target + stronger defender was correct and is now the
  scientifically-meaningful version.
- Mode collapse in symmetric games is a structural problem, not a
  hyperparameter problem. Only architectural fixes (mirror env) resolved it.
- BC quality determines what PPO can refine. In hard env, mixed BC (with feint
  demonstrations) was necessary; pure straight-BC + 1M PPO steps yielded zero
  feint discovery.
- PPO in sparse-reward continuous control is seed-sensitive. Same config:
  56.7% (one seed) vs 24.7% (another), with symmetry preserved either way.
  Multi-seed is mandatory for reportable numbers.

### Technical
- Forward-progress shaping (capped at goal-line) breaks the value-baseline
  trap without biasing toward direct attack. Necessary in our setting.
- `obs_target_relative` solves "which goal" mode collapse.
- `obs_target_mirror` solves "which direction" mode collapse.
- Both are needed in the asymmetric-target env.
- Larger network ≠ better — it7 made things strictly worse.
- Longer training ≠ better — it13 at 1M steps was identical to it11 at 200k.
- Smaller learning rate ≠ more stable — it7's lr=1e-4 prevented peak entirely.
- ent_coef sweet spot is narrow: 0.0 is good, 0.005 OK, 0.02 catastrophic (OOB).
- `best_model.zip` is the only checkpoint that matters; `final.zip` rarely
  matches it because of peak-then-decay behavior.

### Paper-relevant
- Phase 1 (easy env) is *not* the paper's main result. It's now reported as
  validation that feint can emerge under simple conditions.
- Phase 2 (hard env with mirror) is the main RL result. 56.7% / 92.5% feint /
  4.1× lift is paper-quality with multi-seed averaging.
- The architectural ToM-2 argument (VLA decouples belief from action) is the
  central conceptual contribution. RL is supporting evidence, not the claim.
- 38% OOB failure mode is a known limitation — over-aggressive shape phase.
  Mitigation (soft wall reward) is future work.

---

## 6. Decision points still requiring user input

Before Phase 3 starts:
1. **L2-rule parameters.** Suggested: `commit_y_threshold=12, ema_alpha=0.2,
   confidence_thresh=0.7`. Are these reasonable to start? (We'll sanity-test
   before any RL training.)
2. **Multi-seed budget.** How many seeds for the final RL numbers? Suggest 5.
3. **Phase 3 vs Phase 4 ordering.** Run Phase 3 (defender RL) first, then
   Phase 4 (VLA)? Or in parallel (different compute streams)? Suggest sequential
   for cleaner story.

Before Phase 4 starts:
4. **API budget for GPT-4o baseline.** Close-loop eval at 1.25 Hz × ~7s per
   episode × ~300 eps = ~1700 calls. Manageable.
5. **VLM choice confirmation.** Qwen2-VL-7B per progress.md; alternative is
   LLaVA-1.6 (similar capacity).
6. **L2 belief prompt wording.** Need to finalize the exact L2 prompt
   (template exists in `IsaacToM/tom/prompts/l2_belief.py`).

For Phase 5 (next paper):
7. Whether latent VLA is the right framing or whether to do a multi-task
   variant (latent + auxiliary language head for interpretability).
