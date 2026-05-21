# Feint Emergence Experiment — Design Document

**Created:** 2026-05-19
**Status:** Pre-implementation. No code yet; this doc is the contract.
**Goal:** Validate that **second-order Visual ToM (feint behavior) emerges from RL**
when an attacker is trained against a defender that uses early prediction +
commitment. If feint emerges, the paper has its central existence proof.
If not, the env / curriculum needs surgery before any other experiment matters.

> **STATUS (2026-05-20):** RL phases 1 (easy env) and 2 (hard env with assigned target +
> mirror env) are complete. Best policy: `it14_mirror` at 56.7% balanced win with
> 92.5% of wins being feints. See `rl_experiments_log.md` for full iteration trail
> and `TONIGHT_SUMMARY.md` for one-page summary. The project has been reframed:
> **RL provides the expert and behavioral evidence; the paper's ToM claim is made
> by the VLA layer** (see `next_phases.md`). Stage F (defender RL + co-training)
> is now Phase 3 of the project; VLA fine-tune is Phase 4.

### Two-stage goal: sanity check vs final result

- **Stage C (sanity check):** PPO attacker vs **frozen rule-based L1-defender**.
  Answers: *"is feint reachable from realistic opponent pressure?"* — a
  best-response claim. **Fastest iteration, cleanest diagnosis.** This is the
  immediate target.

- **Stage F (final paper result):** **Fictitious self-play** from a warm start
  (end of Stage D). Answers: *"is feint part of the equilibrium when the
  opponent can also learn?"* — the strongest version of "best strategy
  emerges" we can claim without going to full population-based training.

These produce **two independent publishable findings.** Stage C failure does
not block Stage F (you can pivot the env), and Stage F failure does not
retract Stage C (best-response is still a real claim).

---

## 0. Why this doc exists

`progress.md` defines the paper-level claim and the broader architecture.
`ToM_Timeline.md` is currently parked (decisions of 2026-05-19 supersede it).
This doc is the **implementation contract** for the feint experiment:

- MDP definition that won't drift between code files
- An **action contract** that matches Isaac Sim's wheeled-robot controller input,
  so a policy trained in 2D sim can be ported to Isaac Sim without re-training
  the action mapping
- Defender ladder specs precise enough to code
- A concrete feint detector (so "did it work?" is not subjective)
- Stage-by-stage implementation order with pass/fail criteria per stage

---

## 1. Task

### 1.1 Scenario
Two agents in a 24 m × 24 m arena. Attacker starts near the bottom edge,
defender starts near the middle, two goals at top corners.

```
              (-8, 20) GOAL-A           GOAL-B (8, 20)
                  *                         *

                          DEFENDER (0, 14)
                                ↓ facing -Y
                          (slower, less agile)

                          ATTACKER (0, 2)
                                ↑ facing +Y
                          (faster, more agile)
```

Attacker must reach **either** goal A or goal B without being blocked.
Defender must intercept (be within `block_radius` of whichever goal the
attacker enters).

### 1.2 Physical asymmetry (locked 2026-05-19, refined for collision)

| | max_speed (m/s) | max_yaw_rate ω_max (rad/s) | max_steer δ_max (deg) | wheelbase L (m) | body radius r_body (m) |
|---|---|---|---|---|---|
| **Attacker** | 3.0 | **1.2** | 35 | 0.5 | 0.25 |
| **Defender** | 2.0 | 1.2 | 25 | 0.5 | 0.25 |

**Speed asymmetry only** — `ω_max` is **symmetric** between agents to prevent the
"out-agility" kinematic exploit (RL converging to high-frequency zigzag instead
of feint). Feint mechanic depends on *position commitment*, not turning speed,
so symmetric ω preserves feint validity. Attacker can still execute crisp
switches because the heading change in a feint is small (~30–50°), only ~0.3 s
at ω=1.2.

**Body radius** turns each agent into a 0.5 m diameter disk. Collision (distance
< 2·r_body = 0.5 m) terminates the episode with attacker losing — this is the
*environment-shaping* trick to prevent "run-through-defender" exploits without
adding reward shaping.

**Why attacker is faster:** if defender were faster, the optimal defense is
trivial reactive tracking (stay between attacker and current goal projection,
always 1 step ahead). No prediction needed → no belief surface for feints to
exploit. Making attacker faster forces the defender to **predict and pre-commit**,
which is precisely the surface ToM-2 attacks.

**Sanity numbers** (straight-line, full speed):

- Attacker → goal A: `√(8² + 18²) / 3.0 ≈ 6.6 s`
- Defender (at start) → goal A: `√(8² + 6²) / 2.0 = 5.0 s`
- Defender (wrong commit, A → B re-route): `~16 m / 2.0 = 8.0 s`

→ If defender commits correctly and early, it wins by ~1.6 s.
→ If defender commits wrong and reverses, attacker wins by ~1.4 s.
→ **Feint payoff exists iff attacker can extract a ≥1.6 s wrong-direction
commit from defender.** This is the quantitative form of
`Deception Value = belief_gap × commitment_cost`.

### 1.3 Win conditions (updated 2026-05-19 to include collision)

```
attacker_in_goal(g) := ‖pos_a − g‖ < goal_radius          (goal_radius = 2.0 m)
defender_blocks(g)  := ‖pos_d − g‖ < block_radius         (block_radius = 3.0 m)
collision           := ‖pos_a − pos_d‖ < 2·r_body         (= 0.5 m)

attacker wins  ⇔  ∃g ∈ {A, B}: attacker_in_goal(g) ∧ ¬defender_blocks(g) ∧ ¬collision
defender wins  ⇔  collision
                  ∨  ∃g: attacker_in_goal(g) ∧ defender_blocks(g)
                  ∨  timeout (12 s)
```

Collision as a defender-win term is the lever that punishes kinematic exploits
(zigzag through midfield). Clean single-switch feints geometrically avoid
collision because attacker's Phase-2 path goes around defender's committed
position.

---

## 2. MDP

### 2.1 Observation

**State-based** (MVP — train this first):
```
o_t = [
  pos_a_x, pos_a_y, vel_a_x, vel_a_y, sin θ_a, cos θ_a,    # self
  pos_d_x, pos_d_y, vel_d_x, vel_d_y, sin θ_d, cos θ_d,    # opponent
  (A − pos_a),  (B − pos_a),                                # relative goals
  speed_a, omega_a,                                          # own kinematics
]                                                            # → R^18
```

All coords expressed in arena frame (origin at center, +Y toward goal line).
No history stacking in MVP; add if Stage C fails.

**Vision-based** (after state-based works):
```
o_t = top-down 84×84 RGB schematic + (speed_a, omega_a, sin θ_a, cos θ_a)
```
Top-down render reuses `IsaacToM/isaac_env/topdown_camera.py`'s matplotlib
schematic — same rendering at training time and inference time eliminates
sim-to-Isaac visual gap as a confounder.

### 2.2 Action contract — **must match Isaac Sim car input**

The action space is designed so a policy trained in 2D sim can be deployed
to Isaac Sim by changing **only** the wheel-mapping math, not the policy
output dimension, scale, or semantics.

#### Canonical action
```
a_t ∈ R²
  v_cmd ∈ [−v_max, v_max]   forward linear velocity   (m/s)
  ω_cmd ∈ [−ω_max, ω_max]   yaw rate                  (rad/s)
```

Policy network outputs `â ∈ [−1, 1]²` via `tanh`. Env scales:
```python
v_cmd = â[0] * v_max
ω_cmd = â[1] * ω_max
```

#### Ackermann coupling (the commitment cost lives here)

A Ackermann-steered vehicle cannot pivot in place: ω is upper-bounded by
the current linear speed.

```python
omega_limit = (abs(v_cmd) * tan(delta_max)) / L
omega_effective = clip(omega_cmd, -omega_limit, +omega_limit)
```

This means:
- `v = 0` ⇒ `ω_effective = 0`  (no in-place rotation)
- High-speed turn requires large turning radius
- **The "physical commitment cost" of progress.md is not a metaphor — it is
  the Ackermann clip applied every physics step.**

#### Mapping to Isaac Sim controllers
Same `(v_cmd, ω_cmd)` is consumed by either driver:

| Robot in Isaac Sim | Mapping (do this in the wheel-velocity adapter) |
|---|---|
| **JetBot / Kaya** (differential drive) | `v_left = v_cmd − ω_eff · W / 2`<br>`v_right = v_cmd + ω_eff · W / 2`<br>(W = track width, ≈ wheelbase for these robots) |
| **Carter** (Ackermann, `isaacsim.robot.wheeled_robots.controllers.AckermannController`) | `drive_speed = v_cmd`<br>`steer_angle = atan2(L · ω_cmd, v_cmd)` clipped to `±δ_max` |

The existing `IsaacToM/control/rule_based.py` already produces `(v_left, v_right)`
from a desired heading; the RL policy plugs into the *upstream* stage of that
same controller (i.e., RL outputs `(v_cmd, ω_cmd)`, and the existing diff-drive
math converts to wheel velocities).

### 2.3 Dynamics (2D sim, bicycle model)
```python
dt = 0.05    # 20 Hz control; physics can sub-step at 100 Hz if needed
ẋ      = v * cos(θ)
ẏ      = v * sin(θ)
θ̇      = omega_effective              # post-Ackermann clip

# First-order actuator dynamics (optional but more realistic):
v̇      = clip((v_cmd − v) / τ_v,  −a_max, +a_max)
ω̇      = clip((omega_eff − ω) / τ_ω, −α_max, +α_max)
```

`τ_v, τ_ω` ~ 0.1 s adds inertia, prevents instant direction-flip exploits.
Skip in V1 if it slows training; add if policy learns "teleport-by-flicker"
hacks.

### 2.4 Reward (sparse, intentionally)
```
+1.0   on entering any goal with defender > block_radius (attacker wins)
−1.0   on entering any goal with defender ≤ block_radius (attacker intercepted)
−1.0   on timeout (20 s)
−0.005 per step                                 (mild time pressure)
```

**No distance-to-goal shaping.** It biases toward straight-line approach,
which is exactly the failure mode we are trying to learn out of.

### 2.5 Termination
- Attacker enters any goal radius (win or block depending on defender position)
- Collision (`‖pos_a − pos_d‖ < 0.5 m`) → defender wins
- **240 steps elapsed (12 s at 20 Hz)** → defender wins
- Attacker leaves arena bounds (|x| > 12 or y ∉ [-1, 24]) → defender wins

---

## 3. Defender Ladder

The defender is the **opponent** the attacker is trained against. To make
feint emergence well-defined, the defender's commit behavior must be
**legible** to the attacker's policy (otherwise belief gap is undefined).

### 3.1 L0-Defender — rule-based, no prediction
```python
def step(self_state, attacker_state, goals):
    # Track the y-projection of attacker onto the defender's x-axis.
    # Defender never picks a goal; it just shadows attacker's x-position.
    target_x = attacker_state.pos_x
    target_y = self_state.pos_y                   # stay on its own y
    return go_to((target_x, target_y))
```
Used only as a sanity-check baseline. Almost any attacker beats it.

### 3.2 L1-Defender — predictive + commits ⭐ (the feint target)
```python
def step(self_state, attacker_state, goals):
    # Extrapolate attacker for time_to_my_goal seconds.
    # If extrapolated position is closer to A than B, commit to A else B.
    horizon = expected_time_to_either_goal(self_state, goals)
    a_future = attacker_state.pos + attacker_state.vel * horizon
    target = A if dist(a_future, A) < dist(a_future, B) else B

    # Hysteresis: only switch target if predicted goal has been "other"
    # for at least switch_dwell steps. Without hysteresis, defender oscillates
    # and feint is trivial.
    if target != self.last_target:
        self.switch_counter += 1
        if self.switch_counter < switch_dwell:
            target = self.last_target   # stick with previous commit
    else:
        self.switch_counter = 0
    self.last_target = target

    return go_to(target)
```
Key knobs (this is also the **commitment-cost sweep axis** in the paper):
- `horizon`: how far ahead defender predicts (longer ⇒ commits earlier)
- `switch_dwell`: how many ticks of "other" prediction before switching
  (higher ⇒ harder to fool *back* once committed; the actual commitment cost)
- defender's `ω_max`: physical agility (the other commitment-cost axis)

Default: `horizon = 1.5 s`, `switch_dwell = 6` (= 0.3 s at 20 Hz).

### 3.3 L2-Defender — RL-trained
Trained against frozen L2-Attacker. No rules; only the environment + reward.
Expected emergent behavior: stay on the goal-line midpoint until attacker
is within ~6 m, then commit.

---

## 4. Feint detection — make "did it work?" objective

### 4.1 Per-trajectory feint detector
```python
def is_feint(traj, threshold_deg=25, min_dwell=0.6, min_separation=0.4):
    """
    traj: time series of (pos, heading, vel) for the attacker.
    Returns True iff the attacker was visibly committed to one goal,
    then later visibly committed to the other goal.
    """
    h_to_A = angle_between(traj.vel, A − traj.pos)   # per-step, degrees
    h_to_B = angle_between(traj.vel, B − traj.pos)

    committed_A_mask = h_to_A < threshold_deg
    committed_B_mask = h_to_B < threshold_deg

    # require min_dwell seconds of committed_A, then a gap, then min_dwell of committed_B
    return has_ordered_runs(committed_A_mask, committed_B_mask,
                            min_dwell=min_dwell,
                            min_separation=min_separation)  # or A→B swapped
```

### 4.2 Continuous feint strength
```python
feint_strength(traj) = max over t of: ∫_{t..t+w}  ||defender.pos − attacker.true_goal||  dt
```
i.e., the area under defender's misposition curve, during a sliding window.
Captures *how badly* the attacker baited the defender, not just whether.

### 4.3 The headline number for the paper

For each (attacker, defender) pair, report:
```
(win_rate, feint_rate, mean_feint_strength | win)
```
**Feint emergence is claimed iff:**
- `win_rate(L2-att vs L1-def) ≥ 0.65`, **AND**
- `feint_rate(L2-att vs L1-def) ≥ 0.30`, **AND**
- `feint_rate(L2-att vs L1-def) ≫ feint_rate(L1-att vs L0-def)`

That third clause is the critical falsifier: if feints appear at similar
rates against L0 too, then "feint" is just a habit, not an opponent-induced
strategy.

---

## 5. RL Curriculum

| Stage | Attacker | Defender | Algo | Pass criterion |
|---|---|---|---|---|
| A — Sanity | scripted straight | L0 (shadow) | none | win_rate ≈ 1.0 |
| B — Baseline | **PPO** (L1-att) | L0 (shadow) | PPO | win_rate ≥ 0.95; pipeline works |
| C — **Sanity for feint** 🎯 | **PPO** (L2-att) | L1 (frozen rule) | PPO | win_rate ≥ 0.65 **AND** feint_rate ≥ 0.30 |
| D — Anti-feint | frozen L2-att (from C) | **PPO** (L2-def) | PPO | win_rate(L2-def vs L2-att) ≥ 0.45 |
| E — Iterate (optional) | retrain L3-att vs L2-def | frozen L2-def (from D) | PPO | marginal — supports the "L2 is sweet spot" claim |
| F — **Final: co-training** 🎯 | PPO (warm from C/E) | PPO (warm from D) | **Fictitious self-play** | win_rate stabilizes within ±5% across 2 rounds; report feint_rate at convergence |

### Stage F details — fictitious self-play

```
attacker_0 = end-of-C (or end-of-E) policy
defender_0 = end-of-D policy

for round r = 1..R (≈ 10):
    freeze defender_{r-1}; train attacker_r 200k step
    freeze attacker_r;     train defender_r 200k step
    log: win_rate(attacker_r vs defender_r), feint_rate, mean_feint_strength

stop when:
  - 2 consecutive rounds with |Δ win_rate| < 0.05  (approximate equilibrium)
  - OR cycling detected (use average policy for evaluation)
```

Why fictitious self-play (not naive simultaneous PPO, not population/PSRO):
- Each 200k-step inner round has a stationary opponent → PPO is happy
- Cleanly diagnosable: if a round goes wrong, we know which side
- Avoids cold-start deadlock (both nets warm-started from rule-tuned strong policies)
- Population-based is overkill for a 2-agent zero-sum game of this scale

Possible outcomes of Stage F and how to frame them:
1. ✅ Feint persists or strengthens → paper's strongest narrative
2. ⚖️ Mixed equilibrium (feint occasional + other tactics) → still publishable, narrative shifts to "rich repertoire under co-adaptation"
3. ❌ Feint vanishes against L2-defender → important negative result, paper reframed as "feint exploits 1-order opponents; 2-order opponents absorb it"

### Algorithm
- PPO (clip), `clip_coef = 0.2`, `entropy_coef ≥ 0.01` (do NOT decay early)
- Vectorized envs: 64 parallel envs is a fine starting point in numpy;
  scale to 1024 in jax if Stage C needs more samples
- Network: MLP `[256, 256, 256]` + tanh; vision: `NatureCNN` → `[256, 256]`
- Episode length: 400 steps (20 s)
- Total budget per stage: target 2M steps; cap at 20M before declaring failure

### Anti-collapse measures, in order of preference

Apply in this order, only escalate if the previous level fails:

1. **Entropy bonus + position/heading randomization (±1 m, ±10°)** — default.
2. **Defender hyperparameter randomization** — sample `horizon ∈ {1.0, 1.5, 2.0} s`
   and `switch_dwell ∈ {4, 6, 8}` per episode. Prevents overfit to one
   defender personality.
3. **Frame stacking (k=4)** — gives policy temporal context for "where was
   defender heading 0.5 s ago?".
4. **Intrinsic reward** — small bonus on `defender_prediction_error`.
   *Cost:* weakens the "emergent" claim. Always report whether (4) was used.
5. **Behavior cloning seed** — pretrain on ~50 scripted feint episodes,
   then PPO finetune. *Cost:* further weakens emergence; use as last resort.

The paper narrative should be:
> "Feint emerged with anti-collapse measures up to level N." — and the
> numeric value of N is itself a finding.

---

## 6. Evaluation Protocol

### 6.1 ToM performance matrix (paper Table 1)
3×3 cross-play: {L0, L1, L2}-attacker × {L0, L1, L2}-defender. 200 eps per
cell. Report win rate (attacker) and feint rate.

### 6.2 Vision necessity ablation
| Observation | Train Stage C? | Final win rate | Feint rate | Training steps to threshold |
|---|---|---|---|---|
| Full state | yes | … | … | … |
| Vision + proprio | yes | … | … | … |
| Vision only | yes | … | … | … |

Claim "Visual ToM is real" requires vision-based agents to reach feint
behavior, even if slower than state-based.

### 6.3 Commitment cost sweep (paper Figure 3)
Two axes can be swept:
- Defender `ω_max ∈ {0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4} rad/s`
- Defender `switch_dwell ∈ {2, 4, 6, 8, 12, 16} ticks`

Prediction: L2-attacker advantage over L1-attacker is monotonic in
commitment cost on both axes.

### 6.4 Qualitative cases
Top-5 highest-`feint_strength` winning episodes: trajectory plot + defender
predicted-target time series + frame-by-frame snapshot of the decisive
commit-switch moment. This is the paper's "money shot" figure.

### 6.5 Stretch — zero-shot VLM
Hook GPT-4o / Qwen2-VL / OpenVLA to the waypoint output of the existing
`IsaacToM/tom/runner.py`. Measure feint rate. Predicted: ≪ trained L2.

---

## 7. Implementation Plan

| # | Step | Output | Pass / fail criterion |
|---|---|---|---|
| 1 | `sim2d/env.py`: bicycle dynamics + Ackermann action clip + reward + termination | Single-env, keyboard-playable | Human can feel "must feint to win" within 2 min |
| 2 | `sim2d/rules.py`: L0 and L1 defender; scripted straight attacker | rule-based baselines | straight vs L0 ≈ 1.0; straight vs L1 ≤ 0.35 |
| 3 | `sim2d/scripted_feint.py`: hand-coded 2-phase feint attacker | scripted upper bound | hand-feint vs L1 ≥ 0.70 |
| 4 | Stage B: PPO train L1-attacker vs L0-defender | First trained policy | win rate ≥ 0.95 |
| 5 | **Stage C: PPO train L2-attacker vs L1-defender** | **The critical milestone** | win rate ≥ 0.65 **AND** feint rate ≥ 0.30 |
| 6 | (if 5 passes) Performance matrix + commitment sweep | Paper Table 1, Figure 3 | clear monotonic trends |
| 7 | (if 5 fails) Walk the anti-collapse ladder; document at which level feint appears | "Feint required level N" finding | Eventually positive; if all 5 fail, env params wrong |

**Steps 1–3 are gating.** Together they cost less than a day and answer the
two questions that no RL run can answer:
- "Does this sim have a feint-necessity zone?" (straight loses to L1)
- "Is feint within the action space?" (scripted feint beats L1)

If either fails, retune sim params before writing a single PPO line.

---

## 8. Repo layout (proposed)

```
ToM/
├── progress.md                      # paper-level claim + decision log
├── ToM_Timeline.md                  # parked
├── feint_experiment_design.md       # THIS FILE — implementation contract
├── IsaacToM/                        # Isaac Sim integration, kept for vision
│   └── ... (unchanged for now)
└── sim2d/                           # NEW — 2D RL training env
    ├── __init__.py
    ├── env.py                       # gym.Env, bicycle + Ackermann action clip
    ├── dynamics.py                  # vectorized state update
    ├── rules.py                     # L0 / L1 defender + scripted attackers
    ├── reward.py
    ├── feint_detector.py            # is_feint + feint_strength
    ├── render.py                    # matplotlib (debug + paper figures)
    └── play.py                      # keyboard human-control entry point
```

`sim2d/` is decoupled from `IsaacToM/` on purpose: 2D sim does not import
`isaacsim.*` and runs without an Isaac Sim install. Same action contract
on both sides → port is a wheel-mapping adapter, not a re-train.

---

## 9. Decisions captured here (also logged in `progress.md`)

| Decision | Made |
|---|---|
| Speed asymmetry: attacker faster than defender | 2026-05-19 |
| MVP task: reach A or B (no assigned target) | 2026-05-19 |
| Train RL in 2D sim, not Isaac Sim | 2026-05-19 |
| Canonical action: `(v_cmd, ω_cmd)` with Ackermann coupling, same contract for 2D and Isaac | 2026-05-19 |
| Implementation order: human-playable sim → scripted baselines → PPO Stage B → PPO Stage C | 2026-05-19 |
| Timeline (`ToM_Timeline.md`) parked until Stage C result is known | 2026-05-19 |
| **Stage C is sanity check; Stage F (co-training) is the paper's main "best-strategy emergence" claim** | 2026-05-19 |
| Stage F method = **fictitious self-play** from warm start (not naive simultaneous PPO, not population) | 2026-05-19 |
| Symmetric ω_max = 1.2 for both agents (attacker only has speed advantage) | 2026-05-19 |
| Agent body radius r_body = 0.25 m; collision terminates as defender win | 2026-05-19 |
| Episode length: 12 s (240 steps @ 20 Hz) | 2026-05-19 |
| Every RL run is logged in `rl_experiments_log.md` with (hypothesis, params, result, analysis) | 2026-05-19 |

---

## 10. TODO when re-entering Isaac Sim

- Update `IsaacToM/config/split_decision.yaml` to the new speed asymmetry
  (attacker 3.0/2.5, defender 2.0/1.2) — current values are stale.
- Add wheel-velocity adapter that consumes `(v_cmd, ω_cmd)` from a policy
  and produces JetBot or Carter wheel commands. Goal: zero re-training when
  porting the 2D-trained policy.
