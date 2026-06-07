# Visual Theory of Mind for Embodied Adversarial Agents — Project Progress

## Project Overview
**Primary target:** CoRL 2026 (deadline: May 28, 2026) — rush MVP
**Backup target:** ICLR 2026 (deadline: ~Oct 2026) — full version if CoRL quality insufficient
**Working title:** *Visual Theory of Mind: Deceptive Motion Planning through Visual-Temporal Belief Reasoning in Embodied Adversarial Games*
**Start date:** 2026-03-22

---

## Core Claim (Refined 2026-03-23)

**Previous framing (deprecated):** Belief-space adversarial motion planning — plan in opponent's belief space to exploit physical commitment. Problem: "just engineering" if contribution is only the pipeline.

**Current framing:** We propose **Visual Theory of Mind** — a fundamentally different form of ToM where reasoning occurs in visual-temporal space rather than language/symbolic space. In physical adversarial settings, deception is not about transmitting false propositions (like lying in Werewolf); it's about generating misleading motion patterns that exploit the opponent's visual-temporal perception of your intent.

**Three distinguishing features of Visual ToM vs Language/State-based ToM:**
1. **Information carrier = motion pattern, not symbol.** No propositions — only continuous trajectories. Intent is inferred perceptually, not propositionally.
2. **Deception = misleading motion generation, not information transmission.** No communication channel — your body/movement IS the signal. Deception means generating motion that looks like intent-A but executes intent-B.
3. **Reasoning and execution are coupled.** No separate "think then act" — you read opponent motion while generating your own, simultaneously. ToM is embedded in the sensorimotor loop.

**One-sentence pitch:**
> We define Visual Theory of Mind — where embodied agents reason about opponents' beliefs through visual-temporal motion patterns rather than language — and show that second-order Visual ToM enables strategic deception (feints) in adversarial physical settings where Nash equilibrium is intractable.

**Paper story arc (three parts):**
1. **Define the problem:** Visual ToM as a distinct form of ToM; design adversarial environment where it's necessary
2. **Diagnose existing models:** Show current VLAs/RL policies fail at 2nd-order Visual ToM
3. **Propose solution + validate:** New architecture/training that enables Visual ToM; demonstrate transferability across environments

---

## Problem Formulation (Key Insight)

**Deception Value = Belief Gap × Commitment Cost**

- **Belief gap**: divergence between opponent's prediction of my next action and my actual intent
- **Commitment cost**: physical cost for opponent to reverse course (function of velocity, turning radius, reaction delay)
- Feints are only effective when BOTH are large — you need the opponent to believe the wrong thing AND be physically committed
- This explains the two-phase structure of human feints: (1) shaping phase to build belief gap, (2) exploitation phase when commitment window opens

### Formal Definitions

**Game setting:** Two-player Partially Observable Stochastic Game (POSG)
- State: s_t = (pos_e, vel_e, θ_e, pos_p, vel_p, θ_p, obstacles) ∈ S
- Actions: a_t^e ∈ A_e (evader/attacker), a_t^p ∈ A_p (pursuer/defender)
- Observations: each agent sees partial state through ego camera (NOT full state)

**ToM Order Definitions (corrected 2026-03-23):**
- **0th-order (No ToM):** Pure reactive. Act based on physical state, no mental state modeling. "Ball is left → go left." "Opponent is left → go right."
- **1st-order:** I infer your mental state. "I believe you intend to go left" (defender reads attacker's movement). This is basic intent prediction.
- **2nd-order:** I infer your inference about MY mental state. "I believe you will believe I intend to go left" (attacker does feint). This IS the feint — I generate misleading motion because I model how you'll interpret my motion.
- **3rd-order:** "I believe you know I might feint, so you won't trust my leftward motion" → double bluff. Intractable (see Proposition 1 below).

**Key correction:** A feint (shoulder drop, fake shot) IS second-order ToM, not first-order. The attacker models the defender's interpretation of the attacker's own behavior. A defender simply predicting attacker direction IS first-order.

**Belief (Level-k) — formal:**
- Level-0: b⁰(s) — agent's belief about environment state
- Level-1: b¹(a_opp) = P(a^opp_{t+1} | o^self_{1:t}) — my prediction of opponent's next action
- Level-2: b²(b¹_opp(a_me)) — my estimate of "opponent's prediction of MY next action"
  - For rule-based pursuers, this is closed-form computable from pursuer's logic

**Belief Gap:**
- B_gap(t) = D_KL( b²_t || δ(a^actual_t) )
- i.e., divergence between what pursuer predicts I'll do vs what I actually do

**Commitment Cost:**
- C(t) = |v^pursuer_t| / ω_max^pursuer
- Physical: higher speed + lower turning rate = harder to reverse

**Deception Value:**
- V(t) = B_gap(t) × C(t)
- Prediction: escape success correlates with V at the moment of exploitation

### Theoretical Justification: Why 2nd-Order is the Sweet Spot

**Proposition 1 (from I-POMDP theory, Gmytrasiewicz & Doshi 2005):**
In a two-player POSG with discretized state space |S| and action space |A|,
the model space at ToM depth d grows as:

    |Θ_d| = |A| ↑↑ d    (tetration / iterated exponentiation)

Specifically:
- |Θ₀| = |A|^|S|
- |Θ₁| = |A|^(|S| × |Θ₀|)
- |Θ₂| = |A|^(|S| × |Θ₁|)
- ...tower of exponentials

**In our setting** (conservative discretization: |S| ≈ 10⁷, |A| = 5):
- Level-0: 5^(10⁷) possible policies — astronomical but approximable by neural nets
- Level-1: manageable with function approximation
- Level-2: expensive but tractable with VLM approximation (our method)
- Level-3+: model space is a 4-layer exponential tower; **intractable even with approximation**

**Error propagation argument (practical intractability):**
Each level of belief estimation introduces error ε. At depth d:
- Cumulative estimation accuracy ≤ (1-ε)^d
- If Level-2 belief accuracy = 70%, then Level-3 accuracy ≈ 70% × 70% = 49% (near chance)
- Additional depth provides no useful signal — estimation noise dominates

**Cognitive science support:**
- Camerer et al. (2004): humans average ~1.5 levels of strategic reasoning
- Wright & Leyton-Brown (2014): Level-2 effectively exploits lower levels; Level-3+ marginal gain negligible
- Oguntola (2025): empirically confirmed diminishing returns beyond 2nd-order in MARL

**Conclusion:** 2nd-order ToM is the computational sweet spot — highest tractable depth that provides genuine deception capability. This is a principled justification, not just an empirical choice.

---

## Gap Analysis (Literature Landscape)

### What exists:
| Area | Key Work | What They Did | What's Missing |
|------|----------|---------------|----------------|
| ToM-driven deception | Oguntola (CMU 2023/2025) | ToM prediction error as intrinsic reward in MARL; tested on Kuhn Poker, Mafia, Stratego, ParticleWorld | Discrete/simple environments; no visual input; no physical commitment |
| Deception formalization | Alon et al. (Open Mind 2023) | Deception = reducing mutual information between true preferences and actions | Abstract buyer-seller game; not embodied |
| Vision pursuit-evasion | Bajcsy et al. (ICRA 2024) | Quadruped robot with RGB-D, pursuit-evasion in forests | No deception, purely reactive pursuit |
| Deceptive path planning | Adversarial RRT* (RA-L 2022), Bayesian Game DPP (2025), VoI-based DPP (2025) | Plan paths that fool goal recognizers | Gridworld/sim only; no physical commitment; no visual belief estimation |
| Robot deception | Wagner & Arkin (2011), Dragan et al. (RSS 2014) | Physical robot deception (false trails, ambiguous motion) | Very simple environments; not pursuit-evasion; old |
| LLM/VLM ToM | ReCon (ACL 2024), K-Level (NAACL 2025), SymbolicToM (ACL 2023) | 2nd-order ToM prompting/tracking for text games | All text-based; no embodied deployment |
| Embodied robot ToM | LatentToM (CoRL 2025 Oral), HUMAC (ICRA 2025) | Latent ToM for multi-robot coordination | Cooperative, not adversarial |

### Our unique position:
**Belief-space planning + physical commitment exploitation + embodied visual setting**
— this combination does not exist in any published work.

### Key related work to differentiate from:
1. **Oguntola** — we share the ToM-for-deception motivation, but we formalize it as a planning problem with physical commitment (not just intrinsic reward in RL)
2. **Bayesian Game DPP (2025)** — similar formulation (Bayesian belief manipulation), but they're in abstract graphs; we ground it in embodied visual observation
3. **Bajcsy et al.** — same physical setting (visual pursuit-evasion), but they do reactive pursuit with no deception; we do deceptive evasion with belief modeling

---

## Architecture Design

```
┌──────────────────────────────────────────────────────┐
│              Strategy Layer (1-2 Hz)                  │
│  ┌───────────┐   ┌──────────────┐   ┌──────────────┐│
│  │ VLM       │──▶│ Belief       │──▶│ Tactic       ││
│  │ (fine-    │   │ Estimator    │   │ Selector     ││
│  │  tuned)   │   │              │   │              ││
│  └───────────┘   └──────────────┘   └──────────────┘│
│   "opponent      "opponent thinks    "continue      │
│    decelerating,  I'll go left,      shaping" or    │
│    turning right" p=0.7"             "exploit now"  │
└───────────────────────┬──────────────────────────────┘
                        │ high-level intent
                        ▼
┌──────────────────────────────────────────────────────┐
│              Execution Layer (50-100 Hz)              │
│  RL locomotion policy / MPC controller               │
│  Input: intent + local observation                   │
│  Output: velocity commands                           │
└──────────────────────────────────────────────────────┘
```

### Key design decisions:
- **VLM is belief estimator, NOT policy** — it estimates what the opponent thinks, not what the robot should do
- **Two-frequency architecture** — strategic reasoning at 1-2Hz, motor execution at 50-100Hz
- **VLM latency is a feature** — matches human "stop and think" decision rhythm
- **Language reasoning as intermediate representation** — VLM outputs chain-of-thought about opponent belief state; this is both interpretable and potentially functionally useful (ablation: with/without CoT)

---

## Data Generation Pipeline (Rule-Based, No RL Required)

### Phase 1: Rule-based agent pool
- **Pursuers**: Level-0 (direct chase), Level-1 (predictive intercept), Level-1.5 (delayed prediction), Level-2 (wait-and-react)
- **Evaders**: Scripted tactics — classic feint, pause-observe, occlusion exploit, no-feint baseline
- All rule-based, ~hundreds of lines of Python each
- Combinatorial: 4 pursuers × 4 evaders × N environments × M random seeds

### Phase 2: Simulation rendering + auto-labeling
- Render first-person view from evader's head camera
- Auto-label each frame with:
  - Opponent state GT (position, velocity, heading — from simulator)
  - Belief GT (what pursuer predicts about evader — closed-form from pursuer's rule logic)
  - Reasoning chain (template-generated + GPT-4o paraphrase)

### Phase 3: VLM LoRA fine-tuning
- Base model: Qwen2-VL-7B or LLaVA-1.6-7B
- Multi-task: opponent state parsing + belief estimation + reasoning chain
- Hardware: RTX 3080 16GB, LoRA fine-tuning, ~5k-20k samples, ~hours

### Phase 4: Tactic selector training
- Lightweight RL or supervised classification
- Frozen VLM provides belief features as input
- 3 discrete actions: continue shaping / exploit now / observe
- Reward: escape success + belief manipulation bonus + time penalty

---

## Task Design: "Split Decision" Arena

### Scenario
```
    [GOAL-A]                         [GOAL-B]
   (-8, 20)                          (8, 20)
       \                               /
        \                             /
         \___________________________/
                DEFENDER (0, 14)
                    ▲ facing evader
                    |
                    |  ← 8m approach zone
                    |     (defender observes evader)
                    |
                EVADER (0, 2)
                    ▲ facing forward
```

- Evader starts bottom, must reach GOAL-A or GOAL-B
- Defender starts middle, must reach evader's chosen goal first to block
- Defender does NOT know evader's target — infers from visual observation only
- Both cars: Ackermann steering, inertia, turning radius constraints

### ToM Level Mapping (this specific task)
| Agent | Behavior | ToM Level |
|-------|----------|-----------|
| Evader drives straight to goal | No strategy | 0th-order |
| Defender sees evader drift left → goes to Goal-A | Infers evader intent from motion | **1st-order** |
| Evader drifts left to bait defender, then swerves right to Goal-B | Models defender's interpretation of own motion | **2nd-order** |
| Defender knows evader may feint → waits before committing | Models evader's model of defender | **3rd-order** (intractable) |

### Win Condition
- Evader enters goal radius (2m) → check if defender within block radius (3m) of same goal
- Defender present → **defender wins** (intercept)
- Defender absent → **evader wins** (escape)
- Timeout (200 steps @ 10Hz = 20s) → draw

### Physical Parameters (V1)
```
Arena:          24m × 24m flat ground
Goals:          (-8, 20) and (8, 20), radius 2m
Block radius:   3m (defender must be this close to block)

Evader:         start (0, 2), max_speed 2.0 m/s, max_steer 35°
Defender:       start (0, 14), max_speed 2.5 m/s, max_steer 25°

Key asymmetry:  Defender is FASTER but LESS AGILE
                → once committed to a direction, physically cannot reverse in time
                → this makes feints meaningful

Physics:        dt=0.1s (10Hz), wheelbase=0.3m, Ackermann steering
Episode:        max 200 steps (20 seconds)
```

### Observation Space (per agent)
```python
observation = {
    "image": np.array(84, 84, 3),     # front RGB camera, 90° FOV
    "proprioception": {
        "speed": float,                 # own speed (m/s)
        "steering_angle": float,        # current steering
        "heading": float,               # relative to arena north
    }
}
# For state-based ablation: also provide (x,y,v,θ) of both agents
```

### Action Space
```python
action = {
    "throttle": float,   # [-1, 1] — brake to full accel
    "steering": float,   # [-1, 1] — full left to full right
}
```

### Reward (zero-sum)
```python
# Evader
+10  if reached goal and defender NOT blocking
-10  if reached goal and defender IS blocking
-0.05 per step (time pressure — encourages fast decisions)

# Defender = -reward_evader
```

### RL Training Curriculum (generating ToM-leveled agents)
```
Stage 1: Train L1-Defender
  Opponent:  L0-Evader (rule-based: pick random goal, drive straight)
  Learns:    predict evader's goal from motion direction → sprint to intercept
  Expected:  defender watches evader drift, commits to matching goal

Stage 2: Train L2-Evader ← CORE EXPERIMENT
  Opponent:  Frozen L1-Defender from Stage 1
  Learns:    straight approach gets intercepted → must learn feint
  Expected:  drift toward fake goal → defender commits → swerve to real goal

Stage 3 (optional): Train L2-Defender
  Opponent:  Frozen L2-Evader from Stage 2
  Learns:    don't trust early motion signals, wait for true commitment
  Expected:  stays centered longer, only commits at last moment
```

### Environment Scaling Path
| Version | Complexity | When |
|---------|-----------|------|
| V1 | Open arena, 2 goals | Week 1-2 (MVP) |
| V2 | Add 2-3 obstacles in approach zone (occlusion, visual complexity) | Week 3-4 |
| V3 | 3-4 goals, corridors, multi-room | ICLR version |

---

## Experiments & Ablations

### Experiment 1: ToM Level Performance Matrix (CORE)
Cross-play tournament between all trained agent levels.

**Expected result:**
```
                  Defender-L0  Defender-L1  Defender-L2
  Attacker-L0        50%          30%          20%
  Attacker-L1        70%          50%          35%
  Attacker-L2        90%          75%          50%
```
If this matrix shows clear ToM hierarchy → validates the environment and training.

### Experiment 2: Vision Necessity
| Condition | Observation | Tests |
|-----------|-------------|-------|
| Full state | (x,y,v,θ) of both agents | Upper bound — can L2 emerge with perfect info? |
| Vision + proprio | Camera RGB + own speed/heading | Our target setting |
| Vision only | Camera RGB, no proprioception | Pure visual reasoning |

Claim: Vision-based L2 agent exhibits feint behavior through visual-temporal patterns.

### Experiment 3: Commitment Cost Sweep (validates formulation)
Sweep defender's max_steer from 15° to 45°:
- 15° = very committed (can barely turn) → feints very effective
- 45° = very agile (can reverse easily) → feints less effective
Prediction: L2 evader advantage over L1 evader scales with commitment cost.

### Experiment 4 (stretch): Pretrained Model Diagnosis
Zero-shot test OpenVLA/RT-2 in the arena. Expected: complete failure, no strategic behavior.

### Key Metrics
- **Evader win rate** (primary)
- **Feint frequency**: how often does L2 evader change direction mid-approach?
- **Defender commit timing**: at what distance does defender start moving to a goal?
- **Direction switch success rate**: when evader switches, does it lead to escape?
- **Commitment cost correlation**: does evader advantage scale with defender's turning constraint?

---

## MVP Strategy (10-Week CoRL Sprint)

**Philosophy:** Treat 10 weeks as minimum viable validation. If results are strong → submit CoRL. If not → polish for ICLR 2026 (~Oct deadline, ~28 weeks total).

### What MVP must validate:
1. **Environment works + 2nd-order ToM behavior emerges via RL.** Self-play or curriculum produces agents that do feints. If this fails → environment design is wrong.
2. **Quantifiable performance gap between ToM levels.** 2nd-order attacker beats 1st-order defender; 2nd-order defender resists feints. Clear performance matrix.
3. **Vision is necessary.** State-based policy ≠ vision-based policy in meaningful ways. If state-based trivially solves it → "Visual ToM" motivation collapses.
4. **(Stretch) Existing VLA/models fail.** Zero-shot evaluation of pretrained models shows they lack 2nd-order Visual ToM.

### What MVP does NOT need:
- Full 3D Habitat environment (upgrade to 3D for ICLR if needed)
- Real robot demo
- Transferability across environments (ICLR version)
- Comprehensive architectural search

### Environment approach for MVP:
**Decided: Isaac Sim 5.x directly.** No 2D prototype — would waste a week and need redo.
- Task: "Split Decision" arena (2 goals, evader vs defender, Ackermann cars)
- V1 open arena for MVP → V2 with obstacles in week 3-4 → V3 multi-goal for ICLR
- Training: RL curriculum (L0→L1-defender→L2-evader→L2-defender)

---

## Open Questions (To Resolve)
1. ~~**Simulation environment**~~ → **Decided: Isaac Sim 5.x with car agents**
2. **Environment geometry**: How many obstacle layouts? Procedurally generated?
3. **Real robot**: In scope or out of scope for this submission?
4. **Evaluation protocol**: How many episodes per condition? Statistical significance?
5. **Paper structure**: How much space for formalization vs experiments?
6. ~~**Theoretical contribution**~~ → **Resolved: cite I-POMDP tetration growth (Gmytrasiewicz & Doshi 2005) + error propagation argument for practical intractability at depth ≥ 3**

---

## Decision Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-22 | Target CoRL 2026 | Gap fits robotics venue; 10-week timeline feasible for focused paper |
| 2026-03-22 | Contribution = problem formulation, not implementation | "Belief-space adversarial planning" is the novelty, VLM is the tool |
| 2026-03-22 | Rule-based data generation, no RL dependency | Faster, more controllable, belief GT is closed-form |
| 2026-03-22 | VLM fine-tuning required | User requirement; also strengthens method contribution |
| 2026-03-22 | Sim-only for submission, real robot as stretch goal | De-risk timeline; real demo if time permits |
| 2026-03-22 | Isaac Sim 5.x as simulation platform | 3D physics (Ackermann steering = natural commitment), professional visuals, user already has setup |
| 2026-03-22 | Compute: 4080 laptop + 5090 workstation + GMU HPC | Compute not a bottleneck; parallel envs feasible |
| 2026-03-23 | 2nd-order ToM as principled choice, not arbitrary | I-POMDP tetration complexity + error propagation makes 3rd-order+ intractable; cite Gmytrasiewicz & Doshi 2005 |
| 2026-03-23 | Reframed project as "Visual ToM" | Core distinction: ToM in visual-temporal space, not language/symbolic space. Three unique features: motion-as-signal, deception-as-motion-generation, coupled reasoning+execution |
| 2026-03-23 | Paper story = 3 parts | (1) Define Visual ToM + environment (2) Diagnose existing models fail (3) Propose architecture + validate |
| 2026-03-23 | MVP strategy: rush CoRL, fallback ICLR | 10 weeks = validate core hypothesis; if quality insufficient, polish for ICLR Oct 2026 |
| 2026-03-23 | Corrected ToM order definitions | Feint IS 2nd-order (not 1st). Defender predicting direction IS 1st-order (not 0th). |
| 2026-03-23 | Task: "Split Decision" arena | Evader vs Defender, 2 goals, Ackermann cars. Defender faster but less agile = natural commitment. |
| 2026-03-23 | Start directly in Isaac Sim, no 2D prototype | Already have Isaac Sim 5.x setup; 2D would waste time and need redo |
| 2026-03-23 | RL curriculum for ToM levels | L0 rule-based → train L1 defender → freeze → train L2 evader → freeze → train L2 defender |
| 2026-05-19 | **Speed asymmetry inverted: attacker FASTER than defender** | Original (defender faster) admits trivial reactive defense — defender just tracks attacker. With attacker faster, defender *must* predict + commit early, creating the prediction surface that feints exploit. New nominal params: attacker (3.0 m/s, 2.5 rad/s), defender (2.0 m/s, 1.2 rad/s). Commitment cost now comes from defender being unable to reverse course in time after a wrong commit. |
| 2026-05-19 | MVP task = reach A *or* B (no assigned target) | Defer the "assigned target hidden from defender" variant. Reward depends only on reaching a goal unblocked. The "belief being deceived" is defender's prediction of attacker's eventual goal (readable directly from L1 rule), not a hidden true-target variable. Simpler env, same feint semantics. |
| 2026-05-19 | **RL training in 2D sim, not Isaac Sim** | Isaac Sim too slow for Stage C (feint emergence) hyperparameter iteration. 2D numpy/jax sim ~3 orders of magnitude faster. Isaac Sim retained for final vision experiment + paper video, not for training. New module: `sim2d/`. |
| 2026-05-19 | Action contract: `(v_cmd, ω_cmd)` with Ackermann coupling | Direct match to Isaac Sim AckermannController input. ω clipped by `|v|·tan(δ_max)/L` → cannot pivot in place → physical commitment cost is *built into the action space*, not just emergent from dynamics. Same contract works for diff-drive (JetBot) and Ackermann (Carter) in Isaac Sim — only the wheel-mapping math differs. |
| 2026-05-19 | Implementation order: human-playable sim → scripted baselines → PPO Stage B → PPO Stage C ⭐ | Steps 1–3 validate env parameters *before* any RL: if straight-attacker beats L1-defender, or scripted-feint can't beat L1, then sim params are wrong and no RL will help. Spend <1 day on these sanity checks. |
| 2026-05-19 | Timeline put on hold (`ToM_Timeline.md` may be outdated) | Focus is now on validating the core scientific claim (feint emergence). Timeline will be re-derived once Stage C result is known. |
| 2026-05-19 | **Stage C = sanity check, Stage F = paper main result** | Stage C (PPO attacker vs frozen rule L1) answers "is feint reachable?" (best response). Stage F (co-training) answers "is feint part of the equilibrium?" (mutual best response). Two independent publishable findings. |
| 2026-05-19 | Stage F method = fictitious self-play with warm start | Alternating K=200k-step PPO updates, both sides initialized from end-of-D. Each inner round has a stationary opponent → PPO's convergence assumptions hold. Naive simultaneous PPO rejected (non-stationary). Population/PSRO rejected (overkill). |
| 2026-05-19 | **Symmetric ω_max (= 1.2)** for both agents; only speed differs | Asymmetric agility opened a "high-frequency zigzag" exploit that bypassed feint. With symmetric ω, feint still works (depends on position commitment, not turning speed) and RL cannot kinematically dominate. |
| 2026-05-19 | **Add body radius (r_body=0.25 m) and collision = defender win** | Environment-shaping (not reward-shaping) to penalize "run-through-defender" and zigzag exploits. Single-switch feint geometrically avoids collision → it becomes the cleanest path to win. |
| 2026-05-19 | Episode length = 12 s (240 steps @ 20 Hz) | Direct attack 6.6s + feint ~6.4s; 12s gives buffer without rewarding wandering. |
| 2026-05-19 | All RL runs logged in `rl_experiments_log.md` | Per-run (hypothesis, params, result, analysis) for later cross-run analysis when something works or fails. |
| 2026-05-20 | **RL phase 1 (easy env) result: it3 100% at narrow spawn but seed-luck (it5 = 7.3%)** | Same config, different seed → 13× win rate variation. Env was attacker-trivial; no real Nash. User flagged this and triggered pivot to harder env. |
| 2026-05-20 | **Harder env design: assigned target (hidden from defender) + stronger defender (v_max 2.0→2.5)** | Attacker gets per-episode target ∈ {A,B}; entering wrong goal = loss. Defender doesn't see target — must infer. Creates real information asymmetry, no longer attacker-trivial. |
| 2026-05-20 | **Phase 2 breakthrough: `obs_target_mirror` (it14) → 56.7% balanced win, 92.5% feint** | Target=B episodes mirrored in obs+action so policy sees both directions as the same canonical scenario. Solves mode collapse structurally. Per-target win: A=51%, B=53% (balanced). |
| 2026-05-20 | **Mixed BC (50% straight + 50% feint with t* ∈ [1.8, 3.0]) required for harder env** | Pure straight-BC + sparse reward fails to discover feint in harder env within 1M steps. Honest disclosure for paper: feint demonstrations in BC are necessary for hard env, not for easy env. |
| 2026-05-20 | **Project reframed: RL is the expert, VLA is the paper's main result** | Original framing made RL feint emergence the headline. Refined framing: RL provides expert demonstrations + behavioral evidence of 2-ToM-consistent behavior. The ToM claim itself is made by the VLA's architectural commitment to belief representations. |
| 2026-05-20 | **ToM-2 architectural argument** | RL black-box policy cannot cleanly separate "belief about opponent" from "predicted opponent action" (in L1 these coincide). VLA architecture forces belief output as input to subsequent action layer → real belief state in philosophical sense (intentional, truth-evaluable, action-relevant). This is the principled basis for the 2-ToM paper claim. |
| 2026-05-20 | **Defer latent VLA; first paper uses language-VLM ToM** | Language ToM via L1→L2→tactic prompt chain (already prototyped in `IsaacToM/tom/runner.py`). Belief output is categorical {A, B}; confidence via token logits, not verbalized. LoRA fine-tune Qwen2-VL-7B. Latent VLA is the next paper. |
| 2026-05-20 | **Phase 3 plan: train defender RL via L2-rule + BC + fictitious self-play co-training** | Defender RL is harder than attacker (patience policy, smoothed prediction, action ≠ belief). Bootstrap via hand-coded L2-rule defender, BC seed, then iterate against frozen attacker. Target: Nash-ish equilibrium ~40% attacker win. |
