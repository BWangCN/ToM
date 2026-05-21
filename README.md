# Visual Theory of Mind: Deceptive Motion Planning via Visual-Temporal Belief Reasoning

A PhD research project at GMU studying **second-order Theory of Mind (ToM-2) in
embodied adversarial games**, specifically the emergence and reasoning of *feint*
behavior in physical pursuit-evasion. We propose that in physical adversarial
settings the central deceptive primitive is *misleading motion generation* rather
than language-mediated lying — and we use a Split-Decision arena (two goals,
attacker + defender with Ackermann-steered cars) to make this measurable.

> Status (2026-05-21): Phase 1 + Phase 2 RL training **complete**. Best policy
> achieves **56.7 % balanced win rate** against an L1 predict-and-commit
> defender, with **92.5 % of wins detected as feints** (lift 4.1× over
> non-feint episodes). Phase 3 (defender RL + fictitious self-play) is in
> progress; def_v1 currently exploits the frozen attacker's deterministic
> behavior and needs population-of-opponents training to be a healthy
> generalist. Phase 4 will train a VLM for ToM-2 belief reasoning over the same
> game (planned).

---

## Repo layout

```
ToM/
├── progress.md                   Paper-level state, problem formulation,
│                                 decision log spanning the whole project.
├── feint_experiment_design.md    Implementation contract for the RL phases:
│                                 env spec, MDP, action contract, defender
│                                 ladder, eval protocol.
├── rl_experiments_log.md         Per-iteration RL trail (Phase 1 + Phase 2)
│                                 with hypothesis, params, result, analysis,
│                                 lessons learned.
├── TONIGHT_SUMMARY.md            One-page snapshot at the end of Phase 2.
├── next_phases.md                Forward plan: ToM framing argument, defender
│                                 RL plan, VLM fine-tune plan, decision points.
│
├── sim2d/                        Phase 1+2+3 RL environment and training code.
│   ├── env.py                    SplitDecisionEnv + DefenderRLEnv.
│   ├── dynamics.py               Bicycle model with Ackermann coupling.
│   ├── rules.py                  L0/L1/L2 rule defenders + scripted attackers.
│   ├── feint_detector.py         Trajectory-based feint classifier.
│   ├── train.py                  PPO attacker training (Phase 1 + 2).
│   ├── train_defender.py         PPO defender training (Phase 3).
│   ├── sanity_check.py           Pre-RL gating sanity (scripted vs L0/L1).
│   ├── sanity_def_v1.py          Phase-3 defender sanity (vs scripted pool).
│   ├── analyze.py                One-shot eval + conditional analysis.
│   ├── evaluate.py               Per-model evaluation script.
│   ├── visualize.py              Trajectory + outcome distribution figures.
│   └── make_video.py             Animated MP4 of single or batch episodes.
│
├── IsaacToM/                     Original Isaac Sim PoC (deferred).
│   ├── isaac_env/                Isaac Sim scene + headless runner.
│   ├── tom/                      L1 → L2 → tactic prompt chain for VLM.
│   ├── control/                  Rule-based / RL-policy / VLM-direct controllers.
│   ├── ros2/                     ROS 2 + Nav2 bridge for waypoint planning.
│   └── viz/                      Demo video generation.
│
├── logs/runs/                    Curated subset only (most runs gitignored).
│   ├── it14_mirror/best/         The Phase-2 best policy (paper number 56.7 %).
│   ├── it14_mirror/viz/          Demo videos + figures.
│   └── defender/def_v1/best/     Phase-3 attempt #1 (exploit, needs retraining).
│
└── ToM_Robotics_Literature_Landscape_2026.md
                                  ~50-paper literature review covering ToM
                                  in robotics, deceptive planning, VLM-ToM,
                                  pursuit-evasion, etc.
```

---

## Quick start — reproduce the Phase-2 attacker result (it14_mirror)

```bash
cd ToM
pip install -r IsaacToM/requirements.txt
pip install stable-baselines3 gymnasium tensorboard matplotlib

# Sanity check the env is well-posed (scripted attackers vs scripted defenders)
python -m sim2d.sanity_check --test all --episodes 200

# Train the Phase-2 attacker (about 8-10 minutes on a laptop CPU+RAM, no GPU
# needed for this 2D sim).
python -m sim2d.train \
    --run_id it14_mirror_reproduce \
    --defender L1 \
    --total_steps 500000 \
    --n_envs 8 \
    --seed 0 \
    --bc_pretrain --bc_episodes 200 --bc_epochs 15 \
    --bc_expert mixed \
    --forward_progress_reward 0.01 --step_penalty -0.002 \
    --ent_coef 0.005 --learning_rate 3e-4 \
    --env_overrides '{"target_mode": "random",
                      "spawn_xy_noise": 3.0,
                      "defender_params": {"v_max": 2.5, "omega_max": 1.2,
                                          "delta_max": 0.4363, "L": 0.5},
                      "obs_target_relative": true,
                      "obs_target_mirror": true}'

# Evaluate
python -m sim2d.analyze --run_id it14_mirror_reproduce --episodes 300

# Render demo videos + figures
python -m sim2d.visualize    --model logs/runs/it14_mirror_reproduce/best/best_model.zip
python -m sim2d.make_video   --model logs/runs/it14_mirror_reproduce/best/best_model.zip
```

---

## Highlight results (so far)

**Phase 2 (hard env: attacker has hidden assigned target + stronger defender):**

| Metric | Value |
|--------|-------|
| Attacker win rate (single policy, target=random) | **56.7 %** |
| Per-target win rate (target=A / target=B) | 51 % / 53 % (balanced) |
| Feint rate | 74.8 % |
| P(win \| feint) | 65.8 % |
| P(win \| no feint) | 15.9 % |
| Lift (feint vs non-feint) | **4.1×** |
| P(feint \| win) | **92.5 %** |
| Mean feint switch time t* | 1.99 s |

Most wins come from feints; non-feint episodes overwhelmingly lose. This is
**behavioral evidence** that the policy reaches the win condition by performing
ToM-2-consistent feint maneuvers (induce a wrong commitment in the defender's
belief, exploit the commitment-cost gap).

See `logs/runs/it14_mirror/viz/` for trajectory plots, outcome distribution,
and demonstration videos.

---

## Conceptual contribution (paper-level)

The project's central claim is that **VLA architecture is required to make
the ToM-2 claim rigorous** — RL alone can only provide behavioral evidence.
See `next_phases.md` for the full ToM framing argument.

In short:
- RL policy: black-box, no architectural commitment to belief representation.
  Under L1 defender, `defender.belief == defender.action`, so an auxiliary
  belief head can equivalently be interpreted as action prediction (1-ToM) or
  belief prediction (2-ToM). Behavioral evidence is suggestive but not
  conclusive.
- VLA pipeline: VLM produces an *explicit* belief state ("defender believes
  attacker is going to X") which is then *consumed* by a downstream tactic
  selector. This is action-decoupled by construction. The VLM additionally
  carries linguistic priors for belief concepts from pretraining. Under this
  architecture the 2-ToM claim is *architectural* and *functional*.

Phase 4 trains a fine-tuned Qwen2-VL-7B on this pipeline using RL-policy
rollouts as the expert data source. Phase 5 (deferred) replaces the
language belief output with a latent representation per the "motion is the
information" framing.

---

## Project phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ done | Easy env (no assigned target). Result: feint emerges but env was attacker-trivial (Nash near 100 % attacker win, seed-dependent). |
| 2 | ✅ done | Hard env (assigned target hidden from defender + stronger defender + mirror obs). Result: 56.7 % balanced win, 92.5 % of wins are feints. |
| 3 | 🚧 in-progress | Defender RL via L2-rule BC + PPO + fictitious self-play. def_v1 currently it14-OOD exploit; needs opponent-population training. |
| 4 | planned | Language-VLM ToM fine-tune (Qwen2-VL-7B with LoRA). L1 → L2 → tactic prompt chain. |
| 5 | deferred | Latent VLA (motion-as-information, separate follow-up paper). |

---

## License & citation

This is unpublished research. Please cite via the project if you use any of
the code or framing. Author: Beichen Wang, George Mason University, 2026.
