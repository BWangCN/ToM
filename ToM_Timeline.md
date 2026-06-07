# 10-Week Execution Plan — CoRL 2026 (MVP Sprint)

**Deadline: May 28, 2026 (Wednesday)**
**Start: March 22, 2026 (Sunday)**
**Backup: ICLR 2026 (~Oct 2026) if CoRL quality insufficient**

---

## Week 1 (Mar 23 – Mar 29): Isaac Sim Environment Setup

- [ ] Create "Split Decision" arena in Isaac Sim 5.x
  - 24m × 24m flat ground plane
  - Two goal zones: (-8,20) and (8,20) with visual markers
  - Two car agents with Ackermann steering
- [ ] Configure car physics
  - Evader: max_speed=2.0 m/s, max_steer=35°
  - Defender: max_speed=2.5 m/s, max_steer=25°
  - Wheelbase=0.3m, proper inertia/friction
- [ ] Set up front-mounted RGB cameras (84×84, 90° FOV) on both cars
- [ ] Implement episode logic: spawn, goal assignment, win/loss detection, reset
- [ ] Implement rule-based L0 agents
  - L0-Evader: pick random goal, drive straight to it
  - L0-Defender: drive to random goal OR stay still
- [ ] Run 100 episodes with L0 vs L0, verify pipeline: images save, states log, episodes reset

**Deliverable:** Runnable Isaac Sim environment, L0 agents, camera output verified.

---

## Week 2 (Mar 30 – Apr 5): RL Training Pipeline + L1 Defender

- [ ] Set up RL training infrastructure
  - MAPPO or IPPO via Isaac Lab / rl_games / cleanrl
  - Parallel environments (target: 64-256 on 5090)
  - Both state-based and vision-based observation wrappers
- [ ] Train L1-Defender (state-based first, then vision)
  - Opponent: L0-Evader (random goal, straight drive)
  - Reward: +10 if at correct goal when evader arrives, -10 otherwise, -0.05/step
  - Expected behavior: observe evader's drift direction, sprint to matching goal
- [ ] Validate L1-Defender: does it reliably intercept L0-Evader? (target: >70% win rate)
- [ ] Implement L1-Defender as rule-based backup: linear extrapolation of evader velocity → go to predicted goal

**Deliverable:** Trained L1-Defender that intercepts straight-driving evaders. RL pipeline confirmed working.

---

## Week 3 (Apr 6 – Apr 12): Train L2-Evader (CRITICAL MILESTONE)

- [ ] Freeze L1-Defender, train L2-Evader against it
  - Reward: +10 escape, -10 intercepted, -0.05/step
  - L0-Evader (straight) should get ~30% win rate against L1-Defender
  - L2-Evader must learn to beat L1-Defender significantly (target: >65%)
- [ ] **Check: does feint behavior emerge?**
  - Visualize L2-Evader trajectories
  - Look for: initial drift toward one goal → direction switch → arrive at other goal
  - If no feint emerges: adjust reward (add exploration bonus), adjust physics (more commitment), or try curriculum variants
- [ ] Run state-based vs vision-based comparison for L2-Evader
  - Both should learn feints; vision-based may be slower to converge
- [ ] Start L2-Defender training (freeze L2-Evader, train defender against it)

**Deliverable:** L2-Evader that demonstrably feints. Trajectories showing 2-phase structure (shape → exploit). This is the existence proof — if this works, the paper has legs.

---

## Week 4 (Apr 13 – Apr 19): ToM Level Tournament + Vision Ablation

- [ ] Cross-play tournament: all combinations of L0/L1/L2 attackers vs L0/L1/L2 defenders
  - 200+ episodes per pairing
  - Generate performance matrix (evader win rate)
  - **Must show clear hierarchy: L2 > L1 > L0 against each defender level**
- [ ] Vision necessity ablation
  - State-based policy vs vision-based policy at each ToM level
  - Document differences in learning speed, final performance, failure modes
- [ ] Add V2 obstacles (2-3 walls/pillars in approach zone) for visual complexity
- [ ] Collect qualitative examples: top-5 most convincing feint episodes with trajectory plots

**Deliverable:** ToM performance matrix + vision ablation results + qualitative feint visualizations. Core paper results are now in hand.

---

## Week 5 (Apr 20 – Apr 26): Commitment Cost Sweep + Stretch Experiments

- [ ] Sweep defender max_steer: [15°, 20°, 25°, 30°, 35°, 40°, 45°]
  - For each: retrain L1-Defender + L2-Evader (or test fixed agents)
  - Plot: L2 evader advantage vs commitment cost
  - **Must show: advantage decreases as defender becomes more agile**
- [ ] (Stretch) Zero-shot VLA evaluation
  - Load OpenVLA-7B into environment
  - Text prompt: "Drive to the green target while avoiding the red car"
  - Document failure modes
- [ ] (Stretch) Test architecture variants
  - Frame stacking (4 frames) vs single frame
  - Add GRU/LSTM memory module
  - Compare which best supports temporal reasoning

**Deliverable:** Commitment cost curve validating formulation + (optional) VLA failure evidence.

---

## Week 6 (May 3 – May 9): Analysis + Figure Generation

- [ ] Compute all final metrics across all experiments
- [ ] Generate paper figures:
  - Arena diagram (schematic)
  - Trajectory plots (L0 vs L1 vs L2 evader)
  - Performance matrix heatmap
  - Commitment cost curve
  - Vision vs state comparison
  - Qualitative feint sequence (frame-by-frame with annotations)
- [ ] Statistical analysis: confidence intervals, significance tests
- [ ] Start writing Introduction + Related Work

**Deliverable:** All figures and tables finalized. Two sections drafted.

---

## Week 7 (May 4 – May 10): Writing — Problem Formulation + Method

- [ ] Problem Formulation section
  - POSG formalization
  - Visual ToM definition (3 distinguishing features)
  - Deception value = belief gap × commitment cost
  - Proposition 1: 3rd-order+ intractability
  - Why 2nd-order is sweet spot
- [ ] Method section
  - "Split Decision" environment design + justification
  - RL curriculum (L0 → L1 → L2)
  - Observation/action/reward design
- [ ] Related Work (compact: Oguntola, Bajcsy, deceptive planning, NLP ToM)

**Deliverable:** Problem Formulation + Method + Related Work drafted.

---

## Week 8 (May 11 – May 17): Writing — Experiments + Discussion

- [ ] Experiments section
  - ToM performance matrix + analysis
  - Vision necessity ablation
  - Commitment cost sweep + formulation validation
  - Qualitative analysis of emergent feints
  - (Optional) VLA diagnosis results
- [ ] Discussion: limitations, future work (3D, real robot, transferability, architecture)
- [ ] Conclusion
- [ ] Abstract

**Deliverable:** Complete paper draft.

---

## Week 9 (May 18 – May 24): Revision + Polish

- [ ] Internal review: story coherence, claim-evidence alignment
- [ ] Co-author review + feedback
- [ ] Fix any weak experimental results (rerun if needed)
- [ ] Supplementary material: video of feint episodes, full ablation tables, training curves
- [ ] Proofread, formatting, references

**Deliverable:** Camera-ready-quality draft.

---

## Week 10 (May 25 – May 28): Final Polish + Submit

- [ ] Final pass on all figures and text
- [ ] Prepare supplementary video
- [ ] **Submit by May 28 EOD**

**Deliverable:** Submitted paper.

---

## Critical Path

The entire paper hinges on **Week 3**: does L2-Evader learn to feint against L1-Defender?

- If YES → project is viable, continue full plan
- If NO (after debugging) → pivot: try different env parameters, different curriculum, or switch to rule-based L2-Evader as fallback data source
- If STILL NO → save work for ICLR, use extra time to investigate why and fix

---

## Risk Management

| Risk | Impact | Mitigation |
|------|--------|------------|
| Feint doesn't emerge in RL | PROJECT KILLER | Adjust physics (more commitment), try reward shaping, fallback to scripted feints |
| Isaac Sim setup takes >1 week | Delays everything | Have pygame 2D backup ready as emergency prototype |
| RL training too slow on 4080 | Delays week 2-3 | Use 5090 workstation or HPC for training, 4080 for dev |
| Vision-based RL doesn't converge | Weakens "Visual ToM" claim | Train state-based first (always works), then transfer to vision |
| Performance matrix shows no clear hierarchy | Weakens core claim | Check reward, check physics parameters, check training convergence |
| Not enough time for paper writing | Misses deadline | Start writing Method/RelatedWork in week 5 in parallel with experiments |

---

## Weekly Self-Assessment
Each week ends with:
1. Did I hit the deliverable?
2. Is the critical path still on track?
3. What's blocking next week?
4. Should I cut scope or switch to ICLR timeline?
