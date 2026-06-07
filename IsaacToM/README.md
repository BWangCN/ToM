# IsaacToM — Visual Theory-of-Mind PoC on Isaac Sim

Port of AR-LLMs' LLM-driven decision loop to Isaac Sim 5.1, with the scenario
replaced by the **Split Decision** pursuit-evasion task (evader vs defender,
two goals, Ackermann-ish wheeled cars). The VLM now performs explicit ToM
reasoning (L1 opponent intent + L2 opponent's belief about me) instead of
hide-and-seek path weighting.

## Layout

```
IsaacToM/
├── config/split_decision.yaml     # all knobs
├── isaac_env/                     # Isaac Sim scene + runner
│   ├── runner.py                  # entry point (launches SimulationApp)
│   ├── scene_builder.py           # arena + goals + two wheeled robots
│   └── topdown_camera.py          # orthographic top-down render
├── tom/                           # strategy layer (1–2 Hz)
│   ├── runner.py                  # L1 → L2 → tactic, per perspective
│   ├── vlm_client.py              # OpenAI + Qwen2-VL (stub) backends
│   └── prompts/                   # ToM-oriented prompts
├── control/                       # execution layer (physics rate)
│   ├── base.py                    # shared interface + diff-drive utility
│   ├── rule_based.py              # PoC default
│   └── rl_policy.py               # placeholder for RL curriculum
├── viz/replay.py                  # quick trajectory plot from a session log
└── logs/                          # frames + JSONL reasoning traces
```

## Run

```powershell
conda activate env_isaacsim
$env:OPENAI_API_KEY = "sk-..."
cd C:\Users\Wayne\Desktop\GMU\PhD\ToM\IsaacToM
python -m isaac_env.runner -c config/split_decision.yaml --max-seconds 30
```

Dry run with no VLM (controller still operates on default tactics):
```powershell
python -m isaac_env.runner -c config/split_decision.yaml --no-vlm
```

## RL extension path

`isaac_env/runner.py` is structured so the physics loop only touches the
`BaseController.step(tactic, self_state, other_state, goals)` contract.
When the RL curriculum begins (progress.md Stage 1–3), swap
`control/rule_based.py` for `control/rl_policy.py` and optionally wrap the
whole thing in a Gymnasium env (`ManagerBasedRLEnv`) by lifting `runner.py`'s
loop into `_step / _compute_observations / _get_rewards`. The ToM runner stays
as an external strategy layer at 1–2 Hz and can provide `tactic.label` as an
additional observation or goal-conditioning signal.

## Prompt design (ToM-oriented)

- **L1 (`tom/prompts/l1_intent.py`)** — classic 1st-order ToM. Asks the VLM:
  *given the top-down view and state, what goal is the opponent heading for?*
- **L2 (`tom/prompts/l2_belief.py`)** — 2nd-order. Asks: *what does the
  opponent currently believe about MY goal, and how committed does my motion
  look?* Includes a `commitment_evidence` scalar that maps directly to the
  "belief gap × commitment cost" formulation in `progress.md`.
- **Tactic (`tom/prompts/tactic.py`)** — role-specific closed vocabulary
  (evader: shape_* / exploit_* / slow_observe; defender: wait_center /
  track_evader / commit_* / hedge). `rule_based.py` implements every label.

Each call is logged as a single JSONL row containing state, images path, raw
and parsed VLM outputs, and the chosen tactic — suitable for later filtering
into a LoRA fine-tuning dataset.

## Known TODOs

- `topdown_camera.py` uses `UsdGeom.Camera.GetProjectionAttr()` to set ortho
  projection; verify aperture math matches desired half-extent on a first run.
- Ackermann steering via differential-drive JetBot is a PoC approximation.
  Swap in Carter + `AckermannController` (`isaacsim.robot.wheeled_robots.controllers.ackermann_controller`)
  when moving to the commitment-cost sweep experiments.
- `vlm_client._QwenBackend` is a stub — implement against local HF weights
  once we start fine-tuning.
- Add Isaac Lab `ManagerBasedRLEnv` wrapper when we start Stage 1 RL.
