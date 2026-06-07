"""Evaluate a trained PPO model against a defender, computing win/feint rates.

    python -m sim2d.evaluate --model logs/runs/stage_c_run1/best/best_model \
                              --defender L1 --episodes 200

Outputs:
  - prints summary
  - writes logs/runs/<run_id>/eval_<defender>.json with full metrics
  - if --save_traj: also dumps a few example trajectories for visualization
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.rules import L0Defender, L1Defender  # noqa: E402
from sim2d.feint_detector import is_feint  # noqa: E402


def make_defender(env, defender_type: str, horizon_s: float = 1.5, switch_dwell: int = 6):
    if defender_type == 'L0':
        return L0Defender(env.defender_params)
    if defender_type == 'L1':
        return L1Defender(env.defender_params, horizon_s=horizon_s, switch_dwell=switch_dwell)
    raise ValueError(defender_type)


def evaluate(model, env, defender, n_episodes: int, base_seed: int = 0, save_traj: bool = False):
    env.set_defender(defender)
    outcomes = defaultdict(int)
    winners = defaultdict(int)
    ep_lengths = []
    feint_count = 0
    switch_times = []
    saved_traj = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=base_seed + ep)
        defender.reset()

        done = False
        info = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        outcomes[info.get('outcome', 'unknown')] += 1
        winners[info.get('winner', 'unknown')] += 1
        ep_lengths.append(info.get('t_steps', env.t))

        feint, evidence = is_feint(env.history['attacker'], env.goal_A, env.goal_B)
        if feint:
            feint_count += 1
            if evidence is not None:
                A_run = evidence.get('A_run', (0, 0))
                B_run = evidence.get('B_run', (0, 0))
                if evidence.get('order') == 'A_to_B':
                    switch_step = A_run[1]
                else:
                    switch_step = B_run[1]
                switch_times.append(switch_step * env.dt)

        if save_traj and ep < 5:
            saved_traj.append({
                'episode': ep,
                'attacker': [s.tolist() for s in env.history['attacker']],
                'defender': [s.tolist() for s in env.history['defender']],
                'defender_target': list(env.history['defender_target']),
                'outcome': info.get('outcome'),
                'winner': info.get('winner'),
                'is_feint': feint,
            })

    summary = {
        'n_episodes': n_episodes,
        'attacker_win_rate': winners.get('attacker', 0) / n_episodes,
        'defender_win_rate': winners.get('defender', 0) / n_episodes,
        'feint_rate': feint_count / n_episodes,
        'mean_ep_steps': float(np.mean(ep_lengths)),
        'outcomes': dict(outcomes),
        'mean_switch_time_s': float(np.mean(switch_times)) if switch_times else None,
        'std_switch_time_s': float(np.std(switch_times)) if switch_times else None,
        'saved_traj': saved_traj if save_traj else None,
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=str, required=True,
                    help='Path to model .zip (or directory containing best_model.zip)')
    ap.add_argument('--defender', choices=['L0', 'L1'], default='L1')
    ap.add_argument('--horizon_s', type=float, default=1.5)
    ap.add_argument('--switch_dwell', type=int, default=6)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed', type=int, default=10000)
    ap.add_argument('--save_traj', action='store_true')
    ap.add_argument('--out_json', type=str, default=None)
    args = ap.parse_args()

    model_path = Path(args.model)
    if model_path.is_dir():
        model_path = model_path / 'best_model.zip'
    if not model_path.exists():
        candidate = model_path.with_suffix('.zip')
        if candidate.exists():
            model_path = candidate

    model = PPO.load(str(model_path))
    env = SplitDecisionEnv(seed=args.seed)
    defender = make_defender(env, args.defender, args.horizon_s, args.switch_dwell)

    print(f"Evaluating model={model_path}  defender={args.defender}  "
          f"(h={args.horizon_s}, dwell={args.switch_dwell})")
    summary = evaluate(model, env, defender, args.episodes, args.seed, args.save_traj)

    print(f"\nattacker_win_rate = {summary['attacker_win_rate']:.2%}")
    print(f"defender_win_rate = {summary['defender_win_rate']:.2%}")
    print(f"feint_rate        = {summary['feint_rate']:.2%}")
    print(f"mean_ep_steps     = {summary['mean_ep_steps']:.1f}")
    if summary['mean_switch_time_s'] is not None:
        print(f"feint t* (s)      = {summary['mean_switch_time_s']:.2f} ± {summary['std_switch_time_s']:.2f}")
    print(f"outcomes          = {summary['outcomes']}")

    if args.out_json is None:
        run_dir = model_path.parent
        if run_dir.name in ('best', 'final'):
            run_dir = run_dir.parent
        out_path = run_dir / f'eval_{args.defender}.json'
    else:
        out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull metrics -> {out_path}")


if __name__ == '__main__':
    main()
