"""One-shot evaluation + conditional analysis for a trained run.

    python -m sim2d.analyze --run_id it2_ent0 --episodes 300

Prints summary + saves logs/runs/<run_id>/analysis.json.
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
from sim2d.rules import L1Defender  # noqa: E402
from sim2d.feint_detector import is_feint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', required=True)
    ap.add_argument('--episodes', type=int, default=300)
    ap.add_argument('--seed', type=int, default=400000)
    ap.add_argument('--env_overrides', type=str, default='{}')
    ap.add_argument('--defender_random_tiebreak', action='store_true')
    args = ap.parse_args()

    run_dir = _ROOT / 'logs' / 'runs' / args.run_id
    model_path = run_dir / 'best' / 'best_model.zip'
    if not model_path.exists():
        model_path = run_dir / 'final.zip'
    model = PPO.load(str(model_path))

    cfg = {'forward_progress_reward': 0.01, 'step_penalty': -0.002}
    cfg.update(json.loads(args.env_overrides))
    env = SplitDecisionEnv(cfg=cfg, seed=args.seed)
    env.set_defender(L1Defender(env.defender_params, horizon_s=1.5, switch_dwell=6,
                                  randomize_initial_target=args.defender_random_tiebreak))

    eval_npz = run_dir / 'eval' / 'evaluations.npz'
    peak_step = peak_r = final_r = None
    if eval_npz.exists():
        d = np.load(eval_npz)
        rs = d['results'].mean(axis=1)
        idx = int(rs.argmax())
        peak_step = int(d['timesteps'][idx])
        peak_r = float(rs[idx])
        final_r = float(rs[-1])

    cells = defaultdict(int)
    outcomes = defaultdict(int)
    feint_t_stars = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        env.defender_policy.reset()
        done = False
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        won = info.get('winner') == 'attacker'
        outcomes[info.get('outcome', '?')] += 1
        is_f, ev = is_feint(env.history['attacker'], env.goal_A, env.goal_B)
        if is_f and ev is not None:
            order = ev.get('order')
            switch_step = ev['A_run'][1] if order == 'A_to_B' else ev['B_run'][1]
            feint_t_stars.append(switch_step * env.dt)
        cells[('feint' if is_f else 'noFeint') + ('_win' if won else '_loss')] += 1

    n = args.episodes
    n_feint = cells['feint_win'] + cells['feint_loss']
    n_no = cells['noFeint_win'] + cells['noFeint_loss']
    win_total = cells['feint_win'] + cells['noFeint_win']
    pw_f = cells['feint_win'] / max(n_feint, 1)
    pw_n = cells['noFeint_win'] / max(n_no, 1)
    lift = pw_f / max(pw_n, 1e-3)
    mean_t = float(np.mean(feint_t_stars)) if feint_t_stars else None

    out = {
        'run_id': args.run_id,
        'peak_step': peak_step, 'peak_reward': peak_r, 'final_reward': final_r,
        'overall_win': win_total / n,
        'feint_rate': n_feint / n,
        'p_win_given_feint': pw_f,
        'p_win_given_noFeint': pw_n,
        'lift': lift,
        'p_feint_given_win': cells['feint_win'] / max(win_total, 1),
        'mean_t_star_s': mean_t,
        'outcomes': dict(outcomes),
        'cells': dict(cells),
        'n_episodes': n,
    }

    print(f"\n=== {args.run_id} ===")
    print(f"Curve   peak step={peak_step}  peak_r={peak_r:.3f}  final_r={final_r:.3f}"
          if peak_step is not None else "No eval curve")
    print(f"Win rate  = {out['overall_win']:.1%}   Feint rate = {out['feint_rate']:.1%}")
    print(f"P(win|feint)   = {pw_f:.1%}")
    print(f"P(win|noFeint) = {pw_n:.1%}")
    print(f"Lift           = {lift:.1f}x")
    if mean_t is not None:
        print(f"Mean t*        = {mean_t:.2f}s")
    print(f"Outcomes: {dict(outcomes)}")

    out_path = run_dir / 'analysis.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"-> {out_path}")


if __name__ == '__main__':
    main()
