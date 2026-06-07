"""Sanity check def_v1: is it a healthy generalist defender or an it14-OOD exploit?

    python -m sim2d.sanity_def_v1

Tests def_v1 against:
  - it14 RL attacker (the one it was trained against)
  - scripted Straight attacker (any defender should beat this)
  - scripted Feint at multiple t* (the strategy-space probe)

Also renders a 6-episode trajectory grid showing what def_v1 vs it14 looks like.
"""
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.feint_detector import is_feint  # noqa: E402
from sim2d.rules import FeintAttacker, StraightAttacker  # noqa: E402
from sim2d.visualize import plot_trajectory  # noqa: E402


class DefenderRLAsRule:
    """Plug an RL defender model into SplitDecisionEnv's rule-based defender slot."""

    def __init__(self, model):
        self.model = model
        self.last_target = None

    def reset(self):
        self.last_target = None

    def step(self, self_state, attacker_state, goal_A, goal_B):
        ax, ay, atheta, av, aomega = attacker_state
        dx, dy, dtheta, dv, domega = self_state
        obs = np.array([
            dx, dy, np.sin(dtheta), np.cos(dtheta), dv, domega,
            ax, ay, np.sin(atheta), np.cos(atheta), av, aomega,
            goal_A[0] - dx, goal_A[1] - dy,
            goal_B[0] - dx, goal_B[1] - dy,
        ], dtype=np.float32)
        a, _ = self.model.predict(obs, deterministic=True)
        return (float(a[0]), float(a[1]))


def run_eval(env_cfg, n, seed, attacker, def_model):
    env = SplitDecisionEnv(cfg=env_cfg, seed=seed)
    env.set_defender(DefenderRLAsRule(def_model))
    if isinstance(attacker, (StraightAttacker, FeintAttacker)):
        attacker.params = env.attacker_params
    episodes = []
    for ep in range(n):
        obs, _ = env.reset(seed=seed + ep)
        env.defender_policy.reset()
        if hasattr(attacker, 'reset'):
            try:
                attacker.reset(assigned_target=env.assigned_target)
            except TypeError:
                attacker.reset()
        done = False
        info = {}
        while not done:
            if hasattr(attacker, 'predict'):
                a, _ = attacker.predict(obs, deterministic=True)
            else:
                a = attacker.step(env.attacker_state, env.defender_state,
                                   env.goal_A, env.goal_B)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        feint, ev = is_feint(env.history['attacker'], env.goal_A, env.goal_B)
        episodes.append({
            'attacker_traj': np.array(env.history['attacker']),
            'defender_traj': np.array(env.history['defender']),
            'defender_target_history': list(env.history['defender_target']),
            'assigned_target': env.assigned_target,
            'outcome': info.get('outcome', '?'),
            'winner': info.get('winner', '?'),
            'is_feint': feint,
            'feint_ev': ev,
            'n_steps': len(env.history['attacker']),
            'seed': seed + ep,
            'goal_A': env.goal_A.copy(),
            'goal_B': env.goal_B.copy(),
        })
    return episodes


def summarize(label, eps):
    n = len(eps)
    wins = sum(1 for e in eps if e['winner'] == 'attacker')
    feints = sum(1 for e in eps if e['is_feint'])
    outs = defaultdict(int)
    for e in eps:
        outs[e['outcome']] += 1
    print(f'  {label:42s}  att_win={wins/n:>5.1%}  def_win={(n - wins)/n:>5.1%}  '
          f'feint={feints/n:>5.1%}  outcomes={dict(outs)}')
    return wins, feints, outs


def main():
    out_dir = _ROOT / 'logs' / 'runs' / 'defender' / 'def_v1' / 'viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('Loading models...')
    attacker = PPO.load(str(_ROOT / 'logs/runs/it14_mirror/best/best_model.zip'))
    def_v1 = PPO.load(str(_ROOT / 'logs/runs/defender/def_v1/best/best_model.zip'))

    base = {
        'forward_progress_reward': 0.01, 'step_penalty': -0.002,
        'target_mode': 'random', 'spawn_xy_noise': 3.0,
        'defender_params': {'v_max': 2.5, 'omega_max': 1.2, 'delta_max': 0.4363, 'L': 0.5},
        'obs_target_relative': True,
    }
    cfg_mirror = {**base, 'obs_target_mirror': True}
    cfg_no_mirror = {**base, 'obs_target_mirror': False}

    # Dummy params; will be replaced inside run_eval
    dummy = {'v_max': 3.0, 'omega_max': 1.2, 'delta_max': 0.61, 'L': 0.5}

    print()
    print('=== def_v1 vs various attackers (n=200 each) ===')

    eps_it14 = run_eval(cfg_mirror, 200, 90000, attacker, def_v1)
    summarize('vs it14 RL', eps_it14)

    eps_str = run_eval(cfg_no_mirror, 200, 91000,
                        StraightAttacker(dummy, target_goal='B'), def_v1)
    summarize('vs scripted straight', eps_str)

    for ts in [1.0, 1.5, 2.0, 2.4, 3.0]:
        eps_f = run_eval(cfg_no_mirror, 200, 92000 + int(ts * 1000),
                          FeintAttacker(dummy, fake_goal='A', real_goal='B',
                                         switch_time=ts), def_v1)
        summarize(f'vs scripted feint t*={ts}', eps_f)

    print()
    print('=== Building trajectory grid for def_v1 vs it14 ===')
    cats = defaultdict(list)
    for e in eps_it14:
        cats[e['outcome']].append(e)
    print(f'  outcome categories: {dict((k, len(v)) for k, v in cats.items())}')

    picks = []
    for outcome in ['timeout', 'out_of_bounds', 'wrong_goal_A', 'wrong_goal_B',
                    'collision', 'won_at_A', 'won_at_B', 'blocked_at_A', 'blocked_at_B']:
        for ep in cats.get(outcome, [])[:2]:
            picks.append(ep)
            if len(picks) >= 6:
                break
        if len(picks) >= 6:
            break

    n_picks = min(len(picks), 6)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, 11))
    for ax, ep in zip(axes.flatten(), picks[:n_picks]):
        plot_trajectory(ax, ep, ep['goal_A'], ep['goal_B'])
        ax.set_xlabel(f"{ep['outcome']}  |  target={ep['assigned_target']}  "
                       f"|  seed={ep['seed']}  |  steps={ep['n_steps']}",
                       fontsize=9, fontweight='bold')
    for ax in axes.flatten()[n_picks:]:
        ax.set_axis_off()
    plt.suptitle('def_v1 (RL) vs it14 (RL attacker) — 6 representative episodes\n'
                  'red=attacker, blue=defender (def_v1), gold ring=true target, '
                  'gold dots=defender commit switches',
                  fontsize=12)
    plt.tight_layout()
    out_path = out_dir / 'sanity_def_v1_vs_it14.png'
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f'  -> {out_path}')


if __name__ == '__main__':
    main()
