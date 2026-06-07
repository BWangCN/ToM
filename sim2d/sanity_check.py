"""Scripted-vs-scripted sanity tests.

Run BEFORE any RL. If T1/T2/T4 fail, env is broken. If T3 fails or T5 sweep
has no clear peak, env params are wrong (no point training).

    python -m sim2d.sanity_check --test all --episodes 200

Output: prints summary; writes JSON to logs/sanity_results.json.
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

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.rules import (  # noqa: E402
    L0Defender, L1Defender, StraightAttacker, FeintAttacker,
)
from sim2d.feint_detector import is_feint  # noqa: E402


def _run_scenario(env: SplitDecisionEnv, attacker, defender, n_episodes: int, base_seed: int = 0):
    env.set_defender(defender)
    outcomes = defaultdict(int)
    winners = defaultdict(int)
    ep_lengths = []
    feint_count = 0

    for ep in range(n_episodes):
        env.reset(seed=base_seed + ep)
        # Pass assigned target to attacker if env has one (target_mode active).
        attacker.reset(assigned_target=env.assigned_target)
        defender.reset()

        done = False
        info = {}
        while not done:
            action = attacker.step(env.attacker_state, env.defender_state,
                                    env.goal_A, env.goal_B)
            _, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        outcomes[info.get('outcome', 'unknown')] += 1
        winners[info.get('winner', 'unknown')] += 1
        ep_lengths.append(info.get('t_steps', env.t))

        feint, _ = is_feint(env.history['attacker'], env.goal_A, env.goal_B)
        if feint:
            feint_count += 1

    return {
        'n_episodes': n_episodes,
        'outcomes': dict(outcomes),
        'attacker_win_rate': winners.get('attacker', 0) / n_episodes,
        'defender_win_rate': winners.get('defender', 0) / n_episodes,
        'feint_rate': feint_count / n_episodes,
        'mean_ep_steps': float(np.mean(ep_lengths)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', choices=['T1', 'T2', 'T3', 'T4', 'T5', 'all'], default='all')
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    env = SplitDecisionEnv()
    a_params = env.attacker_params
    d_params = env.defender_params

    def make_l1():
        return L1Defender(d_params, horizon_s=1.5, switch_dwell=6)

    def make_l0():
        return L0Defender(d_params)

    results = {}
    tests = []
    if args.test in ('T1', 'all'):
        tests.append(('T1_straight_B_vs_L1',
                       StraightAttacker(a_params, target_goal='B'), make_l1(),
                       'expected defender_win >= 0.80'))
    if args.test in ('T2', 'all'):
        tests.append(('T2_straight_A_vs_L1',
                       StraightAttacker(a_params, target_goal='A'), make_l1(),
                       'expected defender_win >= 0.80'))
    if args.test in ('T3', 'all'):
        tests.append(('T3_feint_A_to_B_t2.4_vs_L1',
                       FeintAttacker(a_params, fake_goal='A', real_goal='B', switch_time=2.4),
                       make_l1(),
                       'expected attacker_win >= 0.60 (feint should work)'))
    if args.test in ('T4', 'all'):
        tests.append(('T4_straight_B_vs_L0',
                       StraightAttacker(a_params, target_goal='B'), make_l0(),
                       'expected attacker_win >= 0.80 (L0 has no prediction)'))

    for name, attacker, defender, note in tests:
        print(f"\n=== {name} ===  ({note})")
        r = _run_scenario(env, attacker, defender, n_episodes=args.episodes, base_seed=args.seed)
        print(f"  attacker_win = {r['attacker_win_rate']:.2%}  "
              f"defender_win = {r['defender_win_rate']:.2%}")
        print(f"  feint_rate   = {r['feint_rate']:.2%}  "
              f"mean_ep_steps = {r['mean_ep_steps']:.1f}")
        print(f"  outcomes     = {dict(r['outcomes'])}")
        results[name] = r

    if args.test in ('T5', 'all'):
        print("\n=== T5: feint switch_time sweep (vs L1 h=1.5, dwell=6) ===")
        sweep = []
        for t_s in [0.5, 1.0, 1.5, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]:
            att = FeintAttacker(a_params, fake_goal='A', real_goal='B', switch_time=t_s)
            r = _run_scenario(env, att, make_l1(), n_episodes=args.episodes, base_seed=args.seed)
            sweep.append({
                'switch_time': t_s,
                'attacker_win': r['attacker_win_rate'],
                'feint_rate': r['feint_rate'],
                'mean_ep_steps': r['mean_ep_steps'],
                'top_outcomes': sorted(r['outcomes'].items(), key=lambda kv: -kv[1])[:3],
            })
            print(f"  t*={t_s:.2f}s  att_win={r['attacker_win_rate']:.2%}  "
                  f"feint_rate={r['feint_rate']:.2%}  ep_steps={r['mean_ep_steps']:.1f}  "
                  f"top: {sweep[-1]['top_outcomes']}")
        results['T5_sweep'] = sweep
        best = max(sweep, key=lambda r: r['attacker_win'])
        print(f"\n  -> best t* = {best['switch_time']}s, attacker_win = {best['attacker_win']:.2%}")

    out_path = _ROOT / 'logs' / 'sanity_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == '__main__':
    main()
