"""Trajectory + outcome visualization for a trained RL policy.

    python -m sim2d.visualize  (defaults to it14_mirror)

Outputs PNGs to logs/runs/<run_id>/viz/.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.rules import L1Defender  # noqa: E402
from sim2d.feint_detector import is_feint  # noqa: E402


def collect_episodes(model, env, n_eps, base_seed):
    episodes = []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=base_seed + ep)
        env.defender_policy.reset()
        assigned = env.assigned_target
        done = False
        info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc

        is_f, ev = is_feint(env.history['attacker'], env.goal_A, env.goal_B)
        episodes.append({
            'seed': base_seed + ep,
            'assigned_target': assigned,
            'attacker_traj': np.array(env.history['attacker']),
            'defender_traj': np.array(env.history['defender']),
            'defender_target_history': list(env.history['defender_target']),
            'outcome': info.get('outcome', '?'),
            'winner': info.get('winner', '?'),
            'is_feint': is_f,
            'feint_ev': ev,
            'n_steps': len(env.history['attacker']),
        })
    return episodes


def _gradient_trail(ax, traj_xy, cmap, label=None):
    pts = traj_xy.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    if len(segs) == 0:
        return
    colors = cmap(np.linspace(0.25, 1.0, len(segs)))
    lc = LineCollection(segs, colors=colors, linewidth=2.0, label=label)
    ax.add_collection(lc)


def _heading_arrow(ax, state, color, scale=1.2):
    x, y, theta = state[0], state[1], state[2]
    ax.arrow(x, y, scale * np.cos(theta), scale * np.sin(theta),
              head_width=0.4, head_length=0.3, fc=color, ec=color, alpha=0.9, zorder=5)


def plot_trajectory(ax, episode, goal_A, goal_B, goal_radius=2.0, block_radius=3.0,
                     arena_x_half=12.0, arena_y_max=24.0, arena_y_min=-1.0):
    """Single episode trajectory."""
    att = episode['attacker_traj']
    defd = episode['defender_traj']
    targets = episode['defender_target_history']

    ax.set_xlim(-arena_x_half - 0.5, arena_x_half + 0.5)
    ax.set_ylim(arena_y_min, arena_y_max + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xticks(np.arange(-arena_x_half, arena_x_half + 1, 4))
    ax.set_yticks(np.arange(0, arena_y_max + 1, 4))

    # Goal areas
    ax.add_patch(Circle(goal_A, goal_radius, color='#3498db', alpha=0.20, zorder=1))
    ax.add_patch(Circle(goal_B, goal_radius, color='#e67e22', alpha=0.20, zorder=1))
    ax.plot(*goal_A, marker='*', markersize=22, color='#2980b9', zorder=2)
    ax.plot(*goal_B, marker='*', markersize=22, color='#d35400', zorder=2)
    ax.text(goal_A[0], goal_A[1] + 0.7, 'A', ha='center', fontsize=11, fontweight='bold')
    ax.text(goal_B[0], goal_B[1] + 0.7, 'B', ha='center', fontsize=11, fontweight='bold')

    # Highlight assigned target with gold ring
    tgt_pos = goal_A if episode['assigned_target'] == 'A' else goal_B
    ax.add_patch(Circle(tgt_pos, goal_radius + 0.3, fill=False, edgecolor='gold',
                         linewidth=2.5, linestyle='--', zorder=3))

    # Trails
    _gradient_trail(ax, att[:, :2], plt.cm.Reds)
    _gradient_trail(ax, defd[:, :2], plt.cm.Blues)

    # Start markers
    ax.plot(*att[0, :2], 'o', color='#c0392b', markersize=10,
             markeredgecolor='black', markeredgewidth=0.8, zorder=4)
    ax.plot(*defd[0, :2], 'o', color='#2980b9', markersize=10,
             markeredgecolor='black', markeredgewidth=0.8, zorder=4)

    # Final positions with heading arrows
    _heading_arrow(ax, att[-1], '#c0392b')
    _heading_arrow(ax, defd[-1], '#2980b9')

    # Defender commit-switch markers (yellow dots wherever last_target changed)
    prev_t = targets[0] if targets else None
    for i in range(1, len(targets)):
        if targets[i] != prev_t and targets[i] is not None:
            ax.plot(*defd[i, :2], 'o', color='gold', markersize=7,
                     markeredgecolor='black', markeredgewidth=0.5, zorder=6)
            prev_t = targets[i]

    # Title
    outcome = episode['outcome']
    is_win = episode['winner'] == 'attacker'
    color = '#2ecc71' if is_win else '#e74c3c'
    feint_tag = ' (feint)' if episode['is_feint'] else ''
    n_s = episode['n_steps']
    title = f"target={episode['assigned_target']} | {outcome}{feint_tag} | {n_s} steps"
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')


def plot_trajectory_grid(episodes, goal_A, goal_B, save_path, title_prefix=''):
    """Pick representative episodes across categories and plot as grid."""
    cats = {
        'feint_win_A':   [e for e in episodes if e['is_feint'] and e['winner']=='attacker' and e['assigned_target']=='A'],
        'feint_win_B':   [e for e in episodes if e['is_feint'] and e['winner']=='attacker' and e['assigned_target']=='B'],
        'feint_loss':    [e for e in episodes if e['is_feint'] and e['winner']=='defender'],
        'nofeint_win':   [e for e in episodes if not e['is_feint'] and e['winner']=='attacker'],
        'nofeint_block': [e for e in episodes if not e['is_feint'] and 'blocked' in e['outcome']],
        'nofeint_oob':   [e for e in episodes if not e['is_feint'] and e['outcome']=='out_of_bounds'],
    }
    cat_labels = ['Feint WIN (target=A)', 'Feint WIN (target=B)', 'Feint LOSS',
                  'No-feint WIN (rare)', 'No-feint BLOCK', 'No-feint OOB']

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, key, label in zip(axes.flatten(), list(cats.keys()), cat_labels):
        if cats[key]:
            plot_trajectory(ax, cats[key][0], goal_A, goal_B)
            ax.set_xlabel(label, fontsize=11, fontweight='bold')
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f'no example\n({label})', ha='center', va='center',
                     transform=ax.transAxes, fontsize=11, color='gray')

    fig.suptitle(f'{title_prefix} | red=attacker, blue=defender, gold ring=assigned target,'
                  ' gold dots=defender commit switches\n'
                  'gradient color = time progression (light→dark)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


def plot_outcome_distribution(episodes, save_path, title_prefix=''):
    """Outcome bars by target, with win/loss color coding + conditional analysis."""
    by_target = defaultdict(lambda: defaultdict(int))
    for ep in episodes:
        by_target[ep['assigned_target']][ep['outcome']] += 1

    all_outcomes = ['won_at_A', 'won_at_B', 'blocked_at_A', 'blocked_at_B',
                    'wrong_goal_A', 'wrong_goal_B', 'out_of_bounds', 'collision', 'timeout']
    colors = {
        'won_at_A': '#27ae60', 'won_at_B': '#2ecc71',
        'blocked_at_A': '#c0392b', 'blocked_at_B': '#e74c3c',
        'wrong_goal_A': '#e67e22', 'wrong_goal_B': '#f39c12',
        'out_of_bounds': '#8e44ad', 'collision': '#9b59b6', 'timeout': '#34495e',
    }
    display_outcomes = [o for o in all_outcomes
                         if by_target['A'].get(o, 0) > 0 or by_target['B'].get(o, 0) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: bars by target
    ax = axes[0]
    width = 0.38
    x_pos = np.arange(len(display_outcomes))
    a_vals = [by_target['A'].get(o, 0) for o in display_outcomes]
    b_vals = [by_target['B'].get(o, 0) for o in display_outcomes]
    a_colors = [colors.get(o, 'gray') for o in display_outcomes]
    b_colors = [colors.get(o, 'gray') for o in display_outcomes]
    bars_a = ax.bar(x_pos - width/2, a_vals, width, label='target=A',
                     color=a_colors, edgecolor='black', linewidth=0.5)
    bars_b = ax.bar(x_pos + width/2, b_vals, width, label='target=B',
                     color=b_colors, edgecolor='black', linewidth=0.5, hatch='//')
    for bars, vals in [(bars_a, a_vals), (bars_b, b_vals)]:
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(v), ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(display_outcomes, rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Episode count')
    ax.set_title('Outcomes by assigned target')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Right: conditional 2x2 (feint vs no-feint) × (win vs loss)
    ax = axes[1]
    cells = defaultdict(int)
    for ep in episodes:
        won = ep['winner'] == 'attacker'
        is_f = ep['is_feint']
        cells[('feint' if is_f else 'noFeint') + ('_win' if won else '_loss')] += 1
    n = len(episodes)
    n_f = cells['feint_win'] + cells['feint_loss']
    n_nf = cells['noFeint_win'] + cells['noFeint_loss']
    pw_f = cells['feint_win'] / max(n_f, 1)
    pw_nf = cells['noFeint_win'] / max(n_nf, 1)

    cats = ['Feint', 'No-feint']
    wins = [cells['feint_win'], cells['noFeint_win']]
    losses = [cells['feint_loss'], cells['noFeint_loss']]
    ax.bar(cats, wins, color='#27ae60', edgecolor='black', label=f'Win')
    ax.bar(cats, losses, bottom=wins, color='#c0392b', edgecolor='black', label='Loss')

    for i, c in enumerate(cats):
        total = wins[i] + losses[i]
        wr = wins[i] / max(total, 1)
        ax.text(i, total + max(wins+losses)*0.02,
                 f'n={total}\nP(win)={wr:.1%}',
                 ha='center', fontsize=10, fontweight='bold')

    lift = pw_f / max(pw_nf, 1e-6)
    ax.set_ylabel('Episode count')
    ax.set_title(f'Conditional: feint vs no-feint  |  lift = {lift:.1f}×')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Summary at bottom
    win_total = sum(1 for e in episodes if e['winner']=='attacker')
    feint_total = sum(1 for e in episodes if e['is_feint'])
    p_feint_given_win = cells['feint_win'] / max(win_total, 1)
    fig.suptitle(
        f'{title_prefix}  |  n={n} ep  |  overall win = {win_total/n:.1%}  |  '
        f'feint rate = {feint_total/n:.1%}  |  P(feint|win) = {p_feint_given_win:.1%}',
        fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='logs/runs/it14_mirror/best/best_model.zip')
    ap.add_argument('--n_eps', type=int, default=300)
    ap.add_argument('--seed', type=int, default=70000)
    ap.add_argument('--out_dir', default=None,
                    help='default: logs/runs/<run_id>/viz/')
    args = ap.parse_args()

    model_path = Path(args.model)
    run_dir = model_path.parent if model_path.parent.name not in ('best', 'final') else model_path.parent.parent
    if args.out_dir is None:
        out_dir = _ROOT / 'logs' / 'runs' / run_dir.name / 'viz'
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ENV_CFG = {
        'forward_progress_reward': 0.01, 'step_penalty': -0.002,
        'target_mode': 'random', 'spawn_xy_noise': 3.0,
        'defender_params': {'v_max': 2.5, 'omega_max': 1.2, 'delta_max': 0.4363, 'L': 0.5},
        'obs_target_relative': True, 'obs_target_mirror': True,
    }
    model = PPO.load(str(model_path))
    env = SplitDecisionEnv(cfg=ENV_CFG, seed=args.seed)
    env.set_defender(L1Defender(env.defender_params, horizon_s=1.5, switch_dwell=6))

    print(f'Collecting {args.n_eps} episodes from {model_path} ...')
    episodes = collect_episodes(model, env, args.n_eps, args.seed)
    n = len(episodes)
    wins = sum(1 for e in episodes if e['winner']=='attacker')
    feints = sum(1 for e in episodes if e['is_feint'])
    print(f'  n={n}  win={wins/n:.1%}  feint={feints/n:.1%}')

    title = f'{run_dir.name}'
    traj_path = out_dir / 'trajectories.png'
    plot_trajectory_grid(episodes, env.goal_A, env.goal_B, traj_path, title_prefix=title)
    print(f'  trajectories -> {traj_path}')

    dist_path = out_dir / 'outcome_distribution.png'
    plot_outcome_distribution(episodes, dist_path, title_prefix=title)
    print(f'  outcomes -> {dist_path}')

    print('done.')


if __name__ == '__main__':
    main()
