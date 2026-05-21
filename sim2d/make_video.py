"""Render one episode of the trained RL policy as MP4.

    python -m sim2d.make_video                       # auto-find feint wins for A and B
    python -m sim2d.make_video --target A --seed 80042
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO  # noqa: E402

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.rules import L1Defender  # noqa: E402
from sim2d.feint_detector import is_feint  # noqa: E402


def build_env(target_mode: str, seed: int) -> SplitDecisionEnv:
    cfg = {
        'forward_progress_reward': 0.01, 'step_penalty': -0.002,
        'target_mode': target_mode, 'spawn_xy_noise': 3.0,
        'defender_params': {'v_max': 2.5, 'omega_max': 1.2, 'delta_max': 0.4363, 'L': 0.5},
        'obs_target_relative': True, 'obs_target_mirror': True,
    }
    env = SplitDecisionEnv(cfg=cfg, seed=seed)
    env.set_defender(L1Defender(env.defender_params, horizon_s=1.5, switch_dwell=6))
    return env


def roll_one(model, env, seed):
    obs, _ = env.reset(seed=seed)
    env.defender_policy.reset()
    done = False
    info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(a)
        done = term or trunc
    att = np.array(env.history['attacker'])
    is_f, _ = is_feint(att, env.goal_A, env.goal_B)
    return {
        'attacker_traj': att,
        'defender_traj': np.array(env.history['defender']),
        'def_targets': list(env.history['defender_target']),
        'info': info,
        'assigned_target': env.assigned_target,
        'goal_A': env.goal_A.copy(),
        'goal_B': env.goal_B.copy(),
        'dt': env.dt,
        'seed': seed,
        'is_feint': is_f,
    }


def categorize(ep):
    """Map an episode to a category string."""
    outcome = ep['info'].get('outcome', 'other')
    is_f = ep['is_feint']
    winner = ep['info'].get('winner')
    target = ep['assigned_target']
    if winner == 'attacker':
        return f'feint_win_{target}' if is_f else f'nofeint_win_{target}'
    if outcome == 'out_of_bounds':
        return 'feint_loss_oob' if is_f else 'nofeint_loss_oob'
    if 'blocked' in outcome:
        return 'feint_loss_block' if is_f else 'nofeint_loss_block'
    if outcome == 'timeout':
        return 'feint_loss_timeout' if is_f else 'nofeint_loss_timeout'
    if outcome == 'collision':
        return 'feint_loss_collision' if is_f else 'nofeint_loss_collision'
    if outcome.startswith('wrong_goal'):
        return f'wrong_goal_{is_f}'
    return f'other_{outcome}'


def find_feint_win(model, target: str, n_search: int = 100, base_seed: int = 80000):
    env = build_env(target_mode=target, seed=base_seed)
    for ep in range(n_search):
        result = roll_one(model, env, base_seed + ep)
        if result['info'].get('winner') != 'attacker':
            continue
        if result['info'].get('goal') != target:
            continue
        is_f, _ = is_feint(result['attacker_traj'], result['goal_A'], result['goal_B'])
        if is_f:
            print(f'found feint WIN target={target}  seed={result["seed"]}  steps={len(result["attacker_traj"])}')
            return result
    return None


def make_animation(ep, out_path, fps=15, trail_len=30,
                    goal_radius=2.0, arena_x_half=12.0,
                    arena_y_max=24.0, arena_y_min=-1.0):
    att = ep['attacker_traj']
    defd = ep['defender_traj']
    targets = ep['def_targets']
    n = len(att)
    dt = ep['dt']

    fig, ax = plt.subplots(figsize=(7, 7.5))
    ax.set_xlim(-arena_x_half - 0.5, arena_x_half + 0.5)
    ax.set_ylim(arena_y_min, arena_y_max + 0.8)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.set_xticks(np.arange(-arena_x_half, arena_x_half + 1, 4))
    ax.set_yticks(np.arange(0, arena_y_max + 1, 4))

    # Static elements
    ax.add_patch(Circle(ep['goal_A'], goal_radius, color='#3498db', alpha=0.20, zorder=1))
    ax.add_patch(Circle(ep['goal_B'], goal_radius, color='#e67e22', alpha=0.20, zorder=1))
    ax.plot(*ep['goal_A'], marker='*', markersize=25, color='#2980b9', zorder=2)
    ax.plot(*ep['goal_B'], marker='*', markersize=25, color='#d35400', zorder=2)
    ax.text(ep['goal_A'][0], ep['goal_A'][1] + 0.8, 'A', ha='center',
             fontsize=13, fontweight='bold')
    ax.text(ep['goal_B'][0], ep['goal_B'][1] + 0.8, 'B', ha='center',
             fontsize=13, fontweight='bold')

    # Gold ring on assigned target
    tgt = ep['goal_A'] if ep['assigned_target'] == 'A' else ep['goal_B']
    ax.add_patch(Circle(tgt, goal_radius + 0.3, fill=False, edgecolor='gold',
                         linewidth=2.5, linestyle='--', zorder=3))
    ax.text(tgt[0], tgt[1] - 3.0, '(true target)', ha='center',
             fontsize=9, color='goldenrod', style='italic')

    # Dynamic
    att_trail, = ax.plot([], [], '-', color='#c0392b', linewidth=2.5, alpha=0.55, zorder=5)
    def_trail, = ax.plot([], [], '-', color='#2980b9', linewidth=2.5, alpha=0.55, zorder=5)
    att_dot, = ax.plot([], [], 'o', color='#c0392b', markersize=14,
                        markeredgecolor='black', markeredgewidth=1.0, zorder=10)
    def_dot, = ax.plot([], [], 'o', color='#2980b9', markersize=14,
                        markeredgecolor='black', markeredgewidth=1.0, zorder=10)
    att_heading, = ax.plot([], [], '-', color='#c0392b', linewidth=3, zorder=11)
    def_heading, = ax.plot([], [], '-', color='#2980b9', linewidth=3, zorder=11)
    def_commit_line, = ax.plot([], [], ':', color='#7f8c8d', linewidth=1.5,
                                 alpha=0.6, zorder=4)

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='#c0392b', markersize=10,
                    linestyle='', label='Attacker (RL policy)'),
        plt.Line2D([0], [0], marker='o', color='#2980b9', markersize=10,
                    linestyle='', label='Defender (L1 rule)'),
        plt.Line2D([0], [0], color='gold', linewidth=2, linestyle='--',
                    label='Attacker\'s assigned target'),
        plt.Line2D([0], [0], color='#7f8c8d', linewidth=1.5, linestyle=':',
                    label='Defender\'s current commit'),
    ]
    ax.legend(handles=legend_handles, loc='upper center',
               bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8, frameon=False)

    title = ax.set_title('', fontsize=11, fontweight='bold')
    info_box = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        fontsize=10, va='top',
                        bbox=dict(boxstyle='round', facecolor='white',
                                  edgecolor='gray', alpha=0.85))

    def init():
        return ()

    def update(i):
        att_dot.set_data([att[i, 0]], [att[i, 1]])
        def_dot.set_data([defd[i, 0]], [defd[i, 1]])

        h = 1.2
        att_heading.set_data([att[i, 0], att[i, 0] + h * np.cos(att[i, 2])],
                              [att[i, 1], att[i, 1] + h * np.sin(att[i, 2])])
        def_heading.set_data([defd[i, 0], defd[i, 0] + h * np.cos(defd[i, 2])],
                              [defd[i, 1], defd[i, 1] + h * np.sin(defd[i, 2])])

        start = max(0, i - trail_len)
        att_trail.set_data(att[start:i + 1, 0], att[start:i + 1, 1])
        def_trail.set_data(defd[start:i + 1, 0], defd[start:i + 1, 1])

        commit = targets[i] if i < len(targets) else None
        if commit == 'A':
            commit_goal = ep['goal_A']
        elif commit == 'B':
            commit_goal = ep['goal_B']
        else:
            commit_goal = None
        if commit_goal is not None:
            def_commit_line.set_data([defd[i, 0], commit_goal[0]],
                                       [defd[i, 1], commit_goal[1]])
        else:
            def_commit_line.set_data([], [])

        t_sec = i * dt
        title.set_text(f't = {t_sec:>4.2f}s     defender commit → {commit}')

        info_str = (f"target (hidden from defender): {ep['assigned_target']}\n"
                     f"step {i+1} / {n}")
        if i == n - 1:
            outcome = ep['info'].get('outcome', '?')
            winner = ep['info'].get('winner', '?')
            mark = '✓ WIN' if winner == 'attacker' else '✗ LOSS'
            info_str += f"\n\noutcome: {outcome}\n{mark}"
        info_box.set_text(info_str)
        return ()

    anim = animation.FuncAnimation(fig, update, frames=n, init_func=init,
                                    blit=False, interval=1000.0 / fps)
    writer = animation.FFMpegWriter(fps=fps, codec='libx264',
                                      extra_args=['-pix_fmt', 'yuv420p'])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close()
    print(f'  wrote {out_path}')


def batch_main(args):
    from collections import defaultdict
    from sim2d.visualize import plot_trajectory

    model = PPO.load(args.model)
    out_dir = _ROOT / args.out_dir / 'cases'
    out_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(target_mode='random', seed=args.base_seed)
    print(f'Rolling {args.n_search} episodes...')
    episodes = []
    for i in range(args.n_search):
        ep = roll_one(model, env, args.base_seed + i)
        episodes.append(ep)

    by_cat = defaultdict(list)
    for ep in episodes:
        by_cat[categorize(ep)].append(ep)

    print('\nCategories found (with counts):')
    cats_order = sorted(by_cat.keys(),
                         key=lambda c: ('win' not in c, c))  # wins first
    for c in cats_order:
        print(f'  {c:30s}  {len(by_cat[c]):4d}')

    selected = []
    for cat in cats_order:
        cat_dir = out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for i, ep in enumerate(by_cat[cat][:args.n_per_cat]):
            out_path = cat_dir / f'{cat}_{i+1}_seed{ep["seed"]}.mp4'
            make_animation(ep, out_path, fps=args.fps)
            selected.append((cat, i + 1, ep))

    # Static summary grid
    n_panels = len(selected)
    if n_panels > 0:
        cols = 4
        rows = (n_panels + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 4.5))
        axes_flat = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        for ax, (cat, idx, ep) in zip(axes_flat, selected):
            plot_trajectory(ax, ep, ep['goal_A'], ep['goal_B'])
            ax.set_xlabel(f'{cat} #{idx}  seed={ep["seed"]}', fontsize=9, fontweight='bold')
        for ax in axes_flat[n_panels:]:
            ax.set_axis_off()
        win_count = sum(1 for ep in episodes if ep['info'].get('winner') == 'attacker')
        feint_count = sum(1 for ep in episodes if ep['is_feint'])
        fig.suptitle(
            f'Case-study cases (n_search={args.n_search})  |  '
            f'win = {win_count/args.n_search:.1%}  |  '
            f'feint rate = {feint_count/args.n_search:.1%}',
            fontsize=13, fontweight='bold')
        plt.tight_layout()
        summary_path = out_dir / 'cases_summary.png'
        plt.savefig(summary_path, dpi=130, bbox_inches='tight')
        plt.close()
        print(f'\nSummary grid -> {summary_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='logs/runs/it14_mirror/best/best_model.zip')
    ap.add_argument('--out_dir', default='logs/runs/it14_mirror/viz')
    ap.add_argument('--mode', choices=['single', 'batch'], default='single')
    ap.add_argument('--target', choices=['A', 'B', 'both'], default='both')
    ap.add_argument('--seed', type=int, default=None,
                    help='if given (single mode), render this specific seed')
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--n_search', type=int, default=400,
                    help='(batch mode) episodes to roll before categorizing')
    ap.add_argument('--base_seed', type=int, default=80000,
                    help='(batch mode) starting seed')
    ap.add_argument('--n_per_cat', type=int, default=2,
                    help='(batch mode) videos per category')
    args = ap.parse_args()

    if args.mode == 'batch':
        batch_main(args)
        return

    model = PPO.load(args.model)
    out_dir = _ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = ['A', 'B'] if args.target == 'both' else [args.target]
    for t in targets:
        if args.seed is not None:
            env = build_env(target_mode=t, seed=args.seed)
            ep = roll_one(model, env, args.seed)
        else:
            ep = find_feint_win(model, t, n_search=200, base_seed=80000)
        if ep is None:
            print(f'no clean feint win found for target={t}')
            continue
        out_path = out_dir / f'feint_target_{t}.mp4'
        make_animation(ep, out_path, fps=args.fps)


if __name__ == '__main__':
    main()
