"""PPO training for the defender RL policy.

    python -m sim2d.train_defender --run_id def_v1 \
        --attacker_model logs/runs/it14_mirror/best/best_model.zip \
        --total_steps 500000

Attacker is loaded as a frozen SB3 PPO model; defender is the trainable agent.
BC seed defender on L2RuleDefender expert (the patience defender) before PPO.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch as th

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402

from sim2d.env import DefenderRLEnv  # noqa: E402
from sim2d.rules import L2RuleDefender  # noqa: E402


def bc_defender_pretrain(model, attacker_policy, env_cfg, n_episodes, n_epochs,
                          batch_size, lr, seed,
                          y_threshold, ema_alpha, conf_thresh):
    """BC the defender policy on L2RuleDefender expert actions."""
    env = DefenderRLEnv(cfg=env_cfg, attacker_policy=attacker_policy, seed=seed)
    expert = L2RuleDefender(env.inner.defender_params,
                             commit_y_threshold=y_threshold,
                             ema_alpha=ema_alpha,
                             confidence_thresh=conf_thresh)
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        expert.reset()
        done = False
        while not done:
            phys_action = expert.step(env.defender_state, env.attacker_state,
                                       env.goal_A, env.goal_B)
            obs_buf.append(obs)
            act_buf.append(np.array(phys_action, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(np.array(phys_action))
            done = term or trunc
    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.float32)
    print(f"BC: collected {len(obs_arr)} (obs, action) pairs from {n_episodes} expert eps.")

    device = model.policy.device
    obs_t = th.tensor(obs_arr, device=device)
    act_t = th.tensor(act_arr, device=device)
    opt = th.optim.Adam(model.policy.parameters(), lr=lr)

    n = len(obs_t)
    for epoch in range(n_epochs):
        perm = th.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ob = obs_t[idx]
            ac = act_t[idx]
            features = model.policy.extract_features(ob)
            if isinstance(features, tuple):
                features = features[0]
            latent_pi = model.policy.mlp_extractor.policy_net(features)
            pred = model.policy.action_net(latent_pi)
            loss = ((pred - ac) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch == 0 or epoch == n_epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"  BC epoch {epoch + 1}/{n_epochs}: loss = {total_loss / n_batches:.5f}")


def make_env_fn(env_cfg, attacker_policy, seed, monitor_dir=None):
    def _f():
        env = DefenderRLEnv(cfg=env_cfg, attacker_policy=attacker_policy, seed=seed)
        if monitor_dir is not None:
            env = Monitor(env, monitor_dir)
        return env
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', type=str, required=True)
    ap.add_argument('--attacker_model', type=str, required=True,
                    help='Path to frozen attacker policy .zip')
    ap.add_argument('--total_steps', type=int, default=500_000)
    ap.add_argument('--n_envs', type=int, default=8)
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--ent_coef', type=float, default=0.005)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_steps', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--clip_range', type=float, default=0.2)
    ap.add_argument('--target_kl', type=float, default=None)
    ap.add_argument('--net_arch', type=str, default='256,256,256')

    # BC config
    ap.add_argument('--bc_pretrain', action='store_true')
    ap.add_argument('--bc_episodes', type=int, default=200)
    ap.add_argument('--bc_epochs', type=int, default=15)
    ap.add_argument('--bc_batch_size', type=int, default=256)
    ap.add_argument('--bc_lr', type=float, default=1e-3)

    # L2 defender expert params
    ap.add_argument('--y_threshold', type=float, default=12.0)
    ap.add_argument('--ema_alpha', type=float, default=0.1)
    ap.add_argument('--conf_thresh', type=float, default=0.3)

    # Env overrides
    ap.add_argument('--env_overrides', type=str, default='{}')
    ap.add_argument('--forward_progress_reward', type=float, default=0.0,
                    help='Defender doesn\'t need forward shaping (it doesn\'t want to move forward).')
    ap.add_argument('--step_penalty', type=float, default=-0.002)
    ap.add_argument('--win_reward', type=float, default=1.0)
    ap.add_argument('--loss_reward', type=float, default=-1.0)

    args = ap.parse_args()

    out_dir = _ROOT / 'logs' / 'runs' / 'defender' / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval').mkdir(exist_ok=True)
    (out_dir / 'best').mkdir(exist_ok=True)

    env_cfg = {
        'forward_progress_reward': args.forward_progress_reward,
        'step_penalty': args.step_penalty,
        'win_reward': args.win_reward,
        'loss_reward': args.loss_reward,
        'target_mode': 'random',
        'spawn_xy_noise': 3.0,
        'defender_params': {'v_max': 2.5, 'omega_max': 1.2,
                             'delta_max': float(np.radians(25)), 'L': 0.5},
        'obs_target_relative': True,
        'obs_target_mirror': True,
    }
    if args.env_overrides:
        env_cfg.update(json.loads(args.env_overrides))

    print(f"Loading frozen attacker from {args.attacker_model}")
    attacker = PPO.load(args.attacker_model)

    train_env_fns = [make_env_fn(env_cfg, attacker, seed=args.seed + i)
                      for i in range(args.n_envs)]
    train_env = DummyVecEnv(train_env_fns)
    eval_env = DummyVecEnv([make_env_fn(env_cfg, attacker, seed=args.seed + 100000)])

    model = PPO(
        'MlpPolicy', train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        policy_kwargs=dict(net_arch=[int(x) for x in args.net_arch.split(',')]),
        verbose=1,
        tensorboard_log=str(out_dir / 'tb'),
        seed=args.seed,
    )

    if args.bc_pretrain:
        bc_defender_pretrain(
            model, attacker, env_cfg,
            n_episodes=args.bc_episodes,
            n_epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            lr=args.bc_lr,
            seed=args.seed + 50000,
            y_threshold=args.y_threshold,
            ema_alpha=args.ema_alpha,
            conf_thresh=args.conf_thresh,
        )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / 'best'),
        log_path=str(out_dir / 'eval'),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    sample_env = DefenderRLEnv(cfg=env_cfg, attacker_policy=attacker)
    config = {
        'role': 'defender',
        'run_id': args.run_id,
        'attacker_model': args.attacker_model,
        'total_steps': args.total_steps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'ent_coef': args.ent_coef,
        'gamma': args.gamma,
        'seed': args.seed,
        'episode_seconds': sample_env.inner.max_episode_seconds,
        'attacker_params': {k: float(v) for k, v in sample_env.inner.attacker_params.items()},
        'defender_params': {k: float(v) for k, v in sample_env.inner.defender_params.items()},
        'bc_expert_params': {'y_threshold': args.y_threshold,
                              'ema_alpha': args.ema_alpha,
                              'conf_thresh': args.conf_thresh},
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nStarting defender training: {args.run_id}, total_steps={args.total_steps}")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                tb_log_name=args.run_id, progress_bar=False)
    elapsed = time.time() - t0

    model.save(str(out_dir / 'final'))
    print(f"\nDefender training done in {elapsed:.1f}s. Final -> {out_dir/'final.zip'}")


if __name__ == '__main__':
    main()
