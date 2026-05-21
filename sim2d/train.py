"""PPO training for the Split Decision attacker.

    python -m sim2d.train --run_id stage_b_run1 --defender L0 --total_steps 500000
    python -m sim2d.train --run_id stage_c_run1 --defender L1 --total_steps 2000000

Outputs in logs/runs/<run_id>/:
  config.json           snapshot of hyperparams + env params
  tb/                   tensorboard logs
  eval/                 eval metrics
  best/                 best model (by mean eval reward)
  final.zip             final model
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
from stable_baselines3.common.vec_env import DummyVecEnv  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402

from sim2d.env import SplitDecisionEnv  # noqa: E402
from sim2d.rules import L0Defender, L1Defender, StraightAttacker, FeintAttacker  # noqa: E402


def bc_pretrain_policy(model, defender_type: str, n_episodes: int, n_epochs: int,
                        batch_size: int, lr: float, seed: int = 999999,
                        env_cfg: dict = None,
                        defender_random_tiebreak: bool = False,
                        bc_expert: str = 'straight'):
    """Behavior cloning on a scripted StraightAttacker (random target A/B).

    Expert target follows env.assigned_target when target_mode is active so
    that BC data matches the env's reward signal.
    """
    env = SplitDecisionEnv(cfg=env_cfg, seed=seed)
    if defender_type == 'L0':
        env.set_defender(L0Defender(env.defender_params))
    else:
        env.set_defender(L1Defender(env.defender_params, horizon_s=1.5, switch_dwell=6,
                                      randomize_initial_target=defender_random_tiebreak))

    rng = np.random.default_rng(seed)
    obs_buf, act_buf = [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        if env.assigned_target is not None:
            target_goal = env.assigned_target
        else:
            target_goal = 'A' if ep % 2 == 0 else 'B'
        # Choose expert per episode based on bc_expert mode.
        use_feint = False
        if bc_expert == 'feint':
            use_feint = True
        elif bc_expert == 'mixed':
            use_feint = bool(rng.random() < 0.5)
        if use_feint:
            fake = 'B' if target_goal == 'A' else 'A'
            t_star = float(rng.uniform(1.8, 3.0))
            expert = FeintAttacker(env.attacker_params, fake_goal=fake,
                                    real_goal=target_goal, switch_time=t_star,
                                    dt=env.dt)
            expert.reset(assigned_target=target_goal)
        else:
            expert = StraightAttacker(env.attacker_params, target_goal=target_goal)
            expert.reset(assigned_target=target_goal)
        done = False
        while not done:
            phys_action = expert.step(env.attacker_state, env.defender_state,
                                       env.goal_A, env.goal_B)
            # Expert reasons in physical frame. If env mirrors, convert to policy frame
            # so BC saves the action the policy is expected to output.
            if env.mirror_active:
                bc_action = (float(phys_action[0]), -float(phys_action[1]))
            else:
                bc_action = (float(phys_action[0]), float(phys_action[1]))
            obs_buf.append(obs)
            act_buf.append(np.array(bc_action, dtype=np.float32))
            obs, _, term, trunc, _ = env.step(np.array(bc_action))
            done = term or trunc

    obs_arr = np.array(obs_buf, dtype=np.float32)
    act_arr = np.array(act_buf, dtype=np.float32)
    print(f"BC: collected {len(obs_arr)} (obs, action) pairs from {n_episodes} expert episodes.")

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


def make_env_fn(defender_type: str, seed: int, env_cfg: dict = None,
                 monitor_dir: str = None, defender_random_tiebreak: bool = False):
    def _f():
        env = SplitDecisionEnv(cfg=env_cfg, seed=seed)
        if defender_type == 'L0':
            env.set_defender(L0Defender(env.defender_params))
        elif defender_type == 'L1':
            env.set_defender(L1Defender(env.defender_params, horizon_s=1.5, switch_dwell=6,
                                          randomize_initial_target=defender_random_tiebreak))
        else:
            raise ValueError(f"unknown defender type: {defender_type}")
        if monitor_dir is not None:
            env = Monitor(env, monitor_dir)
        return env
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_id', type=str, required=True)
    ap.add_argument('--defender', choices=['L0', 'L1'], default='L1')
    ap.add_argument('--total_steps', type=int, default=2_000_000)
    ap.add_argument('--n_envs', type=int, default=8)
    ap.add_argument('--learning_rate', type=float, default=3e-4)
    ap.add_argument('--ent_coef', type=float, default=0.01)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n_steps', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--n_epochs', type=int, default=10)
    ap.add_argument('--forward_bias', type=float, default=0.0,
                    help='Optional bias on v_cmd action mean at init. '
                         'Use 0 if --bc_pretrain is set.')
    ap.add_argument('--bc_pretrain', action='store_true',
                    help='Bootstrap policy via behavior cloning on a scripted '
                         'StraightAttacker (random target A or B). Strongly '
                         'recommended for cold-start in sparse-reward env.')
    ap.add_argument('--bc_episodes', type=int, default=200)
    ap.add_argument('--bc_epochs', type=int, default=20)
    ap.add_argument('--bc_batch_size', type=int, default=256)
    ap.add_argument('--bc_lr', type=float, default=1e-3)
    ap.add_argument('--forward_progress_reward', type=float, default=0.01,
                    help='Per-step shaping for +y movement (0.01 * delta_y). '
                         'Breaks the value-baseline trap in sparse-reward setting. '
                         'Set 0 for pure sparse reward.')
    ap.add_argument('--step_penalty', type=float, default=-0.002)
    ap.add_argument('--target_kl', type=float, default=None,
                    help='Stop policy update step if approx_kl > target_kl. '
                         'Caps damage per update (mitigates peak-then-collapse).')
    ap.add_argument('--win_reward', type=float, default=1.0)
    ap.add_argument('--loss_reward', type=float, default=-1.0)
    ap.add_argument('--env_overrides', type=str, default='{}',
                    help='Extra env config as JSON, merged into env_cfg.')
    ap.add_argument('--clip_range', type=float, default=0.2)
    ap.add_argument('--defender_random_tiebreak', action='store_true')
    ap.add_argument('--net_arch', type=str, default='256,256,256',
                    help='Comma-separated MLP layer sizes')
    ap.add_argument('--bc_expert', choices=['straight', 'mixed', 'feint'], default='straight',
                    help="BC expert. mixed = 50% straight + 50% feint(t* random in [1.8,3.0])")
    args = ap.parse_args()

    out_dir = _ROOT / 'logs' / 'runs' / args.run_id
    if out_dir.exists():
        print(f"WARNING: {out_dir} exists; overwriting.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'eval').mkdir(exist_ok=True)
    (out_dir / 'best').mkdir(exist_ok=True)

    env_cfg = {
        'forward_progress_reward': args.forward_progress_reward,
        'step_penalty': args.step_penalty,
        'win_reward': args.win_reward,
        'loss_reward': args.loss_reward,
    }
    if args.env_overrides:
        env_cfg.update(json.loads(args.env_overrides))
    train_env_fns = [make_env_fn(args.defender, seed=args.seed + i, env_cfg=env_cfg,
                                  defender_random_tiebreak=args.defender_random_tiebreak)
                      for i in range(args.n_envs)]
    train_env = DummyVecEnv(train_env_fns)

    eval_env = DummyVecEnv([make_env_fn(args.defender, seed=args.seed + 100000,
                                          env_cfg=env_cfg,
                                          defender_random_tiebreak=args.defender_random_tiebreak)])

    model = PPO(
        'MlpPolicy',
        train_env,
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

    if args.forward_bias != 0.0:
        with th.no_grad():
            model.policy.action_net.bias.data[0] += args.forward_bias
        print(f"Applied forward_bias={args.forward_bias} to v_cmd action mean.")

    if args.bc_pretrain:
        bc_pretrain_policy(model, args.defender, args.bc_episodes, args.bc_epochs,
                            args.bc_batch_size, args.bc_lr, seed=args.seed + 50000,
                            env_cfg=env_cfg,
                            defender_random_tiebreak=args.defender_random_tiebreak,
                            bc_expert=args.bc_expert)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(out_dir / 'best'),
        log_path=str(out_dir / 'eval'),
        eval_freq=max(10_000 // args.n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        render=False,
    )

    sample_env = SplitDecisionEnv()
    config = {
        'run_id': args.run_id,
        'defender': args.defender,
        'total_steps': args.total_steps,
        'n_envs': args.n_envs,
        'learning_rate': args.learning_rate,
        'ent_coef': args.ent_coef,
        'gamma': args.gamma,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'seed': args.seed,
        'episode_seconds': sample_env.max_episode_seconds,
        'dt': sample_env.dt,
        'attacker_params': {k: float(v) for k, v in sample_env.attacker_params.items()},
        'defender_params': {k: float(v) for k, v in sample_env.defender_params.items()},
        'r_body': sample_env.r_body,
        'goal_A': sample_env.goal_A.tolist(),
        'goal_B': sample_env.goal_B.tolist(),
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nStarting training: run_id={args.run_id}  defender={args.defender}  "
          f"total_steps={args.total_steps}")
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=eval_cb,
                tb_log_name=args.run_id, progress_bar=False)
    elapsed = time.time() - t0

    model.save(str(out_dir / 'final'))
    print(f"\nTraining done in {elapsed:.1f}s. Final model -> {out_dir/'final.zip'}")


if __name__ == '__main__':
    main()
