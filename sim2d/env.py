"""Split Decision env. Gymnasium-compatible interface, plain-Python core
(gymnasium is only required for RL training, not for sanity checks).
"""
from typing import Optional, Tuple

import numpy as np

from .dynamics import step_dynamics

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
    _BASE_CLASS = gym.Env
except ImportError:
    _HAS_GYMNASIUM = False
    _BASE_CLASS = object


class _InjectedDefender:
    """Defender stub that returns whatever action DefenderRLEnv injects."""

    def __init__(self):
        self.next_action = (0.0, 0.0)
        self.last_target = None

    def reset(self):
        self.last_target = None
        self.next_action = (0.0, 0.0)

    def step(self, *args, **kwargs):
        return self.next_action


_DEFAULT_ATTACKER = {
    'v_max': 3.0,
    'omega_max': 1.2,
    'delta_max': np.radians(35),
    'L': 0.5,
}
_DEFAULT_DEFENDER = {
    'v_max': 2.0,
    'omega_max': 1.2,
    'delta_max': np.radians(25),
    'L': 0.5,
}


class SplitDecisionEnv(_BASE_CLASS):
    """Single-agent env: RL controls attacker, defender is set externally.

    reset() -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)

    Signatures match gymnasium so SB3 wraps it directly.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[dict] = None, seed: int = 0):
        if _HAS_GYMNASIUM:
            super().__init__()
        cfg = cfg or {}
        self.attacker_params = dict(cfg.get('attacker_params', _DEFAULT_ATTACKER))
        self.defender_params = dict(cfg.get('defender_params', _DEFAULT_DEFENDER))

        self.dt = float(cfg.get('dt', 1.0 / 20.0))
        self.max_episode_seconds = float(cfg.get('max_episode_seconds', 12.0))
        self.max_steps = int(round(self.max_episode_seconds / self.dt))

        self.goal_A = np.array(cfg.get('goal_A', [-8.0, 20.0]), dtype=np.float32)
        self.goal_B = np.array(cfg.get('goal_B', [+8.0, 20.0]), dtype=np.float32)
        self.goal_radius = float(cfg.get('goal_radius', 2.0))
        self.block_radius = float(cfg.get('block_radius', 3.0))
        self.r_body = float(cfg.get('r_body', 0.25))
        self.collision_radius = 2.0 * self.r_body

        self.attacker_spawn = np.array(cfg.get('attacker_spawn', [0.0, 2.0]), dtype=np.float32)
        self.defender_spawn = np.array(cfg.get('defender_spawn', [0.0, 14.0]), dtype=np.float32)
        self.spawn_xy_noise = float(cfg.get('spawn_xy_noise', 1.0))
        self.spawn_heading_noise = float(cfg.get('spawn_heading_noise', np.radians(10)))

        self.arena_x_half = float(cfg.get('arena_x_half', 12.0))
        self.arena_y_min = float(cfg.get('arena_y_min', -1.0))
        self.arena_y_max = float(cfg.get('arena_y_max', 24.0))

        self.step_penalty = float(cfg.get('step_penalty', -0.005))
        # Small dense reward for +y progress, only applied during training to
        # bootstrap exploration. Default 0 = pure sparse reward (for sanity
        # checks). Set >0 in train.py to escape the value-baseline trap.
        # Capped at y = goal_line (= 20) to prevent the "fly out of bounds" exploit.
        self.forward_progress_reward = float(cfg.get('forward_progress_reward', 0.0))
        self.win_reward = float(cfg.get('win_reward', 1.0))
        self.loss_reward = float(cfg.get('loss_reward', -1.0))
        # target_mode: None (either goal wins), 'A', 'B', or 'random' (assigned per episode)
        self.target_mode = cfg.get('target_mode', None)
        self.assigned_target = None
        # obs_target_relative: if True and target_mode is active, swap goal order in obs
        # so that (goal_assigned, goal_other) appears in fixed slots regardless of physical
        # side. Forces policy to use target info; cleanest fix for mode collapse.
        self.obs_target_relative = bool(cfg.get('obs_target_relative', False))
        # obs_target_mirror: if True and target=B, mirror obs across y-axis AND
        # invert omega action. Makes target=B episodes equivalent to target=A for
        # the policy, ENFORCING symmetric behavior.
        self.obs_target_mirror = bool(cfg.get('obs_target_mirror', False))

        self.defender_policy = None
        self.rng = np.random.default_rng(seed)
        self._seed = seed

        if _HAS_GYMNASIUM:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.t = 0
        self.attacker_state = None
        self.defender_state = None
        self.history = None

    def set_defender(self, defender_policy):
        self.defender_policy = defender_policy

    @property
    def mirror_active(self) -> bool:
        return self.obs_target_mirror and self.assigned_target == 'B'

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._seed = seed

        n = self.spawn_xy_noise
        nh = self.spawn_heading_noise
        att_x = self.attacker_spawn[0] + self.rng.uniform(-n, n)
        att_y = self.attacker_spawn[1] + self.rng.uniform(-0.5, 0.5)
        att_theta = (np.pi / 2.0) + self.rng.uniform(-nh, nh)
        def_x = self.defender_spawn[0] + self.rng.uniform(-n, n)
        def_y = self.defender_spawn[1] + self.rng.uniform(-0.5, 0.5)
        def_theta = (+np.pi / 2.0) + self.rng.uniform(-nh, nh)

        self.attacker_state = np.array([att_x, att_y, att_theta, 0.0, 0.0], dtype=np.float32)
        self.defender_state = np.array([def_x, def_y, def_theta, 0.0, 0.0], dtype=np.float32)

        if self.target_mode == 'random':
            self.assigned_target = 'A' if self.rng.random() < 0.5 else 'B'
        elif self.target_mode in ('A', 'B'):
            self.assigned_target = self.target_mode
        else:
            self.assigned_target = None

        if self.defender_policy is not None:
            self.defender_policy.reset()

        self.t = 0
        self.history = {
            'attacker': [self.attacker_state.copy()],
            'defender': [self.defender_state.copy()],
            'attacker_action': [],
            'defender_action': [],
            'defender_target': [],
        }

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        ax, ay, atheta, av, aomega = self.attacker_state
        dx, dy, dtheta, dv, domega = self.defender_state

        # Mirror across y-axis if active (target=B and obs_target_mirror enabled).
        if self.mirror_active:
            sgn = -1.0
        else:
            sgn = 1.0
        ax_o = sgn * ax
        dx_o = sgn * dx
        cos_at_o = sgn * np.cos(atheta)
        cos_dt_o = sgn * np.cos(dtheta)
        aomega_o = sgn * aomega
        domega_o = sgn * domega

        # Apparent goal positions (after mirror)
        gA_x_o = sgn * self.goal_A[0]
        gB_x_o = sgn * self.goal_B[0]
        gA_y = self.goal_A[1]
        gB_y = self.goal_B[1]

        if self.obs_target_relative and self.assigned_target is not None:
            if self.assigned_target == 'A':
                g1_x, g1_y = gA_x_o, gA_y
                g2_x, g2_y = gB_x_o, gB_y
            else:  # B
                g1_x, g1_y = gB_x_o, gB_y
                g2_x, g2_y = gA_x_o, gA_y
            target_bit = 1.0
        else:
            g1_x, g1_y = gA_x_o, gA_y
            g2_x, g2_y = gB_x_o, gB_y
            if self.assigned_target == 'A':
                target_bit = 1.0
            elif self.assigned_target == 'B':
                target_bit = -1.0
            else:
                target_bit = 0.0

        return np.array([
            ax_o, ay, np.sin(atheta), cos_at_o, av, aomega_o,
            dx_o, dy, np.sin(dtheta), cos_dt_o, dv, domega_o,
            g1_x - ax_o, g1_y - ay,
            g2_x - ax_o, g2_y - ay,
            target_bit,
        ], dtype=np.float32)

    def step(self, attacker_action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.defender_policy is None:
            raise RuntimeError("call set_defender(policy) before step()")

        prev_y = float(self.attacker_state[1])

        def_action = self.defender_policy.step(
            self.defender_state, self.attacker_state, self.goal_A, self.goal_B)

        # Mirror env: policy outputs action in mirrored frame. Convert to physical
        # by flipping omega_cmd. (v_cmd unchanged.)
        physical_attacker_action = tuple(attacker_action)
        if self.mirror_active:
            physical_attacker_action = (float(attacker_action[0]),
                                         -float(attacker_action[1]))

        self.attacker_state = step_dynamics(
            self.attacker_state, physical_attacker_action, self.attacker_params, self.dt)
        self.defender_state = step_dynamics(
            self.defender_state, tuple(def_action), self.defender_params, self.dt)

        # Shaping caps at y = goal_line (= 20). Progress past the goal line
        # earns no shaping, which removes the "fly out the top" exploit.
        y_now = float(self.attacker_state[1])
        cap_y = float(self.goal_A[1])  # 20.0
        delta_y_capped = max(0.0, min(y_now, cap_y) - min(prev_y, cap_y))
        shaping = self.forward_progress_reward * delta_y_capped

        self.t += 1
        terminated, terminal_reward, info = self._check_termination()
        if terminated:
            reward = float(terminal_reward) + shaping
        else:
            reward = float(self.step_penalty) + shaping
        truncated = False

        self.history['attacker'].append(self.attacker_state.copy())
        self.history['defender'].append(self.defender_state.copy())
        self.history['attacker_action'].append(np.array(attacker_action, dtype=np.float32))
        self.history['defender_action'].append(np.array(def_action, dtype=np.float32))
        self.history['defender_target'].append(
            getattr(self.defender_policy, 'last_target', None))

        return self._get_obs(), reward, terminated, truncated, info

    def _check_termination(self):
        a_pos = self.attacker_state[:2]
        d_pos = self.defender_state[:2]
        info = {'t_steps': self.t}

        if float(np.linalg.norm(a_pos - d_pos)) < self.collision_radius:
            info.update({'outcome': 'collision', 'winner': 'defender'})
            return True, self.loss_reward, info

        for g_name, g_pos in (('A', self.goal_A), ('B', self.goal_B)):
            if float(np.linalg.norm(a_pos - g_pos)) < self.goal_radius:
                # Assigned-target mode: entering the wrong goal counts as a loss.
                if self.assigned_target is not None and g_name != self.assigned_target:
                    info.update({'outcome': f'wrong_goal_{g_name}', 'winner': 'defender',
                                 'goal': g_name, 'assigned_target': self.assigned_target})
                    return True, self.loss_reward, info
                if float(np.linalg.norm(d_pos - g_pos)) < self.block_radius:
                    info.update({'outcome': f'blocked_at_{g_name}', 'winner': 'defender',
                                 'goal': g_name, 'assigned_target': self.assigned_target})
                    return True, self.loss_reward, info
                info.update({'outcome': f'won_at_{g_name}', 'winner': 'attacker',
                             'goal': g_name, 'assigned_target': self.assigned_target})
                return True, self.win_reward, info

        if (abs(float(a_pos[0])) > self.arena_x_half
                or float(a_pos[1]) < self.arena_y_min
                or float(a_pos[1]) > self.arena_y_max):
            info.update({'outcome': 'out_of_bounds', 'winner': 'defender'})
            return True, self.loss_reward, info

        if self.t >= self.max_steps:
            info.update({'outcome': 'timeout', 'winner': 'defender'})
            return True, self.loss_reward, info

        return False, 0.0, info


class DefenderRLEnv(_BASE_CLASS):
    """Defender-perspective wrapper. Defender is RL-controlled; attacker is a
    fixed policy (loaded RL model, or any object with .predict(obs)).

    Obs is defender-centric (16-dim): defender state + attacker state + goal
    vectors relative to defender. No target_bit (defender does NOT know the
    assigned target). No mirror (defender has no per-target symmetry).
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Optional[dict] = None, attacker_policy=None, seed: int = 0):
        if _HAS_GYMNASIUM:
            super().__init__()
        self.inner = SplitDecisionEnv(cfg=cfg, seed=seed)
        self._defender_proxy = _InjectedDefender()
        self.inner.set_defender(self._defender_proxy)
        self.attacker_policy = attacker_policy
        if _HAS_GYMNASIUM:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

    def set_attacker_policy(self, attacker_policy):
        self.attacker_policy = attacker_policy

    def _defender_obs(self) -> np.ndarray:
        env = self.inner
        ax, ay, atheta, av, aomega = env.attacker_state
        dx, dy, dtheta, dv, domega = env.defender_state
        return np.array([
            dx, dy, np.sin(dtheta), np.cos(dtheta), dv, domega,
            ax, ay, np.sin(atheta), np.cos(atheta), av, aomega,
            env.goal_A[0] - dx, env.goal_A[1] - dy,
            env.goal_B[0] - dx, env.goal_B[1] - dy,
        ], dtype=np.float32)

    @property
    def history(self):
        return self.inner.history

    @property
    def goal_A(self):
        return self.inner.goal_A

    @property
    def goal_B(self):
        return self.inner.goal_B

    @property
    def attacker_state(self):
        return self.inner.attacker_state

    @property
    def defender_state(self):
        return self.inner.defender_state

    @property
    def assigned_target(self):
        return self.inner.assigned_target

    @property
    def dt(self):
        return self.inner.dt

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.inner.reset(seed=seed)
        return self._defender_obs(), {}

    def step(self, defender_action):
        if self.attacker_policy is None:
            raise RuntimeError("call set_attacker_policy(policy) before step()")
        self._defender_proxy.next_action = (
            float(defender_action[0]), float(defender_action[1]))
        attacker_obs = self.inner._get_obs()
        if hasattr(self.attacker_policy, 'predict'):
            attacker_action, _ = self.attacker_policy.predict(
                attacker_obs, deterministic=True)
        elif callable(self.attacker_policy):
            attacker_action = self.attacker_policy(attacker_obs)
        else:
            raise TypeError(f"attacker_policy must be sb3 model or callable: {self.attacker_policy}")
        _, attacker_reward, terminated, truncated, info = self.inner.step(attacker_action)
        defender_reward = -float(attacker_reward)
        return self._defender_obs(), defender_reward, terminated, truncated, info
