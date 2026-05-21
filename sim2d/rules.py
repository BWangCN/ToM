"""Rule-based defenders (L0, L1) and scripted attackers (straight, feint).

All policies output normalized (v_cmd, omega_cmd) in [-1, 1]^2 so that they're
interchangeable with RL policies through the env's action interface.
"""
from typing import Tuple

import numpy as np


def wrap_pi(angle: float) -> float:
    return ((angle + np.pi) % (2 * np.pi)) - np.pi


def goto_normalized(
    self_state: np.ndarray,
    target_xy: Tuple[float, float],
    params: dict,
    K_omega: float = 3.0,
) -> Tuple[float, float]:
    """Heading-P with speed scale-down on heading error."""
    x, y, theta = float(self_state[0]), float(self_state[1]), float(self_state[2])
    dx = target_xy[0] - x
    dy = target_xy[1] - y
    dist = float(np.hypot(dx, dy))

    if dist < 0.2:
        return (0.0, 0.0)

    desired_h = float(np.arctan2(dy, dx))
    h_err = wrap_pi(desired_h - theta)
    abs_err = abs(h_err)

    omega_cmd_phys = float(np.clip(K_omega * h_err, -params['omega_max'], +params['omega_max']))

    # Drive at v_max so Ackermann coupling does not throttle omega.
    # Only slow down for heading errors > 90 deg, where moving forward
    # is actively counterproductive while turning.
    if abs_err < np.pi / 2.0:
        speed_scale = 1.0
    else:
        speed_scale = 0.5
    v_cmd_phys = speed_scale * params['v_max']

    return (v_cmd_phys / params['v_max'], omega_cmd_phys / params['omega_max'])


class L0Defender:
    """Shadow attacker's x-coordinate. Has no goal prediction."""

    def __init__(self, params: dict):
        self.params = params

    def reset(self):
        pass

    def step(self, self_state, attacker_state, goal_A, goal_B):
        target_xy = (float(attacker_state[0]), float(self_state[1]))
        return goto_normalized(self_state, target_xy, self.params)


class L1Defender:
    """Constant-velocity goal predictor with switch hysteresis.

    Stage C's frozen opponent. Strategy knobs (horizon_s, switch_dwell) are the
    commitment-cost sweep axes for paper figure 3.
    """

    def __init__(self, params: dict, horizon_s: float = 1.5, switch_dwell: int = 6,
                 randomize_initial_target: bool = False):
        self.params = params
        self.horizon_s = horizon_s
        self.switch_dwell = switch_dwell
        self.randomize_initial_target = randomize_initial_target
        self._rng = np.random.default_rng()
        self.last_target = "A"
        self.switch_counter = 0

    def reset(self):
        self.switch_counter = 0
        if self.randomize_initial_target:
            self.last_target = "A" if self._rng.random() < 0.5 else "B"
        else:
            self.last_target = "A"

    def step(self, self_state, attacker_state, goal_A, goal_B):
        ax = float(attacker_state[0])
        ay = float(attacker_state[1])
        atheta = float(attacker_state[2])
        av = float(attacker_state[3])
        vx = av * np.cos(atheta)
        vy = av * np.sin(atheta)

        pred_x = ax + vx * self.horizon_s
        pred_y = ay + vy * self.horizon_s
        d_A = float(np.hypot(pred_x - goal_A[0], pred_y - goal_A[1]))
        d_B = float(np.hypot(pred_x - goal_B[0], pred_y - goal_B[1]))
        predicted = "A" if d_A <= d_B else "B"

        if predicted == self.last_target:
            self.switch_counter = 0
        else:
            self.switch_counter += 1
            if self.switch_counter >= self.switch_dwell:
                self.last_target = predicted
                self.switch_counter = 0

        target_xy = (float(goal_A[0]), float(goal_A[1])) if self.last_target == "A" \
                    else (float(goal_B[0]), float(goal_B[1]))
        return goto_normalized(self_state, target_xy, self.params)


class L2RuleDefender:
    """Patience + EMA-smoothed velocity prediction. Behaviorally 2nd-order:
    resists short shape phases of feint via smoothing and a y-threshold gate
    before committing to a goal.

    Key property: action != belief during the uncommitted phase (defender hedges
    near midline while uncommitted). This is the env property under which the
    paper's 2-ToM claim becomes rigorous (action and belief decouple).

    Knobs (tuned via sanity tests):
        commit_y_threshold: attacker must cross this y before commitment is allowed
        ema_alpha:          velocity smoothing rate (smaller = smoother / more patient)
        confidence_thresh:  ratio |d_A - d_B| / (d_A + d_B) gate before commit
        horizon_s:          prediction lookahead in seconds
    """

    def __init__(self, params: dict,
                 commit_y_threshold: float = 12.0,
                 ema_alpha: float = 0.1,
                 confidence_thresh: float = 0.3,
                 horizon_s: float = 1.0):
        self.params = params
        self.commit_y_threshold = commit_y_threshold
        self.ema_alpha = ema_alpha
        self.confidence_thresh = confidence_thresh
        self.horizon_s = horizon_s
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        self.last_target = None    # None = uncommitted (hedging)

    def reset(self):
        self.smoothed_vx = 0.0
        self.smoothed_vy = 0.0
        self.last_target = None

    def step(self, self_state, attacker_state, goal_A, goal_B):
        ax = float(attacker_state[0])
        ay = float(attacker_state[1])
        atheta = float(attacker_state[2])
        av = float(attacker_state[3])
        vx = av * np.cos(atheta)
        vy = av * np.sin(atheta)

        a = self.ema_alpha
        self.smoothed_vx = a * vx + (1.0 - a) * self.smoothed_vx
        self.smoothed_vy = a * vy + (1.0 - a) * self.smoothed_vy

        def _hedge():
            target_xy = (0.5 * ax, float(self_state[1]))
            self.last_target = None
            return goto_normalized(self_state, target_xy, self.params)

        if ay < self.commit_y_threshold:
            return _hedge()

        pred_x = ax + self.smoothed_vx * self.horizon_s
        pred_y = ay + self.smoothed_vy * self.horizon_s
        d_A = float(np.hypot(pred_x - goal_A[0], pred_y - goal_A[1]))
        d_B = float(np.hypot(pred_x - goal_B[0], pred_y - goal_B[1]))
        confidence = abs(d_A - d_B) / (d_A + d_B + 1e-6)

        if confidence < self.confidence_thresh:
            return _hedge()

        self.last_target = "A" if d_A < d_B else "B"
        target_xy = (float(goal_A[0]), float(goal_A[1])) if self.last_target == "A" \
                    else (float(goal_B[0]), float(goal_B[1]))
        return goto_normalized(self_state, target_xy, self.params)


class StraightAttacker:
    """Drives directly to a fixed goal."""

    def __init__(self, params: dict, target_goal: str = "B"):
        if target_goal not in ("A", "B"):
            raise ValueError(target_goal)
        self.params = params
        self.target_goal = target_goal

    def reset(self, assigned_target: str = None):
        if assigned_target is not None:
            self.target_goal = assigned_target

    def step(self, self_state, defender_state, goal_A, goal_B):
        target_xy = (float(goal_A[0]), float(goal_A[1])) if self.target_goal == "A" \
                    else (float(goal_B[0]), float(goal_B[1]))
        return goto_normalized(self_state, target_xy, self.params)


class FeintAttacker:
    """Phase 1: head to fake_goal. Phase 2 (t >= switch_time): head to real_goal."""

    def __init__(self, params: dict, fake_goal: str = "A", real_goal: str = "B",
                 switch_time: float = 2.4, dt: float = 1.0 / 20.0):
        if fake_goal == real_goal:
            raise ValueError("fake_goal and real_goal must differ")
        if fake_goal not in ("A", "B") or real_goal not in ("A", "B"):
            raise ValueError("goals must be 'A' or 'B'")
        self.params = params
        self.fake_goal = fake_goal
        self.real_goal = real_goal
        self.switch_time = switch_time
        self.dt = dt
        self.t = 0.0

    def reset(self, assigned_target: str = None):
        if assigned_target is not None:
            self.real_goal = assigned_target
            self.fake_goal = 'B' if assigned_target == 'A' else 'A'
        self.t = 0.0

    def step(self, self_state, defender_state, goal_A, goal_B):
        chosen = self.fake_goal if self.t < self.switch_time else self.real_goal
        target_xy = (float(goal_A[0]), float(goal_A[1])) if chosen == "A" \
                    else (float(goal_B[0]), float(goal_B[1]))
        self.t += self.dt
        return goto_normalized(self_state, target_xy, self.params)
