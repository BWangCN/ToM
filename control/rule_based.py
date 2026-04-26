"""Rule-based controllers used as the default execution layer during PoC.

The controller maps a HighLevelTactic into (v, omega) and then into wheel
velocities via differential drive kinematics. The tactic vocabulary matches the
JSON we ask the VLM to emit, so this file also serves as the reference spec.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .base import BaseController, HighLevelTactic


def _angle_to(target_xy: np.ndarray, self_xy: np.ndarray) -> float:
    d = target_xy - self_xy
    return math.atan2(d[1], d[0])


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


class RuleBasedController(BaseController):
    EVADER_TACTICS = {"go_straight", "shape_toward_A", "shape_toward_B",
                      "exploit_swerve_A", "exploit_swerve_B", "slow_observe"}
    DEFENDER_TACTICS = {"wait_center", "track_evader",
                        "commit_A", "commit_B", "hedge"}

    def __init__(self, role, cfg):
        super().__init__(role, cfg)
        self._kp_heading = 2.0       # gain for heading error → yaw rate
        self._cruise_frac = 0.85     # fraction of max_speed in normal driving
        self._shape_lateral_bias = 0.35  # how much a shape_* tactic pulls off-axis

    # ------------------------------------------------------------------
    def step(self, tactic: HighLevelTactic, self_state, other_state, goals) -> Tuple[float, float]:
        self_xy = np.asarray(self_state["pos_xy"])
        heading = self_state["heading_rad"]

        if self.role == "evader":
            target_xy, speed_frac = self._evader_target(tactic, self_xy, goals)
        else:
            target_xy, speed_frac = self._defender_target(tactic, self_xy,
                                                          np.asarray(other_state["pos_xy"]),
                                                          goals)

        desired_heading = _angle_to(target_xy, self_xy)
        heading_err = _wrap(desired_heading - heading)
        omega = float(np.clip(self._kp_heading * heading_err, -self.max_omega, self.max_omega))

        # Slow down when we need to turn hard (prevents overshoot + models commitment cost).
        turn_penalty = max(0.0, 1.0 - abs(heading_err) / math.pi)
        v = self.max_speed * self._cruise_frac * speed_frac * turn_penalty

        return self.diff_drive(v, omega)

    # ------------------------------------------------------------------
    def _evader_target(self, tactic, self_xy, goals) -> Tuple[np.ndarray, float]:
        gA = np.asarray(goals["A"])
        gB = np.asarray(goals["B"])
        label = tactic.label if tactic.label in self.EVADER_TACTICS else "go_straight"

        if label == "go_straight":
            target = gB if tactic.target_goal == "B" else gA
            return target, 1.0
        if label == "slow_observe":
            return self_xy + np.array([0.0, 1.0]), 0.3
        if label.startswith("shape_toward_"):
            goal = label.split("_")[-1]
            anchor = gA if goal == "A" else gB
            # pull toward the anchor but keep the option open: aim 35% of the way
            target = self_xy + self._shape_lateral_bias * (anchor - self_xy) + np.array([0.0, 4.0])
            return target, 0.9
        if label.startswith("exploit_swerve_"):
            goal = label.split("_")[-1]
            return (gA if goal == "A" else gB), 1.0
        return gA, 0.8

    def _defender_target(self, tactic, self_xy, evader_xy, goals) -> Tuple[np.ndarray, float]:
        gA = np.asarray(goals["A"])
        gB = np.asarray(goals["B"])
        label = tactic.label if tactic.label in self.DEFENDER_TACTICS else "wait_center"

        if label == "wait_center":
            return self_xy + np.array([0.0, -0.5]), 0.05  # almost stationary
        if label == "track_evader":
            # Stay on the line between evader and arena top
            target = np.array([evader_xy[0] * 0.8, self_xy[1]])
            return target, 0.4
        if label == "commit_A":
            return gA, 1.0
        if label == "commit_B":
            return gB, 1.0
        if label == "hedge":
            midpoint = 0.5 * (gA + gB)
            return np.array([midpoint[0] + 0.4 * (evader_xy[0] - midpoint[0]), midpoint[1]]), 0.7
        return self_xy, 0.0
