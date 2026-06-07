"""Controller interface. All controllers (rule-based, RL, VLM-direct) share this
contract so the runner is agnostic to implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class HighLevelTactic:
    """Output of the strategy layer (ToM / policy manager). Execution layer
    interprets this into wheel commands."""
    role: str                       # "evader" | "defender"
    label: str                      # e.g. shape_toward_A, exploit_swerve_B, wait_center, commit_A
    target_goal: Optional[str] = None
    confidence: float = 1.0
    rationale: str = ""             # VLM reasoning chain (for logging / fine-tuning)
    extra: Dict = field(default_factory=dict)
    waypoint: Optional[Tuple[float, float]] = None   # user go-to-point override (x, y) in world frame


class BaseController:
    def __init__(self, role: str, cfg: dict):
        self.role = role
        self.cfg = cfg
        agent_cfg = cfg["agents"][role]
        self.max_speed = agent_cfg["max_speed"]
        self.max_omega = agent_cfg["max_omega"]

    def step(self, tactic: HighLevelTactic, self_state: dict,
             other_state: dict, goals: dict) -> Tuple[float, float]:
        """Return (left_wheel_vel, right_wheel_vel) in rad/s."""
        raise NotImplementedError

    # --- differential drive utility --------------------------------------
    # JetBot's wheel motors saturate well below what `(max_speed, max_omega)`
    # in YAML implies. Empirically, at ~1.7 m/s commanded, both wheels clip
    # at the same value → differential collapses, the agent drives straight
    # without turning. The fix is to scale (v_l, v_r) jointly so the worst-
    # offending wheel sits at WHEEL_VEL_CAP, preserving the v:omega ratio.
    WHEEL_VEL_CAP_RAD_S = 11.0

    @staticmethod
    def diff_drive(v: float, omega: float, wheel_radius: float = 0.0325,
                   wheel_base: float = 0.1185) -> Tuple[float, float]:
        """Convert (linear_vel m/s, yaw_rate rad/s) → (left, right) wheel rad/s.
        Defaults match JetBot. Override for other robots.

        Joint-rate-aware: rescales both wheels proportionally if either would
        exceed BaseController.WHEEL_VEL_CAP_RAD_S, so heading control is not
        lost when the strategy layer asks for more speed than the motors can
        deliver. The agent just travels slower than commanded."""
        v_l = (v - omega * wheel_base / 2.0) / wheel_radius
        v_r = (v + omega * wheel_base / 2.0) / wheel_radius
        cap = BaseController.WHEEL_VEL_CAP_RAD_S
        peak = max(abs(v_l), abs(v_r))
        if peak > cap and peak > 1e-9:
            s = cap / peak
            v_l *= s
            v_r *= s
        return v_l, v_r
