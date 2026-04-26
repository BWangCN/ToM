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
    @staticmethod
    def diff_drive(v: float, omega: float, wheel_radius: float = 0.0325,
                   wheel_base: float = 0.1185) -> Tuple[float, float]:
        """Convert (linear_vel m/s, yaw_rate rad/s) → (left, right) wheel rad/s.
        Defaults match JetBot. Override for other robots."""
        v_l = (v - omega * wheel_base / 2.0) / wheel_radius
        v_r = (v + omega * wheel_base / 2.0) / wheel_radius
        return v_l, v_r
