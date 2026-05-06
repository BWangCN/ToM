"""Drop-in replacement for ToMRunner that uses scripted waypoints instead of
calling a real VLM. Produces the same HighLevelTactic(label="goto_waypoint",
waypoint=(x, y), extra={"speed_frac": s, "phase": p}) values that the real
GPT-4o stack would produce, so the rest of the runner is unchanged.

Use case: rollout the Isaac Sim physics + matplotlib schematic with no
OPENAI_API_KEY and no compute spent on inference, just to inspect that the
waypoint navigator actually drives the cars in the way we expect.

Same public interface as ToMRunner:
    runner = MockOracleRunner(cfg, scenario="feint_left_to_right",
                              goal_a=goal_a, goal_b=goal_b)
    tactics = runner.reason_step(step, state, image_path, image_rgb)
    runner.close(winner=...)
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from control.base import HighLevelTactic
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from control.base import HighLevelTactic


# ---------------------------------------------------------------------------
# Scripted oracles. Mirrors viz/preset_demo.py so the trajectories produced
# in Isaac Sim match the standalone 2D demo (modulo physics differences).
# ---------------------------------------------------------------------------

def _goal_xy(arena: dict, label: str) -> Tuple[float, float]:
    return tuple(arena[f"goal_{label.lower()}_pos"])


class _FeintOracle:
    """Position-driven (not time-driven) so the same scenario produces the
    same trajectory shape regardless of how fast the physics actually runs.
    This matters in Isaac Sim where the JetBot tops out at ~0.4 m/s no matter
    what max_speed says.

    Evader phase machine:
        shaping     y_evader < y_exploit       drift toward fake goal
        exploiting  y_exploit ≤ y < y_commit   swerve hard toward real goal
        committing  y ≥ y_commit               head straight at real goal

    Defender phase machine:
        holding     y_evader < y_lock          stay near (0, 15)
        committing  y_evader ≥ y_lock,         drive at the goal it locked to
                    evader still on locked side
        recovering  evader has crossed midline drive at the OPPOSITE goal,
                                               but commitment cost (slow turn)
                                               makes it usually too late
    """

    def __init__(self, arena: dict, fake_goal: str = "A", real_goal: str = "B",
                 y_lock: float = 4.0, y_exploit: float = 4.5, y_commit: float = 6.5):
        self.arena = arena
        self.fake = fake_goal
        self.real = real_goal
        self.y_lock = y_lock          # defender locks its commit at this evader y
        self.y_exploit = y_exploit    # evader switches from shaping to exploit
        self.y_commit = y_commit      # evader stops swerving and drives at real goal
        self._defender_locked_to: Optional[str] = None

    def evader(self, state: dict, t_sim: float):
        ex, ey = state["evader"]["pos_xy"]
        fake_x = _goal_xy(self.arena, self.fake)[0]
        real_x = _goal_xy(self.arena, self.real)[0]
        if ey < self.y_exploit:
            tx = ex + 0.5 * (fake_x - ex)
            ty = ey + 3.5
            return (tx, ty), 0.85, "shaping"
        if ey < self.y_commit:
            tx = ex + 0.7 * (real_x - ex)
            ty = ey + 3.0
            return (tx, ty), 1.0, "exploiting"
        return _goal_xy(self.arena, self.real), 1.0, "committing"

    def defender(self, state: dict, t_sim: float):
        ex, _ = state["evader"]["pos_xy"]
        ey = state["evader"]["pos_xy"][1]
        if ey < self.y_lock:
            return (0.0, 15.0), 0.30, "holding"
        if self._defender_locked_to is None:
            self._defender_locked_to = "A" if ex < 0 else "B"
        ev_actual = "A" if ex < 0 else "B"
        if ev_actual == self._defender_locked_to:
            return _goal_xy(self.arena, self._defender_locked_to), 1.0, "committing"
        return _goal_xy(self.arena, ev_actual), 1.0, "recovering"


# ---------------------------------------------------------------------------
# Same clamping logic as ToMRunner._decide_waypoint.
# ---------------------------------------------------------------------------

def _clamp_waypoint(target_xy: Tuple[float, float], self_xy: Tuple[float, float],
                    arena: dict, max_horizon: float
                    ) -> Tuple[Tuple[float, float], bool]:
    x_max = arena["size_x"] / 2.0
    x_min = -x_max
    y_min, y_max = 0.0, float(arena["size_y"])
    tx, ty = target_xy
    tx = max(x_min, min(x_max, tx))
    ty = max(y_min, min(y_max, ty))
    dx, dy = tx - self_xy[0], ty - self_xy[1]
    dist = math.hypot(dx, dy)
    if dist > max_horizon and dist > 1e-6:
        s = max_horizon / dist
        return (self_xy[0] + dx * s, self_xy[1] + dy * s), True
    return (tx, ty), False


# ---------------------------------------------------------------------------

class MockOracleRunner:
    """Same .reason_step / .close interface as ToMRunner. No VLM, no API key."""

    SCENARIOS = {
        "feint_left_to_right": dict(fake_goal="A", real_goal="B"),
        "feint_right_to_left": dict(fake_goal="B", real_goal="A"),
    }

    def __init__(self, cfg: dict, scenario: str = "feint_left_to_right",
                 goal_a=None, goal_b=None):
        if scenario not in self.SCENARIOS:
            raise ValueError(f"unknown mock scenario {scenario!r}; "
                             f"choices: {list(self.SCENARIOS)}")
        self.cfg = cfg
        self.arena = cfg["arena"]
        self.scenario = scenario
        self.max_horizon = float(cfg["tom"].get("waypoint_max_horizon", 6.0))
        params = self.SCENARIOS[scenario]
        self.oracle = _FeintOracle(self.arena, **params)

        log_dir = Path(cfg["tom"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"mock_{scenario}_{ts}.jsonl"
        self._log_fp = open(self.log_path, "w", encoding="utf-8")
        print(f"[mock] scenario={scenario} max_horizon={self.max_horizon}m  "
              f"log -> {self.log_path}")

    def reason_step(self, step: int, state: dict,
                    image_path: Optional[str] = None,
                    image_rgb=None) -> Dict[str, HighLevelTactic]:
        t_sim = float(state.get("t_sim", 0.0))
        out: Dict[str, HighLevelTactic] = {}
        for role in ("evader", "defender"):
            raw_target, speed_frac, phase = getattr(self.oracle, role)(state, t_sim)
            self_xy = state[role]["pos_xy"]
            (tx, ty), clamped = _clamp_waypoint(raw_target, self_xy,
                                                self.arena, self.max_horizon)

            tac = HighLevelTactic(
                role=role, label="goto_waypoint",
                target_goal=None, waypoint=(tx, ty),
                confidence=1.0,
                rationale=f"mock-oracle scenario={self.scenario} phase={phase}",
                extra={
                    "speed_frac": speed_frac,
                    "phase": phase,
                    "raw_target_xy": [float(raw_target[0]), float(raw_target[1])],
                    "clamped_horizon": clamped,
                    "mock": True,
                },
            )
            out[role] = tac

            entry = {
                "step": step, "t_sim": round(t_sim, 3), "self_role": role,
                "scenario": self.scenario, "mock": True,
                "self_pos": [round(self_xy[0], 3), round(self_xy[1], 3)],
                "raw_target_xy": [round(raw_target[0], 3), round(raw_target[1], 3)],
                "clamped_target_xy": [round(tx, 3), round(ty, 3)],
                "speed_frac": speed_frac,
                "phase": phase,
                "clamped_horizon": clamped,
            }
            self._log_fp.write(json.dumps(entry) + "\n")

            print(f"[mock] t={t_sim:5.2f} {role:8s} {phase:11s} -> "
                  f"({tx:+.2f},{ty:+.2f}) v={speed_frac:.2f}"
                  + ("  [clamp]" if clamped else ""))

        self._log_fp.flush()
        return out

    def close(self, winner: Optional[str] = None):
        self._log_fp.write(json.dumps({
            "event": "session_end", "winner": winner,
            "ts": dt.datetime.now().isoformat(),
        }) + "\n")
        self._log_fp.close()
        print(f"[mock] closed; winner={winner}")
