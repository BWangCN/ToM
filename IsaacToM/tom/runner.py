"""ToM strategy loop. For each configured perspective (evader / defender),
runs L1 → L2 → tactic sequentially. All inputs/outputs are logged as JSON for
later visualization and fine-tuning data curation (same pattern as AR-LLMs
response_history)."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .vlm_client import VLMClient
from .prompts.common import format_state_block
from .prompts.l1_intent import SYSTEM_L1, USER_L1_TEMPLATE, SCHEMA_L1
from .prompts.l2_belief import SYSTEM_L2, USER_L2_TEMPLATE, SCHEMA_L2
from .prompts.tactic import (
    SYSTEM_TACTIC, USER_TACTIC_TEMPLATE, SCHEMA_TACTIC,
    EVADER_TACTICS, DEFENDER_TACTICS,
)
from .prompts.waypoint import (
    SYSTEM_WAYPOINT, USER_WAYPOINT_TEMPLATE, SCHEMA_WAYPOINT,
)

try:
    from control.base import HighLevelTactic
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from control.base import HighLevelTactic


class ToMRunner:
    def __init__(self, cfg: dict, goal_a: np.ndarray, goal_b: np.ndarray):
        self.cfg = cfg
        self.tom_cfg = cfg["tom"]
        self.perspectives: List[str] = list(self.tom_cfg["perspectives"])
        self.goal_a = np.asarray(goal_a)
        self.goal_b = np.asarray(goal_b)

        self.client = VLMClient(cfg)
        self.output_mode = str(self.tom_cfg.get("output_mode", "tactic")).lower()
        if self.output_mode not in ("tactic", "waypoint"):
            raise ValueError(f"tom.output_mode must be 'tactic' or 'waypoint', got {self.output_mode!r}")
        self.waypoint_max_horizon = float(self.tom_cfg.get("waypoint_max_horizon", 6.0))
        self.log_dir = Path(self.tom_cfg["log_dir"])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"tom_session_{ts}.jsonl"
        self._log_fp = open(self.log_path, "w", encoding="utf-8")
        self._history: Dict[str, List[dict]] = {p: [] for p in self.perspectives}

    # ------------------------------------------------------------------
    def reason_step(self, step: int, state: dict,
                    image_path: Optional[str], image_rgb) -> Dict[str, HighLevelTactic]:
        tactics_out: Dict[str, HighLevelTactic] = {}
        image_list = [image_path] if image_path else []
        for role in self.perspectives:
            tac = self._reason_one(step=step, self_role=role, state=state, image_paths=image_list)
            tactics_out[role] = tac
        return tactics_out

    def _reason_one(self, step: int, self_role: str, state: dict,
                    image_paths: List[str]) -> HighLevelTactic:
        opp_role = "defender" if self_role == "evader" else "evader"
        state_block = format_state_block(state, self_role, self._history[self_role])
        state_json = json.dumps(state_block, indent=2)

        # --- L1 ---
        user_l1 = USER_L1_TEMPLATE.format(state_json=state_json,
                                          self_role=self_role, opp_role=opp_role)
        l1 = self.client.query(SYSTEM_L1, user_l1, image_paths=image_paths,
                               response_schema=SCHEMA_L1).parsed

        # --- L2 ---
        user_l2 = USER_L2_TEMPLATE.format(state_json=state_json,
                                          l1_json=json.dumps(l1, indent=2),
                                          self_role=self_role)
        l2 = self.client.query(SYSTEM_L2, user_l2, image_paths=image_paths,
                               response_schema=SCHEMA_L2).parsed

        # --- Decision layer: vocabulary tactic OR free waypoint ---
        if self.output_mode == "waypoint":
            return self._decide_waypoint(step, self_role, state_block, state_json,
                                         l1, l2, image_paths, state)
        return self._decide_tactic(step, self_role, state_block, state_json,
                                   l1, l2, image_paths)

    # ------------------------------------------------------------------
    def _decide_tactic(self, step, self_role, state_block, state_json,
                       l1, l2, image_paths) -> HighLevelTactic:
        tactic_list = EVADER_TACTICS if self_role == "evader" else DEFENDER_TACTICS
        user_tac = USER_TACTIC_TEMPLATE.format(
            state_json=state_json,
            l1_json=json.dumps(l1, indent=2),
            l2_json=json.dumps(l2, indent=2),
            self_role=self_role,
            tactic_list=json.dumps(tactic_list),
        )
        tac_resp = self.client.query(SYSTEM_TACTIC, user_tac, image_paths=image_paths,
                                     response_schema=SCHEMA_TACTIC).parsed

        label = tac_resp.get("tactic", "go_straight" if self_role == "evader" else "wait_center")
        target_goal = tac_resp.get("target_goal")
        if label not in tactic_list:
            label = tactic_list[0]

        hlt = HighLevelTactic(
            role=self_role, label=label, target_goal=target_goal,
            confidence=float(tac_resp.get("confidence", 0.5)),
            rationale=tac_resp.get("rationale", ""),
            extra={"l1": l1, "l2": l2},
        )

        self._history[self_role].append({
            "step": step, "label": label, "target_goal": target_goal,
            "l1_intent": l1.get("opponent_intent"),
            "l2_belief_p_A": (l2.get("opponent_belief") or {}).get("p_goal_A"),
            "l2_belief_p_B": (l2.get("opponent_belief") or {}).get("p_goal_B"),
        })
        self._write_log({
            "step": step, "self_role": self_role, "state": state_block,
            "image_path": image_paths[0] if image_paths else None,
            "l1": l1, "l2": l2, "tactic": tac_resp,
            "ts": dt.datetime.now().isoformat(),
        })
        return hlt

    # ------------------------------------------------------------------
    def _decide_waypoint(self, step, self_role, state_block, state_json,
                         l1, l2, image_paths, state) -> HighLevelTactic:
        """VLM picks a free (x, y) waypoint instead of a tactic-vocabulary label.
        Output is clamped to arena bounds and to a max distance from the agent
        before being handed to the goto_waypoint executor in rule_based.py.
        """
        arena = self.cfg["arena"]
        x_max = arena["size_x"] / 2.0
        x_min = -x_max
        y_min, y_max = 0.0, float(arena["size_y"])
        self_pos = state[self_role]["pos_xy"]
        self_pos_rounded = [round(self_pos[0], 2), round(self_pos[1], 2)]

        user_wp = USER_WAYPOINT_TEMPLATE.format(
            state_json=state_json,
            l1_json=json.dumps(l1, indent=2),
            l2_json=json.dumps(l2, indent=2),
            self_role=self_role,
            self_pos=self_pos_rounded,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            goal_a=state_block["goal_A_xy"],
            goal_b=state_block["goal_B_xy"],
            max_horizon=self.waypoint_max_horizon,
        )
        wp_resp = self.client.query(SYSTEM_WAYPOINT, user_wp, image_paths=image_paths,
                                    response_schema=SCHEMA_WAYPOINT).parsed

        # --- parse + sanitize target ---
        raw_target = wp_resp.get("target_xy")
        if isinstance(raw_target, list) and len(raw_target) == 2:
            try:
                tx, ty = float(raw_target[0]), float(raw_target[1])
            except (TypeError, ValueError):
                tx, ty = self_pos[0], self_pos[1]
        else:
            tx, ty = self_pos[0], self_pos[1]

        # Clamp to arena bounds first.
        tx = max(x_min, min(x_max, tx))
        ty = max(y_min, min(y_max, ty))

        # Clamp horizon: if the VLM tried to teleport far away, project the
        # waypoint back onto a circle of radius max_horizon around the agent.
        dx, dy = tx - self_pos[0], ty - self_pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        clamped_horizon = False
        if dist > self.waypoint_max_horizon and dist > 1e-6:
            scale = self.waypoint_max_horizon / dist
            tx = self_pos[0] + dx * scale
            ty = self_pos[1] + dy * scale
            clamped_horizon = True

        speed_frac = wp_resp.get("speed_frac", 1.0)
        try:
            speed_frac = float(speed_frac)
        except (TypeError, ValueError):
            speed_frac = 1.0
        speed_frac = max(0.0, min(1.0, speed_frac))

        phase = wp_resp.get("phase", "")
        rationale = wp_resp.get("rationale", "")

        hlt = HighLevelTactic(
            role=self_role,
            label="goto_waypoint",
            target_goal=None,
            waypoint=(tx, ty),
            confidence=1.0,
            rationale=rationale,
            extra={
                "l1": l1, "l2": l2,
                "speed_frac": speed_frac,
                "phase": phase,
                "raw_target_xy": raw_target,
                "clamped_horizon": clamped_horizon,
            },
        )

        self._history[self_role].append({
            "step": step,
            "label": "goto_waypoint",
            "waypoint": [round(tx, 2), round(ty, 2)],
            "speed_frac": round(speed_frac, 2),
            "phase": phase,
            "l1_intent": l1.get("opponent_intent"),
            "l2_belief_p_A": (l2.get("opponent_belief") or {}).get("p_goal_A"),
            "l2_belief_p_B": (l2.get("opponent_belief") or {}).get("p_goal_B"),
        })
        self._write_log({
            "step": step, "self_role": self_role, "state": state_block,
            "image_path": image_paths[0] if image_paths else None,
            "l1": l1, "l2": l2,
            "waypoint": {
                "target_xy": [tx, ty], "speed_frac": speed_frac,
                "phase": phase, "rationale": rationale,
                "raw_target_xy": raw_target, "clamped_horizon": clamped_horizon,
            },
            "ts": dt.datetime.now().isoformat(),
        })
        print(f"[tom] {self_role:8s} -> waypoint=({tx:+.2f},{ty:+.2f}) "
              f"speed={speed_frac:.2f} phase={phase}"
              + ("  [clamped]" if clamped_horizon else ""))
        return hlt

    # ------------------------------------------------------------------
    def _write_log(self, entry: dict) -> None:
        self._log_fp.write(json.dumps(entry) + "\n")
        self._log_fp.flush()

    def close(self, winner: Optional[str] = None):
        self._write_log({"event": "session_end", "winner": winner,
                         "ts": dt.datetime.now().isoformat()})
        self._log_fp.close()
