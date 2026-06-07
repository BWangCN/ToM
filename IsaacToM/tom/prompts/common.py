"""Shared scenario text + state formatting utilities.

Design mirrors AR-LLMs `prompt.py`: a static scenario block + a dynamically
formatted state block, all handed to the VLM as a single user message plus one
or more images. We keep text/state as plain dicts so they're easy to diff, log,
and later reuse as fine-tuning targets.
"""
from __future__ import annotations

import math
from typing import Dict


SCENARIO_DESCRIPTION = (
    "You are analyzing a two-player adversarial driving game called \"Split Decision\". "
    "Two small wheeled robots operate on a flat 24 m × 24 m arena. There are two goal "
    "regions at the top of the arena: Goal A on the left (green cylinder) and Goal B on "
    "the right (orange cylinder). The EVADER starts near the bottom and must reach either "
    "goal. The DEFENDER starts near the middle and must physically arrive at the evader's "
    "chosen goal first to block. The DEFENDER is FASTER but has a LARGER turning radius, so "
    "once it commits to a direction it cannot easily reverse — this makes feints strategically "
    "valuable for the evader."
)

ARENA_CONVENTIONS = (
    "Top-down image conventions: +X is right (toward Goal B), -X is left (toward Goal A), "
    "+Y is up in world coordinates (toward the goals), -Y is down (toward the evader's start). "
    "In the rendered top-down image the scene is viewed from above with north (+Y) at the top."
)


def format_agent_state(tag: str, state: Dict) -> Dict:
    x, y = state["pos_xy"]
    return {
        "role": tag,
        "pos_xy": [round(x, 2), round(y, 2)],
        "heading_deg": round(state["heading_deg"], 1),
        "speed_m_s": round(state["speed"], 2),
        "yaw_rate_rad_s": round(state["yaw_rate"], 2),
    }


def format_state_block(state: Dict, self_role: str, history: list) -> Dict:
    other_role = "defender" if self_role == "evader" else "evader"
    return {
        "scenario_summary": SCENARIO_DESCRIPTION,
        "arena_conventions": ARENA_CONVENTIONS,
        "self": format_agent_state(self_role, state[self_role]),
        "opponent": format_agent_state(other_role, state[other_role]),
        "goal_A_xy": [round(v, 2) for v in state["goal_a"]],
        "goal_B_xy": [round(v, 2) for v in state["goal_b"]],
        "sim_time_sec": round(state["t_sim"], 2),
        "recent_decisions": history[-5:],
    }
