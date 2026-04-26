"""Tactic selection prompt. Fuses L1 + L2 outputs into a concrete high-level
action label consumed by control/rule_based.py.

Evader vocabulary: mirrors the deception two-phase structure (shape then
exploit). Defender vocabulary: commit vs wait, with the "hedge" option tied to
commitment cost.
"""
from __future__ import annotations

SYSTEM_TACTIC = (
    "You are the strategy layer for a driving game agent. You already have first-order "
    "and second-order Theory-of-Mind estimates. Choose the next high-level tactic from the "
    "allowed list for your role. Do not invent new tactics. Output strict JSON only."
)

USER_TACTIC_TEMPLATE = (
    "State (JSON):\n{state_json}\n\n"
    "Level-1 (opponent intent):\n{l1_json}\n\n"
    "Level-2 (opponent's belief about me):\n{l2_json}\n\n"
    "You are the {self_role}. Pick the single best tactic from: {tactic_list}. "
    "If you are the evader: prefer shape_* while the opponent's belief_gap is still small; "
    "exploit_* only when opponent_belief has concentrated on the fake goal AND the "
    "opponent's commitment_evidence about YOU is high. "
    "If you are the defender: prefer track_evader or wait_center while belief is ambiguous; "
    "commit_* only when your level-1 confidence is high and level-2 says the evader is not "
    "setting you up."
)

EVADER_TACTICS = [
    "go_straight", "shape_toward_A", "shape_toward_B",
    "exploit_swerve_A", "exploit_swerve_B", "slow_observe",
]
DEFENDER_TACTICS = [
    "wait_center", "track_evader", "commit_A", "commit_B", "hedge",
]

SCHEMA_TACTIC = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tactic", "target_goal", "confidence", "rationale"],
    "properties": {
        "tactic": {"type": "string"},
        "target_goal": {"type": ["string", "null"], "enum": ["A", "B", None]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "rationale": {"type": "string"},
    },
}
