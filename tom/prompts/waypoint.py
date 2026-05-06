"""Free-form waypoint output. Replaces the fixed tactic vocabulary so the VLM
can express continuous spatial strategy (off-axis feints, holding lines, slow
observation) instead of being funnelled into a handful of stereotyped labels.

Runs as the third query after L1+L2. Output is consumed directly by
control/rule_based.py via the existing goto_waypoint path:
    HighLevelTactic(label="goto_waypoint", waypoint=(x, y), extra={"speed_frac": s})
"""
from __future__ import annotations

SYSTEM_WAYPOINT = (
    "You are the strategy layer for an embodied agent in a two-player driving game. "
    "You have just produced first-order and second-order Theory-of-Mind estimates of "
    "the opponent. Now choose the NEXT WAYPOINT — a single (x, y) point in arena "
    "world coordinates that the agent will drive toward over roughly the next 0.8 "
    "seconds. The execution layer will steer in a straight line toward this point at "
    "the speed you specify, then ask you again. Use the waypoint to ENACT strategy, "
    "not to repeat the goal: shape the opponent's belief, exploit a commitment, hold "
    "a defensive line, or close distance. Do NOT default to picking the literal goal "
    "every step — off-axis intermediate points are how feints, blocks, and pre-empts "
    "are made physical. Output strict JSON only."
)

USER_WAYPOINT_TEMPLATE = (
    "State (JSON):\n{state_json}\n\n"
    "Level-1 (your estimate of opponent's intent):\n{l1_json}\n\n"
    "Level-2 (opponent's belief about you):\n{l2_json}\n\n"
    "You are the {self_role}. Your current position is {self_pos}.\n"
    "Arena bounds: x ∈ [{x_min:.1f}, {x_max:.1f}], y ∈ [{y_min:.1f}, {y_max:.1f}].\n"
    "Goal A (left, green) is at {goal_a}. Goal B (right, orange) is at {goal_b}.\n\n"
    "Pick a waypoint within {max_horizon:.1f} m of your current position. Smaller "
    "moves (1-3 m) suit shaping or observing; larger moves (3-{max_horizon:.0f} m) "
    "suit committing. Coordinates are in METERS, world frame.\n\n"
    "Strategy hints for your role:\n"
    "  EVADER — during the SHAPING phase, place the waypoint biased toward the FAKE "
    "  goal so your motion looks committed there, even though you have not yet "
    "  decided. Watch the opponent's commitment_evidence about you (from L2). When "
    "  it is high AND the opponent's L1-predicted intent matches the fake goal, "
    "  switch the waypoint hard toward the REAL goal — this is the EXPLOIT phase. "
    "  A straight line at the goal is a 0th-order behaviour and almost always loses.\n"
    "  DEFENDER — when L2 belief is ambiguous, hold a centered waypoint near "
    "  (0, ~12-14) with small forward/backward steps so you preserve the option to "
    "  go either way. Only commit toward a goal when the evader's motion is "
    "  consistently aligned with one goal AND your L1 confidence is high. If L2 "
    "  suggests the evader is shaping you (their commitment_evidence is high but "
    "  their position contradicts the L1 inference), STAY CENTERED — do not bite.\n\n"
    "speed_frac controls cruise speed (0.0 = halt, 1.0 = max). Slowing down is a "
    "valid tactic — it gives you more time to read the opponent's motion."
)

SCHEMA_WAYPOINT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["target_xy", "speed_frac", "phase", "rationale"],
    "properties": {
        "target_xy": {
            "type": "array",
            "minItems": 2, "maxItems": 2,
            "items": {"type": "number"},
            "description": "World-frame target [x, y] in meters",
        },
        "speed_frac": {
            "type": "number",
            "minimum": 0.0, "maximum": 1.0,
            "description": "Cruise speed as fraction of max_speed",
        },
        "phase": {
            "type": "string",
            "enum": ["shaping", "exploiting", "committing", "holding", "observing"],
            "description": "Strategic phase label — for telemetry, does not change execution",
        },
        "rationale": {"type": "string"},
    },
}
