"""Trajectory feint detector.

is_feint(traj) = True iff attacker had a 'committed to goal X' phase of at
least min_dwell_steps, followed by a 'committed to goal Y' phase of at least
min_dwell_steps, with X != Y.
"""
from typing import List, Optional, Tuple

import numpy as np


def _heading_alignment_to_goal(
    positions: np.ndarray, headings: np.ndarray, goal: np.ndarray,
) -> np.ndarray:
    diffs = goal - positions
    dists = np.linalg.norm(diffs, axis=1)
    goal_dirs = diffs / np.maximum(dists[:, None], 1e-6)
    heading_vecs = np.stack([np.cos(headings), np.sin(headings)], axis=1)
    cos_angle = np.sum(goal_dirs * heading_vecs, axis=1)
    return np.arccos(np.clip(cos_angle, -1.0, 1.0))


def _find_runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def is_feint(
    states: List[np.ndarray],
    goal_A: np.ndarray,
    goal_B: np.ndarray,
    threshold_deg: float = 25.0,
    min_dwell_steps: int = 12,
) -> Tuple[bool, Optional[dict]]:
    if len(states) < 2 * min_dwell_steps:
        return False, None

    arr = np.array(states)
    positions = arr[:, :2]
    headings = arr[:, 2]
    threshold_rad = np.radians(threshold_deg)

    angles_A = _heading_alignment_to_goal(positions, headings, np.asarray(goal_A))
    angles_B = _heading_alignment_to_goal(positions, headings, np.asarray(goal_B))

    committed_A = angles_A < threshold_rad
    committed_B = angles_B < threshold_rad

    runs_A = _find_runs(committed_A, min_dwell_steps)
    runs_B = _find_runs(committed_B, min_dwell_steps)

    for ra in runs_A:
        for rb in runs_B:
            if rb[0] >= ra[1]:
                return True, {'order': 'A_to_B', 'A_run': ra, 'B_run': rb}
    for rb in runs_B:
        for ra in runs_A:
            if ra[0] >= rb[1]:
                return True, {'order': 'B_to_A', 'A_run': ra, 'B_run': rb}
    return False, None


def feint_strength(
    attacker_states: List[np.ndarray],
    defender_states: List[np.ndarray],
    actual_goal_xy: np.ndarray,
    window: int = 20,
) -> float:
    """Max over time of (avg distance from defender to attacker's actual goal)
    within a sliding window. Captures 'how badly defender was lured away'."""
    d_arr = np.array(defender_states)[:, :2]
    misposition = np.linalg.norm(d_arr - np.asarray(actual_goal_xy), axis=1)
    if len(misposition) < window:
        return float(misposition.mean())
    sliding = np.convolve(misposition, np.ones(window) / window, mode='valid')
    return float(sliding.max())
