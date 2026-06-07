"""Bicycle dynamics with Ackermann coupling.

State = [x, y, theta, v, omega]   pose + last applied (v, omega)
Action = (v_cmd, omega_cmd) normalized in [-1, 1]^2
"""
from typing import Tuple

import numpy as np


def step_dynamics(
    state: np.ndarray,
    action_normalized: Tuple[float, float],
    params: dict,
    dt: float,
) -> np.ndarray:
    x, y, theta, _, _ = state
    v_cmd_norm, omega_cmd_norm = action_normalized

    v_cmd = float(np.clip(v_cmd_norm, -1.0, 1.0)) * params['v_max']
    omega_cmd = float(np.clip(omega_cmd_norm, -1.0, 1.0)) * params['omega_max']

    # Ackermann coupling: yaw rate bounded by speed -> can't pivot in place.
    # This is what makes the commitment cost physical.
    omega_limit = abs(v_cmd) * np.tan(params['delta_max']) / params['L']
    omega_effective = float(np.clip(omega_cmd, -omega_limit, +omega_limit))

    x_new = x + v_cmd * np.cos(theta) * dt
    y_new = y + v_cmd * np.sin(theta) * dt
    theta_new = theta + omega_effective * dt
    theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi

    return np.array([x_new, y_new, theta_new, v_cmd, omega_effective], dtype=np.float32)
