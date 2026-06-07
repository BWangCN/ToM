"""Coordinate frame conversions between arena world frame and robot body frame.

World frame:  origin at arena center, +X right, +Y toward goals.
Body frame:   origin at robot, +X forward (along heading), +Y left.
"""
from __future__ import annotations

import math
from typing import Tuple


def body_to_world(body_xy: Tuple[float, float],
                  robot_xy: Tuple[float, float],
                  robot_heading_rad: float) -> Tuple[float, float]:
    """Transform a point given in robot body frame into arena world frame."""
    bx, by = body_xy
    c, s = math.cos(robot_heading_rad), math.sin(robot_heading_rad)
    wx = robot_xy[0] + c * bx - s * by
    wy = robot_xy[1] + s * bx + c * by
    return wx, wy


def world_to_body(world_xy: Tuple[float, float],
                  robot_xy: Tuple[float, float],
                  robot_heading_rad: float) -> Tuple[float, float]:
    """Transform a point given in arena world frame into robot body frame."""
    dx = world_xy[0] - robot_xy[0]
    dy = world_xy[1] - robot_xy[1]
    c, s = math.cos(robot_heading_rad), math.sin(robot_heading_rad)
    bx =  c * dx + s * dy
    by = -s * dx + c * dy
    return bx, by
