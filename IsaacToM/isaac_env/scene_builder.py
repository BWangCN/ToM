"""Build the Split Decision arena: ground plane, two goal markers, two wheeled cars.

Must be imported only AFTER SimulationApp is alive.

Uses the low-level ``isaacsim.core.api.robots.Robot`` + ``add_reference_to_stage``
path (articulation only) instead of ``WheeledRobot``. Rationale: the
``WheeledRobot`` class lives in the ``isaacsim.robot.wheeled_robots`` extension
which pulls in ``isaacsim.gui.components`` → ``omni.kit.viewport.window`` → the
hydra viewport widget that segfaults on this hybrid-GPU machine. We don't need
the wrapper; driving wheel DOFs with joint velocity commands is two lines.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


ASSET_PATHS = {
    "jetbot": "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
    "carter": "/Isaac/Robots/NVIDIA/Carter/nova_carter.usd",
    "kaya":   "/Isaac/Robots/NVIDIA/Kaya/kaya.usd",
}

WHEEL_DOFS = {
    "jetbot": ["left_wheel_joint", "right_wheel_joint"],
    "carter": ["joint_wheel_left", "joint_wheel_right"],
    "kaya":   ["axle_0_joint", "axle_1_joint", "axle_2_joint"],
}


@dataclass
class AgentHandle:
    name: str                # "evader" | "defender"
    prim_path: str
    robot: object            # isaacsim.core.api.robots.Robot (SingleArticulation)
    wheel_dofs: Tuple[str, ...]
    wheel_dof_idx: Tuple[int, ...]  # resolved at initialize-time, filled post-reset
    max_speed: float
    max_omega: float


def build_scene(world, cfg: dict) -> Tuple[AgentHandle, AgentHandle, np.ndarray, np.ndarray]:
    """Populate the world and return (evader, defender, goal_a, goal_b)."""
    from isaacsim.core.api.objects import VisualCylinder, GroundPlane
    from isaacsim.core.api.robots import Robot
    from isaacsim.core.utils.stage import add_reference_to_stage
    from isaacsim.storage.native import get_assets_root_path

    world.scene.add(GroundPlane(prim_path="/World/Ground",
                                size=max(cfg["arena"]["size_x"], cfg["arena"]["size_y"])))

    ga = np.array(cfg["arena"]["goal_a_pos"] + [0.05], dtype=np.float32)
    gb = np.array(cfg["arena"]["goal_b_pos"] + [0.05], dtype=np.float32)
    world.scene.add(VisualCylinder(
        prim_path="/World/GoalA", name="goal_a",
        position=ga, radius=cfg["arena"]["goal_radius"], height=0.1,
        color=np.array([0.2, 0.9, 0.2])))
    world.scene.add(VisualCylinder(
        prim_path="/World/GoalB", name="goal_b",
        position=gb, radius=cfg["arena"]["goal_radius"], height=0.1,
        color=np.array([0.9, 0.6, 0.1])))

    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Could not resolve Isaac Sim assets root. Check local asset server.")

    agents = {}
    for name, spec in cfg["agents"].items():
        asset_key = spec["asset"]
        usd_path = assets_root + ASSET_PATHS[asset_key]
        wheel_dofs = tuple(WHEEL_DOFS[asset_key])
        pos = np.array(spec["start_pos"] + [0.0], dtype=np.float32)
        heading_rad = np.deg2rad(spec["start_heading_deg"])
        orientation = np.array([np.cos(heading_rad / 2), 0.0, 0.0, np.sin(heading_rad / 2)],
                               dtype=np.float32)
        prim_path = f"/World/{name.capitalize()}"
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)
        robot = world.scene.add(Robot(
            prim_path=prim_path, name=name,
            position=pos, orientation=orientation,
        ))
        agents[name] = AgentHandle(
            name=name, prim_path=prim_path, robot=robot,
            wheel_dofs=wheel_dofs, wheel_dof_idx=(),
            max_speed=spec["max_speed"], max_omega=spec["max_omega"],
        )

    return agents["evader"], agents["defender"], ga[:2], gb[:2]


def resolve_wheel_indices(handle: AgentHandle) -> None:
    """Call AFTER world.reset() so articulation dof names are populated."""
    dof_names = list(handle.robot.dof_names)
    idx = []
    for n in handle.wheel_dofs:
        if n not in dof_names:
            raise RuntimeError(f"[scene] wheel dof '{n}' not found on {handle.name}. "
                               f"Available dofs: {dof_names}")
        idx.append(dof_names.index(n))
    handle.wheel_dof_idx = tuple(idx)
