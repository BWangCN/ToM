"""Standalone Isaac Sim runner for the Split Decision arena.

Launches SimulationApp, builds the scene, spins the physics loop, and calls the
ToM + controller stack at configurable cadence. Designed so the physics loop is
unaware of whether the controller is rule-based, RL, or VLM-driven.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Dict

import numpy as np
import yaml


def _quat_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_agent_state(handle) -> Dict[str, float]:
    pos, quat = handle.robot.get_world_pose()
    lin_vel = handle.robot.get_linear_velocity()
    ang_vel = handle.robot.get_angular_velocity()
    yaw = _quat_to_yaw(np.asarray(quat))
    return {
        "pos_xy": (float(pos[0]), float(pos[1])),
        "heading_rad": yaw,
        "heading_deg": math.degrees(yaw),
        "speed": float(np.linalg.norm(lin_vel[:2])),
        "yaw_rate": float(ang_vel[2]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/split_decision.yaml")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM calls; controller still runs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    # 1) Boot Isaac Sim FIRST, before any isaacsim.* imports.
    # Use Isaac Lab's AppLauncher — it picks the correct kit experience for
    # headless+cameras (isaaclab.python.headless.rendering.kit) which avoids
    # the _wait_for_viewport deadlock seen with raw SimulationApp on Windows.
    from isaaclab.app import AppLauncher
    headless = cfg["simulation"]["headless"]
    # enable_cameras=False: the top-down view is rendered via matplotlib from the
    # physics state (see topdown_camera.py). The Isaac Sim hydra viewport path
    # segfaults on hybrid-GPU Windows systems, and a schematic is a better
    # VLM input for ToM anyway (unambiguous positions + heading arrows).
    app_launcher = AppLauncher(headless=headless, enable_cameras=False)
    simulation_app = app_launcher.app

    # The asset-fetch helper lives in a non-default extension. Enable before use.
    import omni.kit.app
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    ext_mgr.set_extension_enabled_immediate("isaacsim.storage.native", True)

    # Imports that need an alive SimulationApp.
    from isaacsim.core.api import World
    from .scene_builder import build_scene, resolve_wheel_indices
    from .topdown_camera import TopDownCamera

    # Parent-package-relative imports: fall back to absolute if run as script.
    try:
        from control.rule_based import RuleBasedController
        from control.base import HighLevelTactic
        from tom.runner import ToMRunner
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from control.rule_based import RuleBasedController
        from control.base import HighLevelTactic
        from tom.runner import ToMRunner

    np.random.seed(args.seed)

    physics_dt = 1.0 / cfg["simulation"]["physics_hz"]
    render_dt = 1.0 / cfg["simulation"]["render_hz"]
    world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=render_dt)

    evader, defender, goal_a, goal_b = build_scene(world, cfg)
    world.reset()
    resolve_wheel_indices(evader)
    resolve_wheel_indices(defender)

    cam = TopDownCamera(cfg)
    cam.initialize()

    controllers = {
        "evader":   RuleBasedController(role="evader",   cfg=cfg),
        "defender": RuleBasedController(role="defender", cfg=cfg),
    }

    tom = None if args.no_vlm else ToMRunner(cfg, goal_a=goal_a, goal_b=goal_b)

    last_tom_time = -1e9
    tom_interval = cfg["tom"]["reason_interval_sec"]

    # Holds the latest tactic per agent; controllers consume whichever is newest.
    current_tactic: Dict[str, HighLevelTactic] = {
        "evader":   HighLevelTactic(role="evader",   label="go_straight", target_goal="B"),
        "defender": HighLevelTactic(role="defender", label="wait_center",  target_goal=None),
    }

    t_start = time.time()
    step = 0
    episode_done = False
    winner = None

    while simulation_app.is_running() and (time.time() - t_start) < args.max_seconds and not episode_done:
        # --- perception snapshot ---
        state = {
            "evader":   read_agent_state(evader),
            "defender": read_agent_state(defender),
            "goal_a":   goal_a.tolist(),
            "goal_b":   goal_b.tolist(),
            "t_sim":    step * physics_dt,
        }

        # Feed state to the schematic renderer so it has something to draw.
        cam.update_state(state)

        # --- strategy layer (low frequency) ---
        sim_t = state["t_sim"]
        if tom is not None and (sim_t - last_tom_time) >= tom_interval:
            frame = cam.get_frame()
            if frame is not None:
                png_path = cam.save_frame(step)
                tactics = tom.reason_step(step=step, state=state, image_path=png_path, image_rgb=frame)
                for role, tac in tactics.items():
                    current_tactic[role] = tac
            last_tom_time = sim_t

        # --- always save a frame for MP4 replay (every N steps) ---
        if step % 3 == 0:  # ~20 fps at 60Hz physics
            cam.save_frame(step)

        # --- execution layer (every physics step) ---
        for role, handle in (("evader", evader), ("defender", defender)):
            v_left, v_right = controllers[role].step(
                tactic=current_tactic[role],
                self_state=state[role],
                other_state=state["defender" if role == "evader" else "evader"],
                goals={"A": goal_a, "B": goal_b},
            )
            _apply_wheel_velocities(handle, np.array([v_left, v_right], dtype=np.float32))

        # render=False because we render via matplotlib, not the Isaac Sim pipeline.
        world.step(render=False)
        step += 1

        # --- termination check ---
        episode_done, winner = check_termination(state, cfg)

    print(f"[runner] finished after {step} steps, winner={winner}")
    if tom is not None:
        tom.close(winner=winner)
    cam.close()

    simulation_app.close()


def _apply_wheel_velocities(handle, wheel_vel: np.ndarray) -> None:
    """Drive only the wheel DOFs on the articulation; leave others untouched."""
    from isaacsim.core.utils.types import ArticulationAction
    n_dof = handle.robot.num_dof
    full = np.full(n_dof, np.nan, dtype=np.float32)
    for i, idx in enumerate(handle.wheel_dof_idx):
        full[idx] = float(wheel_vel[i]) if i < len(wheel_vel) else 0.0
    # If the robot has more wheels than the controller returned (e.g. 3-wheel kaya),
    # repeat the last velocity for any remaining wheel DOFs.
    if len(handle.wheel_dof_idx) > len(wheel_vel):
        for i in range(len(wheel_vel), len(handle.wheel_dof_idx)):
            full[handle.wheel_dof_idx[i]] = float(wheel_vel[-1])
    handle.robot.apply_action(ArticulationAction(joint_velocities=full))


def check_termination(state, cfg):
    ga = np.asarray(cfg["arena"]["goal_a_pos"])
    gb = np.asarray(cfg["arena"]["goal_b_pos"])
    gr = cfg["arena"]["goal_radius"]
    br = cfg["arena"]["block_radius"]
    ev = np.asarray(state["evader"]["pos_xy"])
    de = np.asarray(state["defender"]["pos_xy"])

    for g_name, g in (("A", ga), ("B", gb)):
        if np.linalg.norm(ev - g) < gr:
            if np.linalg.norm(de - g) < br:
                return True, "defender"
            return True, "evader"
    return False, None


if __name__ == "__main__":
    main()
