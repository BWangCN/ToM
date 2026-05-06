"""Standalone Isaac Sim runner for the Split Decision arena.

Launches SimulationApp, builds the scene, spins the physics loop, and calls the
ToM + controller stack at configurable cadence. Designed so the physics loop is
unaware of whether the controller is rule-based, RL, or VLM-driven.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml


def _stdin_reader(q: "queue.Queue[str]") -> None:
    """Background thread: push every stripped line from stdin into a queue."""
    for line in sys.stdin:
        line = line.strip()
        if line:
            q.put(line)


def _parse_goto_command(line: str, cfg: dict, state: dict
                        ) -> Tuple[Optional[str], Optional[Tuple[float, float]], Optional[str]]:
    """Parse one stdin line into a go-to-point command.

    NOTE: This is *not* navigation — there is no planner, costmap, or obstacle
    avoidance. The runner simply drives a heading-P controller at the world-
    frame target. See README for the Nav2/ROS-bridge integration that adds
    real navigation.

    Returns (role, world_xy, error_msg).

    Grammar:
        <role> <bx> <by>     -> body-frame target, converted to world
        <role> A | B          -> alias for goal positions (world frame)
        <role> stop           -> hold position
        <role> auto           -> release back to VLM/rule-based control
    role ∈ {evader, defender}.
    """
    try:
        from control.frames import body_to_world
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from control.frames import body_to_world

    parts = line.split()
    if len(parts) < 2:
        return None, None, f"bad command: {line!r} (expected: <role> <bx> <by> | A | B | stop | auto)"
    role = parts[0].lower()
    if role not in ("evader", "defender"):
        return None, None, f"unknown role {role!r}"

    keyword = parts[1].lower()
    if keyword == "stop":
        return role, tuple(state[role]["pos_xy"]), None  # waypoint = current pos => stop
    if keyword == "auto":
        return role, None, None  # signal: release override
    if keyword in ("a", "b"):
        wx, wy = cfg["arena"][f"goal_{keyword}_pos"]
        return role, (float(wx), float(wy)), None

    if len(parts) != 3:
        return None, None, f"bad command: {line!r} (expected: <role> <bx> <by>)"
    try:
        bx, by = float(parts[1]), float(parts[2])
    except ValueError:
        return None, None, f"non-numeric coords in {line!r}"

    rxy = state[role]["pos_xy"]
    heading = state[role]["heading_rad"]
    wx, wy = body_to_world((bx, by), rxy, heading)

    half_x = cfg["arena"]["size_x"] / 2.0
    # Arena Y spans [0, size_y] — origin sits at the bottom edge in this layout
    # (evader starts at y=2, goals at y=20). Bound from 0 to size_y.
    y_lo, y_hi = 0.0, float(cfg["arena"]["size_y"])
    if not (-half_x <= wx <= half_x and y_lo <= wy <= y_hi):
        return None, None, (f"waypoint out of arena: body=({bx:.2f},{by:.2f}) -> "
                            f"world=({wx:.2f},{wy:.2f}); bounds x∈[{-half_x},{half_x}], "
                            f"y∈[{y_lo},{y_hi}]")
    return role, (wx, wy), None


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
    parser.add_argument("--mock-scenario", default=None,
                        choices=["feint_left_to_right", "feint_right_to_left"],
                        help="Use scripted mock-VLM oracle (no API key). "
                             "Same waypoint contract as the real ToM stack.")
    parser.add_argument("--defender-faces-north", action="store_true",
                        help="Override defender start_heading_deg to 90° so "
                             "it begins pre-aimed at the goal line. Useful for "
                             "mock-oracle demos where commitment cost should "
                             "come from lateral turning, not a 180° pivot.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-nav2", action="store_true",
                        help="Route evader goto_waypoint targets through Nav2 in WSL2 "
                             "instead of the heading-P controller. Requires "
                             "ros2/launch/split_decision_nav2.launch.py running in WSL2.")
    parser.add_argument("--nav2-host", default="127.0.0.1",
                        help="Host where nav2_goal_proxy.py listens (WSL2 default = 127.0.0.1).")
    parser.add_argument("--nav2-port", type=int, default=5005)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    # CLI overrides on cfg (must happen before scene_builder reads them).
    if args.defender_faces_north:
        old = cfg["agents"]["defender"]["start_heading_deg"]
        cfg["agents"]["defender"]["start_heading_deg"] = 90.0
        print(f"[runner] override: defender start_heading_deg {old} -> 90.0")

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
        from tom.mock_oracle import MockOracleRunner
    except ImportError:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from control.rule_based import RuleBasedController
        from control.base import HighLevelTactic
        from tom.runner import ToMRunner
        from tom.mock_oracle import MockOracleRunner

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

    # ROS 2 / Nav2 wiring (evader only). The graph publishes /evader/odom and
    # /tf, and subscribes /evader/cmd_vel directly into the articulation. When
    # Nav2 is active, we bypass the heading-P controller for the evader and
    # let Nav2's controller_server drive the wheels via cmd_vel.
    nav2_client = None
    if args.use_nav2:
        from .ros2_bridge import enable_ros2_bridge, build_evader_graph, GoalProxyClient
        enable_ros2_bridge()
        build_evader_graph(evader_prim=evader.prim_path)
        nav2_client = GoalProxyClient(host=args.nav2_host, port=args.nav2_port)
        print(f"[runner] Nav2 mode ON. evader cmd_vel <- /evader/cmd_vel; "
              f"goals -> tcp://{args.nav2_host}:{args.nav2_port}")

    controllers = {
        "evader":   RuleBasedController(role="evader",   cfg=cfg),
        "defender": RuleBasedController(role="defender", cfg=cfg),
    }

    if args.mock_scenario:
        tom = MockOracleRunner(cfg, scenario=args.mock_scenario)
    elif args.no_vlm:
        tom = None
    else:
        tom = ToMRunner(cfg, goal_a=goal_a, goal_b=goal_b)

    last_tom_time = -1e9
    tom_interval = cfg["tom"]["reason_interval_sec"]

    # Holds the latest tactic per agent; controllers consume whichever is newest.
    current_tactic: Dict[str, HighLevelTactic] = {
        "evader":   HighLevelTactic(role="evader",   label="go_straight", target_goal="B"),
        "defender": HighLevelTactic(role="defender", label="wait_center",  target_goal=None),
    }
    # Roles currently locked to a user-issued go-to-point target. Strategy
    # layer cannot overwrite their tactic until the user types `<role> auto`.
    goto_override: Dict[str, bool] = {"evader": False, "defender": False}

    goto_queue: "queue.Queue[str]" = queue.Queue()
    goto_thread = threading.Thread(target=_stdin_reader, args=(goto_queue,), daemon=True)
    goto_thread.start()
    print("[runner] stdin go-to-point active (heading-P only, no planner).")
    print("  evader 0 5         # 5m forward in body frame")
    print("  defender 2 -1      # 2m forward, 1m right (body frame)")
    print("  evader A | B       # go to goal A or B (world frame)")
    print("  evader stop | auto # hold | release to strategy layer")

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

        # Feed state + current tactics to the schematic renderer so it can
        # overlay each agent's waypoint × marker.
        cam.update_state(state, tactics=current_tactic)

        # --- user go-to-point input (drains queue; overrides strategy layer) ---
        while True:
            try:
                line = goto_queue.get_nowait()
            except queue.Empty:
                break
            role, wxy, err = _parse_goto_command(line, cfg, state)
            if err is not None:
                print(f"[goto] ERROR: {err}", file=sys.stderr)
                continue
            if wxy is None:
                # `<role> auto` — release override, next strategy tick takes over
                goto_override[role] = False
                if nav2_client is not None and role == "evader":
                    nav2_client.cancel()
                print(f"[goto] {role} released to auto control")
                continue
            current_tactic[role] = HighLevelTactic(
                role=role, label="goto_waypoint",
                target_goal=None, waypoint=wxy,
                rationale=f"user goto cmd: {line}",
            )
            goto_override[role] = True
            # In Nav2 mode the evader is driven by /evader/cmd_vel, so the
            # tactic above is informational only — the actual goal is sent to
            # Nav2's NavigateToPose action via the WSL2 goal proxy.
            if nav2_client is not None and role == "evader":
                if nav2_client.send_goal(wxy[0], wxy[1], 0.0):
                    print(f"[goto] evader -> Nav2 goal ({wxy[0]:.2f}, {wxy[1]:.2f})")
                else:
                    print("[goto] WARN: Nav2 goal proxy unreachable; "
                          "falling back to heading-P for this command")
                    goto_override[role] = True
            else:
                print(f"[goto] {role} -> world ({wxy[0]:.2f}, {wxy[1]:.2f})")

        # --- strategy layer (low frequency) ---
        sim_t = state["t_sim"]
        if tom is not None and (sim_t - last_tom_time) >= tom_interval:
            frame = cam.get_frame()
            if frame is not None:
                png_path = cam.save_frame(step)
                tactics = tom.reason_step(step=step, state=state, image_path=png_path, image_rgb=frame)
                for role, tac in tactics.items():
                    if not goto_override[role]:
                        current_tactic[role] = tac
            last_tom_time = sim_t

        # --- always save a frame for MP4 replay (every N steps) ---
        if step % 3 == 0:  # ~20 fps at 60Hz physics
            cam.save_frame(step)

        # --- execution layer (every physics step) ---
        for role, handle in (("evader", evader), ("defender", defender)):
            # In Nav2 mode the evader's wheels are driven by the OmniGraph
            # (cmd_vel -> diff drive -> articulation), so we skip the local
            # heading-P controller for it. Defender always stays local.
            if args.use_nav2 and role == "evader":
                continue
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
    if nav2_client is not None:
        nav2_client.close()
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
