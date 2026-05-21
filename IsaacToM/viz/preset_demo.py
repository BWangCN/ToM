"""Standalone visualization of the waypoint navigator with a SCRIPTED mock-VLM
oracle. No Isaac Sim, no OpenAI API key — just numpy + matplotlib + pyyaml +
(optional) ffmpeg.

The simulator integrates a unicycle model whose math is intentionally identical
to control/rule_based.py (kp_heading, cruise_frac, turn_penalty, dist<0.3 stop
band). The only thing replaced is:
    - perception (we use ground-truth state, no camera)
    - the VLM (we use a hand-scripted oracle)

This lets the user inspect the system's behaviour under the new waypoint
contract before having a working VLM environment. The same waypoints would be
fed to control/rule_based.py at runtime, so the trajectories you see here are
representative of what the real stack will produce when given the same outputs.

Scenarios
    feint_left_to_right   evader feints toward Goal A, then exploits to Goal B
    feint_right_to_left   mirror of the above
    straight_baseline     evader drives straight to Goal A; defender tracks
                          and intercepts — the "head-on" contrast case

Usage
    python -m viz.preset_demo                                     # default feint
    python -m viz.preset_demo --scenario straight_baseline
    python -m viz.preset_demo --duration 10 --out-dir logs/demo2
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# Physics — mirrors control/rule_based.py
# ===========================================================================

KP_HEADING = 2.0
CRUISE_FRAC = 0.85
STOP_BAND = 0.3


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def step_agent(state: dict, target_xy: Tuple[float, float], speed_frac: float,
               max_speed: float, max_omega: float, dt: float) -> dict:
    """One physics tick. Same control law as RuleBasedController.step's
    goto_waypoint branch, but integrated as a unicycle (skip the wheel-velocity
    step since we don't have an articulation here)."""
    sx, sy = state["pos_xy"]
    heading = state["heading_rad"]
    dx, dy = target_xy[0] - sx, target_xy[1] - sy
    dist = math.hypot(dx, dy)

    if dist < STOP_BAND:
        v, omega = 0.0, 0.0
    else:
        desired = math.atan2(dy, dx)
        err = _wrap(desired - heading)
        omega = max(-max_omega, min(max_omega, KP_HEADING * err))
        turn_penalty = max(0.0, 1.0 - abs(err) / math.pi)
        v = max_speed * CRUISE_FRAC * speed_frac * turn_penalty

    return {
        "pos_xy": (sx + v * math.cos(heading) * dt,
                   sy + v * math.sin(heading) * dt),
        "heading_rad": _wrap(heading + omega * dt),
        "speed": v,
        "yaw_rate": omega,
    }


# ===========================================================================
# Mock-VLM oracles
# ===========================================================================

@dataclass
class WaypointDecision:
    target_xy: Tuple[float, float]
    speed_frac: float
    phase: str
    rationale: str = ""


def _goal_xy(arena: dict, label: str) -> Tuple[float, float]:
    return tuple(arena[f"goal_{label.lower()}_pos"])


class FeintOracle:
    """L2 evader vs L1 defender. Evader scripts a feint; defender locks its
    commit at t=lock_t based on the evader's lateral position at that moment
    and only switches when the evader's position is unambiguously on the
    other side. The slow defender max_omega + the late switch are what
    naturally produce 'committed too far, can't recover'."""

    def __init__(self, arena: dict, fake_goal: str = "A", real_goal: str = "B",
                 lock_t: float = 1.5, exploit_t: float = 2.0, commit_t: float = 3.2):
        assert fake_goal != real_goal and fake_goal in ("A", "B")
        self.arena = arena
        self.fake = fake_goal
        self.real = real_goal
        self.lock_t = lock_t
        self.exploit_t = exploit_t
        self.commit_t = commit_t
        self._defender_locked_to: Optional[str] = None

    # -- evader ----------------------------------------------------------
    def evader(self, state: dict, t_sim: float) -> WaypointDecision:
        ex, ey = state["evader"]["pos_xy"]
        fake_x = _goal_xy(self.arena, self.fake)[0]
        real_x = _goal_xy(self.arena, self.real)[0]

        if t_sim < self.exploit_t:
            # SHAPING: drift ~3 m toward fake goal, modest forward.
            tx = ex + 0.45 * (fake_x - ex)        # bias 45% toward fake side
            ty = ey + 4.0
            return WaypointDecision(
                (tx, ty), speed_frac=0.85, phase="shaping",
                rationale=f"shape opponent belief toward fake goal {self.fake}")
        elif t_sim < self.commit_t:
            # EXPLOIT: hard switch — push past mid-line toward real side.
            tx = ex + 0.65 * (real_x - ex)
            ty = ey + 3.5
            return WaypointDecision(
                (tx, ty), speed_frac=1.0, phase="exploiting",
                rationale=f"defender now committed to {self.fake}, swerve to {self.real}")
        else:
            # COMMIT: drive at the real goal directly.
            return WaypointDecision(
                _goal_xy(self.arena, self.real), speed_frac=1.0, phase="committing",
                rationale=f"final approach to goal {self.real}")

    # -- defender --------------------------------------------------------
    def defender(self, state: dict, t_sim: float) -> WaypointDecision:
        ex, _ = state["evader"]["pos_xy"]

        if t_sim < self.lock_t:
            # HOLDING: stay near (0, 15) — small forward step, low speed.
            # Defender does *not* try to read the evader yet, only watches.
            return WaypointDecision(
                (0.0, 15.0), speed_frac=0.30, phase="holding",
                rationale="evader trajectory ambiguous — stay centered")

        # First commit decision (locks based on evader's drift at lock_t).
        if self._defender_locked_to is None:
            self._defender_locked_to = "A" if ex < 0 else "B"

        ev_actual = "A" if ex < 0 else "B"
        committed_goal = _goal_xy(self.arena, self._defender_locked_to)

        if ev_actual == self._defender_locked_to:
            return WaypointDecision(
                committed_goal, speed_frac=1.0, phase="committing",
                rationale=f"L1: evader appears headed for {self._defender_locked_to}")

        # Evader is now on the OTHER side. Try to recover — but our max_omega
        # is the smaller one, so the heading turn alone burns a lot of time.
        actual_goal = _goal_xy(self.arena, ev_actual)
        return WaypointDecision(
            actual_goal, speed_frac=1.0, phase="recovering",
            rationale=f"evader switched to {ev_actual} — recover (commitment cost!)")


class StraightOracle:
    """Baseline contrast: naive evader drives straight, defender tracks-then-
    commits. Produces the head-on collision near the goal."""

    def __init__(self, arena: dict, target_goal: str = "A"):
        self.arena = arena
        self.target = target_goal

    def evader(self, state: dict, t_sim: float) -> WaypointDecision:
        return WaypointDecision(
            _goal_xy(self.arena, self.target), speed_frac=1.0, phase="committing",
            rationale=f"L0: drive straight to goal {self.target}")

    def defender(self, state: dict, t_sim: float) -> WaypointDecision:
        ex, _ = state["evader"]["pos_xy"]
        if t_sim < 0.6:
            return WaypointDecision((0.0, 15.0), 0.3, "holding", "observing")
        if t_sim < 1.6:
            tx = max(-6.0, min(6.0, ex * 0.9))
            return WaypointDecision((tx, 18.0), 0.6, "tracking",
                                    "lateral shadow on evader x-position")
        target = "A" if ex < 0 else "B"
        return WaypointDecision(_goal_xy(self.arena, target), 1.0, "committing",
                                f"intercept at goal {target}")


# ===========================================================================
# Same clamping the production runner applies after the VLM call.
# ===========================================================================

def clamp_waypoint(wp: WaypointDecision, self_xy: Tuple[float, float],
                   arena: dict, max_horizon: float
                   ) -> Tuple[Tuple[float, float], bool]:
    x_max = arena["size_x"] / 2.0
    x_min = -x_max
    y_min, y_max = 0.0, float(arena["size_y"])
    tx, ty = wp.target_xy
    tx = max(x_min, min(x_max, tx))
    ty = max(y_min, min(y_max, ty))
    dx, dy = tx - self_xy[0], ty - self_xy[1]
    dist = math.hypot(dx, dy)
    clamped = False
    if dist > max_horizon and dist > 1e-6:
        s = max_horizon / dist
        tx = self_xy[0] + dx * s
        ty = self_xy[1] + dy * s
        clamped = True
    return (tx, ty), clamped


# ===========================================================================
# Renderer
# ===========================================================================

class Renderer:
    COLORS = {"evader": "#81c784", "defender": "#e57373"}
    GOAL_COLORS = {"A": "#4fc3f7", "B": "#ffb74d"}

    def __init__(self, cfg: dict, save_dir: Path):
        self.cfg = cfg
        self.arena = cfg["arena"]
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)
        self.fig.patch.set_facecolor("#1a1a22")
        self.trails: Dict[str, list] = {"evader": [], "defender": []}

    def render(self, state: dict, waypoints: Dict[str, WaypointDecision],
               clamped: Dict[str, Tuple[Tuple[float, float], bool]],
               step: int, scenario: str, winner: Optional[str]) -> Path:
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#1a1a22")
        ax.set_xlim(-14, 14)
        ax.set_ylim(-2, 26)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

        sx = self.arena["size_x"]; sy = self.arena["size_y"]
        ax.plot([-sx/2, sx/2, sx/2, -sx/2, -sx/2], [0, 0, sy, sy, 0],
                color="#4a4a5a", linewidth=1.5)

        # Goals
        for label in ("A", "B"):
            g = np.asarray(self.arena[f"goal_{label.lower()}_pos"])
            gr = self.arena["goal_radius"]
            ax.add_patch(plt.Circle(g, gr, color=self.GOAL_COLORS[label], alpha=0.35, zorder=1))
            ax.text(g[0], g[1], label, ha="center", va="center",
                    color="white", fontsize=16, fontweight="bold", zorder=2)

        # Trails
        for role, color in self.COLORS.items():
            self.trails[role].append(state[role]["pos_xy"])
            tr = np.array(self.trails[role])
            if len(tr) > 1:
                ax.plot(tr[:, 0], tr[:, 1], color=color, alpha=0.45,
                        linewidth=1.5, zorder=2)

        # Agents + waypoints
        for role, color in self.COLORS.items():
            s = state[role]
            x, y = s["pos_xy"]
            yaw = s["heading_rad"]

            wp = waypoints.get(role)
            cl = clamped.get(role)
            if wp is not None and cl is not None:
                (wx, wy), was_clamped = cl
                ax.plot([x, wx], [y, wy], color=color, alpha=0.5,
                        linewidth=1.0, linestyle=":", zorder=3)
                ax.plot(wx, wy, marker="x", color=color, markersize=11, mew=2.2, zorder=4)
                if was_clamped:
                    ax.plot(wx, wy, marker="o", color=color, markersize=14,
                            fillstyle="none", mew=1.0, alpha=0.6, zorder=4)

            ax.add_patch(plt.Circle((x, y), 0.5, color=color, zorder=5))
            dx_, dy_ = 1.4 * math.cos(yaw), 1.4 * math.sin(yaw)
            ax.annotate("", xy=(x + dx_, y + dy_), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.2),
                        zorder=6)

        # Per-role status panel (top-right) so the phase + rationale is readable
        # even when the agent is moving fast.
        lines = [f"scenario: {scenario}", f"t = {state['t_sim']:5.2f} s"]
        for role in ("evader", "defender"):
            wp = waypoints.get(role)
            if wp is None:
                continue
            wx, wy = clamped[role][0]
            tag = "E" if role == "evader" else "D"
            lines.append(f"{tag}: {wp.phase:11s} -> ({wx:+5.2f},{wy:+5.2f}) v={wp.speed_frac:.2f}")
        if winner is not None:
            lines.append("")
            lines.append(f"*** {winner.upper()} WINS ***")
        ax.text(0.98, 0.97, "\n".join(lines),
                transform=ax.transAxes, color="#dddddd", fontsize=9,
                family="monospace", va="top", ha="right",
                bbox=dict(facecolor="#000000aa", edgecolor="none", boxstyle="round,pad=0.4"))

        ax.text(0.02, 0.03, "× = mock-VLM waypoint     ◯ = clamped to horizon",
                transform=ax.transAxes, color="#888888", fontsize=8,
                va="bottom", ha="left")

        path = self.save_dir / f"step_{step:06d}.png"
        self.fig.savefig(path, facecolor=self.fig.get_facecolor(), bbox_inches="tight")
        return path

    def close(self):
        plt.close(self.fig)


# ===========================================================================
# Main
# ===========================================================================

def _check_termination(state: dict, arena: dict) -> Optional[str]:
    ev = np.asarray(state["evader"]["pos_xy"])
    de = np.asarray(state["defender"]["pos_xy"])
    gr = arena["goal_radius"]
    br = arena["block_radius"]
    for label in ("A", "B"):
        g = np.asarray(arena[f"goal_{label.lower()}_pos"])
        if np.linalg.norm(ev - g) < gr:
            if np.linalg.norm(de - g) < br:
                return "defender"
            return "evader"
    return None


def _build_oracle(scenario: str, arena: dict):
    if scenario == "feint_left_to_right":
        return FeintOracle(arena, fake_goal="A", real_goal="B")
    if scenario == "feint_right_to_left":
        return FeintOracle(arena, fake_goal="B", real_goal="A")
    if scenario == "straight_baseline":
        return StraightOracle(arena, target_goal="A")
    raise ValueError(f"unknown scenario: {scenario}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="config/split_decision.yaml")
    parser.add_argument("--scenario", default="feint_left_to_right",
                        choices=["feint_left_to_right", "feint_right_to_left",
                                 "straight_baseline"])
    parser.add_argument("--duration", type=float, default=14.0, help="seconds")
    parser.add_argument("--out-dir", default="logs/preset_demo")
    parser.add_argument("--mp4", default=None,
                        help="output mp4 path; default <out-dir>/<scenario>.mp4")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--render-stride", type=int, default=3,
                        help="render every Nth physics step (60Hz / 3 = 20fps)")
    parser.add_argument("--vlm-interval", type=float, default=0.8)
    parser.add_argument("--max-horizon", type=float, default=6.0)
    parser.add_argument("--defender-faces-north", action="store_true", default=True,
                        help="override defender start_heading to face goals (90°). "
                             "Off would model defender starting nose-toward-evader, "
                             "which spends ~2s just turning around.")
    parser.add_argument("--no-mp4", action="store_true",
                        help="skip ffmpeg, leave PNGs only")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    arena = cfg["arena"]
    ev_cfg = cfg["agents"]["evader"]
    de_cfg = cfg["agents"]["defender"]

    state = {
        "evader": {
            "pos_xy": tuple(ev_cfg["start_pos"]),
            "heading_rad": math.radians(ev_cfg["start_heading_deg"]),
            "speed": 0.0, "yaw_rate": 0.0,
        },
        "defender": {
            "pos_xy": tuple(de_cfg["start_pos"]),
            "heading_rad": math.radians(
                90.0 if args.defender_faces_north else de_cfg["start_heading_deg"]),
            "speed": 0.0, "yaw_rate": 0.0,
        },
        "t_sim": 0.0,
    }

    oracle = _build_oracle(args.scenario, arena)

    # Clean output dir each run so frame counts don't drift.
    # Use per-file unlink instead of rmtree — on Windows, an external file
    # viewer holding a handle on a PNG (e.g. the previous run's preview)
    # makes rmtree fail with WinError 32.
    out_dir = Path(args.out_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(frames_dir.glob("step_*.png")) + list(frames_dir.glob("concat.txt")):
        try:
            stale.unlink()
        except OSError:
            pass
    renderer = Renderer(cfg, frames_dir)

    physics_dt = 1.0 / cfg["simulation"]["physics_hz"]
    n_steps = int(args.duration / physics_dt)

    waypoints: Dict[str, Optional[WaypointDecision]] = {"evader": None, "defender": None}
    clamped: Dict[str, Tuple[Tuple[float, float], bool]] = {}
    last_query_t = -1e9
    winner: Optional[str] = None
    frame_count = 0

    print(f"[preset_demo] scenario={args.scenario} duration={args.duration}s "
          f"physics_dt={physics_dt:.4f}s")

    for step in range(n_steps):
        t_sim = step * physics_dt
        state["t_sim"] = t_sim

        # Mock-VLM tick
        if (t_sim - last_query_t) >= args.vlm_interval:
            for role in ("evader", "defender"):
                wp = getattr(oracle, role)(state, t_sim)
                self_xy = state[role]["pos_xy"]
                tgt, was_clamped = clamp_waypoint(wp, self_xy, arena, args.max_horizon)
                waypoints[role] = wp
                clamped[role] = (tgt, was_clamped)
            last_query_t = t_sim
            ew, dw = waypoints["evader"], waypoints["defender"]
            print(f"[t={t_sim:5.2f}s] "
                  f"E {ew.phase:11s} -> {clamped['evader'][0][0]:+5.2f},{clamped['evader'][0][1]:+5.2f} v={ew.speed_frac:.2f}"
                  + ("  [clamp]" if clamped['evader'][1] else "        ")
                  + f" | D {dw.phase:11s} -> {clamped['defender'][0][0]:+5.2f},{clamped['defender'][0][1]:+5.2f} v={dw.speed_frac:.2f}"
                  + ("  [clamp]" if clamped['defender'][1] else ""))

        # Physics tick (uses CLAMPED waypoint, same as production)
        for role, role_cfg in (("evader", ev_cfg), ("defender", de_cfg)):
            wp = waypoints[role]
            if wp is None:
                continue
            tgt = clamped[role][0]
            state[role] = step_agent(state[role], tgt, wp.speed_frac,
                                     role_cfg["max_speed"], role_cfg["max_omega"],
                                     physics_dt)

        # Render frame
        if step % args.render_stride == 0:
            renderer.render(state, waypoints, clamped, frame_count, args.scenario, winner)
            frame_count += 1

        # Termination
        if winner is None:
            winner = _check_termination(state, arena)
            if winner is not None:
                print(f"[preset_demo] {winner.upper()} wins at t={t_sim:.2f}s")
                # Hold the final frame for ~1 s so the result is readable.
                for _ in range(args.fps):
                    renderer.render(state, waypoints, clamped, frame_count,
                                    args.scenario, winner)
                    frame_count += 1
                break

    renderer.close()
    print(f"[preset_demo] wrote {frame_count} frames to {frames_dir}")

    # Encode MP4 (optional)
    if args.no_mp4:
        return
    mp4_path = Path(args.mp4) if args.mp4 else out_dir / f"{args.scenario}.mp4"
    pngs = sorted(frames_dir.glob("step_*.png"))
    if not pngs:
        print("[preset_demo] no frames to encode")
        return
    concat = frames_dir / "concat.txt"
    step_dur = 1.0 / args.fps
    with concat.open("w") as f:
        f.write("ffconcat version 1.0\n")
        for p in pngs:
            f.write(f"file '{p.name}'\nduration {step_dur:.4f}\n")
        f.write(f"file '{pngs[-1].name}'\n")
    # Run ffmpeg from inside frames_dir so the bare `step_*.png` filenames
    # in concat.txt resolve. The concat file is referenced by name only;
    # mp4 output is written to an absolute path so it lands where requested.
    # h264 requires even dimensions; matplotlib's bbox_inches='tight' gives
    # us odd ones (e.g. 481x482), so the scale filter rounds both DOWN to
    # the nearest even pixel.
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
           "-i", concat.name, "-c:v", "libx264",
           "-pix_fmt", "yuv420p",
           "-vf", f"fps={args.fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
           str(mp4_path.resolve())]
    try:
        subprocess.run(cmd, check=True, cwd=str(frames_dir.resolve()),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[preset_demo] wrote {mp4_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"[preset_demo] ffmpeg unavailable or failed ({e}); "
              f"PNG frames are still in {frames_dir}")


if __name__ == "__main__":
    main()
