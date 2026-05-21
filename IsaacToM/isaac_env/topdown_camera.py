"""Schematic top-down renderer.

Renders a 2D schematic (arena, goals, agents with heading arrows) using
matplotlib from the physics state. Used instead of an Isaac Sim Camera because
the hybrid-GPU hydra viewport path segfaults on this dev box. For VLM ToM
reasoning a clean schematic is arguably *better* than a photorealistic render:
positions and headings are unambiguous.

The class mimics the old Camera interface so runner.py stays the same:
  cam.initialize(), cam.update_state(state), cam.get_frame(), cam.save_frame(step)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np


class TopDownCamera:
    def __init__(self, cfg: dict):
        self.cfg = cfg["topdown_camera"]
        self.arena = cfg["arena"]
        self.resolution = tuple(self.cfg["resolution"])
        self.ortho_half = float(self.cfg["orthographic_size"])
        self.save_dir = Path(self.cfg["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self._plt = plt

        dpi = 100
        fig_w = self.resolution[0] / dpi
        fig_h = self.resolution[1] / dpi
        self._fig, self._ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        self._latest_state: Optional[dict] = None
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_tactics: Optional[dict] = None
        # Cumulative position history per agent — drawn as a fading polyline
        # so the viewer can see the *shape* of the trajectory (e.g. an S-curve
        # feint) without having to scrub through frames.
        self._trails: dict = {"evader": [], "defender": []}
        self._trail_stride = 4  # only sample every Nth update to keep it light

    def initialize(self):
        return

    def update_state(self, state: dict, tactics: Optional[dict] = None):
        """Cache the latest physics state (and optionally per-role tactics, so
        the renderer can overlay current waypoints)."""
        self._latest_state = state
        if tactics is not None:
            self._latest_tactics = tactics
        # Append to trails every Nth update_state call. Track sample count on
        # the camera (not via len(buf), which doesn't grow if we ever stop
        # appending — that was a real bug).
        self._trail_count = getattr(self, "_trail_count", 0) + 1
        if self._trail_count % self._trail_stride == 0:
            for role in ("evader", "defender"):
                agent = state.get(role)
                if agent is None:
                    continue
                self._trails[role].append(tuple(agent["pos_xy"]))
                if len(self._trails[role]) > 2000:
                    self._trails[role] = self._trails[role][-1500:]
        self._latest_rgb = None  # invalidate render cache

    def _render(self, state: dict) -> np.ndarray:
        ax = self._ax
        ax.clear()
        ax.set_facecolor("#1a1a22")
        ax.set_xlim(-self.ortho_half, self.ortho_half)
        ax.set_ylim(-2.0, 2.0 * self.ortho_half - 2.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        arena = self.arena
        ax.plot(
            [-arena["size_x"] / 2, arena["size_x"] / 2, arena["size_x"] / 2,
             -arena["size_x"] / 2, -arena["size_x"] / 2],
            [0, 0, arena["size_y"], arena["size_y"], 0],
            color="#4a4a5a", linewidth=1.5,
        )

        goal_a = np.asarray(arena["goal_a_pos"])
        goal_b = np.asarray(arena["goal_b_pos"])
        gr = arena["goal_radius"]
        ax.add_patch(self._plt.Circle(goal_a, gr, color="#4fc3f7", alpha=0.35, zorder=1))
        ax.add_patch(self._plt.Circle(goal_b, gr, color="#ffb74d", alpha=0.35, zorder=1))
        ax.text(goal_a[0], goal_a[1], "A", ha="center", va="center",
                color="white", fontsize=16, fontweight="bold", zorder=2)
        ax.text(goal_b[0], goal_b[1], "B", ha="center", va="center",
                color="white", fontsize=16, fontweight="bold", zorder=2)

        # Trails first so agents draw on top of them.
        for role, color in (("evader", "#81c784"), ("defender", "#e57373")):
            tr = self._trails.get(role) or []
            if len(tr) > 1:
                arr = np.asarray(tr)
                ax.plot(arr[:, 0], arr[:, 1], color=color, alpha=0.5,
                        linewidth=1.5, zorder=2)

        for role, color in (("evader", "#81c784"), ("defender", "#e57373")):
            s = state[role]
            x, y = s["pos_xy"]
            yaw = s["heading_rad"]

            # Optional waypoint overlay — × marker + dotted line from agent.
            tac = (self._latest_tactics or {}).get(role)
            if tac is not None and getattr(tac, "label", None) == "goto_waypoint" \
                    and getattr(tac, "waypoint", None) is not None:
                wx, wy = tac.waypoint
                ax.plot([x, wx], [y, wy], color=color, alpha=0.5,
                        linewidth=1.0, linestyle=":", zorder=2)
                ax.plot(wx, wy, marker="x", color=color,
                        markersize=10, mew=2.0, zorder=4)
                if (tac.extra or {}).get("clamped_horizon"):
                    ax.plot(wx, wy, marker="o", color=color,
                            markersize=13, fillstyle="none", mew=1.0,
                            alpha=0.6, zorder=4)

            ax.add_patch(self._plt.Circle((x, y), 0.5, color=color, zorder=3))
            dx, dy = 1.2 * math.cos(yaw), 1.2 * math.sin(yaw)
            ax.annotate(
                "", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2),
                zorder=4,
            )
            ax.text(x + 0.7, y + 0.7, role[0].upper(),
                    color=color, fontsize=11, fontweight="bold", zorder=4)

        # Status header (top-right) — shows phase + clamped target per role.
        lines = [f"t={state['t_sim']:.2f}s"]
        for role in ("evader", "defender"):
            tac = (self._latest_tactics or {}).get(role)
            if tac is None:
                continue
            phase = (tac.extra or {}).get("phase") if hasattr(tac, "extra") else None
            tag = "E" if role == "evader" else "D"
            if tac.label == "goto_waypoint" and tac.waypoint is not None:
                wx, wy = tac.waypoint
                sf = (tac.extra or {}).get("speed_frac", 1.0) if hasattr(tac, "extra") else 1.0
                lines.append(f"{tag}:{phase or tac.label:11s}->({wx:+5.2f},{wy:+5.2f}) v={sf:.2f}")
            else:
                lines.append(f"{tag}:{tac.label}")
        ax.text(0.98, 0.97, "\n".join(lines),
                transform=ax.transAxes, color="#dddddd", fontsize=8,
                family="monospace", va="top", ha="right",
                bbox=dict(facecolor="#000000aa", edgecolor="none",
                          boxstyle="round,pad=0.3"))

        self._fig.canvas.draw()
        w, h = self._fig.canvas.get_width_height()
        buf = np.frombuffer(self._fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgb = buf.reshape(h, w, 4)[..., :3].copy()
        self._latest_rgb = rgb
        return rgb

    def get_frame(self) -> Optional[np.ndarray]:
        if self._latest_state is None:
            return None
        if self._latest_rgb is None:
            self._render(self._latest_state)
        return self._latest_rgb

    def save_frame(self, step: int) -> Optional[str]:
        rgb = self.get_frame()
        if rgb is None:
            return None
        from PIL import Image
        path = self.save_dir / f"step_{step:06d}.png"
        Image.fromarray(rgb).save(path)
        return str(path)

    def close(self):
        self._plt.close(self._fig)
