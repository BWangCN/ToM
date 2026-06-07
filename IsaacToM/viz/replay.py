"""ToM session viewer.

Plots each role's decision-time trajectory and every waypoint emitted by the
strategy layer. Waypoint X markers get more saturated over session time, so old
targets fade toward gray and late-session targets read as the role color.
"""
from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_session(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _state_time(row: dict) -> float:
    state = row.get("state", {})
    if "sim_time_sec" in state:
        return float(state["sim_time_sec"])
    return float(row.get("step", 0))


def _role_color(role: str, progress: float):
    """Return a role hue with saturation increasing over session progress."""
    hue = 0.58 if role == "evader" else 0.0
    sat = 0.18 + 0.82 * max(0.0, min(1.0, progress))
    val = 0.88
    return colorsys.hsv_to_rgb(hue, sat, val)


def _goal_xy(rows, key: str, default):
    for row in rows:
        goal = row.get("state", {}).get(key)
        if goal and len(goal) == 2:
            return goal
    return default


def _waypoint_xy(row: dict):
    waypoint = row.get("waypoint") or {}
    target = waypoint.get("target_xy")
    if isinstance(target, list) and len(target) == 2:
        return target
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--arena-size", type=float, default=24.0)
    args = parser.parse_args()
    rows = load_session(args.session)
    rows.sort(key=lambda r: (_state_time(r), str(r.get("self_role", ""))))

    times = [_state_time(r) for r in rows]
    t_min = min(times) if times else 0.0
    t_max = max(times) if times else 1.0
    t_span = max(1e-9, t_max - t_min)

    fig, ax = plt.subplots(figsize=(9, 9))

    for role in ("evader", "defender"):
        xs, ys = [], []
        for r in rows:
            if r.get("self_role") != role:
                continue
            state = r.get("state", {})
            self_state = state.get("self", {})
            pos = self_state.get("pos_xy")
            if not pos:
                continue
            xs.append(pos[0])
            ys.append(pos[1])
        if xs:
            base_color = _role_color(role, 1.0)
            ax.plot(xs, ys, color=base_color, linewidth=2.0, alpha=0.7)
            ax.scatter(xs, ys, color=base_color, s=18, alpha=0.45, zorder=3)
            ax.scatter(xs[0], ys[0], color=base_color, s=70, marker="o",
                       edgecolors="white", linewidths=1.0, zorder=5)
            ax.scatter(xs[-1], ys[-1], color=base_color, s=90, marker="s",
                       edgecolors="white", linewidths=1.0, zorder=5)

    for r in rows:
        role = r.get("self_role")
        if role not in ("evader", "defender"):
            continue
        pos = (r.get("state", {}).get("self") or {}).get("pos_xy")
        target = _waypoint_xy(r)
        if not pos or not target:
            continue

        progress = (_state_time(r) - t_min) / t_span
        color = _role_color(role, progress)
        ax.plot([pos[0], target[0]], [pos[1], target[1]],
                color=color, linestyle=":", linewidth=1.0, alpha=0.35, zorder=1)
        ax.scatter([target[0]], [target[1]], marker="x", s=45, color=color,
                   linewidths=2.2, alpha=0.95, zorder=6)

        phase = (r.get("waypoint") or {}).get("phase")
        if phase:
            ax.text(target[0] + 0.15, target[1] + 0.15, str(r.get("step", "")),
                    color=color, fontsize=7, alpha=0.85)

    goal_a = _goal_xy(rows, "goal_A_xy", [-8.0, 20.0])
    goal_b = _goal_xy(rows, "goal_B_xy", [8.0, 20.0])
    ax.axhline(goal_a[1], color="gray", linestyle="--", alpha=0.3)
    ax.scatter([goal_a[0], goal_b[0]], [goal_a[1], goal_b[1]],
               marker="*", s=320, c=["green", "orange"], zorder=5)
    ax.text(goal_a[0], goal_a[1] + 0.7, "A", ha="center", va="bottom")
    ax.text(goal_b[0], goal_b[1] + 0.7, "B", ha="center", va="bottom")

    half = args.arena_size / 2.0
    ax.set_xlim(-half - 2.0, half + 2.0)
    ax.set_ylim(-2, args.arena_size)
    ax.set_aspect("equal")
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (m)")
    ax.set_title(f"ToM session waypoints: {args.session.name}\n"
                 "X markers are waypoints; saturation increases with time")
    ax.legend(handles=[
        Line2D([0], [0], color=_role_color("evader", 1.0), lw=2, label="evader trajectory"),
        Line2D([0], [0], color=_role_color("defender", 1.0), lw=2, label="defender trajectory"),
        Line2D([0], [0], marker="x", color="black", lw=0, markersize=3, label="waypoint"),
        Line2D([0], [0], marker="o", color="black", lw=0, markersize=7, label="start"),
        Line2D([0], [0], marker="s", color="black", lw=0, markersize=7, label="latest"),
    ], loc="lower right")
    ax.grid(True, alpha=0.3)

    out = args.out or args.session.with_name(f"{args.session.stem}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
