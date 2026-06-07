"""Minimal log viewer: plot evader vs defender trajectories + overlay L1/L2
beliefs at each decision step. Mirrors AR-LLMs visualization.py role."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_session(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args()
    rows = load_session(args.session)

    fig, ax = plt.subplots(figsize=(8, 8))

    for role, color in (("evader", "C0"), ("defender", "C3")):
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
            ax.plot(xs, ys, color=color, marker="o", markersize=4, label=role)

    ax.axhline(20, color="gray", linestyle="--", alpha=0.3)
    ax.scatter([-8, 8], [20, 20], marker="*", s=300, c=["green", "orange"], zorder=5)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-2, 24)
    ax.set_aspect("equal")
    ax.set_title(f"ToM session: {args.session.name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    out = args.session.with_suffix(".png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
