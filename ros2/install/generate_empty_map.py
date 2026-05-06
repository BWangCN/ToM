"""Generate an empty 24x24m occupancy grid for the Split Decision arena.

Run inside WSL2 (or anywhere with numpy + Pillow). Output:
    ros2/maps/empty_arena.pgm
    ros2/maps/empty_arena.yaml

Resolution 0.05 m/cell -> 480x480 cells. All free space (value 254).
Origin = (-12, 0) so world frame matches what runner.py uses
(x ∈ [-12, 12], y ∈ [0, 24]).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

RES = 0.05
SIZE_X = 24.0
SIZE_Y = 24.0
ORIGIN = (-12.0, 0.0)  # x_min, y_min — must match runner.py arena bounds

OUT_DIR = Path(__file__).resolve().parents[1] / "maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

w = int(SIZE_X / RES)
h = int(SIZE_Y / RES)
# Nav2 occupancy grid pgm convention: 254 = free, 0 = occupied, 205 = unknown.
img = np.full((h, w), 254, dtype=np.uint8)
Image.fromarray(img, mode="L").save(OUT_DIR / "empty_arena.pgm")

(OUT_DIR / "empty_arena.yaml").write_text(
    f"""image: empty_arena.pgm
mode: trinary
resolution: {RES}
origin: [{ORIGIN[0]}, {ORIGIN[1]}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
""",
    encoding="utf-8",
)
print(f"Wrote {OUT_DIR / 'empty_arena.pgm'} ({w}x{h}) and yaml.")
