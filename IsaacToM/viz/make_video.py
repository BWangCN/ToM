"""Encode logs/frames/step_*.png into an MP4 via ffmpeg.

    python -m viz.make_video --frames logs/frames --out split_decision_demo.mp4 --fps 20
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="logs/frames")
    parser.add_argument("--out", default="split_decision_demo.mp4")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    frames_dir = Path(args.frames).resolve()
    out_path = Path(args.out).resolve()
    pngs = sorted(frames_dir.glob("step_*.png"))
    if not pngs:
        raise SystemExit(f"no step_*.png frames found in {frames_dir}")

    concat = frames_dir / "concat.txt"
    step = 1.0 / args.fps
    with concat.open("w") as f:
        f.write("ffconcat version 1.0\n")
        for p in pngs:
            f.write(f"file '{p.name}'\nduration {step:.4f}\n")
        # ffmpeg concat demuxer requires the last frame to be repeated
        f.write(f"file '{pngs[-1].name}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-vf", f"fps={args.fps}",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, cwd=str(frames_dir))
    print(f"wrote {out_path} ({len(pngs)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
