"""Dataset validator (CLAUDE.md §3.3). Runs inside `make smoke`."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.action_quant import bins_to_chunk, chunk_to_bins, half_bin_width
from dvla.toy_env import DECISIONS


def validate_file(jsonl_path: Path, root: Path, slotlog: dict) -> int:
    n = 0
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            eid = r["episode_id"]
            chunk = np.asarray(r["action_chunk"], dtype=np.float64)
            assert chunk.shape == (8, 2), f"{eid}: chunk shape {chunk.shape}"
            assert (chunk >= -1).all() and (chunk <= 1).all(), f"{eid}: chunk out of range"
            assert r["decision"] in DECISIONS, f"{eid}: bad decision {r['decision']}"
            for fp in r["frames"]:
                assert (root / fp).exists(), f"{eid}: missing frame {fp}"
            rt = bins_to_chunk(chunk_to_bins(chunk))
            err = np.abs(rt - chunk).max()
            assert err <= half_bin_width() + 1e-12, f"{eid}: round-trip err {err}"

            # causal-locality leakage check against the generator's slot log
            slots = slotlog[eid]
            used_vals = set(slots["used"].values())
            text = r["reasoning"]
            for bad in slots["forbidden"]:
                if bad in used_vals:
                    continue  # same string legitimately appears as a <=keyframe slot
                assert bad not in text, f"{eid}: post-keyframe value '{bad}' leaked into reasoning"
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/toy")
    args = ap.parse_args()
    root = Path(args.root)

    slotlog = {e["episode_id"]: e for e in json.loads((root / "slotlog.json").read_text())}
    total = 0
    for name in ["train.jsonl", "val.jsonl"]:
        n = validate_file(root / name, root, slotlog)
        print(f"{name}: {n} samples OK")
        total += n
    print(f"VALIDATOR PASS ({total} samples)")


if __name__ == "__main__":
    main()
