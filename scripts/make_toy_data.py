"""Toy data generator (CLAUDE.md §3.2).

One JSONL line per decision keyframe (one per episode). Reasoning comes from
templates keyed by decision, slots filled ONLY from state <= keyframe_t
(causal locality by construction). A slot log records used slot values and
formatted post-keyframe values for the validator's leakage check.

Deterministic under seed 0; --check-determinism regenerates and compares hashes.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.toy_env import (CHUNK_LEN, DECISIONS, EPISODE_LEN, FALSE_BELIEFS,
                          make_episode_env, outcome_deceived, render_state)
from dvla.common import record_timing

TASK_PROMPT = ("You are agent A (blue). Reach the green goal square; "
               "agent B (red) is guarding. Plan your next 8 steps (0.8 s).")

TEMPLATES = {
    "HONEST_DIRECT": [
        "B sits {bearing} at range {dist}. The {open} side is already open, so no deception is "
        "needed — I drive straight for the goal along the {open} corridor. Decision: honest direct.",
        "Range to B is {dist} and he is {bearing}. With the {open} lane clear I simply commit to it "
        "and push for the goal. Decision: go direct, no feint.",
        "B tracks me with a ~{lag}-step lag but he is {bearing} and the {open} corridor is open. "
        "I take the direct route on the {open} side. Decision: honest direct.",
        "No trick required: B is {bearing}, {dist} away, and the {open} side is free. I head "
        "straight through on the {open} corridor. Decision: direct approach.",
    ],
    "FEINT_LEFT_GO_RIGHT": [
        "B mirrors my heading with a ~{lag}-step lag and sits {bearing}. If I step left now he will "
        "commit left, opening the right side for about two steps. Decision: feint left, then cut right.",
        "Distance to B is {dist} and his pursuit runs about {lag} steps behind my true motion. A short "
        "burst leftward should drag his commitment left; I then cut right through the gap. "
        "Decision: feint left, go right.",
        "B is {bearing} at range {dist}, steering off my stale heading (~{lag} steps old). I sell a "
        "move to the left to pull him across, then break right past him. Decision: feint left, cut right.",
        "Because B lags my motion by roughly {lag} steps, a left feint now makes him drift left while "
        "I swing right around him ({dist} away, {bearing}). Decision: feint left, then go right.",
    ],
    "FEINT_RIGHT_GO_LEFT": [
        "B mirrors my heading with a ~{lag}-step lag and sits {bearing}. If I step right now he will "
        "commit right, opening the left side for about two steps. Decision: feint right, then cut left.",
        "Distance to B is {dist} and his pursuit runs about {lag} steps behind my true motion. A short "
        "burst rightward should drag his commitment right; I then cut left through the gap. "
        "Decision: feint right, go left.",
        "B is {bearing} at range {dist}, steering off my stale heading (~{lag} steps old). I sell a "
        "move to the right to pull him across, then break left past him. Decision: feint right, cut left.",
        "Because B lags my motion by roughly {lag} steps, a right feint now makes him drift right while "
        "I swing left around him ({dist} away, {bearing}). Decision: feint right, then go left.",
    ],
    "HESITATION_FAKE": [
        "B reads my heading with a ~{lag}-step delay and sits {bearing} at {dist}. If I stall for a "
        "couple of steps he gets no fresh heading to commit to, then I burst {open} before he can "
        "react. Decision: fake hesitation, then burst {open}.",
        "B is {bearing}, range {dist}. I freeze briefly so his lagged estimate goes stale and his "
        "commitment stalls, then I explode toward the {open} side. Decision: hesitate, then burst {open}.",
        "His tracker runs ~{lag} steps behind me, so a short stall starves it of signal. After the "
        "pause I sprint {open} past him ({bearing}, {dist} away). Decision: hesitation fake, go {open}.",
        "A brief stutter now leaves B ({bearing}, {dist}) without a heading to chase; I then commit "
        "hard to the {open} corridor. Decision: fake hesitation, burst {open}.",
    ],
}


def side_word(x):
    return "left" if x < 0 else "right"


def gen_episode(episode_idx, seed, image_size, render=True, out_root=None):
    env, policy, decision, rng = make_episode_env(episode_idx, seed=seed, image_size=image_size)
    obs = env.reset()
    actions = []
    for _ in range(EPISODE_LEN):
        a = np.clip(policy.act(obs), -1.0, 1.0)
        actions.append(a)
        obs, _, done, _ = env.step(a)
    kf = policy.keyframe
    assert kf is not None and 8 <= kf <= 24, f"bad keyframe {kf} ep {episode_idx}"
    chunk = np.stack(actions[kf:kf + CHUNK_LEN])  # commanded (clipped) actions

    ego = np.asarray(env.ego_hist)
    opp = np.asarray(env.opp_hist)
    deceived = outcome_deceived(ego, opp, kf)

    # ---- reasoning from decision-keyed templates, slots from state <= kf only
    e_kf, o_kf = ego[kf], opp[kf]
    slots = {
        "bearing": f"{side_word(o_kf[0] - 0.5)}-of-center",
        "dist": f"{np.linalg.norm(e_kf - o_kf):.2f}",
        "lag": "3",
        "open": side_word(policy.open_side if policy.open_side is not None else 1.0),
    }
    variants = TEMPLATES[decision]
    reasoning = variants[int(rng.integers(0, len(variants)))].format(**slots)

    # post-keyframe values that must NOT appear in the reasoning (leakage check)
    k5, k8 = min(kf + 5, EPISODE_LEN), min(kf + 8, EPISODE_LEN)
    forbidden = sorted({
        f"{np.linalg.norm(ego[k5] - opp[k5]):.2f}",
        f"{opp[k5, 0]:.2f}", f"{ego[k8, 0]:.2f}", f"{ego[k8, 1]:.2f}",
        "deceived", "succeeded",
    })

    ep_id = f"ep{episode_idx:04d}"
    frame_paths = [f"frames/{ep_id}/t{t:02d}.png" for t in range(kf - 3, kf + 1)]
    if render and out_root is not None:
        fdir = out_root / "frames" / ep_id
        fdir.mkdir(parents=True, exist_ok=True)
        for t in range(kf - 3, kf + 1):
            render_state(ego[t], opp[t], image_size).save(out_root / f"frames/{ep_id}/t{t:02d}.png")

    record = {
        "episode_id": ep_id,
        "t": int(kf),
        "keyframe_t": int(kf),
        "frames": frame_paths,
        "task_prompt": TASK_PROMPT,
        "decision": decision,
        "target_false_belief": FALSE_BELIEFS[decision],
        "reasoning": reasoning,
        "action_chunk": [[round(float(v), 6) for v in row] for row in chunk],
        "outcome_deceived": deceived,
    }
    slot_entry = {"episode_id": ep_id, "used": slots, "forbidden": forbidden}
    return record, slot_entry


def generate(n_train, n_val, seed, image_size, out_root, render=True):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    records, slotlog = [], []
    for i in range(n_train + n_val):
        rec, slot = gen_episode(i, seed, image_size, render=render, out_root=out_root)
        records.append(rec)
        slotlog.append(slot)
    train, val = records[:n_train], records[n_train:]
    if render:
        for name, recs in [("train.jsonl", train), ("val.jsonl", val)]:
            with open(out_root / name, "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        with open(out_root / "slotlog.json", "w") as f:
            json.dump(slotlog, f, indent=1)
        dec_counts = {d: sum(r["decision"] == d for r in records) for d in DECISIONS}
        dec_rate = {d: float(np.mean([r["outcome_deceived"] for r in records if r["decision"] == d]))
                    for d in DECISIONS}
        meta = {"seed": seed, "n_train": n_train, "n_val": n_val, "image_size": image_size,
                "decision_counts": dec_counts, "deceived_rate_by_decision": dec_rate}
        with open(out_root / "meta.json", "w") as f:
            json.dump(meta, f, indent=1)
    return records


def records_hash(records):
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/toy")
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--check-determinism", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    recs = generate(args.n_train, args.n_val, args.seed, args.image_size, args.out, render=True)
    h1 = records_hash(recs)
    print(f"generated {len(recs)} samples -> {args.out}  DATA_HASH={h1}")

    if args.check_determinism:
        recs2 = generate(args.n_train, args.n_val, args.seed, args.image_size, args.out, render=False)
        h2 = records_hash(recs2)
        assert h1 == h2, f"determinism check FAILED: {h1} != {h2}"
        print(f"determinism check PASS (regenerated hash {h2})")

    dec_rate = float(np.mean([r["outcome_deceived"] for r in recs if r["decision"] != "HONEST_DIRECT"]))
    print(f"deceived rate over deceptive decisions: {dec_rate:.2f}")
    record_timing("data_gen", time.time() - t0, {"n": len(recs), "hash": h1})


if __name__ == "__main__":
    main()
