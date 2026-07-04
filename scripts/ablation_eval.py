"""Three-arm ablation harness (CLAUDE.md §7.11).

  (a) discrete-only : default ckpt, decoder bypassed, argmax dequantization
  (b) AR1-style     : default ckpt, flow decode, condition = think+act spans
  (c) pi0.5-style   : think_only ckpt, flow decode, condition = think span;
                      plus timing of the emit_discrete_at_inference=false path

Open-loop: 20 val samples (chunk L2 vs GT). Closed-loop: 20 toy-env rollouts
per arm (seed 0), model in the loop, replanning every 8 steps. Perturbation
ratio per §7.9 (arm (a) is deterministic -> noise=0, we report the raw shift).

Pass = the harness completes and the table is present; no criterion on which
arm wins (smoke validates the harness, not the science).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.collator import build_messages, detect_think_autopen, load_frames
from dvla.common import REPO_ROOT, Timer, load_config, record_timing, seed_everything
from dvla.constrained_decode import discrete_chunk, generate_reason_act
from dvla.model_wrapper import DVLA
from dvla.toy_env import EPISODE_LEN, make_episode_env, render_state, rollout_metrics
from perturb_test import run_perturb

CHUNK = 8


def encode_prompt_batch(dvla, images_lists, task_prompts):
    texts = []
    for imgs, tp in zip(images_lists, task_prompts):
        msgs = build_messages(tp, len(imgs))
        texts.append(dvla.processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                        tokenize=False))
    tok = dvla.tokenizer
    old = tok.padding_side
    tok.padding_side = "left"
    enc = dvla.processor(text=texts, images=images_lists, return_tensors="pt", padding=True)
    tok.padding_side = old
    return enc


def flow_from_generated(dvla, seqs, prompt_enc, rows, condition, flow_seed=0):
    pad_id = dvla.tokenizer.pad_token_id
    att = (seqs != pad_id).long()
    mm = None
    if "mm_token_type_ids" in prompt_enc and prompt_enc["mm_token_type_ids"] is not None:
        pm = prompt_enc["mm_token_type_ids"]
        mm = torch.cat([pm, torch.zeros(seqs.shape[0], seqs.shape[1] - pm.shape[1],
                                        dtype=pm.dtype)], dim=1)
    t0 = time.time()
    h = dvla.hidden_states_for(seqs, att, prompt_enc["pixel_values"],
                               prompt_enc["image_grid_thw"], mm)
    t_hidden = time.time() - t0
    spans, valid = [], []
    for r in rows:
        s = r["spans"]
        if condition == "think_and_act" and s.get("act_end", -1) > 0:
            spans.append((s["think_open"], s["act_end"] + 1)); valid.append(True)
        elif condition == "think_only" and s.get("think_close", -1) > 0:
            spans.append((s["think_open"], s["think_close"] + 1)); valid.append(True)
        else:
            spans.append((max(s.get("think_open", 0), 0), max(s.get("think_open", 0), 0) + 1))
            valid.append(False)
    t0 = time.time()
    chunks = dvla.flow_decode(h, torch.tensor(spans, device=dvla.device), seed=flow_seed)
    t_flow = time.time() - t0
    out = []
    for i, ok in enumerate(valid):
        out.append(chunks[i].float().cpu().numpy() if ok else np.zeros((CHUNK, 2)))
    return out, valid, t_hidden, t_flow


def open_loop(dvla, recs, data_root, cfg, arms, bs=5):
    """arms subset of {a,b,c-like}: returns per-arm {l2, ms_per_step, malformed}."""
    res = {arm: {"l2": [], "wall": 0.0, "malformed": 0} for arm in arms}
    think_budget = cfg["train"]["reasoning_max_new_tokens"]
    for i in range(0, len(recs), bs):
        grp = recs[i:i + bs]
        imgs = [load_frames(r, data_root, cfg["inputs"]["n_frames"]) for r in grp]
        enc = encode_prompt_batch(dvla, imgs, [r["task_prompt"] for r in grp])
        out = generate_reason_act(dvla, enc, think_budget=think_budget, greedy=True)
        gts = [np.asarray(r["action_chunk"]) for r in grp]
        if "a" in arms:
            res["a"]["wall"] += out["wall"]
            for row, gt in zip(out["rows"], gts):
                ch = discrete_chunk(row)
                if ch is None:
                    res["a"]["malformed"] += 1
                    ch = np.zeros((CHUNK, 2))
                res["a"]["l2"].append(float(np.linalg.norm(ch - gt)))
        flow_arm = "b" if "b" in arms else ("c" if "c" in arms else None)
        if flow_arm:
            cond = "think_and_act" if flow_arm == "b" else "think_only"
            chunks, valid, th, tf = flow_from_generated(dvla, out["sequences"], enc,
                                                        out["rows"], cond)
            res[flow_arm]["wall"] += out["wall"] + th + tf
            for ch, ok, gt in zip(chunks, valid, gts):
                if not ok:
                    res[flow_arm]["malformed"] += 1
                res[flow_arm]["l2"].append(float(np.linalg.norm(ch - gt)))
    n = len(recs)
    for arm in arms:
        r = res[arm]
        r["chunk_l2_mean"] = round(float(np.mean(r["l2"])), 4)
        r["ms_per_step"] = round(r["wall"] / (n * CHUNK) * 1000, 1)
        del r["l2"], r["wall"]
    return res


def emit_false_timing(dvla, recs, data_root, cfg, bs=5):
    """Arm (c) extra: generation stops at </think>; decoder conditions on think only."""
    wall = 0.0
    n = len(recs)
    for i in range(0, n, bs):
        grp = recs[i:i + bs]
        imgs = [load_frames(r, data_root, cfg["inputs"]["n_frames"]) for r in grp]
        enc = encode_prompt_batch(dvla, imgs, [r["task_prompt"] for r in grp])
        out = generate_reason_act(dvla, enc, think_budget=cfg["train"]["reasoning_max_new_tokens"],
                                  emit_discrete=False, greedy=True)
        _, _, th, tf = flow_from_generated(dvla, out["sequences"], enc, out["rows"], "think_only")
        wall += out["wall"] + th + tf
    return round(wall / (n * CHUNK) * 1000, 1)


def closed_loop(dvla, cfg, arm, n_rollouts=20, bs=5, base_idx=10000):
    from dvla.toy_env import ToyEnv  # noqa: F401 (doc pointer)
    from dvla.collator import SYSTEM_PROMPT  # noqa: F401
    task_prompt = ("You are agent A (blue). Reach the green goal square; "
                   "agent B (red) is guarding. Plan your next 8 steps (0.8 s).")
    think_budget = cfg["train"]["reasoning_max_new_tokens"]
    cond = {"a": None, "b": "think_and_act", "c": "think_only"}[arm]

    envs = []
    for i in range(n_rollouts):
        env, _, _, _ = make_episode_env(base_idx + i, seed=cfg["seed"],
                                        image_size=cfg["inputs"]["image_size"])
        env.reset()
        envs.append(env)

    def frames_of(env):
        idx = env.t
        ids = [max(0, idx - k) for k in (3, 2, 1, 0)]
        return [render_state(env.ego_hist[k], env.opp_hist[k], cfg["inputs"]["image_size"])
                for k in ids]

    wall = 0.0
    n_malformed = 0
    for plan_t in range(0, EPISODE_LEN, CHUNK):
        chunks_all = []
        for g in range(0, n_rollouts, bs):
            grp = envs[g:g + bs]
            imgs = [frames_of(e) for e in grp]
            enc = encode_prompt_batch(dvla, imgs, [task_prompt] * len(grp))
            t0 = time.time()
            out = generate_reason_act(dvla, enc, think_budget=think_budget, greedy=True)
            if arm == "a":
                wall += time.time() - t0
                for row in out["rows"]:
                    ch = discrete_chunk(row)
                    if ch is None:
                        n_malformed += 1
                        ch = np.zeros((CHUNK, 2))
                    chunks_all.append(ch)
            else:
                chunks, valid, th, tf = flow_from_generated(dvla, out["sequences"], enc,
                                                            out["rows"], cond)
                wall += time.time() - t0  # includes gen; hidden+flow inside t0..now
                n_malformed += sum(not v for v in valid)
                chunks_all.extend(chunks)
        for env, ch in zip(envs, chunks_all):
            for k in range(CHUNK):
                env.step(ch[k])
    mets = [rollout_metrics(e.ego_hist, e.opp_hist) for e in envs]
    return {
        "deceived_rate": round(float(np.mean([m["deceived"] for m in mets])), 3),
        "goal_rate": round(float(np.mean([m["goal_reached"] for m in mets])), 3),
        "ms_per_step": round(wall / (n_rollouts * EPISODE_LEN) * 1000, 1),
        "malformed": n_malformed,
    }


def discrete_perturb_shift(dvla, val_recs, data_root, cfg, n_samples=8):
    """Arm (a): teacher-force true vs swapped reasoning, constrained greedy act
    decode (deterministic -> noise=0); report the raw chunk shift."""
    shifts = []
    autopen = detect_think_autopen(dvla.processor)
    for i in range(n_samples):
        rec = val_recs[i]
        j = next(k % len(val_recs) for k in range(i + 1, i + len(val_recs))
                 if val_recs[k % len(val_recs)]["decision"] != rec["decision"])
        chunks = []
        for reasoning in (rec["reasoning"], val_recs[j]["reasoning"]):
            imgs = [load_frames(rec, data_root, cfg["inputs"]["n_frames"])]
            msgs = build_messages(rec["task_prompt"], len(imgs[0]))
            prompt = dvla.processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                        tokenize=False)
            prefix = prompt + ("" if autopen else "<think>\n") + reasoning + "\n</think>\n\n"
            enc = dvla.processor(text=[prefix], images=imgs, return_tensors="pt")
            out = generate_reason_act(dvla, enc, think_budget=1, greedy=True)
            ch = discrete_chunk(out["rows"][0])
            chunks.append(ch if ch is not None else np.zeros((CHUNK, 2)))
        shifts.append(float(np.linalg.norm(chunks[0] - chunks[1])))
    return round(float(np.median(shifts)), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/laptop.yaml"))
    ap.add_argument("--default-ckpt", required=True)
    ap.add_argument("--thinkonly-ckpt", required=True)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports/ablation.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    data_root = REPO_ROOT / cfg["data"]["root"]
    val_recs = [json.loads(l) for l in (data_root / "val.jsonl").read_text().splitlines()]
    val_n = cfg["eval"]["ablation_val_n"]
    n_roll = cfg["eval"]["ablation_rollouts"]
    timer = Timer()
    results = {}

    # ---------------- phase 1: default checkpoint (arms a, b)
    print("[ablation] loading default checkpoint (arms a+b)")
    dvla = DVLA.load_trained(cfg, args.default_ckpt)
    dvla.model.eval(); dvla.decoder.eval()
    ol = open_loop(dvla, val_recs[:val_n], data_root, cfg, arms=("a", "b"))
    results["a"] = {**ol["a"], "checkpoint": "default", "pathway": "discrete argmax"}
    results["b"] = {**ol["b"], "checkpoint": "default", "pathway": "flow, cond=think+act"}
    results["b"]["perturb"] = run_perturb(dvla, val_recs, data_root, cfg, "think_and_act",
                                          cfg["eval"]["perturb_n_samples"],
                                          cfg["eval"]["perturb_n_seeds"])
    results["a"]["perturb_shift_median"] = discrete_perturb_shift(dvla, val_recs, data_root, cfg)
    print("[ablation] closed-loop arm a"); results["a"]["rollout"] = closed_loop(dvla, cfg, "a", n_roll)
    print("[ablation] closed-loop arm b"); results["b"]["rollout"] = closed_loop(dvla, cfg, "b", n_roll)
    del dvla
    torch.cuda.empty_cache()

    # ---------------- phase 2: think_only checkpoint (arm c)
    print("[ablation] loading think_only checkpoint (arm c)")
    cfg_c = load_config(args.config)
    cfg_c["decoder"]["condition"] = "think_only"
    dvla = DVLA.load_trained(cfg_c, args.thinkonly_ckpt)
    dvla.model.eval(); dvla.decoder.eval()
    ol = open_loop(dvla, val_recs[:val_n], data_root, cfg_c, arms=("c",))
    results["c"] = {**ol["c"], "checkpoint": "think_only", "pathway": "flow, cond=think"}
    results["c"]["perturb"] = run_perturb(dvla, val_recs, data_root, cfg_c, "think_only",
                                          cfg["eval"]["perturb_n_samples"],
                                          cfg["eval"]["perturb_n_seeds"])
    results["c"]["emit_false_ms_per_step"] = emit_false_timing(dvla, val_recs[:val_n],
                                                               data_root, cfg_c)
    print("[ablation] closed-loop arm c"); results["c"]["rollout"] = closed_loop(dvla, cfg_c, "c", n_roll)
    del dvla
    torch.cuda.empty_cache()

    # ---------------- table
    rows = []
    head = ("| arm | pathway | chunk L2 vs GT | rollout deceived | rollout goal | "
            "perturb ratio | open-loop ms/step | rollout ms/step | notes |")
    rows.append(head)
    rows.append("|" + "---|" * 9)
    for arm in ("a", "b", "c"):
        r = results[arm]
        ratio = (f"{r['perturb']['median_ratio']}" if "perturb" in r
                 else f"shift={r['perturb_shift_median']} (det.)")
        notes = []
        if r.get("malformed"):
            notes.append(f"{r['malformed']} malformed(open)")
        if r["rollout"].get("malformed"):
            notes.append(f"{r['rollout']['malformed']} malformed(roll)")
        if arm == "c":
            notes.append(f"emit_discrete=false: {r['emit_false_ms_per_step']} ms/step")
        rows.append(f"| ({arm}) | {r['pathway']} | {r['chunk_l2_mean']} | "
                    f"{r['rollout']['deceived_rate']} | {r['rollout']['goal_rate']} | "
                    f"{ratio} | {r['ms_per_step']} | {r['rollout']['ms_per_step']} | "
                    f"{'; '.join(notes) or '—'} |")
    table = "\n".join(rows)
    (REPO_ROOT / "reports/ablation_table.md").write_text(table + "\n")
    Path(args.out).write_text(json.dumps(results, indent=1))
    record_timing("ablation_eval", timer.elapsed())
    print(table)
    print(f"[ablation] DONE in {timer.elapsed_min():.1f} min")


if __name__ == "__main__":
    main()
