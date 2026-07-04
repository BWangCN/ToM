"""Conditioning perturbation test (CLAUDE.md §7.9) — the seed of the Stage-C
consistency reward.

Operationalization (recorded in the report): the swap measures the CAUSAL
pathway reasoning -> action. For each val sample we teacher-force the reasoning
(true vs. swapped-in from a different-decision sample), let the model generate
its discrete action tokens under constrained greedy decoding, then flow-decode
5x with different flow seeds:

  noise = mean pairwise L2 among the 5 true-reasoning chunks
  shift = L2 between the mean chunks of true vs swapped reasoning
  PASS: median(shift/noise) >= 3

A reasoning-only swap that keeps the ORIGINAL GT act tokens in the conditioning
is also reported ("decoder_direct") — under think_and_act conditioning the act
tokens are a sufficient statistic for the chunk, so this secondary number
isolates how much the decoder reads the think span *beyond* the act tokens.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.collator import (build_messages, collate, detect_think_autopen,
                           encode_sample, load_frames)
from dvla.common import REPO_ROOT, load_config, record_metric, seed_everything
from dvla.constrained_decode import generate_reason_act
from dvla.model_wrapper import DVLA


def _pairwise_l2(chunks):
    n = len(chunks)
    ds = [float(np.linalg.norm(chunks[i] - chunks[j]))
          for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(ds))


def _flow_seeds_from_row(dvla, seqs, enc, row, condition, seeds):
    """Flow-decode one generated row with several flow seeds."""
    pad_id = dvla.tokenizer.pad_token_id
    att = (seqs != pad_id).long()
    mm = None
    if "mm_token_type_ids" in enc and enc["mm_token_type_ids"] is not None:
        pm = enc["mm_token_type_ids"]
        mm = torch.cat([pm, torch.zeros(seqs.shape[0], seqs.shape[1] - pm.shape[1],
                                        dtype=pm.dtype)], dim=1)
    h = dvla.hidden_states_for(seqs, att, enc["pixel_values"], enc["image_grid_thw"], mm)
    s = row["spans"]
    if condition == "think_and_act":
        span = (s["think_open"], s["act_end"] + 1)
    else:
        span = (s["think_open"], s["think_close"] + 1)
    spans = torch.tensor([span], device=dvla.device)
    return [dvla.flow_decode(h, spans, seed=sd)[0].float().cpu().numpy() for sd in seeds]


def _gen_acts_for_reasoning(dvla, rec, reasoning, data_root, cfg, autopen):
    imgs = [load_frames(rec, data_root, cfg["inputs"]["n_frames"])]
    msgs = build_messages(rec["task_prompt"], len(imgs[0]))
    prompt = dvla.processor.apply_chat_template(msgs, add_generation_prompt=True,
                                                tokenize=False)
    prefix = prompt + ("" if autopen else "<think>\n") + reasoning + "\n</think>\n\n"
    enc = dvla.processor(text=[prefix], images=imgs, return_tensors="pt")
    out = generate_reason_act(dvla, enc, think_budget=8, greedy=True)
    return enc, out


def run_perturb(dvla, val_recs, data_root, cfg, condition, n_samples=8, n_seeds=5):
    autopen = detect_think_autopen(dvla.processor)
    seeds = [100 + k for k in range(n_seeds)]
    per_sample = []
    for i in range(n_samples):
        rec = val_recs[i]
        j = next(k % len(val_recs) for k in range(i + 1, i + len(val_recs))
                 if val_recs[k % len(val_recs)]["decision"] != rec["decision"])
        swap_rec = val_recs[j]

        chunks, bins = {}, {}
        for name, reasoning in (("true", rec["reasoning"]), ("swap", swap_rec["reasoning"])):
            enc, out = _gen_acts_for_reasoning(dvla, rec, reasoning, data_root, cfg, autopen)
            row = out["rows"][0]
            bins[name] = row["bins"]
            chunks[name] = _flow_seeds_from_row(dvla, out["sequences"], enc, row,
                                                condition, seeds)
        noise = _pairwise_l2(chunks["true"])
        shift = float(np.linalg.norm(np.mean(chunks["true"], 0) - np.mean(chunks["swap"], 0)))
        ratio = shift / max(noise, 1e-9)

        # secondary: decoder-direct sensitivity (teacher-forced GT act tokens)
        enc_t = encode_sample(rec, dvla.processor, dvla.tok_info, data_root,
                              n_frames=cfg["inputs"]["n_frames"], stage="B", autopen=autopen)
        swapped = dict(rec)
        swapped["reasoning"] = swap_rec["reasoning"]
        enc_s = encode_sample(swapped, dvla.processor, dvla.tok_info, data_root,
                              n_frames=cfg["inputs"]["n_frames"], stage="B", autopen=autopen)
        dd = {}
        for name, e in (("true", enc_t), ("swap", enc_s)):
            b = collate([e], dvla.tokenizer.pad_token_id)
            h = dvla.hidden_states_for(b["input_ids"], b["attention_mask"],
                                       b["pixel_values"], b["image_grid_thw"],
                                       b.get("mm_token_type_ids"))
            spans = b[dvla.cond_span_key(condition)].to(dvla.device)
            dd[name] = [dvla.flow_decode(h, spans, seed=sd)[0].float().cpu().numpy()
                        for sd in seeds]
        dd_noise = _pairwise_l2(dd["true"])
        dd_shift = float(np.linalg.norm(np.mean(dd["true"], 0) - np.mean(dd["swap"], 0)))

        per_sample.append({
            "episode_id": rec["episode_id"], "decision": rec["decision"],
            "swap_from": swap_rec["episode_id"], "swap_decision": swap_rec["decision"],
            "acts_changed": bins["true"] != bins["swap"],
            "noise": round(noise, 6), "shift": round(shift, 6), "ratio": round(ratio, 3),
            "decoder_direct_noise": round(dd_noise, 6),
            "decoder_direct_shift": round(dd_shift, 6),
            "decoder_direct_ratio": round(dd_shift / max(dd_noise, 1e-9), 3),
        })
    ratios = [s["ratio"] for s in per_sample]
    return {
        "condition": condition,
        "method": "causal-pathway swap: teacher-forced reasoning -> constrained "
                  "greedy act generation -> flow decode (see module docstring)",
        "n_samples": n_samples, "n_seeds": n_seeds,
        "per_sample": per_sample,
        "median_ratio": round(float(np.median(ratios)), 3),
        "median_noise": round(float(np.median([s["noise"] for s in per_sample])), 6),
        "median_shift": round(float(np.median([s["shift"] for s in per_sample])), 6),
        "median_decoder_direct_ratio": round(float(np.median(
            [s["decoder_direct_ratio"] for s in per_sample])), 3),
        "acts_changed_frac": round(float(np.mean(
            [s["acts_changed"] for s in per_sample])), 3),
        "pass": bool(np.median(ratios) >= cfg["eval"]["perturb_pass_ratio"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/laptop.yaml"))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports/perturb.json"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    condition = args.condition or cfg["decoder"]["condition"]
    data_root = REPO_ROOT / cfg["data"]["root"]
    val_recs = [json.loads(l) for l in (data_root / "val.jsonl").read_text().splitlines()]

    dvla = DVLA.load_trained(cfg, args.checkpoint)
    dvla.model.eval()
    dvla.decoder.eval()
    res = run_perturb(dvla, val_recs, data_root, cfg, condition,
                      n_samples=cfg["eval"]["perturb_n_samples"],
                      n_seeds=cfg["eval"]["perturb_n_seeds"])
    Path(args.out).write_text(json.dumps(res, indent=1))
    record_metric("perturb_median_ratio", res["median_ratio"])
    print(f"PERTURB median(shift/noise) = {res['median_ratio']} "
          f"(noise {res['median_noise']}, shift {res['median_shift']}; "
          f"acts changed {res['acts_changed_frac']}; decoder-direct ratio "
          f"{res['median_decoder_direct_ratio']}) -> "
          f"{'PASS' if res['pass'] else 'FAIL'}")
    if not res["pass"]:
        sys.exit(3)


if __name__ == "__main__":
    main()
