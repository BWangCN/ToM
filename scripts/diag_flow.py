"""Diagnostic: decompose the flow-matching MSE by timestep t and measure the
quantity that actually matters — Euler-sampled chunk error vs GT.

Rectified flow with x0~N(0,I) has an IRREDUCIBLE loss component as t->1: given
x_t ~= x1, the target v = x1 - x0 contains x0, which is unrecoverable (variance
~1 per element). If the low-t buckets are ~0 while high-t buckets carry the
residual, the decoder has converged; the remaining MSE is the floor, not error.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.collator import collate, detect_think_autopen, encode_sample
from dvla.common import REPO_ROOT, load_config, seed_everything
from dvla.model_wrapper import DVLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/stage_a/ckpt_final")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(0)
    data_root = REPO_ROOT / cfg["data"]["root"]
    recs = [json.loads(l) for l in (data_root / "train.jsonl").read_text().splitlines()][: args.n]

    dvla = DVLA.load_trained(cfg, REPO_ROOT / args.checkpoint)
    dvla.model.eval()
    dvla.decoder.eval()
    autopen = detect_think_autopen(dvla.processor)

    t_grid = [0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99]
    per_t = {t: [] for t in t_grid}
    samp_l2, samp_max = [], []
    g = torch.Generator(device=dvla.device)

    for i, rec in enumerate(recs):
        enc = encode_sample(rec, dvla.processor, dvla.tok_info, data_root,
                            n_frames=cfg["inputs"]["n_frames"], stage=args.stage,
                            autopen=autopen)
        b = collate([enc], dvla.tokenizer.pad_token_id)
        h = dvla.hidden_states_for(b["input_ids"], b["attention_mask"], b["pixel_values"],
                                   b["image_grid_thw"], b.get("mm_token_type_ids"))
        spans = b[dvla.cond_span_key()].to(dvla.device)
        cond, mask = dvla._slice_cond(h, spans, detach=True)
        x1 = b["action_chunk"].to(dvla.device).float()

        with torch.no_grad():
            for t in t_grid:
                errs = []
                for s in range(5):
                    g.manual_seed(1000 * i + s)
                    x0 = torch.randn(x1.shape, device=dvla.device, generator=g)
                    tt = torch.full((1,), t, device=dvla.device)
                    x_t = (1 - t) * x0 + t * x1
                    v = dvla.decoder(x_t, tt, cond, mask)
                    errs.append(float(((v - (x1 - x0)) ** 2).mean()))
                per_t[t].append(float(np.mean(errs)))
            chunk = dvla.flow_decode(h, spans, seed=0)[0]
            d = (chunk - x1[0]).float()
            samp_l2.append(float(d.norm()))
            samp_max.append(float(d.abs().max()))

    print("flow MSE by timestep t (mean over samples; 5 fixed x0 draws each):")
    for t in t_grid:
        print(f"  t={t:>4}: {np.mean(per_t[t]):.4f}")
    lo = float(np.mean([np.mean(per_t[t]) for t in (0.05, 0.2, 0.4, 0.6)]))
    hi = float(np.mean([np.mean(per_t[t]) for t in (0.9, 0.95, 0.99)]))
    print(f"low-t (<=0.6) mean: {lo:.4f} | high-t (>=0.9) mean: {hi:.4f}")
    print(f"Euler-10 sampled chunk vs GT: L2 mean {np.mean(samp_l2):.4f} "
          f"(over 8x2 chunk), max-abs mean {np.mean(samp_max):.4f}")
    print(f"(reference: quantization half-bin = {1/128:.4f}; GT chunk RMS "
          f"norm ~ {np.mean([np.linalg.norm(r['action_chunk']) for r in recs]):.3f})")


if __name__ == "__main__":
    main()
