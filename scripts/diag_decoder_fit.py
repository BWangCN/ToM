"""Discriminating experiment: can the flow decoder fit the 32 Stage-A chunks
when trained ALONE on frozen (detached) conditioning states?

- converges  -> architecture/conditioning fine; the joint smoke simply gives the
  from-scratch DiT too few updates -> add decoder inner-loop updates.
- plateaus   -> the conditioning pathway itself is broken -> debug spans/cond.
"""

import argparse
import json
import sys
import time
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
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--logit-normal-t", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(0)
    data_root = REPO_ROOT / cfg["data"]["root"]
    recs = [json.loads(l) for l in (data_root / "train.jsonl").read_text().splitlines()][:32]

    dvla = DVLA.load_trained(cfg, REPO_ROOT / args.checkpoint)
    dvla.model.eval()
    autopen = detect_think_autopen(dvla.processor)

    # precompute frozen conditioning for all samples
    conds, masks, chunks = [], [], []
    for rec in recs:
        enc = encode_sample(rec, dvla.processor, dvla.tok_info, data_root,
                            n_frames=cfg["inputs"]["n_frames"], stage="A", autopen=autopen)
        b = collate([enc], dvla.tokenizer.pad_token_id)
        h = dvla.hidden_states_for(b["input_ids"], b["attention_mask"], b["pixel_values"],
                                   b["image_grid_thw"], b.get("mm_token_type_ids"))
        cond, mask = dvla._slice_cond(h, b[dvla.cond_span_key()].to(dvla.device), detach=True)
        conds.append(cond[0].float())
        masks.append(mask[0])
        chunks.append(b["action_chunk"][0].to(dvla.device).float())
    Lc = max(c.shape[0] for c in conds)
    C = torch.zeros(len(conds), Lc, conds[0].shape[-1], device=dvla.device)
    M = torch.ones(len(conds), Lc, dtype=torch.bool, device=dvla.device)
    for i, (c, m) in enumerate(zip(conds, masks)):
        C[i, : c.shape[0]] = c
        M[i, : c.shape[0]] = m
    X1 = torch.stack(chunks)
    print(f"cond bank: {tuple(C.shape)}, chunks {tuple(X1.shape)}")

    dec = dvla.decoder.train()
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, fused=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    g = torch.Generator(device=dvla.device).manual_seed(0)
    rng = np.random.default_rng(0)

    def probe():
        dec.eval()
        gp = torch.Generator(device=dvla.device).manual_seed(777)
        with torch.no_grad():
            losses = [float(dec.flow_loss(X1[i:i+8], C[i:i+8], M[i:i+8], generator=gp))
                      for i in range(0, 32, 8)]
            l2 = []
            for i in range(0, 32, 8):
                s = dec.sample(C[i:i+8], M[i:i+8], steps=10,
                               generator=torch.Generator(device=dvla.device).manual_seed(0))
                l2 += [float((s[j] - X1[i+j]).norm()) for j in range(s.shape[0])]
        dec.train()
        return float(np.mean(losses)), float(np.mean(l2))

    f0, l0 = probe()
    print(f"start: flow probe {f0:.4f} | sampled L2 {l0:.4f}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.tensor(rng.choice(32, size=args.bs, replace=False), device=dvla.device)
        x1 = X1[idx]
        if args.logit_normal_t:
            t = torch.sigmoid(torch.randn(args.bs, device=dvla.device, generator=g))
            x0 = torch.randn(x1.shape, device=dvla.device, generator=g)
            x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
            v_pred = dec(x_t, t, C[idx], M[idx])
            loss = torch.nn.functional.mse_loss(v_pred, x1 - x0)
        else:
            loss = dec.flow_loss(x1, C[idx], M[idx], generator=g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 400 == 0 or step == args.steps:
            f, l2 = probe()
            print(f"step {step}: flow probe {f:.4f} | sampled L2 {l2:.4f} | "
                  f"{(time.time()-t0)/step*1000:.0f} ms/step")


if __name__ == "__main__":
    main()
