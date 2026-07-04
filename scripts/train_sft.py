"""Stage A/B SFT smoke trainer — one code path, stages differ only in data and
loss weights (CLAUDE.md §4). batch=1 x grad-accum 8, gradient checkpointing ON.

Also implements: the 1-step no-OOM check (first micro step), peak-VRAM logging,
wall-clock guard (<= cfg.train.wall_clock_limit_min), checkpoint save/resume
with probe-loss continuity, canary hooks, and the stage acceptance criteria.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.canary import run_canary
from dvla.collator import collate, detect_think_autopen, encode_sample
from dvla.common import (REPO_ROOT, Timer, load_config, log_deviation,
                         peak_vram_gb, record_metric, record_timing, seed_everything)
from dvla.constrained_decode import generate_reason_act
from dvla.model_wrapper import DVLA

STAGE_LAMBDAS = {
    "A": {"lambda_reason": 0.0, "lambda_act_ce": 1.0, "lambda_flow": 1.0},
    "B": {"lambda_reason": 1.0, "lambda_act_ce": 1.0, "lambda_flow": 1.0},
}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def build_param_groups(dvla, cfg):
    lora, token, other = [], [], []
    for n, p in dvla.model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora.append(p)
        elif "trainable_tokens" in n or "embed_tokens" in n or "lm_head" in n:
            token.append(p)
        else:
            other.append(p)
    assert not other, f"unclassified trainable params: {len(other)}"
    assert lora and token, "missing lora or token-row params"
    groups = [
        {"params": lora, "lr": cfg["optim"]["lora_lr"], "name": "lora"},
        {"params": token, "lr": cfg["optim"]["token_lr"], "name": "token_rows"},
        {"params": list(dvla.decoder.parameters()), "lr": cfg["optim"]["decoder_lr"],
         "name": "decoder"},
    ]
    return groups


def cast_trainables_fp32(dvla):
    for p in dvla.model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()


@torch.no_grad()
def probe_loss(dvla, probe_batch):
    dvla.model.eval()
    dvla.decoder.eval()
    g = torch.Generator(device=dvla.device).manual_seed(1234)
    total, parts = dvla.forward_train(probe_batch, flow_generator=g)
    dvla.model.train()
    dvla.decoder.train()
    return float(total)


def rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_ckpt(dvla, opt, sched, out_dir, step, probe, flow_gen_state, init_losses=None):
    out_dir = Path(out_dir)
    dvla.save(out_dir)
    torch.save({
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "step": step,
        "probe_loss": probe,
        "rng": rng_state(),
        "flow_gen": flow_gen_state,
        "init_losses": init_losses,
    }, out_dir / "state.pt")


@torch.no_grad()
def eval_probes(dvla, encoded, lambdas):
    """Deterministic eval-mode criteria measurement: token-weighted act/reason CE
    plus flow MSE averaged over all samples with a FIXED noise/t sequence —
    removes dropout and t-sampling noise from the per-step training averages."""
    dvla.model.eval()
    dvla.decoder.eval()
    g = torch.Generator(device=dvla.device).manual_seed(777)
    ce_a, n_a, ce_r, n_r, flows = 0.0, 0, 0.0, 0, []
    for enc in encoded:
        b = collate([enc], dvla.tokenizer.pad_token_id)
        b["lambdas"] = lambdas
        _, parts = dvla.forward_train(b, flow_generator=g)
        ce_a += parts["act_ce"] * parts["n_act_tokens"]
        n_a += parts["n_act_tokens"]
        ce_r += parts["reason_ce"] * parts["n_reason_tokens"]
        n_r += parts["n_reason_tokens"]
        flows.append(parts["flow_mse"])
    dvla.model.train()
    dvla.decoder.train()
    return {"act_ce": ce_a / max(n_a, 1), "reason_ce": ce_r / max(n_r, 1),
            "flow_mse": float(np.mean(flows))}


def greedy_match_count(dvla, samples, data_root, cfg, bs=5):
    """Constrained greedy decode; count exact 16-bin matches vs GT."""
    from dvla.action_quant import chunk_to_bins
    from dvla.collator import build_messages, load_frames

    dvla.model.eval()
    tok = dvla.tokenizer
    old_side = tok.padding_side
    tok.padding_side = "left"
    n_match = 0
    for i in range(0, len(samples), bs):
        chunk_recs = samples[i:i + bs]
        texts, images = [], []
        for r in chunk_recs:
            imgs = load_frames(r, data_root, cfg["inputs"]["n_frames"])
            msgs = build_messages(r["task_prompt"], len(imgs))
            texts.append(dvla.processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False))
            images.append(imgs)
        enc = dvla.processor(text=texts, images=images, return_tensors="pt", padding=True)
        out = generate_reason_act(dvla, enc,
                                  think_budget=cfg["train"]["reasoning_max_new_tokens"],
                                  emit_discrete=True, greedy=True)
        for r, row in zip(chunk_recs, out["rows"]):
            gt = chunk_to_bins(np.asarray(r["action_chunk"])).tolist()
            if row["bins"] is not None and row["bins"] == gt:
                n_match += 1
    tok.padding_side = old_side
    dvla.model.train()
    return n_match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/laptop.yaml"))
    ap.add_argument("--stage", choices=["A", "B"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-from", default=None, help="checkpoint dir to start from")
    ap.add_argument("--resume", default=None, help="checkpoint dir to resume (with state.pt)")
    ap.add_argument("--resume-extra-steps", type=int, default=10)
    ap.add_argument("--save-mid", type=int, default=None, help="save full state at this opt step")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--decoder-condition", default=None,
                    choices=[None, "think_and_act", "think_only"])
    ap.add_argument("--canary", default="none", choices=["none", "pre", "post", "both"])
    ap.add_argument("--canary-pre-tag", default="pre")
    ap.add_argument("--canary-post-tag", default="post")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--init-losses-json", default=None,
                    help="JSON dict of initial losses (for resuming checkpoints "
                         "saved before init_losses was recorded)")
    ap.add_argument("--no-criteria-gate", action="store_true",
                    help="do not exit(3) when stage criteria are unmet — for the "
                         "§7.12 resume check, which only asserts loss continuity")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.decoder_condition:
        cfg["decoder"]["condition"] = args.decoder_condition
    seed_everything(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True

    stage = args.stage
    scfg = cfg["stage_a" if stage == "A" else "stage_b"]
    tag = args.tag or f"stage_{stage.lower()}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = REPO_ROOT / cfg["data"]["root"]
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)

    timer = Timer()
    torch.cuda.reset_peak_memory_stats()

    # ---------------- model
    resume_state = None
    if args.resume:
        dvla = DVLA.load_trained(cfg, args.resume)
        resume_state = torch.load(Path(args.resume) / "state.pt", weights_only=False)
    elif args.init_from:
        dvla = DVLA.load_trained(cfg, args.init_from)
    else:
        dvla = DVLA.build(cfg)
    if cfg["optim"]["grad_checkpointing"]:
        dvla.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        dvla.model.enable_input_require_grads()
    cast_trainables_fp32(dvla)
    dvla.model.train()
    dvla.decoder.train()

    # ---------------- data
    off = scfg.get("sample_offset", 0)
    recs = load_jsonl(data_root / "train.jsonl")[off: off + scfg["n_samples"]]
    autopen = detect_think_autopen(dvla.processor)
    print(f"[data] {len(recs)} samples, stage {stage}, think-autopen={autopen}")
    encoded = [encode_sample(r, dvla.processor, dvla.tok_info, data_root,
                             n_frames=cfg["inputs"]["n_frames"], stage=stage,
                             autopen=autopen) for r in recs]
    lambdas = STAGE_LAMBDAS[stage]

    # ---------------- grad test (§7.5) on sample 0
    def batch_builder():
        b = collate([encoded[0]], dvla.tokenizer.pad_token_id)
        merged = b["labels_reason"].clone()
        act_pos = b["labels_act"] != -100
        merged[act_pos] = b["labels_act"][act_pos]
        out = {
            "input_ids": b["input_ids"].to(dvla.device),
            "attention_mask": b["attention_mask"].to(dvla.device),
            "pixel_values": b["pixel_values"].to(dvla.device),
            "image_grid_thw": b["image_grid_thw"].to(dvla.device),
            "labels": merged.to(dvla.device),
        }
        if b.get("mm_token_type_ids") is not None:
            out["mm_token_type_ids"] = b["mm_token_type_ids"].to(dvla.device)
        return out

    if not args.resume:
        hits = dvla.grad_test(batch_builder)
        print(f"[grad-test] PASS — trainable grads confined to LoRA + 130 token rows "
              f"({len(hits)} tensors touched)")

    # ---------------- canary pre
    canary_path = REPO_ROOT / "configs/canary_prompts.json"
    if args.canary in ("pre", "both"):
        res = run_canary(dvla, canary_path, data_root)
        (reports / f"canary_{args.canary_pre_tag}.json").write_text(json.dumps(res, indent=1))
        print(f"[canary:{args.canary_pre_tag}] auto_pass={res['auto_pass']}")

    # ---------------- optimizer
    from transformers import get_cosine_schedule_with_warmup

    max_steps = args.max_steps or scfg["max_opt_steps"]
    groups = build_param_groups(dvla, cfg)
    # fused=True: single-kernel AdamW, no foreach transient buffers — the foreach
    # path allocates ~full-param-size temporaries that push peak VRAM over the
    # 12 GB edge on this laptop (and expandable_segments is broken on WSL2).
    try:
        opt = torch.optim.AdamW(groups, weight_decay=cfg["optim"]["weight_decay"], fused=True)
    except (RuntimeError, ValueError):
        opt = torch.optim.AdamW(groups, weight_decay=cfg["optim"]["weight_decay"], foreach=False)
    torch.cuda.empty_cache()
    sched = get_cosine_schedule_with_warmup(opt, cfg["optim"]["warmup_steps"], max_steps)
    accum = cfg["optim"]["grad_accum"]
    flow_gen = torch.Generator(device=dvla.device).manual_seed(cfg["seed"])

    start_step = 0
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
        sched.load_state_dict(resume_state["scheduler"])
        start_step = resume_state["step"]
        flow_gen.set_state(resume_state["flow_gen"])
        probe_now = probe_loss(dvla, collate([encoded[0]], dvla.tokenizer.pad_token_id)
                               | {"lambdas": lambdas})
        saved = resume_state["probe_loss"]
        rel = abs(probe_now - saved) / max(abs(saved), 1e-8)
        print(f"[resume] probe pre-save={saved:.6f} post-resume={probe_now:.6f} rel={rel:.2e}")
        assert rel < 1e-2, "resume probe-loss continuity FAILED"
        print("RESUME_PROBE_OK")
        set_rng_state(resume_state["rng"])
        max_steps = min(max_steps, start_step + args.resume_extra_steps)

    restored_init = resume_state.get("init_losses") if resume_state else None
    if restored_init is None and args.init_losses_json:
        restored_init = json.loads(args.init_losses_json)

    # ---------------- training loop
    rng = np.random.default_rng(cfg["seed"])
    order = []
    log_f = open(out_dir / "train_log.jsonl", "a")
    init_losses, last_parts = restored_init, None
    inner_n = cfg["optim"].get("decoder_inner_steps", 0)
    inner_bs = cfg["optim"].get("decoder_inner_bs", 8)
    cond_bank = {}
    if inner_n:
        log_deviation(
            f"decoder inner-loop: {inner_n} decoder-only flow steps (bs {inner_bs}) "
            "per optimizer step, on cached detached conditioning",
            "the from-scratch ~55M DiT needs ~10x more updates than the smoke's "
            "backbone step budget provides (scripts/diag_decoder_fit.py: decoder-only "
            "training on frozen cond converges flow 0.47->0.04 in 2000 steps while the "
            "joint 300-step run plateaued at ~0.4); conditioning stays detached, so the "
            "VLM gradient pathway (knowledge insulation) is unchanged")
    micro = 0
    criteria_met = False
    greedy_ok = 0 if stage == "B" else None
    limit_s = cfg["train"]["wall_clock_limit_min"] * 60
    step_times = []
    stop_reason = "max_steps"
    last_step = start_step

    for step in range(start_step, max_steps):
        last_step = step + 1
        t_step = time.time()
        opt.zero_grad(set_to_none=True)
        parts_acc = []
        for _ in range(accum):
            if not order:
                order = list(rng.permutation(len(encoded)))
            si = order.pop()
            b = collate([encoded[si]], dvla.tokenizer.pad_token_id)
            b["lambdas"] = lambdas
            if inner_n:
                b["want_cond_cache"] = True
            total, parts = dvla.forward_train(b, flow_generator=flow_gen)
            (total / accum).backward()
            if "cond_cache" in parts:
                cond_bank[si] = parts.pop("cond_cache")
            parts_acc.append(parts)
            micro += 1
            if micro == 1:
                print(f"[no-oom] first fwd/bwd OK | peak VRAM {peak_vram_gb():.2f} GB")
                record_metric(f"{tag}_first_step_vram_gb", round(peak_vram_gb(), 2))
        torch.nn.utils.clip_grad_norm_(
            [p for g in groups for p in g["params"]], cfg["optim"]["max_grad_norm"])
        opt.step()
        sched.step()

        # decoder-only inner loop on cached detached conditioning (see deviation)
        if inner_n and len(cond_bank) >= inner_bs:
            keys = list(cond_bank.keys())
            for _ in range(inner_n):
                pick = [keys[j] for j in rng.choice(len(keys), size=inner_bs, replace=False)]
                Lc = max(cond_bank[k][0].shape[1] for k in pick)
                c0 = cond_bank[pick[0]][0]
                Cb = torch.zeros(inner_bs, Lc, c0.shape[-1], device=c0.device, dtype=c0.dtype)
                Mb = torch.ones(inner_bs, Lc, dtype=torch.bool, device=c0.device)
                X1 = torch.stack([cond_bank[k][2][0] for k in pick])
                for r, k in enumerate(pick):
                    c, m, _ = cond_bank[k]
                    Cb[r, : c.shape[1]] = c[0]
                    Mb[r, : c.shape[1]] = m[0]
                floss = dvla.decoder.flow_loss(X1, Cb, Mb, generator=flow_gen)
                opt.zero_grad(set_to_none=True)
                floss.backward()
                torch.nn.utils.clip_grad_norm_(dvla.decoder.parameters(),
                                               cfg["optim"]["max_grad_norm"])
                opt.step()  # only decoder params have grads here

        avg = {k: float(np.mean([p[k] for p in parts_acc]))
               for k in ("reason_ce", "act_ce", "flow_mse", "total")}
        if init_losses is None:
            init_losses = dict(avg)
        last_parts = avg
        step_times.append(time.time() - t_step)
        rec = {"step": step + 1, "micro": micro, **{k: round(v, 5) for k, v in avg.items()},
               "lr": sched.get_last_lr()[0], "t_step": round(step_times[-1], 2),
               "vram_gb": round(peak_vram_gb(), 2)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        if (step + 1) % 10 == 0 or step == start_step:
            print(f"[{tag}] step {step+1}/{max_steps} " +
                  " ".join(f"{k}={v:.4f}" for k, v in avg.items()) +
                  f" | {step_times[-1]:.2f}s/step | VRAM {peak_vram_gb():.2f}GB")

        # ---- stage criteria
        if stage == "A":
            # deterministic eval-mode probes (train-step averages are noisy:
            # dropout + per-draw flow t-sampling); checked when close, gate at 25s
            close = avg["act_ce"] <= scfg["target_act_ce"] * 2.0
            if close and (step + 1) % 25 == 0:
                probes = eval_probes(dvla, encoded, lambdas)
                criteria_met = (probes["act_ce"] <= scfg["target_act_ce"]
                                and probes["flow_mse"] <= scfg["target_flow_frac"] * init_losses["flow_mse"])
                print(f"[{tag}] probes act_ce={probes['act_ce']:.4f} "
                      f"flow={probes['flow_mse']:.4f} (targets {scfg['target_act_ce']}, "
                      f"{scfg['target_flow_frac'] * init_losses['flow_mse']:.4f}) "
                      f"-> criteria_met={criteria_met}")
        else:
            drop_ok = all(avg[k] <= (1 - scfg["target_loss_drop"]) * init_losses[k]
                          for k in ("reason_ce", "act_ce", "flow_mse"))
            if drop_ok and (step + 1) % scfg["greedy_check_every"] == 0:
                greedy_ok = greedy_match_count(dvla, recs, data_root, cfg)
                print(f"[{tag}] greedy match {greedy_ok}/{len(recs)}")
                criteria_met = greedy_ok >= scfg["target_greedy_match"]

        if args.save_mid and (step + 1) == args.save_mid:
            p = probe_loss(dvla, collate([encoded[0]], dvla.tokenizer.pad_token_id)
                           | {"lambdas": lambdas})
            save_ckpt(dvla, opt, sched, out_dir / "ckpt_mid", step + 1, p,
                      flow_gen.get_state(), init_losses)
            print(f"[ckpt] mid-save at step {step+1} probe={p:.6f}")

        if criteria_met and (stage == "B" or (step + 1) >= 300):
            stop_reason = "criteria_met"
            break
        eta = timer.elapsed() + float(np.mean(step_times[-20:]))
        if eta > limit_s:
            stop_reason = "wall_clock"
            log_deviation(
                f"stage {stage} stopped at {step+1} opt steps (< {max_steps})",
                f"wall-clock guard {cfg['train']['wall_clock_limit_min']} min (hard rule #5); "
                f"criteria evaluated on the saved checkpoint")
            break

    # stage A: authoritative end-of-run probe (deterministic, eval mode)
    final_probes = None
    if stage == "A":
        final_probes = eval_probes(dvla, encoded, lambdas)
        criteria_met = (final_probes["act_ce"] <= scfg["target_act_ce"]
                        and final_probes["flow_mse"]
                        <= scfg["target_flow_frac"] * init_losses["flow_mse"])
        print(f"[{tag}] FINAL probes act_ce={final_probes['act_ce']:.4f} "
              f"flow={final_probes['flow_mse']:.4f} vs targets "
              f"({scfg['target_act_ce']}, "
              f"{scfg['target_flow_frac'] * init_losses['flow_mse']:.4f}) "
              f"-> criteria_met={criteria_met}")

    # stage B: final greedy check if not measured at stop step
    if stage == "B" and greedy_ok is not None and not criteria_met \
            and not args.no_criteria_gate:
        greedy_ok = greedy_match_count(dvla, recs, data_root, cfg)
        criteria_met = (greedy_ok >= scfg["target_greedy_match"] and
                        all(last_parts[k] <= (1 - scfg["target_loss_drop"]) * init_losses[k]
                            for k in ("reason_ce", "act_ce", "flow_mse")))
        print(f"[{tag}] final greedy match {greedy_ok}/{len(recs)}")

    # ---------------- final save + canary
    p = probe_loss(dvla, collate([encoded[0]], dvla.tokenizer.pad_token_id)
                   | {"lambdas": lambdas})
    save_ckpt(dvla, opt, sched, out_dir / "ckpt_final", last_step, p,
              flow_gen.get_state(), init_losses)

    if args.canary in ("post", "both"):
        res = run_canary(dvla, canary_path, data_root)
        (reports / f"canary_{args.canary_post_tag}.json").write_text(json.dumps(res, indent=1))
        print(f"[canary:{args.canary_post_tag}] auto_pass={res['auto_pass']}")

    summary = {
        "stage": stage, "tag": tag, "condition": cfg["decoder"]["condition"],
        "steps": last_step, "micro_steps": micro,
        "stop_reason": stop_reason,
        "init_losses": init_losses, "final_losses": last_parts,
        "final_probes": final_probes,
        "criteria_met": bool(criteria_met),
        "greedy_match": greedy_ok,
        "peak_vram_gb": round(peak_vram_gb(), 2),
        "wall_min": round(timer.elapsed_min(), 1),
        "mean_step_s": round(float(np.mean(step_times)), 2) if step_times else None,
        "precision": cfg["model"]["precision"], "n_frames": cfg["inputs"]["n_frames"],
        "resumed": bool(args.resume),
    }
    (reports / f"summary_{tag}.json").write_text(json.dumps(summary, indent=1))
    record_timing(f"train_{tag}", timer.elapsed(), {"steps": summary["steps"]})
    print(f"[{tag}] DONE criteria_met={criteria_met} stop={stop_reason} "
          f"peakVRAM={summary['peak_vram_gb']}GB wall={summary['wall_min']}min")
    if not criteria_met and not args.no_criteria_gate:
        sys.exit(3)


if __name__ == "__main__":
    main()
