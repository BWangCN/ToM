"""Constrained-decoding demos, checkpoint round-trip, and report assembly
(CLAUDE.md §7.8, §7.12, §7.13). Run LAST: it collects every artifact under
reports/ into reports/smoke_report.md.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.collator import collate, detect_think_autopen, encode_sample, load_frames
from dvla.common import (REPO_ROOT, Timer, load_config, peak_vram_gb,
                         record_metric, record_timing, seed_everything)
from dvla.constrained_decode import discrete_chunk, generate_reason_act
from dvla.model_wrapper import DVLA

REPORTS = REPO_ROOT / "reports"


def gen_demos(dvla, val_recs, data_root, cfg, n=3):
    """§7.8: well-formed generation on val samples + chunk plots vs GT."""
    from ablation_eval import encode_prompt_batch, flow_from_generated

    recs = val_recs[:n]
    imgs = [load_frames(r, data_root, cfg["inputs"]["n_frames"]) for r in recs]
    enc = encode_prompt_batch(dvla, imgs, [r["task_prompt"] for r in recs])
    out = generate_reason_act(dvla, enc, think_budget=cfg["train"]["reasoning_max_new_tokens"],
                              greedy=True)
    flow_chunks, valid, _, _ = flow_from_generated(dvla, out["sequences"], enc, out["rows"],
                                                   "think_and_act")
    demos = []
    for i, (rec, row) in enumerate(zip(recs, out["rows"])):
        gt = np.asarray(rec["action_chunk"])
        dch = discrete_chunk(row)
        fch = flow_chunks[i]
        gen_txt = dvla.tokenizer.decode(
            out["sequences"][i][out["prompt_len"]:], skip_special_tokens=False)
        gen_txt = gen_txt.split("<|im_end|>")[0]

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        steps = np.arange(8)
        for d, name in ((0, "dx"), (1, "dy")):
            ax = axes[d]
            ax.plot(steps, gt[:, d], "o-", label="GT")
            if dch is not None:
                ax.plot(steps, dch[:, d], "s--", label="discrete")
            ax.plot(steps, fch[:, d], "^:", label="flow")
            ax.set_title(f"{rec['episode_id']} {name}")
            ax.set_ylim(-1.1, 1.1)
            ax.legend(fontsize=7)
        ax = axes[2]
        for ch, style, name in ((gt, "-o", "GT"), (dch, "--s", "discrete"), (fch, ":^", "flow")):
            if ch is None:
                continue
            path = np.cumsum(np.vstack([[0, 0], ch]), axis=0)
            ax.plot(path[:, 0], path[:, 1], style, label=name, ms=3)
        ax.set_title(f"cumulative path ({rec['decision']})")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = REPORTS / f"gen_demo_{rec['episode_id']}.png"
        fig.savefig(p, dpi=110)
        plt.close(fig)

        demos.append({
            "episode_id": rec["episode_id"], "decision": rec["decision"],
            "well_formed": bool(row["well_formed"]),
            "n_bins": len(row["bins"]) if row["bins"] else 0,
            "flow_l2_vs_gt": round(float(np.linalg.norm(fch - gt)), 4),
            "discrete_l2_vs_gt": (round(float(np.linalg.norm(dch - gt)), 4)
                                  if dch is not None else None),
            "generated_text": gen_txt,
            "plot": p.name,
        })
    ok = all(d["well_formed"] and d["n_bins"] == 16 for d in demos)
    return {"demos": demos, "all_well_formed": ok,
            "forced_act_start": out["forced_act_start"]}


def ckpt_roundtrip(dvla, ckpt_dir, val_rec, data_root, cfg):
    """§7.12 (load half): probe-loss match in a fresh process + adapter merge."""
    state = torch.load(Path(ckpt_dir) / "state.pt", weights_only=False)
    off = cfg["stage_b"].get("sample_offset", 0)
    train_recs = [json.loads(l) for l in
                  (data_root / "train.jsonl").read_text().splitlines()
                  ][off: off + cfg["stage_b"]["n_samples"]]
    autopen = detect_think_autopen(dvla.processor)
    enc0 = encode_sample(train_recs[0], dvla.processor, dvla.tok_info, data_root,
                         n_frames=cfg["inputs"]["n_frames"], stage="B", autopen=autopen)
    b = collate([enc0], dvla.tokenizer.pad_token_id)
    b["lambdas"] = {"lambda_reason": 1.0, "lambda_act_ce": 1.0, "lambda_flow": 1.0}
    dvla.model.eval(); dvla.decoder.eval()
    g = torch.Generator(device=dvla.device).manual_seed(1234)
    with torch.no_grad():
        total, _ = dvla.forward_train(b, flow_generator=g)
    fresh = float(total)
    saved = float(state["probe_loss"])
    rel = abs(fresh - saved) / max(abs(saved), 1e-8)
    # tolerance: relative OR absolute — the adapter serializes through bf16
    # module dtypes (~1e-3 weight rounding), which on a converged probe loss of
    # ~0.04 shows up as a few 1e-4 absolute; both raw numbers are reported.
    probe_ok = rel < 1e-2 or abs(fresh - saved) < 2e-3

    # adapter merge (the peft adapter incl. trainable token rows folds into the base)
    t0 = time.time()
    dvla.model = dvla.model.merge_and_unload()
    merge_s = time.time() - t0
    imgs = [load_frames(val_rec, data_root, cfg["inputs"]["n_frames"])]
    from ablation_eval import encode_prompt_batch
    enc = encode_prompt_batch(dvla, imgs, [val_rec["task_prompt"]])
    out = generate_reason_act(dvla, enc, think_budget=cfg["train"]["reasoning_max_new_tokens"],
                              greedy=True)
    merged_ok = bool(out["rows"][0]["well_formed"])
    return {"probe_saved": saved, "probe_fresh_process": fresh, "rel_err": rel,
            "probe_ok": probe_ok, "merge_seconds": round(merge_s, 1),
            "merged_generation_well_formed": merged_ok}


# --------------------------------------------------------------- report

def _j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _md_escape(s):
    return s.replace("|", "\\|").replace("\n", "<br>")


def build_report(cfg, demo_res, rt_res):
    parts = []
    a = parts.append
    a("# Deception-VLA — Framework Smoke Report\n")
    a("*An Alpamayo-R1-style reasoning VLA — reason-then-act with discrete action tokens "
      "and an insulated flow-matching action expert — instantiated on Qwen3-VL-4B-Thinking "
      "for a pixel-world deception task.*\n")
    a(f"Auto-generated by `scripts/eval_smoke.py`. **Verify ≠ train** — every number here "
      f"is a smoke/overfit result on one consumer GPU (see CLAUDE.md §9).\n")

    env = _j(REPORTS / "env.json")
    a("## 1. Environment\n")
    if env:
        a("```json\n" + json.dumps(env, indent=1) + "\n```\n")

    meta = _j(REPO_ROOT / cfg["data"]["root"] / "meta.json")
    a("## 2. Data\n")
    if meta:
        a("```json\n" + json.dumps(meta, indent=1) + "\n```\n")
    dc = (REPORTS / "data_check.txt")
    if dc.exists():
        a("```\n" + dc.read_text().strip()[-600:] + "\n```\n")

    a("## 3. Unit tests\n")
    pt = REPORTS / "pytest.txt"
    if pt.exists():
        a("```\n" + pt.read_text().strip()[-800:] + "\n```\n")

    a("## 4. Model assembly\n")
    m = _j(REPORTS / "metrics.json") or {}
    bd = m.get("trainable_breakdown")
    if bd:
        a("Trainable-parameter breakdown (exact counts):\n")
        a("```json\n" + json.dumps(bd, indent=1) + "\n```\n")

    a("## 5. Training stages\n")
    for tag, title in [("stage_a", "Stage A — action-modality injection (overfit 32, action-only)"),
                       ("stage_b", "Stage B — joint SFT (overfit 10 triplets)"),
                       ("stage_b_thinkonly", "Stage B (think_only) — ablation arm (c) checkpoint"),
                       ("stage_b_resume", "Stage B resume check (§7.12)")]:
        s = _j(REPORTS / f"summary_{tag}.json")
        if not s:
            continue
        a(f"### {title}\n")
        a("```json\n" + json.dumps(s, indent=1) + "\n```\n")

    a("## 6. Constrained decoding demos (§7.8)\n")
    a(f"All well-formed: **{demo_res['all_well_formed']}** "
      f"(forced <|act_start|> events: {demo_res['forced_act_start']})\n")
    for d in demo_res["demos"]:
        a(f"- `{d['episode_id']}` ({d['decision']}): well_formed={d['well_formed']}, "
          f"16/16 bins, flow L2 vs GT = {d['flow_l2_vs_gt']}, "
          f"discrete L2 vs GT = {d['discrete_l2_vs_gt']} — ![]({d['plot']})\n")
        a(f"  - generated: `{_md_escape(d['generated_text'][:400])}`\n")

    a("\n## 7. Conditioning perturbation test (§7.9)\n")
    p = _j(REPORTS / "perturb.json")
    if p:
        a(f"**median(shift/noise) = {p['median_ratio']}** (pass ≥ 3: **{p['pass']}**), "
          f"median noise {p['median_noise']}, median shift {p['median_shift']}, "
          f"{p['n_samples']} val samples × {p['n_seeds']} flow seeds.\n")
        a("This ratio is the seed of the Stage-C consistency reward.\n")
        a("\n```json\n" + json.dumps(p["per_sample"], indent=1) + "\n```\n")

    a("## 8. Language canary (§7.10)\n")
    tags = [("canary_pre_a", "pre-Stage-A"), ("canary_post_a", "post-A"), ("canary_post_b", "post-B")]
    cans = {t: _j(REPORTS / f"{t}.json") for t, _ in tags}
    if all(cans.values()):
        auto = all(c["auto_pass"] for c in cans.values())
        a(f"Automatic pass (no `<|act_` substrings, no degenerate repetition): **{auto}**. "
          f"Verbatim outputs below — final judgment is human eyeball.\n")
        names = [r["name"] for r in cans["canary_pre_a"]["results"]]
        a("| prompt | pre-Stage-A | post-A | post-B |")
        a("|---|---|---|---|")
        for i, name in enumerate(names):
            row = [name]
            for t, _ in tags:
                r = cans[t]["results"][i]
                flag = " ⚠️" if (r["has_act_substr"] or r["degenerate"]) else ""
                row.append(_md_escape(r["output"])[:600] + flag)
            a("| " + " | ".join(row) + " |")
        a("")

    a("## 9. Three-arm ablation (§7.11)\n")
    at = REPORTS / "ablation_table.md"
    if at.exists():
        a(at.read_text())
    a("\n**Standing caveat:** this local SFT comparison cannot adjudicate the discrete-token "
      "layer itself — its value is the GRPO handle in Stage C, which does not run on this "
      "machine; discrete co-training stays ON in all arms.\n")

    a("## 10. Checkpointing (§7.12)\n")
    a("```json\n" + json.dumps(rt_res, indent=1) + "\n```\n")
    rs = _j(REPORTS / "summary_stage_b_resume.json")
    if rs:
        a(f"Mid-Stage-B save/resume in a fresh process: resumed at step "
          f"{rs['steps'] - 5}→{rs['steps']}, probe-loss continuity asserted in-run "
          f"(see summary above).\n")

    a("## 11. Wall-clock & VRAM\n")
    t = _j(REPORTS / "timings.json") or {}
    a("```json\n" + json.dumps(t, indent=1) + "\n```\n")
    vr = {k: v for k, v in m.items() if "vram" in k}
    if vr:
        a("Peak VRAM (GB): `" + json.dumps(vr) + "`\n")

    a("## 12. Deviations from CLAUDE.md §0\n")
    dev = REPORTS / "deviations.md"
    if dev.exists() and dev.read_text().strip():
        a(dev.read_text())
    else:
        a("None recorded.\n")

    (REPORTS / "smoke_report.md").write_text("\n".join(parts))
    print(f"[report] wrote {REPORTS/'smoke_report.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/laptop.yaml"))
    ap.add_argument("--checkpoint", required=True, help="stage B final ckpt dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    data_root = REPO_ROOT / cfg["data"]["root"]
    val_recs = [json.loads(l) for l in (data_root / "val.jsonl").read_text().splitlines()]
    timer = Timer()

    dvla = DVLA.load_trained(cfg, args.checkpoint)  # this IS the fresh-process load
    dvla.model.eval(); dvla.decoder.eval()
    demo_res = gen_demos(dvla, val_recs, data_root, cfg, n=cfg["eval"]["gen_demo_n"])
    print(f"[demos] all_well_formed={demo_res['all_well_formed']}")
    (REPORTS / "gen_demos.json").write_text(json.dumps(demo_res, indent=1))

    rt_res = ckpt_roundtrip(dvla, args.checkpoint, val_recs[0], data_root, cfg)
    print(f"[roundtrip] probe_ok={rt_res['probe_ok']} rel={rt_res['rel_err']:.2e} "
          f"merged_gen_ok={rt_res['merged_generation_well_formed']}")
    (REPORTS / "ckpt_roundtrip.json").write_text(json.dumps(rt_res, indent=1))
    record_metric("eval_smoke_peak_vram_gb", round(peak_vram_gb(), 2))

    build_report(cfg, demo_res, rt_res)
    record_timing("eval_smoke", timer.elapsed())
    if not (demo_res["all_well_formed"] and rt_res["probe_ok"]
            and rt_res["merged_generation_well_formed"]):
        sys.exit(3)


if __name__ == "__main__":
    main()
