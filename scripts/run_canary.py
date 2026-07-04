"""Standalone language-canary runner (§7.10) — its own process so training runs
stay under the 30-minute hard rule.

--init fresh   : base + fresh adapters (seed 0) == exact pre-Stage-A state
--init <ckpt>  : a saved checkpoint dir (post-A, post-B)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dvla.canary import run_canary
from dvla.common import REPO_ROOT, Timer, load_config, record_timing, seed_everything
from dvla.model_wrapper import DVLA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/laptop.yaml"))
    ap.add_argument("--init", required=True, help="'fresh' or a checkpoint dir")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    timer = Timer()
    if args.init == "fresh":
        dvla = DVLA.build(cfg)
    else:
        dvla = DVLA.load_trained(cfg, args.init)
    dvla.model.eval()

    res = run_canary(dvla, REPO_ROOT / "configs/canary_prompts.json",
                     REPO_ROOT / cfg["data"]["root"])
    out = REPO_ROOT / "reports" / f"{args.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    record_timing(f"canary_{args.tag}", timer.elapsed())
    print(f"[canary:{args.tag}] auto_pass={res['auto_pass']} -> {out}")
    for r in res["results"]:
        flag = " [ACT-LEAK]" if r["has_act_substr"] else ("" if not r["degenerate"] else " [DEGEN]")
        print(f"  - {r['name']}: {r['output'][:100].replace(chr(10), ' / ')}{flag}")


if __name__ == "__main__":
    main()
