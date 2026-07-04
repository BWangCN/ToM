"""Shared utilities: config load, seeding, timing, deviation log, VRAM."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS = REPO_ROOT / "reports"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "configs" / "laptop.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def log_deviation(what: str, why: str) -> None:
    """Append a deviation from CLAUDE.md §0 to reports/deviations.md."""
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / "deviations.md"
    line = f"- **{what}** — {why}\n"
    existing = p.read_text() if p.exists() else ""
    if line not in existing:
        with open(p, "a") as f:
            f.write(line)


def record_timing(key: str, seconds: float, extra: dict | None = None) -> None:
    """Accumulate wall-clock entries for the report."""
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / "timings.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    entry = {"seconds": round(seconds, 1)}
    if extra:
        entry.update(extra)
    data[key] = entry
    p.write_text(json.dumps(data, indent=2))


def record_metric(key: str, value) -> None:
    """Accumulate metrics for the report (reports/metrics.json)."""
    REPORTS.mkdir(exist_ok=True)
    p = REPORTS / "metrics.json"
    data = json.loads(p.read_text()) if p.exists() else {}
    data[key] = value
    p.write_text(json.dumps(data, indent=2))


def peak_vram_gb() -> float:
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def elapsed(self) -> float:
        return time.time() - self.t0

    def elapsed_min(self) -> float:
        return self.elapsed() / 60.0


def env_report() -> dict:
    import torch
    import transformers
    import peft

    info = {
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "python": os.sys.version.split()[0],
    }
    return info
