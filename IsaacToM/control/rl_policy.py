"""Placeholder RL controller. Keeps the same interface as RuleBasedController
so the runner and ToM stack require zero changes when we swap this in during
the RL curriculum (Stage 1–3 in progress.md).

When training, the RL policy consumes:
  - local observation (ego camera or state dict from runner.read_agent_state)
  - optional high-level intent from the VLM ToM layer (`tactic.label`)

When deploying (evaluation), this controller loads a checkpoint and maps
(obs, tactic) -> wheel commands.
"""
from __future__ import annotations

from typing import Tuple

from .base import BaseController, HighLevelTactic


class RLPolicyController(BaseController):
    def __init__(self, role, cfg, checkpoint_path: str | None = None):
        super().__init__(role, cfg)
        self.checkpoint_path = checkpoint_path
        self.policy = None
        if checkpoint_path is not None:
            self._load_policy(checkpoint_path)

    def _load_policy(self, path: str):
        # TODO: plug RSL-RL / SKRL checkpoint loader here once Stage 1 training is ready.
        raise NotImplementedError("RL checkpoint loading not wired up yet.")

    def step(self, tactic: HighLevelTactic, self_state, other_state, goals) -> Tuple[float, float]:
        if self.policy is None:
            # Fallback: behave like a stationary agent so it's obvious the RL path isn't live.
            return 0.0, 0.0
        raise NotImplementedError
