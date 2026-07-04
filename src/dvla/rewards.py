"""Stage-C GRPO reward interfaces — STUBS ONLY (CLAUDE.md §4, out of scope here).

Stage C trains the VLM token pathway only (decoder frozen, AR1 precedent).
These signatures are the contract for the cluster-side implementation.
"""


def reasoning_quality(sample) -> float:
    """Reward for CoC-style reasoning quality (closed-set decision named,
    causal locality respected, belief target stated)."""
    raise NotImplementedError("Stage C runs on the cluster, not this machine")


def consistency(reasoning: str, decision: str, chunk, outcome: bool) -> float:
    """Reward for intent-motion-outcome consistency. Seeded by the perturbation
    ratio of §7.9: the chunk must causally follow the stated reasoning."""
    raise NotImplementedError("Stage C runs on the cluster, not this machine")


def outcome_reward(sample) -> float:
    """Task/deception outcome reward (e.g. outcome_deceived, goal reached)."""
    raise NotImplementedError("Stage C runs on the cluster, not this machine")
