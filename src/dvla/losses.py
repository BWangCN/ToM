"""Three-component loss with separate, token-count-normalized logging (§0 Losses)."""

import torch
import torch.nn.functional as F


def masked_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Causal-shifted CE over labeled (!=-100) positions, mean per token.

    logits (B,S,V), labels (B,S). Returns (loss, n_tokens); loss=0 if span empty.
    """
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    n = int(mask.sum())
    if n == 0:
        return logits.new_zeros(()), 0
    loss = F.cross_entropy(
        shift_logits[mask].float(), shift_labels[mask], reduction="mean"
    )
    return loss, n


def combined_loss(logits, labels_reason, labels_act, flow_mse, lambdas: dict):
    """Returns (total, parts) where parts logs each component separately."""
    reason_ce, n_r = masked_ce(logits, labels_reason)
    act_ce, n_a = masked_ce(logits, labels_act)
    total = (lambdas["lambda_reason"] * reason_ce
             + lambdas["lambda_act_ce"] * act_ce
             + lambdas["lambda_flow"] * flow_mse)
    parts = {
        "reason_ce": float(reason_ce.detach()),
        "act_ce": float(act_ce.detach()),
        "flow_mse": float(flow_mse.detach()),
        "total": float(total.detach()),
        "n_reason_tokens": n_r,
        "n_act_tokens": n_a,
    }
    return total, parts
