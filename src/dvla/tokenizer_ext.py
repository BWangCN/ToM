"""Vocabulary extension + LoRA/trainable-token assembly (CLAUDE.md §2.3).

130 new tokens: <|act_0|>..<|act_127|>, <|act_start|>, <|act_end|>.
New-row training goes through peft `trainable_token_indices`. If the installed
peft rejects the kwarg, we fall back to an explicit gradient mask that zeroes
embedding/lm_head grads for all rows EXCEPT the 130 new ones.

HARD RULE: never modules_to_save=["embed_tokens","lm_head"].
"""

import re
from dataclasses import dataclass, field

import torch

N_BINS = 128
ACT_START = "<|act_start|>"
ACT_END = "<|act_end|>"


def act_token_list():
    return [f"<|act_{i}|>" for i in range(N_BINS)] + [ACT_START, ACT_END]


@dataclass
class TokInfo:
    act_ids: list = field(default_factory=list)  # 128 bin-token ids, index = bin
    act_start_id: int = -1
    act_end_id: int = -1
    new_ids: list = field(default_factory=list)  # all 130
    think_open_id: int = -1
    think_close_id: int = -1
    im_end_id: int = -1
    tied: bool = True


def extend_tokenizer_only(tokenizer) -> TokInfo:
    """Add the 130 tokens to the tokenizer and build the id map (no model ops)."""
    new_tokens = act_token_list()
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    assert n_added == len(new_tokens), f"expected {len(new_tokens)} new tokens, got {n_added}"

    info = TokInfo()
    info.act_ids = [tokenizer.convert_tokens_to_ids(f"<|act_{i}|>") for i in range(N_BINS)]
    info.act_start_id = tokenizer.convert_tokens_to_ids(ACT_START)
    info.act_end_id = tokenizer.convert_tokens_to_ids(ACT_END)
    info.new_ids = info.act_ids + [info.act_start_id, info.act_end_id]
    assert all(i >= 0 for i in info.new_ids) and len(set(info.new_ids)) == 130
    info.think_open_id = tokenizer.convert_tokens_to_ids("<think>")
    info.think_close_id = tokenizer.convert_tokens_to_ids("</think>")
    info.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assert info.think_close_id is not None and info.think_close_id >= 0, "no </think> token"
    return info


def extend_tokenizer_and_model(model, tokenizer, seed: int = 0) -> TokInfo:
    """Add the 130 tokens, resize embeddings, mean-init the new rows."""
    info = extend_tokenizer_only(tokenizer)
    model.resize_token_embeddings(len(tokenizer))

    text_cfg = getattr(model.config, "text_config", model.config)
    info.tied = bool(getattr(text_cfg, "tie_word_embeddings",
                             getattr(model.config, "tie_word_embeddings", True)))

    # Mean-init the new rows (they may be recycled, never-trained padding rows).
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        old = emb[: min(info.new_ids)]
        mu, sd = old.mean(0), old.std(0)
        g = torch.Generator(device="cpu").manual_seed(seed)
        for tid in info.new_ids:
            noise = torch.randn(mu.shape, generator=g, dtype=torch.float32).to(mu.device, mu.dtype)
            emb[tid] = mu + 0.02 * sd * noise
        if not info.tied:
            head = model.get_output_embeddings().weight
            hmu, hsd = head[: min(info.new_ids)].mean(0), head[: min(info.new_ids)].std(0)
            for tid in info.new_ids:
                noise = torch.randn(hmu.shape, generator=g, dtype=torch.float32).to(hmu.device, hmu.dtype)
                head[tid] = hmu + 0.02 * hsd * noise
    return info


LM_TARGET_REGEX = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)


def build_peft_model(model, lora_cfg: dict, tok_info: TokInfo):
    """Wrap with LoRA (LM-only targets) + trainable token rows.

    Returns (peft_model, used_fallback: bool).
    """
    from peft import LoraConfig, get_peft_model

    base_kwargs = dict(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        bias="none",
        target_modules=LM_TARGET_REGEX,
        task_type="CAUSAL_LM",
    )

    used_fallback = False
    try:
        cfg = LoraConfig(trainable_token_indices={"embed_tokens": tok_info.new_ids}, **base_kwargs)
        peft_model = get_peft_model(model, cfg)
    except TypeError:
        used_fallback = True
        cfg = LoraConfig(**base_kwargs)
        peft_model = get_peft_model(model, cfg)
        _install_grad_mask(peft_model, tok_info)

    # LoRA must not have touched the vision tower.
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A"):
            assert "visual" not in name, f"LoRA leaked into vision tower: {name}"
    return peft_model, used_fallback


def _install_grad_mask(model, tok_info: TokInfo):
    """Fallback: train full embed (+lm_head if untied) rows but zero all grads
    except the 130 new rows, via grad hooks."""
    new = torch.tensor(tok_info.new_ids)

    def make_hook(n_rows):
        mask = torch.zeros(n_rows, 1)
        mask[new] = 1.0

        def hook(grad):
            return grad * mask.to(grad.device, grad.dtype)

        return hook

    emb = model.get_input_embeddings().weight
    emb.requires_grad_(True)
    emb.register_hook(make_hook(emb.shape[0]))
    if not tok_info.tied:
        head = model.get_output_embeddings().weight
        head.requires_grad_(True)
        head.register_hook(make_hook(head.shape[0]))


def trainable_breakdown(peft_model, decoder=None) -> dict:
    """Exact trainable-parameter counts: LoRA / token rows / decoder / other."""
    counts = {"lora": 0, "token_rows": 0, "other": 0}
    for name, p in peft_model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in name:
            counts["lora"] += p.numel()
        elif "trainable_tokens" in name:
            counts["token_rows"] += p.numel()
        elif "embed_tokens" in name or "lm_head" in name:
            # grad-mask fallback: whole matrix requires_grad, but only 130 rows update
            d = p.shape[-1]
            counts["token_rows"] += 130 * d
            counts["other"] += p.numel() - 130 * d  # masked to zero — report visibility
        else:
            counts["other"] += p.numel()
    if decoder is not None:
        counts["decoder"] = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    counts["total_trainable"] = sum(v for k, v in counts.items())
    return counts


def assert_row_freeze(peft_model, tok_info: TokInfo, batch_builder):
    """Grad test (§7.5): after one fwd/bwd, embedding grads are nonzero only on
    the 130 new rows; base weights (non-LoRA, non-token) have no grads.

    batch_builder() -> a kwargs dict the model accepts (with labels).
    """
    peft_model.train()
    batch = batch_builder()
    out = peft_model(**batch)
    loss = out.loss
    loss.backward()

    new = set(tok_info.new_ids)
    hits = []
    for name, p in peft_model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad
        if "trainable_tokens" in name:
            hits.append(name)
            continue
        if ("embed_tokens.weight" in name or "lm_head.weight" in name) and g.dim() == 2 \
                and g.shape[0] > max(new):
            nz = g.abs().sum(dim=1).nonzero().flatten().tolist()
            bad = [i for i in nz if i not in new]
            assert not bad, f"{name}: grads on non-new rows {bad[:5]}..."
            hits.append(name)
        elif "lora_" in name:
            hits.append(name)
        else:
            assert g.abs().max() == 0, f"unexpected nonzero grad on {name}"
    peft_model.zero_grad(set_to_none=True)
    assert hits, "grad test saw no trainable grads at all"
    return hits
