"""Vocab-freeze test (§7.5) on a TINY randomly-initialized Qwen3-VL so the
peft wiring (trainable_token_indices + LM-only LoRA + tied lm_head mirroring)
is verified quickly on CPU. The full-size assertions re-run at train init."""

import numpy as np
import pytest
import torch

from dvla.tokenizer_ext import (assert_row_freeze, build_peft_model,
                                extend_tokenizer_and_model)

MODEL = "Qwen/Qwen3-VL-4B-Thinking"
LORA = {"r": 4, "alpha": 8, "dropout": 0.0}


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import (AutoConfig, AutoModelForImageTextToText,
                              AutoProcessor)

    cfg = AutoConfig.from_pretrained(MODEL)
    tc = cfg.text_config
    tc.hidden_size = 64
    tc.num_hidden_layers = 4
    tc.num_attention_heads = 4
    tc.num_key_value_heads = 2
    tc.head_dim = 16
    tc.intermediate_size = 128
    if getattr(tc, "rope_scaling", None) and "mrope_section" in tc.rope_scaling:
        tc.rope_scaling["mrope_section"] = [4, 2, 2]  # sums to head_dim/2
    vc = cfg.vision_config
    vc.depth = 4
    vc.hidden_size = 32
    vc.num_heads = 2
    vc.intermediate_size = 64
    if hasattr(vc, "out_hidden_size"):
        vc.out_hidden_size = tc.hidden_size
    if getattr(vc, "deepstack_visual_indexes", None):
        vc.deepstack_visual_indexes = [1, 2, 3]
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(cfg)
    tok = AutoProcessor.from_pretrained(MODEL).tokenizer
    return model, tok


def test_vocab_freeze_and_lora_scope(tiny_model):
    model, tok = tiny_model
    for p in model.parameters():
        p.requires_grad_(False)
    info = extend_tokenizer_and_model(model, tok, seed=0)
    assert len(info.new_ids) == 130
    assert model.get_input_embeddings().num_embeddings == len(tok)

    peft_model, used_fallback = build_peft_model(model, LORA, info)
    # LoRA only on LM projection layers, never the vision tower
    lora_names = [n for n, m in peft_model.named_modules() if hasattr(m, "lora_A")]
    assert lora_names, "no LoRA modules injected"
    assert all("language_model" in n for n in lora_names)
    assert not any("visual" in n for n in lora_names)

    ids = torch.randint(100, 1000, (1, 12))
    ids[0, -3] = info.act_start_id
    ids[0, -2] = info.act_ids[5]
    ids[0, -1] = info.act_end_id

    def builder():
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

    hits = assert_row_freeze(peft_model, info, builder)
    assert any("trainable_tokens" in h or "embed_tokens" in h for h in hits) or used_fallback

    # tied lm_head must mirror the trainable-token delta: bump the delta for one
    # new token (absent from the input) and require its logit column to move.
    tid = info.act_ids[7]
    text_ids = torch.randint(100, 1000, (1, 8))
    peft_model.eval()
    with torch.no_grad():
        before = peft_model(input_ids=text_ids, attention_mask=torch.ones_like(text_ids)
                            ).logits[0, -1, tid].item()
        bumped = False
        for n, p in peft_model.named_parameters():
            if p.requires_grad and ("trainable_tokens" in n or "embed_tokens" in n):
                if p.dim() == 2 and p.shape[0] == 130:
                    p[info.new_ids.index(tid)] += 5.0
                    bumped = True
                elif p.dim() == 2 and p.shape[0] > max(info.new_ids):
                    p[tid] += 5.0
                    bumped = True
        assert bumped, "could not locate trainable token rows to bump"
        after = peft_model(input_ids=text_ids, attention_mask=torch.ones_like(text_ids)
                           ).logits[0, -1, tid].item()
    assert not np.isclose(before, after), \
        "lm_head did not reflect the token-row update (tie mirroring broken)"
