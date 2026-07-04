"""Collator/mask unit test (CLAUDE.md §2.4 / §7.4): decode a collated batch and
assert exact label-mask boundaries and token counts, against the REAL Qwen3-VL
processor (Thinking chat template)."""

import numpy as np
import pytest
import torch
from PIL import Image

from dvla.action_quant import chunk_to_bins
from dvla.collator import (build_messages, collate, completion_text,
                           detect_think_autopen, encode_sample)
from dvla.tokenizer_ext import extend_tokenizer_only

MODEL = "Qwen/Qwen3-VL-4B-Thinking"


@pytest.fixture(scope="module")
def proc_tok():
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL)
    tok_info = extend_tokenizer_only(proc.tokenizer)
    return proc, tok_info


@pytest.fixture()
def rec(tmp_path):
    rng = np.random.default_rng(0)
    frames = []
    for i in range(4):
        img = Image.fromarray(rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8))
        p = tmp_path / f"t{i}.png"
        img.save(p)
        frames.append(p.name)
    return {
        "episode_id": "test", "keyframe_t": 17,
        "frames": frames,
        "task_prompt": "Reach the goal; B is guarding.",
        "decision": "FEINT_LEFT_GO_RIGHT",
        "reasoning": "B lags my heading by ~3 steps. I feint left, then cut right.",
        "action_chunk": rng.uniform(-1, 1, size=(8, 2)).tolist(),
    }


def test_autopen_detected(proc_tok):
    proc, _ = proc_tok
    assert detect_think_autopen(proc) in (True, False)  # must not crash
    # For the Thinking variant, the generation prompt opens the think block.
    p = proc.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "x"}]}],
                                 add_generation_prompt=True, tokenize=False)
    assert p.rstrip("\n").endswith("<think>") == detect_think_autopen(proc)


def test_native_template_equivalence(proc_tok, rec, tmp_path):
    """prompt(+gen) + completion must equal the native full-conversation render."""
    proc, ti = proc_tok
    autopen = detect_think_autopen(proc)
    bins_seq = chunk_to_bins(np.asarray(rec["action_chunk"]))
    comp = completion_text(rec["reasoning"], bins_seq, ti, autopen)
    msgs = build_messages(rec["task_prompt"], 4)
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    asst = ("<think>\n" if autopen else "") + comp.removesuffix("<|im_end|>")
    full_native = proc.apply_chat_template(
        msgs + [{"role": "assistant", "content": [{"type": "text", "text": asst}]}],
        add_generation_prompt=False, tokenize=False)
    assert prompt + comp == full_native.removesuffix("\n"), \
        "prompt+completion drifted from the native chat template"


@pytest.mark.parametrize("stage", ["A", "B"])
def test_mask_boundaries(proc_tok, rec, tmp_path, stage):
    proc, ti = proc_tok
    enc = encode_sample(rec, proc, ti, tmp_path, n_frames=4, stage=stage)
    ids = enc["input_ids"]
    lr, la = enc["labels_reason"], enc["labels_act"]
    pos = enc["pos"]
    pl = enc["prompt_len"]

    # prompt fully unsupervised in both label tensors
    assert (lr[:pl] == -100).all() and (la[:pl] == -100).all()

    # exactly 16 act tokens strictly between the markers, all act ids
    assert pos["act_end"] - pos["act_start"] - 1 == 16
    act_set = set(ti.act_ids)
    gt_bins = chunk_to_bins(np.asarray(rec["action_chunk"]))
    span_ids = ids[pos["act_start"] + 1: pos["act_end"]].tolist()
    assert [ti.act_ids.index(t) for t in span_ids] == gt_bins.tolist()
    assert all(t in act_set for t in span_ids)

    # act labels = markers + 16 bins + <|im_end|> = 19 supervised positions
    assert int((la != -100).sum()) == 19
    assert la[pos["act_start"]] == ti.act_start_id
    assert la[pos["act_end"]] == ti.act_end_id
    assert la[pos["im_end"]] == ti.im_end_id

    # reason span: [prompt_len, act_start), no overlap with act span
    n_r = int((lr != -100).sum())
    assert n_r == pos["act_start"] - pl
    assert ((lr != -100) & (la != -100)).sum() == 0

    reason_text = proc.tokenizer.decode(ids[pl:pos["act_start"]])
    if stage == "B":
        assert rec["reasoning"] in reason_text
    else:
        assert rec["reasoning"] not in reason_text  # action-only stage: empty think

    # conditioning spans: think_only nested inside think_and_act, both end-exclusive
    s_ta, e_ta = enc["cond_span_think_and_act"]
    s_to, e_to = enc["cond_span_think_only"]
    assert s_ta == s_to == pos["think_open"]
    assert e_to == pos["think_close"] + 1 <= e_ta == pos["act_end"] + 1

    # collate keeps everything aligned
    batch = collate([enc, enc], proc.tokenizer.pad_token_id)
    assert batch["input_ids"].shape == batch["labels_act"].shape
    assert int((batch["labels_act"][0] != -100).sum()) == 19
    assert batch["action_chunk"].shape == (2, 8, 2)
