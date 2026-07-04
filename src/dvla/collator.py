"""Sequence building + loss masks (CLAUDE.md §2.4).

Native chat template only (processor.apply_chat_template builds the prompt; we
append the assistant completion as a tokenized continuation — the standard SFT
prompt+completion pattern). At import-of-model time, `verify_template` checks
the Thinking variant's <think> handling directly against processor output:
Qwen3-*-Thinking templates auto-open the think block in the generation prompt
(prompt ends with "<think>\n"), in which case the opening tag lives in the
(unsupervised) prompt and the completion starts with the reasoning text.

Label layout over the full sequence:
  [prompt: system/user/vision ...............] -> -100 everywhere
  [reasoning ... </think> glue]               -> labels_reason  (lambda_reason)
  [<|act_start|> a1..a16 <|act_end|> <|im_end|>] -> labels_act  (lambda_act_ce)

The act span carries exactly 16 bin tokens strictly between the two markers;
<|im_end|> is supervised under the act weight so the model learns to stop.

Decoder-conditioning spans (token index ranges over the full sequence):
  think_and_act: [pos(<think>), pos(<|act_end|>)]      (AR1-style, default)
  think_only:    [pos(<think>), pos(</think>)]         (pi0.5-style, arm c)
Both are returned; the model wrapper picks one via config.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .action_quant import chunk_to_bins
from .tokenizer_ext import TokInfo

SYSTEM_PROMPT = (
    "You control agent A, the blue disc in a top-down 2D arena. Agent B (red disc) "
    "guards the green goal square. First reason briefly about B's belief and your plan, "
    "then output your next 8 actions as one action-token block."
)


def detect_think_autopen(processor) -> bool:
    """Does the chat template auto-open <think> in the generation prompt?"""
    msgs = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
    p = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return p.rstrip("\n").endswith("<think>")


def build_messages(task_prompt: str, n_images: int):
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image"} for _ in range(n_images)]
                                    + [{"type": "text", "text": task_prompt}]},
    ]


def completion_text(reasoning: str, bins_seq, tok_info: TokInfo, autopen: bool) -> str:
    think = ("" if autopen else "<think>\n") \
        + (reasoning + "\n" if reasoning else "") + "</think>\n\n"
    acts = "<|act_start|>" + "".join(f"<|act_{int(b)}|>" for b in bins_seq) + "<|act_end|>"
    return think + acts + "<|im_end|>"


def load_frames(rec: dict, data_root, n_frames: int):
    paths = rec["frames"][-n_frames:]
    root = Path(data_root)
    return [Image.open(root / p).convert("RGB") for p in paths]


def encode_sample(rec: dict, processor, tok_info: TokInfo, data_root: Path,
                  n_frames: int = 4, stage: str = "B", autopen: bool | None = None):
    """One training sample -> dict of tensors + span indices (no padding)."""
    if autopen is None:
        autopen = detect_think_autopen(processor)
    images = load_frames(rec, data_root, n_frames)
    msgs = build_messages(rec["task_prompt"], len(images))
    prompt_text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    reasoning = "" if stage == "A" else rec["reasoning"]
    bins_seq = chunk_to_bins(np.asarray(rec["action_chunk"], dtype=np.float64))
    comp_text = completion_text(reasoning, bins_seq, tok_info, autopen)

    # Prompt ids come from the processor (native template + image expansion);
    # completion ids are tokenized separately and CONCATENATED — this matches
    # generation exactly (the model continues from the prompt ids; joint text
    # is never re-tokenized, so no cross-boundary BPE merges).
    enc_prompt = processor(text=[prompt_text], images=images, return_tensors="pt")
    tok = processor.tokenizer
    comp_ids = tok(comp_text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    prompt_len = enc_prompt.input_ids.shape[1]
    ids = torch.cat([enc_prompt.input_ids[0], comp_ids])
    assert tok.decode(ids[prompt_len:]) == comp_text, "completion ids do not round-trip"

    pos = _span_positions(ids, tok_info, prompt_len)

    labels_reason = torch.full_like(ids, -100)
    labels_act = torch.full_like(ids, -100)
    labels_reason[pos["reason_start"]:pos["act_start"]] = ids[pos["reason_start"]:pos["act_start"]]
    labels_act[pos["act_start"]:pos["act_end"] + 1] = ids[pos["act_start"]:pos["act_end"] + 1]
    labels_act[pos["im_end"]] = ids[pos["im_end"]]

    n_comp = len(comp_ids)
    mm = None
    if hasattr(enc_prompt, "mm_token_type_ids"):
        mm = torch.cat([enc_prompt.mm_token_type_ids[0],
                        torch.zeros(n_comp, dtype=enc_prompt.mm_token_type_ids.dtype)])
    return {
        "input_ids": ids,
        "attention_mask": torch.cat([enc_prompt.attention_mask[0],
                                     torch.ones(n_comp, dtype=enc_prompt.attention_mask.dtype)]),
        "mm_token_type_ids": mm,
        "pixel_values": enc_prompt.pixel_values,
        "image_grid_thw": enc_prompt.image_grid_thw,
        "labels_reason": labels_reason,
        "labels_act": labels_act,
        "prompt_len": prompt_len,
        "pos": pos,
        "cond_span_think_and_act": (pos["think_open"], pos["act_end"] + 1),  # end-exclusive
        "cond_span_think_only": (pos["think_open"], pos["think_close"] + 1),
        "action_chunk": torch.tensor(np.asarray(rec["action_chunk"]), dtype=torch.float32),
        "episode_id": rec["episode_id"],
        "decision": rec["decision"],
    }


def _span_positions(ids: torch.Tensor, tok_info: TokInfo, prompt_len: int) -> dict:
    def find(tid, start=0):
        hits = (ids[start:] == tid).nonzero().flatten()
        assert len(hits) >= 1, f"token id {tid} not found from {start}"
        return int(hits[0]) + start

    think_open = find(tok_info.think_open_id)          # may sit inside the prompt (autopen)
    think_close = find(tok_info.think_close_id, prompt_len)
    act_start = find(tok_info.act_start_id, prompt_len)
    act_end = find(tok_info.act_end_id, act_start)
    im_end = find(tok_info.im_end_id, act_end)
    n_bins_between = act_end - act_start - 1
    assert n_bins_between == 16, f"expected 16 act tokens between markers, got {n_bins_between}"
    act_id_set = set(tok_info.act_ids)
    for i in range(act_start + 1, act_end):
        assert int(ids[i]) in act_id_set, f"non-act token inside act span at {i}"
    assert think_open < think_close < act_start < act_end < im_end
    return {
        "reason_start": prompt_len,
        "think_open": think_open,
        "think_close": think_close,
        "act_start": act_start,
        "act_end": act_end,
        "im_end": im_end,
    }


def collate(batch: list[dict], pad_id: int):
    """Right-pad a list of encoded samples into one batch (training)."""
    L = max(len(b["input_ids"]) for b in batch)

    def pad1(x, v):
        out = torch.full((len(batch), L), v, dtype=x[0].dtype)
        for i, t in enumerate(x):
            out[i, : len(t)] = t
        return out

    ids = pad1([b["input_ids"] for b in batch], pad_id)
    att = pad1([b["attention_mask"] for b in batch], 0)
    lr = pad1([b["labels_reason"] for b in batch], -100)
    la = pad1([b["labels_act"] for b in batch], -100)
    out = {
        "input_ids": ids,
        "attention_mask": att,
        "labels_reason": lr,
        "labels_act": la,
        "mm_token_type_ids": (pad1([b["mm_token_type_ids"] for b in batch], 0)
                              if batch[0].get("mm_token_type_ids") is not None else None),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch], dim=0),
        "cond_span_think_and_act": torch.tensor([b["cond_span_think_and_act"] for b in batch]),
        "cond_span_think_only": torch.tensor([b["cond_span_think_only"] for b in batch]),
        "action_chunk": torch.stack([b["action_chunk"] for b in batch]),
    }
    return out
