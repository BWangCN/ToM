"""Constrained decoding (CLAUDE.md §2.5).

Between <|act_start|> and <|act_end|>: logits masked to the 128 act tokens;
<|act_end|> forced after exactly 16 bins; <|im_end|> forced after <|act_end|>.
If the think budget is exhausted before the model opens the act block on its
own, <|act_start|> is forced (the event is counted and reported).

emit_discrete_at_inference=false (think_only only): generation simply stops at
</think> — implemented by using </think> as the eos id, no constraint needed.
"""

import time

import numpy as np
import torch
from transformers import LogitsProcessor

from .action_quant import bins_to_chunk
from .tokenizer_ext import TokInfo


class ActConstraintProcessor(LogitsProcessor):
    def __init__(self, tok_info: TokInfo, prompt_len: int, n_act: int = 16,
                 think_budget: int = 256):
        self.ti = tok_info
        self.prompt_len = prompt_len
        self.n_act = n_act
        self.think_budget = think_budget
        self.act_ids_t = torch.tensor(tok_info.act_ids)
        self.forced_act_start = 0

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        neg = torch.finfo(scores.dtype).min
        for b in range(input_ids.shape[0]):
            row = input_ids[b]
            gen = row[self.prompt_len:]
            starts = (gen == self.ti.act_start_id).nonzero().flatten()
            if len(starts) == 0:
                # still thinking; force the act block open when the budget runs out
                if len(gen) >= self.think_budget:
                    scores[b, :] = neg
                    scores[b, self.ti.act_start_id] = 0.0
                    self.forced_act_start += 1
                continue
            after = gen[int(starts[0]) + 1:]
            if (after == self.ti.act_end_id).any():
                # act block closed -> force <|im_end|> to terminate cleanly
                scores[b, :] = neg
                scores[b, self.ti.im_end_id] = 0.0
                continue
            n_bins = len(after)
            if n_bins >= self.n_act:
                scores[b, :] = neg
                scores[b, self.ti.act_end_id] = 0.0
            else:
                # mask to the 128 act tokens, PRESERVING their original logits
                # (overwriting them would flatten the distribution and make
                # greedy decoding collapse to the lowest act id)
                allowed = self.act_ids_t.to(scores.device)
                keep = scores[b, allowed].clone()
                scores[b, :] = neg
                scores[b, allowed] = keep
        return scores


@torch.no_grad()
def generate_reason_act(dvla, prompt_inputs, think_budget=256, emit_discrete=True,
                        greedy=True):
    """prompt_inputs: processor output for a (left-padded) batch of prompts.

    Returns dict with sequences, per-row spans/bins/text, wall seconds.
    """
    model, ti = dvla.model, dvla.tok_info
    ids = prompt_inputs["input_ids"].to(dvla.device)
    prompt_len = ids.shape[1]
    kwargs = dict(
        input_ids=ids,
        attention_mask=prompt_inputs["attention_mask"].to(dvla.device),
        max_new_tokens=think_budget + 24,
        do_sample=not greedy,
    )
    for k in ("pixel_values", "image_grid_thw", "mm_token_type_ids"):
        if k in prompt_inputs and prompt_inputs[k] is not None:
            kwargs[k] = prompt_inputs[k].to(dvla.device)

    proc = None
    if emit_discrete:
        proc = ActConstraintProcessor(ti, prompt_len, think_budget=think_budget)
        kwargs["logits_processor"] = [proc]
        kwargs["eos_token_id"] = ti.im_end_id
    else:
        kwargs["eos_token_id"] = ti.think_close_id  # stop right after </think>

    t0 = time.time()
    seqs = model.generate(**kwargs)
    wall = time.time() - t0

    rows = []
    bin_of_id = {tid: i for i, tid in enumerate(ti.act_ids)}
    for b in range(seqs.shape[0]):
        row = seqs[b]
        gen = row[prompt_len:]
        entry = {"well_formed": False, "bins": None, "spans": {}, "gen_len": int((gen != dvla.tokenizer.pad_token_id).sum())}
        tc = (row == ti.think_close_id).nonzero().flatten()
        entry["spans"]["think_close"] = int(tc[-1]) if len(tc) else -1
        to = (row == ti.think_open_id).nonzero().flatten()
        entry["spans"]["think_open"] = int(to[0]) if len(to) else -1
        if emit_discrete:
            s = (gen == ti.act_start_id).nonzero().flatten()
            e = (gen == ti.act_end_id).nonzero().flatten()
            if len(s) and len(e) and int(e[0]) - int(s[0]) == 17:
                bins = [bin_of_id.get(int(t), -1) for t in gen[int(s[0]) + 1: int(e[0])]]
                if all(v >= 0 for v in bins):
                    entry["bins"] = bins
                    entry["well_formed"] = entry["spans"]["think_open"] >= 0 and entry["spans"]["think_close"] > 0
            entry["spans"]["act_start"] = prompt_len + int(s[0]) if len(s) else -1
            entry["spans"]["act_end"] = prompt_len + int(e[0]) if len(e) else -1
        else:
            entry["well_formed"] = entry["spans"]["think_close"] > 0
        rows.append(entry)
    return {"sequences": seqs, "rows": rows, "wall": wall, "prompt_len": prompt_len,
            "forced_act_start": getattr(proc, "forced_act_start", 0)}


def discrete_chunk(row_entry):
    if row_entry["bins"] is None:
        return None
    return bins_to_chunk(np.array(row_entry["bins"]))
