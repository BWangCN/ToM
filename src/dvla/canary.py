"""Language canary (§7.10): 5 fixed prompts, identical greedy decoding at
pre-A / post-A / post-B checkpoints. Verbatim outputs go into the report."""

import json
from pathlib import Path

import torch
from PIL import Image


def _degenerate(ids: list[int]) -> bool:
    """Consecutive n-gram repetition check (n=1..3, >= ~12 repeated tokens)."""
    for n in (1, 2, 3):
        need = max(12 // n, 4)
        run, prev = 1, None
        grams = [tuple(ids[i:i + n]) for i in range(0, len(ids) - n + 1, n)]
        for g in grams:
            run = run + 1 if g == prev else 1
            prev = g
            if run >= need:
                return True
    return False


@torch.no_grad()
def run_canary(dvla, prompts_path: Path, data_root: Path, max_new_tokens=256):
    prompts = json.loads(Path(prompts_path).read_text())
    results = []
    for p in prompts:
        content = []
        images = None
        if p["type"] == "vqa":
            frame_dir = data_root / p["image_dir"]
            img_path = sorted(frame_dir.glob("*.png"))[0]
            images = [Image.open(img_path).convert("RGB")]
            content.append({"type": "image"})
        content.append({"type": "text", "text": p["prompt"]})
        msgs = [{"role": "user", "content": content}]
        text = dvla.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        enc = dvla.processor(text=[text], images=images, return_tensors="pt")
        kwargs = {k: v.to(dvla.device) for k, v in enc.items()}
        out = dvla.model.generate(**kwargs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = out[0, enc["input_ids"].shape[1]:]
        text_out = dvla.tokenizer.decode(gen, skip_special_tokens=False)
        text_out = text_out.replace("<|im_end|>", "").strip()
        results.append({
            "name": p["name"],
            "prompt": p["prompt"],
            "output": text_out,
            "has_act_substr": "<|act_" in text_out,
            "degenerate": _degenerate(gen.tolist()),
        })
    auto_pass = not any(r["has_act_substr"] or r["degenerate"] for r in results)
    return {"results": results, "auto_pass": auto_pass}
