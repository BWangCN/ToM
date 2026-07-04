"""Backbone + LoRA + token rows + flow decoder assembly (CLAUDE.md §2).

Load path (fresh training):    DVLA.build(cfg)
Load path (trained checkpoint): DVLA.load_trained(cfg, ckpt_dir)

Knowledge insulation: decoder conditions on last-layer hidden states of the
assistant span; detach_condition=True stops any decoder gradient reaching the
VLM — the VLM trains only through the two CE losses.
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from .collator import detect_think_autopen
from .decoder_flow import ActionExpert
from .losses import combined_loss
from .tokenizer_ext import (TokInfo, assert_row_freeze, build_peft_model,
                            extend_tokenizer_and_model, trainable_breakdown)


def _load_base(cfg, device):
    mcfg = cfg["model"]
    kwargs = dict(attn_implementation=mcfg["attn_implementation"])
    if mcfg["precision"] == "bf16":
        kwargs["dtype"] = torch.bfloat16
    elif mcfg["precision"] == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["dtype"] = torch.bfloat16
    else:
        raise ValueError(mcfg["precision"])
    model = AutoModelForImageTextToText.from_pretrained(mcfg["name_or_path"], **kwargs)
    if mcfg["precision"] == "bf16":
        model.to(device)
    return model


def _cap_pixels(processor, cfg):
    ip = getattr(processor, "image_processor", None)
    cap = cfg["inputs"]["max_pixels_per_frame"]
    if ip is not None:
        for attr in ("max_pixels",):
            if hasattr(ip, attr):
                setattr(ip, attr, cap)
        if getattr(ip, "size", None) and isinstance(ip.size, dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = cap  # qwen-style size dict is in pixels (area)


class DVLA:
    def __init__(self, cfg, processor, model, decoder, tok_info: TokInfo, device="cuda"):
        self.cfg = cfg
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model
        self.decoder = decoder
        self.tok_info = tok_info
        self.device = device
        self.autopen = detect_think_autopen(processor)

    # ------------------------------------------------------------ build/load

    @classmethod
    def build(cls, cfg, device="cuda"):
        """Fresh assembly: base + 130 tokens + LoRA + decoder (seeded init)."""
        processor = AutoProcessor.from_pretrained(cfg["model"]["name_or_path"])
        _cap_pixels(processor, cfg)
        model = _load_base(cfg, device)
        for p in model.parameters():
            p.requires_grad_(False)

        tok_info = extend_tokenizer_and_model(model, processor.tokenizer, seed=cfg["seed"])
        model, used_fallback = build_peft_model(model, cfg["lora"], tok_info)
        if used_fallback:
            from .common import log_deviation
            log_deviation("tokenizer_ext fallback",
                          "peft rejected trainable_token_indices; explicit grad mask used")

        if cfg["optim"]["grad_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()

        # vision tower frozen (belt and braces — LoRA regex already excludes it)
        for name, p in model.named_parameters():
            if ".visual." in name and p.requires_grad:
                p.requires_grad_(False)

        text_cfg = getattr(model.config, "text_config", model.config)
        d_vlm = text_cfg.hidden_size
        dcfg = cfg["decoder"]
        decoder = ActionExpert(d_cond=d_vlm, d_model=dcfg["d_model"], depth=dcfg["depth"],
                               n_heads=dcfg["n_heads"], mlp_ratio=dcfg["mlp_ratio"],
                               chunk_len=cfg["action"]["chunk_len"],
                               action_dim=cfg["action"]["dim"]).float().to(device)
        n_dec = decoder.n_params()
        assert 30e6 <= n_dec <= 60e6, f"decoder {n_dec/1e6:.1f}M outside 30-60M band"

        obj = cls(cfg, processor, model, decoder, tok_info, device)
        bd = trainable_breakdown(model, decoder)
        print(f"[init] decoder params: {n_dec/1e6:.2f}M | d_vlm={d_vlm} | "
              f"tie_word_embeddings={tok_info.tied} | peft_fallback={used_fallback}")
        print(f"[init] trainable breakdown: " + ", ".join(f"{k}={v/1e6:.2f}M" for k, v in bd.items()))
        from .common import record_metric
        record_metric("trainable_breakdown", {**bd, "decoder_exact": n_dec,
                                              "tied": tok_info.tied,
                                              "peft_fallback": used_fallback})
        return obj

    @classmethod
    def load_trained(cls, cfg, ckpt_dir, device="cuda"):
        """Fresh-process restore: rebuild base, re-extend (same seed -> same base
        rows), attach saved adapter, load decoder + saved tokenizer."""
        from peft import PeftModel

        ckpt_dir = Path(ckpt_dir)
        processor = AutoProcessor.from_pretrained(ckpt_dir / "tokenizer")
        _cap_pixels(processor, cfg)
        model = _load_base(cfg, device)
        for p in model.parameters():
            p.requires_grad_(False)
        tok_info = extend_tokenizer_and_model(model, AutoProcessor.from_pretrained(
            cfg["model"]["name_or_path"]).tokenizer, seed=cfg["seed"])
        saved_ids = json.loads((ckpt_dir / "tok_info.json").read_text())
        assert saved_ids["new_ids"] == tok_info.new_ids, "token id mismatch on reload"

        model = PeftModel.from_pretrained(model, ckpt_dir / "adapter", is_trainable=True)
        text_cfg = getattr(model.config, "text_config", model.config)
        dcfg = cfg["decoder"]
        decoder = ActionExpert(d_cond=text_cfg.hidden_size, d_model=dcfg["d_model"],
                               depth=dcfg["depth"], n_heads=dcfg["n_heads"],
                               mlp_ratio=dcfg["mlp_ratio"], chunk_len=cfg["action"]["chunk_len"],
                               action_dim=cfg["action"]["dim"]).float().to(device)
        decoder.load_state_dict(torch.load(ckpt_dir / "decoder.pt", map_location=device))
        return cls(cfg, processor, model, decoder, tok_info, device)

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(out_dir / "adapter")
        self.processor.save_pretrained(out_dir / "tokenizer")
        torch.save(self.decoder.state_dict(), out_dir / "decoder.pt")
        (out_dir / "tok_info.json").write_text(json.dumps({
            "new_ids": self.tok_info.new_ids, "act_start_id": self.tok_info.act_start_id,
            "act_end_id": self.tok_info.act_end_id, "tied": self.tok_info.tied}))

    # ------------------------------------------------------------ training fwd

    def _slice_cond(self, h, spans, detach=True):
        """h (B,S,D); spans (B,2) [start, end_exclusive] -> padded cond + PAD mask."""
        if detach:
            h = h.detach()
        B = h.shape[0]
        lens = (spans[:, 1] - spans[:, 0]).tolist()
        Lc = max(lens)
        cond = h.new_zeros(B, Lc, h.shape[-1])
        mask = torch.ones(B, Lc, dtype=torch.bool, device=h.device)  # True = PAD
        for i in range(B):
            s, e = int(spans[i, 0]), int(spans[i, 1])
            cond[i, : e - s] = h[i, s:e]
            mask[i, : e - s] = False
        return cond, mask

    def cond_span_key(self, condition=None):
        condition = condition or self.cfg["decoder"]["condition"]
        return {"think_and_act": "cond_span_think_and_act",
                "think_only": "cond_span_think_only"}[condition]

    def forward_train(self, batch, flow_generator=None):
        dev = self.device
        kwargs = {}
        if batch.get("mm_token_type_ids") is not None:
            kwargs["mm_token_type_ids"] = batch["mm_token_type_ids"].to(dev)
        out = self.model(
            input_ids=batch["input_ids"].to(dev),
            attention_mask=batch["attention_mask"].to(dev),
            pixel_values=batch["pixel_values"].to(dev),
            image_grid_thw=batch["image_grid_thw"].to(dev),
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        h = out.hidden_states[-1]
        cond, mask = self._slice_cond(h, batch[self.cond_span_key()].to(dev),
                                      detach=self.cfg["decoder"]["detach_condition"])
        flow = self.decoder.flow_loss(batch["action_chunk"].to(dev).float(), cond, mask,
                                      generator=flow_generator)
        lambdas = batch.get("lambdas") or self.cfg["losses"]
        total, parts = combined_loss(out.logits, batch["labels_reason"].to(dev),
                                     batch["labels_act"].to(dev), flow, lambdas)
        if batch.get("want_cond_cache"):
            parts["cond_cache"] = (cond.detach(), mask,
                                   batch["action_chunk"].to(dev).float())
        return total, parts

    # ------------------------------------------------------------ inference

    @torch.no_grad()
    def hidden_states_for(self, input_ids, attention_mask, pixel_values, image_grid_thw,
                          mm_token_type_ids=None):
        kwargs = {}
        if mm_token_type_ids is not None:
            kwargs["mm_token_type_ids"] = mm_token_type_ids.to(self.device)
        out = self.model(input_ids=input_ids.to(self.device),
                         attention_mask=attention_mask.to(self.device),
                         pixel_values=pixel_values.to(self.device) if pixel_values is not None else None,
                         image_grid_thw=image_grid_thw.to(self.device) if image_grid_thw is not None else None,
                         output_hidden_states=True, use_cache=False, **kwargs)
        return out.hidden_states[-1]

    def flow_decode(self, h, spans, euler_steps=None, seed=0, condition=None):
        cond, mask = self._slice_cond(h, spans, detach=True)
        g = torch.Generator(device=self.device).manual_seed(seed)
        steps = euler_steps or self.cfg["decoder"]["euler_steps"]
        return self.decoder.sample(cond, mask, steps=steps, generator=g)

    def grad_test(self, batch_builder):
        return assert_row_freeze(self.model, self.tok_info, batch_builder)
