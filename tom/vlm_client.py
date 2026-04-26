"""VLM client abstraction.

Swap the backend by changing `cfg.tom.vlm_backend` (openai | qwen). The
contract: `.query(messages, image_paths, response_json_schema) -> dict`.
Backends are responsible for their own image encoding + retry.
"""
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


@dataclass
class VLMResponse:
    raw: str
    parsed: Dict[str, Any]
    backend: str
    model: str
    usage: Dict[str, int]


class VLMClient:
    """Backend-agnostic entry point. Instantiate once per process."""

    def __init__(self, cfg: dict):
        tom_cfg = cfg["tom"]
        backend = tom_cfg["vlm_backend"].lower()
        self.backend_name = backend
        self.model = tom_cfg["vlm_model"]
        if backend == "openai":
            self.impl = _OpenAIBackend(self.model)
        elif backend == "qwen":
            self.impl = _QwenBackend(self.model)
        else:
            raise ValueError(f"Unknown VLM backend: {backend}")

    def query(self, system_prompt: str, user_prompt: str,
              image_paths: Optional[List[str]] = None,
              response_schema: Optional[Dict] = None,
              temperature: float = 0.2) -> VLMResponse:
        return self.impl.query(system_prompt=system_prompt,
                               user_prompt=user_prompt,
                               image_paths=image_paths or [],
                               response_schema=response_schema,
                               temperature=temperature)


# ---------------------------------------------------------------------------
class _OpenAIBackend:
    def __init__(self, model: str):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.40 in the Isaac env") from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def query(self, system_prompt, user_prompt, image_paths, response_schema, temperature):
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for p in image_paths:
            b64 = _encode_image(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=temperature,
        )
        if response_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "tom_reasoning", "schema": response_schema, "strict": True},
            }

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
        raw = resp.choices[0].message.content or "{}"
        parsed = _safe_json_loads(raw)
        usage = resp.usage.model_dump() if hasattr(resp, "usage") and resp.usage else {}
        return VLMResponse(raw=raw, parsed=parsed, backend="openai",
                           model=self.model, usage=usage)


# ---------------------------------------------------------------------------
class _QwenBackend:
    """Stub for Qwen2-VL. Implement against the local HF model or Dashscope API
    when we move off OpenAI. Contract must match _OpenAIBackend.query."""

    def __init__(self, model: str):
        self.model = model
        # TODO: from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        raise NotImplementedError("Qwen backend not wired yet — keep vlm_backend='openai'.")

    def query(self, *args, **kwargs):
        raise NotImplementedError


def _safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Strip code fences if GPT ignored JSON-mode.
        s2 = s.strip().strip("`")
        if s2.startswith("json"):
            s2 = s2[4:]
        try:
            return json.loads(s2)
        except Exception:
            return {"_raw": s}
