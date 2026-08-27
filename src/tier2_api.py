"""Layer 2: judgment from free-tier hosted models.

Division of labour: this layer is called for semantic judgment and for
adjudicating disagreement between the deterministic layers. It is NOT used for
bulk measurement, for three reasons -- free-tier quotas are small, hosted models
cannot return masks or pixel areas, and their volume limits rule out the
test-time augmentation that the confidence model relies on.

Two things are worth noting for the memo:

  * Where a provider is OpenAI-compatible it may expose token `logprobs`. That
    turns a "yes" into P(yes)=0.83 -- a real probability rather than a
    verbalised one. Support is PROBED at runtime and recorded, never assumed.
  * Every response is cached by a content hash, so re-runs cost no quota and a
    reviewer without keys can reproduce the run from the committed cache.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import config
from .cache import Cache, key_of

_cache = Cache(config.API_CACHE)
TIMEOUT = (10, 120)


@dataclass
class Provider:
    name: str
    env_key: str
    model: str
    kind: str          # "gemini" or "openai"  (wire format)
    url: str
    rpm: float         # self-imposed pacing, below the published free-tier limit

    @property
    def key(self) -> Optional[str]:
        return os.environ.get(self.env_key)

    @property
    def available(self) -> bool:
        return bool(self.key)


# Ordered by preference. Gemini has the best judgment; NIM has no daily cap and
# is therefore the volume-safe fallback; Groq and OpenRouter are further backstops.
PROVIDERS = [
    Provider("gemini", "GEMINI_API_KEY", "gemini-2.5-flash", "gemini",
             "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
             rpm=8),
    Provider("nvidia", "NVIDIA_API_KEY", "meta/llama-3.2-90b-vision-instruct",
             "openai", "https://integrate.api.nvidia.com/v1/chat/completions", rpm=30),
    Provider("groq", "GROQ_API_KEY", "meta-llama/llama-4-scout-17b-16e-instruct",
             "openai", "https://api.groq.com/openai/v1/chat/completions", rpm=25),
    Provider("openrouter", "OPENROUTER_API_KEY",
             "meta-llama/llama-3.2-11b-vision-instruct:free", "openai",
             "https://openrouter.ai/api/v1/chat/completions", rpm=15),
]

_last_call: Dict[str, float] = {}
# Runtime record of which providers actually returned logprobs. Reported in the
# memo rather than assumed from documentation.
LOGPROB_SUPPORT: Dict[str, bool] = {}


def available_providers() -> List[Provider]:
    return [p for p in PROVIDERS if p.available]


def _pace(p: Provider) -> None:
    gap = time.time() - _last_call.get(p.name, 0.0)
    need = 60.0 / max(p.rpm, 1e-6)
    if gap < need:
        time.sleep(need - gap)
    _last_call[p.name] = time.time()


def _b64(image: bytes) -> str:
    return base64.b64encode(image).decode()


def _build_request(p: Provider, prompt: str, image: bytes,
                   want_logprobs: bool, max_tokens: int) -> Tuple[str, Dict, Dict]:
    if p.kind == "gemini":
        url = p.url.format(model=p.model)
        headers = {"x-goog-api-key": p.key, "Content-Type": "application/json"}
        body = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": _b64(image)}},
                {"text": prompt},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max_tokens},
        }
        return url, headers, body

    headers = {"Authorization": f"Bearer {p.key}", "Content-Type": "application/json"}
    body: Dict[str, Any] = {
        "model": p.model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_b64(image)}"}},
        ]}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    if want_logprobs:
        body["logprobs"] = True
        body["top_logprobs"] = 8
    return p.url, headers, body


def _extract_text(p: Provider, payload: Dict[str, Any]) -> str:
    if p.kind == "gemini":
        cands = payload.get("candidates") or []
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", []) or []
        return "".join(part.get("text", "") for part in parts)
    choices = payload.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content") or ""


def _extract_yes_logprob(payload: Dict[str, Any]) -> Optional[float]:
    """P(yes) from the first token's top-k logprobs, renormalised over yes/no.

    This is the signal the whole open-weight argument is about: a probability
    the model actually assigned, rather than a number it wrote down.
    """
    try:
        content = payload["choices"][0]["logprobs"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    if not content:
        return None
    p_yes = p_no = 0.0
    for entry in content[:2]:
        for alt in entry.get("top_logprobs", []) or []:
            tok = (alt.get("token") or "").strip().lower().strip(".,:'\"")
            prob = math.exp(alt.get("logprob", -99))
            if tok in ("yes", "y", "true"):
                p_yes += prob
            elif tok in ("no", "n", "false"):
                p_no += prob
        if p_yes + p_no > 0:
            break
    if p_yes + p_no <= 0:
        return None
    return p_yes / (p_yes + p_no)


def call(prompt: str, image: bytes, want_logprobs: bool = False,
         max_tokens: int = 700, offline: bool = False,
         providers: Optional[List[Provider]] = None) -> Optional[Dict[str, Any]]:
    """Ask the first available provider that answers. Cached by content hash."""
    provs = providers if providers is not None else available_providers()
    ck = key_of("api_v1", prompt, key_of(image[:4096], len(image)),
                want_logprobs, [p.name + ":" + p.model for p in provs][:1])
    cached = _cache.get_json(ck)
    if cached is not None:
        return cached
    if offline or not provs:
        return None

    errors = []
    for p in provs:
        try:
            _pace(p)
            url, headers, body = _build_request(p, prompt, image, want_logprobs,
                                                max_tokens)
            r = requests.post(url, headers=headers, json=body, timeout=TIMEOUT)
            if r.status_code == 429:
                errors.append(f"{p.name}: rate limited")
                continue
            r.raise_for_status()
            payload = r.json()
            text = _extract_text(p, payload)
            yes_lp = _extract_yes_logprob(payload) if want_logprobs else None
            if want_logprobs:
                LOGPROB_SUPPORT[p.name] = yes_lp is not None
            out = {
                "provider": p.name, "model": p.model, "text": text,
                "p_yes": yes_lp, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime()),
            }
            _cache.put_json(ck, out)
            return out
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{p.name}: {type(exc).__name__}")
    # Every provider failing is a degraded run, not a crash: the confidence
    # model takes n_sources_available as a feature, so losing this layer lowers
    # confidence honestly instead of silently.
    return {"provider": None, "error": "; ".join(errors), "text": "", "p_yes": None}


def parse_json_response(text: str) -> Optional[Dict[str, Any]]:
    """Models wrap JSON in prose or code fences often enough to handle it here."""
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def status() -> Dict[str, Any]:
    return {
        "available": [p.name for p in available_providers()],
        "configured_but_missing_key": [p.name for p in PROVIDERS if not p.available],
        "logprob_support_observed": dict(LOGPROB_SUPPORT),
        "cache": _cache.stats(),
    }
