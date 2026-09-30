"""One chat call, fully accounted: tokens, cost, latency, and any time lost to Groq 429s.

The agent and the workflow both go through `chat`, with the same model and the
same pinned params, so the race compares the control flow and nothing else.
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import groq
from dotenv import load_dotenv

from app.generation.llm import GEN_PARAMS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(ROOT, ".env"), override=True)

MODEL = "openai/gpt-oss-20b"
# The app's pinned Week 5 params, plus low reasoning effort: the free tier allows
# 8,000 tokens/minute, and a lap that reasons for 1,000 tokens eats that fast.
PARAMS: Dict[str, Any] = {**GEN_PARAMS, "reasoning_effort": os.environ.get("HR_AGENT_REASONING", "low")}

# USD per 1M tokens, Groq model page for openai/gpt-oss-20b (checked 2026-09-29).
PRICE_PER_M = {"input": 0.075, "cached_input": 0.037, "output": 0.30}

# max_retries=0: the SDK's own silent 429 retries would hide waits inside latency.
_client = groq.Groq(max_retries=0) if os.environ.get("GROQ_API_KEY") else None


class DeadlineExceeded(Exception):
    """The caller's wall-clock deadline passed (or would pass) during this call."""


def cost_usd(prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    return ((prompt_tokens - cached_tokens) * PRICE_PER_M["input"]
            + cached_tokens * PRICE_PER_M["cached_input"]
            + completion_tokens * PRICE_PER_M["output"]) / 1e6


@dataclass
class Call:
    message: Any
    finish_reason: Optional[str]
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_s: float            # wall time of the call, excluding 429 sleeps
    rate_limit_wait_s: float    # time slept on 429s before the call succeeded
    cost_usd: float = field(init=False)

    def __post_init__(self):
        self.cost_usd = cost_usd(self.prompt_tokens, self.completion_tokens, self.cached_tokens)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


# Token-bucket state from the last response headers, used only to pace BETWEEN runs.
_bucket = {"limit": None, "remaining": None, "at": 0.0}


def _record_headers(headers) -> None:
    try:
        _bucket.update(limit=int(headers["x-ratelimit-limit-tokens"]),
                       remaining=int(headers["x-ratelimit-remaining-tokens"]), at=time.monotonic())
    except (KeyError, TypeError, ValueError):
        pass


def wait_for_full_bucket(log=print) -> float:
    """Sleep until the per-minute token bucket has refilled. Called between race runs, outside
    any timed region, so a run never starts half-throttled by the previous one."""
    if not _bucket["limit"]:
        return 0.0
    rate = _bucket["limit"] / 60.0
    missing = _bucket["limit"] - _bucket["remaining"] - rate * (time.monotonic() - _bucket["at"])
    wait = max(0.0, missing / rate + 1.0) if missing > 0 else 0.0
    if wait:
        log(f"  (pacing: waiting {wait:.0f}s for the Groq token bucket to refill)")
        time.sleep(wait)
    return wait


def _retry_after(err: Exception) -> float:
    headers = getattr(getattr(err, "response", None), "headers", {}) or {}
    if headers.get("retry-after"):
        return float(headers["retry-after"]) + 0.5
    m = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(err))
    return (int(m.group(1) or 0) * 60 + float(m.group(2)) + 0.5) if m else 5.0


def chat(messages: List[Dict[str, Any]], *, tools=None, max_tokens: Optional[int] = None,
         response_format=None, deadline: Optional[float] = None) -> Call:
    """One completion. `deadline` (time.monotonic()) bounds the request timeout and any 429 wait."""
    if _client is None:
        raise RuntimeError("GROQ_API_KEY is not set")
    params = dict(PARAMS)
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    extra = {}
    if tools:
        extra["tools"] = tools
    if response_format:
        extra["response_format"] = response_format

    waited = 0.0
    for _ in range(6):
        timeout = None
        if deadline is not None:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                raise DeadlineExceeded("deadline passed before the model call")
        t0 = time.monotonic()
        try:
            raw = _client.chat.completions.with_raw_response.create(
                model=MODEL, messages=messages, timeout=timeout, **params, **extra)
        except groq.RateLimitError as e:
            wait = _retry_after(e)
            if deadline is not None and time.monotonic() + wait >= deadline:
                raise DeadlineExceeded(f"429 asks for {wait:.0f}s, past the wall-clock deadline") from e
            time.sleep(wait)
            waited += wait
            continue
        except groq.APITimeoutError as e:
            raise DeadlineExceeded("model call timed out at the wall-clock deadline") from e
        latency = time.monotonic() - t0
        _record_headers(raw.headers)
        resp = raw.parse()
        u = resp.usage
        details = getattr(u, "prompt_tokens_details", None)
        return Call(message=resp.choices[0].message, finish_reason=resp.choices[0].finish_reason,
                    prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                    cached_tokens=(getattr(details, "cached_tokens", 0) or 0) if details else 0,
                    latency_s=latency, rate_limit_wait_s=waited)
    raise RuntimeError("rate-limited 6 times in a row")
