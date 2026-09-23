import os
from typing import Any, Dict, Optional
from groq import Groq

try:
    client = Groq()
except Exception:
    client = None

# Week 5: generation parameters are pinned and recorded in every trace, so a
# trace can be replayed without guessing what the provider defaults were.
GEN_PARAMS: Dict[str, Any] = {
    "temperature": 0.2,
    "top_p": 1.0,
    "max_tokens": 1024,
    "seed": 20260923,
}

_NOT_CONFIGURED = "Groq client is not initialized. Please set GROQ_API_KEY environment variable."


def generate_response_full(
    prompt: str,
    model: str = "openai/gpt-oss-20b",
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate, returning the raw output alongside the exact params used."""
    params = dict(GEN_PARAMS if params is None else params)
    if not client:
        return {"text": _NOT_CONFIGURED, "params": params, "finish_reason": "error", "usage": None}

    completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        **params,
    )
    choice = completion.choices[0]
    usage = getattr(completion, "usage", None)
    return {
        "text": choice.message.content,
        "params": params,
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        } if usage else None,
    }


def generate_response(prompt: str, model: str = "openai/gpt-oss-20b") -> str:
    return generate_response_full(prompt, model=model)["text"]
