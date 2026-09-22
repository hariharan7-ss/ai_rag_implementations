import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

TRACE_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "traces",
    "traces.jsonl",
)


def log_trace(query: str, retrieved: List[Dict[str, Any]], prompt: str, model: str, answer: str) -> str:
    """Persist one full, replayable record of a query: what was asked, what was
    retrieved, what was sent to the LLM, and what came back."""
    os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
    trace_id = f"trace_{uuid.uuid4().hex[:10]}"
    record = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved": [
            {
                "chunk_id": r.get("payload", {}).get("chunk_id"),
                "document_id": r.get("payload", {}).get("document_id"),
                "source": r.get("payload", {}).get("source"),
                "title": r.get("payload", {}).get("title"),
                "score": r.get("score"),
                "text": r.get("payload", {}).get("text"),
            }
            for r in retrieved
        ],
        "prompt": prompt,
        "model": model,
        "answer": answer,
    }
    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return trace_id
