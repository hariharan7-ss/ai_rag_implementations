import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.services.redaction import redact_record

TRACE_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "traces",
    "traces.jsonl",
)

# Bumped when the trace record shape changes. Week 5 replay audit found the v0
# records were missing prompt_version, model params and retrieval params, so a
# trace could not be replayed from itself without guessing.
TRACE_SCHEMA_VERSION = "trace-v2"


def log_trace(
    query: str,
    retrieved: List[Dict[str, Any]],
    prompt: str,
    model: str,
    answer: str,
    prompt_version: Optional[str] = None,
    model_params: Optional[Dict[str, Any]] = None,
    retrieval_params: Optional[Dict[str, Any]] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist one fully replayable record of a query.

    Everything needed to re-run the generation without the live app is stored:
    the rendered prompt and its version, the retrieved chunk ids and scores, the
    model and the exact parameters it was called with, and the raw output.

    Employee identifiers are redacted here, before the record is serialised and
    written -- not by a later pass over the file.
    """
    os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
    trace_id = f"trace_{uuid.uuid4().hex[:10]}"
    record = {
        "trace_id": trace_id,
        "schema_version": TRACE_SCHEMA_VERSION,
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
        "retrieval_params": retrieval_params,
        "prompt_version": prompt_version,
        "prompt": prompt,
        "model": model,
        "model_params": model_params,
        "finish_reason": finish_reason,
        "usage": usage,
        "raw_output": answer,
        "answer": answer,
    }

    # Redact BEFORE write. Nothing unredacted ever touches the file.
    record, audit = redact_record(record)
    record["redaction"] = audit

    with open(TRACE_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return trace_id
