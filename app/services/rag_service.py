import os
import re
from datetime import date
from typing import Dict, Any, List
from app.services.retrieval_service import retrieval_service
from app.vectorstore.pinecone_store import pinecone_store
from app.services.trace_logger import log_trace
from app.generation.llm import generate_response_full
from app.generation.prompts import RAG_PROMPT_TEMPLATE, RAG_PROMPT_VERSION, REFUSAL

LLM_MODEL = "openai/gpt-oss-20b"

# Recorded in every trace so retrieval can be reasoned about from the trace alone.
RETRIEVAL_PARAMS = {"initial_k": 10, "final_k": 4, "strategy": "vector+rerank"}

_HEADER_END = re.compile(r"\s(?:Owner:|Applies to:|1\.\s)")


def _document_labels(retrieved: List[Dict[str, Any]]) -> Dict[str, str]:
    """document_id -> its header ("<title> Effective: <date>"), read from the doc's chunk 0.

    A mid-document chunk does not say which policy or which version it belongs to,
    so the model could not cite either. Chunk 0 of every document starts with that
    header, and chunk ids are deterministic, so one fetch recovers it.
    """
    doc_ids = {r.get("payload", {}).get("document_id") for r in retrieved} - {None}
    try:
        heads = pinecone_store.fetch_chunks([f"{d}_chunk_0" for d in doc_ids])
    except Exception:
        heads = {}
    labels = {}
    for d in doc_ids:
        text = (heads.get(f"{d}_chunk_0") or {}).get("text", "")
        m = _HEADER_END.search(text)
        header = (text[: m.start()] if m else text[:120]).replace("�", "-").strip()
        labels[d] = header
    return labels


class RAGService:
    async def answer_query(self, query: str) -> Dict[str, Any]:
        retrieved = retrieval_service.retrieve(
            query,
            initial_k=RETRIEVAL_PARAMS["initial_k"],
            final_k=RETRIEVAL_PARAMS["final_k"],
        )

        context_parts = []
        sources = []
        labels = _document_labels(retrieved)

        for r in retrieved:
            payload = r.get("payload", {})
            text = payload.get("text", "")
            label = labels.get(payload.get("document_id")) or payload.get("source") or "unknown"
            context_parts.append(f"[Document: {label}]\n{text}")

            source_info = {
                "document_id": payload.get("document_id"),
                "source": payload.get("source"),
                "source_type": payload.get("source_type"),
                "title": payload.get("title"),
                "chunk_id": payload.get("chunk_id")
            }
            if source_info not in sources:
                sources.append(source_info)

        context = "\n\n---\n\n".join(context_parts)
        # RAG_TODAY pins the date for reproducible evals; otherwise the real date.
        today = os.environ.get("RAG_TODAY") or date.today().isoformat()
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query, today=today, refusal=REFUSAL)
        gen = generate_response_full(prompt, model=LLM_MODEL)
        answer = gen["text"]

        trace_id = log_trace(
            query=query,
            retrieved=retrieved,
            prompt=prompt,
            model=LLM_MODEL,
            answer=answer,
            prompt_version=RAG_PROMPT_VERSION,
            model_params=gen["params"],
            retrieval_params=RETRIEVAL_PARAMS,
            finish_reason=gen["finish_reason"],
            usage=gen["usage"],
        )

        return {
            "answer": answer,
            "sources": sources,
            "trace_id": trace_id,
        }
