from typing import Dict, Any
from app.services.retrieval_service import retrieval_service
from app.services.trace_logger import log_trace
from app.generation.llm import generate_response_full
from app.generation.prompts import RAG_PROMPT_TEMPLATE, RAG_PROMPT_VERSION

LLM_MODEL = "openai/gpt-oss-20b"

# Recorded in every trace so retrieval can be reasoned about from the trace alone.
RETRIEVAL_PARAMS = {"initial_k": 10, "final_k": 4, "strategy": "vector+rerank"}


class RAGService:
    async def answer_query(self, query: str) -> Dict[str, Any]:
        retrieved = retrieval_service.retrieve(
            query,
            initial_k=RETRIEVAL_PARAMS["initial_k"],
            final_k=RETRIEVAL_PARAMS["final_k"],
        )

        context_parts = []
        sources = []

        for r in retrieved:
            payload = r.get("payload", {})
            text = payload.get("text", "")
            context_parts.append(text)

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
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=query)
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
