from typing import Dict, Any
from app.services.retrieval_service import retrieval_service
from app.generation.llm import generate_response
from app.generation.prompts import RAG_PROMPT_TEMPLATE

class RAGService:
    async def answer_query(self, query: str) -> Dict[str, Any]:
        retrieved = retrieval_service.retrieve(query)
        
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
        answer = generate_response(prompt)
        
        return {
            "answer": answer,
            "sources": sources
        }
