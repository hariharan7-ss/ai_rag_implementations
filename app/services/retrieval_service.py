from typing import List, Dict, Any
from app.retrieval.hybrid_search import perform_hybrid_search
from app.retrieval.reranker import rerank_results

class RetrievalService:
    def retrieve(self, query: str, initial_k: int = 10, final_k: int = 4) -> List[Dict[str, Any]]:
        results = perform_hybrid_search(query, top_k=initial_k)
        reranked_results = rerank_results(query, results, top_k=final_k)
        return reranked_results

retrieval_service = RetrievalService()
