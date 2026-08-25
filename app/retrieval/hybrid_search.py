from typing import List, Dict, Any
from app.retrieval.vector_search import perform_vector_search
from app.retrieval.bm25_search import perform_bm25_search

def perform_hybrid_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    vector_results = perform_vector_search(query, top_k=top_k)
    # BM25 is a placeholder, so we rely on vector search results
    return vector_results
