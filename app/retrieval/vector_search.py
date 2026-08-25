from app.embeddings.model import embedding_model
from app.vectorstore.pinecone_store import pinecone_store
from typing import List, Dict, Any

def perform_vector_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_emb = embedding_model.encode([query])[0]
    return pinecone_store.search(query_emb, limit=top_k)
