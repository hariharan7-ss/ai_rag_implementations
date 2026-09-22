import os
import uuid
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec

class PineconeStore:
    def __init__(self, index_name: str = "documents", namespace: str = None):
        self.index_name = index_name
        # "" is Pinecone's default namespace, so the running app is unaffected
        # unless PINECONE_NAMESPACE is set (the eval harness sets it).
        self.namespace = namespace if namespace is not None else os.environ.get("PINECONE_NAMESPACE", "")
        self._pc = None
        self._index = None

    @property
    def pc(self):
        if self._pc is None:
            api_key = os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable is not set.")
            self._pc = Pinecone(api_key=api_key)
        return self._pc

    @property
    def index(self):
        if self._index is None:
            # Check if index exists, create if not
            existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            self._index = self.pc.Index(self.index_name)
        return self._index

    def upsert_chunks(self, chunks_metadata: List[Dict[str, Any]], embeddings: List[List[float]]):
        vectors = []
        for meta, emb in zip(chunks_metadata, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, meta["chunk_id"]))
            # Ensure metadata values are compatible with Pinecone (strings, numbers, booleans, lists of strings)
            safe_meta = {k: v for k, v in meta.items() if v is not None}
            vectors.append((point_id, emb, safe_meta))
            
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[i:i+batch_size], namespace=self.namespace)

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        response = self.index.query(
            vector=query_embedding,
            top_k=limit,
            include_metadata=True,
            namespace=self.namespace
        )
        
        results = []
        for match in response.matches:
            results.append({
                "score": match.score,
                "payload": match.metadata
            })
        return results

    def delete_document(self, document_id: str):
        try:
            self.index.delete(filter={"document_id": {"$eq": document_id}}, namespace=self.namespace)
        except Exception as e:
            print(f"Error deleting document {document_id} from Pinecone: {e}")

    def list_documents(self) -> List[Dict[str, Any]]:
        return [{"status": "Listing documents not natively supported via simple API in Pinecone"}]

pinecone_store = PineconeStore()
