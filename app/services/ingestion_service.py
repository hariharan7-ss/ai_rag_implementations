import uuid
from typing import Dict, Any
from fastapi import UploadFile
from app.ingestion.loaders import load_document
from app.ingestion.web_scraper import scrape_url
from app.ingestion.chunker import recursive_chunk
from app.ingestion.metadata import create_metadata
from app.embeddings.model import embedding_model
from app.vectorstore.pinecone_store import pinecone_store

class IngestionService:
    async def process_file(self, file: UploadFile) -> Dict[str, Any]:
        document_id = f"doc_{uuid.uuid4().hex[:8]}"
        text = await load_document(file)
        chunks = recursive_chunk(text)
        
        if not chunks:
            return {"document_id": document_id, "file_name": file.filename, "status": "failed", "chunks_created": 0}
            
        metadata = create_metadata(document_id, file.filename, "file", file.filename, chunks)
        embeddings = embedding_model.encode(chunks)
        
        pinecone_store.upsert_chunks(metadata, embeddings)
        
        return {
            "document_id": document_id,
            "file_name": file.filename,
            "status": "indexed",
            "chunks_created": len(chunks)
        }

    async def process_url(self, url: str) -> Dict[str, Any]:
        # Handle HttpUrl from pydantic by converting to str
        url_str = str(url)
        document_id = f"web_{uuid.uuid4().hex[:8]}"
        text = scrape_url(url_str)
        chunks = recursive_chunk(text)
        
        if not chunks:
            return {"document_id": document_id, "url": url_str, "status": "failed", "chunks_created": 0}
            
        metadata = create_metadata(document_id, url_str, "web", url_str, chunks)
        embeddings = embedding_model.encode(chunks)
        
        pinecone_store.upsert_chunks(metadata, embeddings)
        
        return {
            "document_id": document_id,
            "url": url_str,
            "status": "indexed",
            "chunks_created": len(chunks)
        }

    def delete_document(self, document_id: str):
        pinecone_store.delete_document(document_id)

    def list_documents(self):
        return pinecone_store.list_documents()
