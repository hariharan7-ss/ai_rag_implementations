from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
from app.schemas.document import ScrapeRequest, DocumentResponse
from app.services.ingestion_service import IngestionService

router = APIRouter()
ingestion_service = IngestionService()

@router.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        # Save temp and process
        result = await ingestion_service.process_file(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

@router.post("/documents/scrape", response_model=DocumentResponse)
async def scrape_url(request: ScrapeRequest):
    try:
        result = await ingestion_service.process_url(request.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=repr(e))

@router.get("/documents")
async def list_documents():
    try:
        return ingestion_service.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        ingestion_service.delete_document(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
