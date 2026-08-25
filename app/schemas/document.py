from pydantic import BaseModel
from typing import Optional

class ScrapeRequest(BaseModel):
    url: str

class DocumentResponse(BaseModel):
    document_id: str
    file_name: Optional[str] = None
    url: Optional[str] = None
    status: str
    chunks_created: int
