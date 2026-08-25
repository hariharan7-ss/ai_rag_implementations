import tempfile
import os
from pypdf import PdfReader
from fastapi import UploadFile
from app.ingestion.cleaner import clean_text

async def load_document(file: UploadFile) -> str:
    filename = file.filename.lower()
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp:
        content = await file.read()
        temp.write(content)
        temp_path = temp.name
        
    try:
        text = ""
        if filename.endswith(".pdf"):
            reader = PdfReader(temp_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif filename.endswith(".txt"):
            with open(temp_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file type")
            
        return clean_text(text)
    finally:
        os.remove(temp_path)
