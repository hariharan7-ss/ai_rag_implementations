from typing import Dict, Any, List

def create_metadata(document_id: str, source: str, source_type: str, title: str, chunks: List[str]) -> List[Dict[str, Any]]:
    metadata_list = []
    for i, chunk in enumerate(chunks):
        metadata_list.append({
            "document_id": document_id,
            "source": source,
            "source_type": source_type,
            "title": title,
            "chunk_id": f"{document_id}_chunk_{i}",
            "text": chunk
        })
    return metadata_list
