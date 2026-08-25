from typing import List

def recursive_chunk(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    # Simple sliding window chunker
    chunks = []
    start = 0
    text_length = len(text)
    
    if text_length == 0:
        return []
        
    while start < text_length:
        end = min(start + chunk_size, text_length)
        # Try to find a space to break at instead of breaking mid-word
        if end < text_length:
            last_space = text.rfind(' ', start, end)
            if last_space != -1 and last_space > start + chunk_size // 2:
                end = last_space
                
        chunks.append(text[start:end].strip())
        
        if end == text_length:
            break
            
        start = end - chunk_overlap
        
    return chunks
