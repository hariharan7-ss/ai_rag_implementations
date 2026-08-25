import re

def clean_text(text: str) -> str:
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
