import os
from dotenv import load_dotenv

load_dotenv()
key = os.environ.get("GROQ_API_KEY", "")
print("Loaded Key Info:")
print(f"Starts with 'gsk_': {key.startswith('gsk_')}")
print(f"Length: {len(key)}")
has_quotes = '"' in key or "'" in key
print(f"Contains quotes: {has_quotes}")
print(f"Contains spaces: {' ' in key}")
