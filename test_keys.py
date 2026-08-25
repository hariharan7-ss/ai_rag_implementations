import os
from dotenv import load_dotenv
import traceback

load_dotenv(override=True)

print("--- Testing Groq ---")
try:
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    groq_client.models.list()
    print("Groq API Key is VALID!")
except Exception as e:
    print(f"Groq failed: {e}")

print("\n--- Testing Pinecone ---")
try:
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pc.list_indexes()
    print("Pinecone API Key is VALID!")
except Exception as e:
    print(f"Pinecone failed: {e}")
