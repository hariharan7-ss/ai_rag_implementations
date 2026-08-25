import os
from groq import Groq

try:
    client = Groq()
except Exception:
    client = None

def generate_response(prompt: str, model: str = "openai/gpt-oss-20b") -> str:
    if not client:
        return "Groq client is not initialized. Please set GROQ_API_KEY environment variable."
        
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content
