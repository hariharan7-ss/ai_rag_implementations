RAG_PROMPT_TEMPLATE = """
You are a helpful assistant answering questions based on the provided context.
If the answer is not contained in the context, say "I cannot answer this based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

QUESTION_GENERATION_PROMPT = """
You are a helpful assistant. Based on the following retrieved document chunks, generate 4 to 5 highly relevant questions that can be answered using *only* this context.
Output the questions as a list, one per line. Do not include answers, numbering is fine, but no extra conversational text.

Context:
{context}

Questions:
"""
