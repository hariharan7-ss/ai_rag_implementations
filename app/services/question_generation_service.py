from typing import Dict, Any
from app.services.retrieval_service import retrieval_service
from app.generation.llm import generate_response
from app.generation.prompts import QUESTION_GENERATION_PROMPT

class QuestionGenerationService:
    async def generate(self, query: str) -> Dict[str, Any]:
        retrieved = retrieval_service.retrieve(query, initial_k=5, final_k=3)
        
        context_parts = []
        for r in retrieved:
            payload = r.get("payload", {})
            text = payload.get("text", "")
            context_parts.append(text)
            
        context = "\n\n---\n\n".join(context_parts)
        prompt = QUESTION_GENERATION_PROMPT.format(context=context)
        
        response_text = generate_response(prompt)
        questions = [q.strip() for q in response_text.split("\n") if q.strip()]
        
        cleaned_questions = []
        for q in questions:
            if q[0].isdigit() and len(q) > 2 and (q[1] == '.' or q[1] == ')'):
                cleaned_questions.append(q[2:].strip())
            elif q.startswith("- ") or q.startswith("* "):
                cleaned_questions.append(q[2:].strip())
            else:
                cleaned_questions.append(q)
        
        return {
            "questions": cleaned_questions
        }
