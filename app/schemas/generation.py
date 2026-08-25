from pydantic import BaseModel
from typing import List

class QuestionGenerationRequest(BaseModel):
    query: str

class QuestionGenerationResponse(BaseModel):
    questions: List[str]
