from fastapi import APIRouter, HTTPException
from app.schemas.generation import QuestionGenerationRequest, QuestionGenerationResponse
from app.services.question_generation_service import QuestionGenerationService

router = APIRouter()
question_gen_service = QuestionGenerationService()

@router.post("/generate/questions", response_model=QuestionGenerationResponse)
async def generate_questions(request: QuestionGenerationRequest):
    try:
        result = await question_gen_service.generate(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
