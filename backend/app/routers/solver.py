from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.schemas import SolveRequest, SolveResponse, VerifyRequest, VerifyResponse
from app.services.solver import solve_calculus, verify_solution
from app.models.models import Question

router = APIRouter(prefix="/solve", tags=["Solver"])


@router.post("", response_model=SolveResponse)
async def solve_problem(request: SolveRequest, db: AsyncSession = Depends(get_db)):
    response = solve_calculus(request)

    question = Question(
        question=response.question,
        question_latex=response.question_latex,
        answer=response.answer,
        answer_latex=response.answer_latex,
        steps=[s.model_dump() for s in response.steps],
        topic=response.topic,
        difficulty=response.difficulty,
        alternative_methods=[m.model_dump() for m in response.alternative_methods],
        formulas_used=[f.model_dump() for f in response.formulas_used],
        ai_confidence=response.ai_confidence,
        graph_data=response.graph_data.model_dump() if response.graph_data else None,
    )
    db.add(question)

    return response


@router.post("/verify", response_model=VerifyResponse)
async def verify_problem(request: VerifyRequest):
    return verify_solution(request)
