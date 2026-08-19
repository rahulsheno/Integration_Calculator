from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas.schemas import SolveRequest, SolveResponse, VerifyRequest, VerifyResponse, StepDetail, FormulaUsed
from app.services.series_integral_engine import try_series_integral
from app.services.solver import solve_calculus, verify_solution
from app.models.models import Question

router = APIRouter(prefix="/solve", tags=["Solver"])


@router.post("", response_model=SolveResponse)
async def solve_problem(request: SolveRequest, db: AsyncSession = Depends(get_db)):
    # FIXED: Intercept series integrals before the generic solver flattens them
    series_hit = try_series_integral(request.expression or "")
    if series_hit:
        response = SolveResponse(
            question=request.expression,
            question_latex=series_hit["question_latex"],
            topic="Series Integrals",
            answer=series_hit["answer"],
            answer_latex=series_hit["answer_latex"],
            steps=[
                StepDetail(step_number=i, description=d, expression_latex=e, justification=j, result_latex=r)
                for i, (d, e, j, r) in enumerate(series_hit["steps"], 1)
            ],
            difficulty="Medium",
            alternative_methods=[],
            formulas_used=[
                FormulaUsed(name="Geometric Series", formula_latex=r"\sum_{n=k}^{\infty} x^n = \frac{x^k}{1-x},\ |x|<1", description="Closed form of the geometric series."),
                FormulaUsed(name="Interchange of Summation and Integration", formula_latex=r"\int_a^b \sum_n f_n(x)\,dx = \sum_n \int_a^b f_n(x)\,dx", description="Term-by-term integration of a uniformly convergent series."),
            ],
            verification=series_hit["verification"],
            ai_confidence=0.95,
        )
    else:
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
