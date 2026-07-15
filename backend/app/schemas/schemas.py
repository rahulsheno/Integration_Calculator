from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StepDetail(BaseModel):
    step_number: int
    description: str
    expression: Optional[str] = None
    expression_latex: Optional[str] = None
    result: Optional[str] = None
    result_latex: Optional[str] = None
    justification: Optional[str] = None


class AlternativeMethod(BaseModel):
    name: str
    steps: list[str] = []
    final_answer: Optional[str] = None
    final_answer_latex: Optional[str] = None


class FormulaUsed(BaseModel):
    name: str
    formula_latex: str
    description: Optional[str] = None


class GraphData(BaseModel):
    graph_type: str
    data: dict = {}
    plotly_json: Optional[dict] = None


class MathContext(BaseModel):
    normalized_expression: str
    latex: Optional[str] = None
    tokens: list[str] = []
    operation: str
    ast: Optional[str] = None
    variables: list[str] = []
    selected_method: Optional[str] = None
    previous_variables: list[str] = []
    previous_answer: Optional[str] = None


class SolveRequest(BaseModel):
    expression: str = Field(..., description="Mathematical expression in LaTeX or plain text")
    topic_hint: Optional[str] = Field(None, description="Optional topic hint")
    include_graph: bool = Field(True, description="Include graph if applicable")
    session_id: Optional[str] = Field(None, description="Optional conversation/session id")


class SolveResponse(BaseModel):
    question: str
    question_latex: Optional[str] = None
    extracted_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    topic: str
    answer: str
    answer_latex: Optional[str] = None
    steps: list[StepDetail] = []
    difficulty: Optional[str] = None
    alternative_methods: list[AlternativeMethod] = []
    formulas_used: list[FormulaUsed] = []
    verification: Optional[str] = None
    verification_latex: Optional[str] = None
    graph_data: Optional[GraphData] = None
    math_context: Optional[MathContext] = None
    ai_confidence: float = 0.0


class VerifyRequest(BaseModel):
    expression: str = Field(..., description="Original problem")
    student_solution: str = Field(..., description="Student's solution to verify")


class VerificationStep(BaseModel):
    step_number: int
    is_correct: bool
    student_step: str
    feedback: str
    correction: Optional[str] = None


class VerifyResponse(BaseModel):
    original_question: str
    student_solution: str
    is_correct: bool
    final_answer_correct: bool
    steps_analysis: list[VerificationStep] = []
    overall_feedback: str
    score: float = 0.0


class QuestionCreate(BaseModel):
    question: str
    question_latex: Optional[str] = None
    answer: str
    answer_latex: Optional[str] = None
    steps: list[StepDetail] = []
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    tags: list[str] = []
    alternative_methods: list[AlternativeMethod] = []
    formulas_used: list[FormulaUsed] = []
    ai_confidence: Optional[float] = None
    graph_data: Optional[GraphData] = None
    notes: Optional[str] = None
    collection_id: Optional[str] = None


class QuestionUpdate(BaseModel):
    question: Optional[str] = None
    question_latex: Optional[str] = None
    answer: Optional[str] = None
    answer_latex: Optional[str] = None
    steps: Optional[list[StepDetail]] = None
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    tags: Optional[list[str]] = None
    alternative_methods: Optional[list[AlternativeMethod]] = None
    formulas_used: Optional[list[FormulaUsed]] = None
    ai_confidence: Optional[float] = None
    graph_data: Optional[GraphData] = None
    notes: Optional[str] = None
    collection_id: Optional[str] = None
    is_favorite: Optional[bool] = None
    is_pinned: Optional[bool] = None


class QuestionResponse(BaseModel):
    id: str
    question: str
    question_latex: Optional[str] = None
    answer: str
    answer_latex: Optional[str] = None
    steps: list[StepDetail] = []
    topic: Optional[str] = None
    difficulty: Optional[str] = None
    tags: list[str] = []
    alternative_methods: list[AlternativeMethod] = []
    formulas_used: list[FormulaUsed] = []
    ai_confidence: Optional[float] = None
    graph_data: Optional[GraphData] = None
    is_favorite: bool = False
    is_pinned: bool = False
    is_archived: bool = False
    view_count: int = 0
    notes: Optional[str] = None
    collection_id: Optional[str] = None
    collection_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class QuestionListResponse(BaseModel):
    items: list[QuestionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    color_label: Optional[str] = None
    icon: Optional[str] = None


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    color_label: Optional[str] = None
    icon: Optional[str] = None
    sort_order: Optional[int] = None


class CollectionResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None
    color_label: Optional[str] = None
    icon: Optional[str] = None
    sort_order: int = 0
    question_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class BulkActionRequest(BaseModel):
    question_ids: list[str]


class BulkTagRequest(BaseModel):
    question_ids: list[str]
    tags: list[str]


class ImportRequest(BaseModel):
    collection_id: Optional[str] = None
    questions: list[QuestionCreate]


class QuestionStats(BaseModel):
    total_questions: int
    questions_per_topic: dict[str, int] = {}
    difficulty_distribution: dict[str, int] = {}
    recently_added: list[QuestionResponse] = []
    recently_viewed: list[QuestionResponse] = []
    most_solved_topics: list[tuple[str, int]] = []


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool

    model_config = {"from_attributes": True}
