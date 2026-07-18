import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, Boolean, Integer, Enum, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.database import Base


class Difficulty(str, enum.Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"
    EXPERT = "Expert"


class Topic(str, enum.Enum):
    LIMITS = "Limits"
    CONTINUITY = "Continuity"
    DIFFERENTIATION = "Differentiation"
    IMPLICIT_DIFFERENTIATION = "Implicit Differentiation"
    HIGHER_ORDER_DERIVATIVES = "Higher-Order Derivatives"
    PARTIAL_DERIVATIVES = "Partial Derivatives"
    CHAIN_RULE = "Chain Rule"
    PRODUCT_QUOTIENT_RULES = "Product & Quotient Rules"
    OPTIMIZATION = "Optimization"
    RELATED_RATES = "Related Rates"
    INDEFINITE_INTEGRALS = "Indefinite Integrals"
    DEFINITE_INTEGRALS = "Definite Integrals"
    INTEGRATION_BY_PARTS = "Integration by Parts"
    U_SUBSTITUTION = "U-Substitution"
    PARTIAL_FRACTIONS = "Partial Fractions"
    TRIGONOMETRIC_INTEGRALS = "Trigonometric Integrals"
    TRIGONOMETRIC_SUBSTITUTION = "Trigonometric Substitution"
    IMPROPER_INTEGRALS = "Improper Integrals"
    MULTIPLE_INTEGRALS = "Multiple Integrals"
    VECTOR_CALCULUS = "Vector Calculus"
    LINE_INTEGRALS = "Line Integrals"
    SURFACE_INTEGRALS = "Surface Integrals"
    GREENS_THEOREM = "Green's Theorem"
    STOKES_THEOREM = "Stokes' Theorem"
    DIVERGENCE_THEOREM = "Divergence Theorem"
    DIFFERENTIAL_EQUATIONS = "Differential Equations"
    TAYLOR_MACLAURIN_SERIES = "Taylor & Maclaurin Series"
    FOURIER_SERIES = "Fourier Series"
    LAPLACE_TRANSFORMS = "Laplace Transforms"
    POLAR_COORDINATES = "Polar Coordinates"
    PARAMETRIC_EQUATIONS = "Parametric Equations"


class Question(Base):
    __tablename__ = "questions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(Text, nullable=False)
    question_latex = Column(Text, nullable=True)
    answer = Column(Text, nullable=False)
    answer_latex = Column(Text, nullable=True)
    steps = Column(JSON, nullable=True, default=list)
    topic = Column(String(100), nullable=True)
    difficulty = Column(String(20), nullable=True)
    tags = Column(JSON, nullable=True, default=list)
    alternative_methods = Column(JSON, nullable=True, default=list)
    formulas_used = Column(JSON, nullable=True, default=list)
    ai_confidence = Column(Float, nullable=True)
    graph_data = Column(JSON, nullable=True)
    is_favorite = Column(Boolean, default=False)
    is_pinned = Column(Boolean, default=False)
    is_archived = Column(Boolean, default=False)
    view_count = Column(Integer, default=0)
    notes = Column(Text, nullable=True)
    collection_id = Column(String(36), ForeignKey("collections.id", ondelete="SET NULL"), nullable=True)
    # Hash of the normalized question text, used to detect and reuse
    # duplicate saves instead of storing the same solved question
    # repeatedly. Indexed (not unique) since duplicates can legitimately
    # live in different collections.
    content_hash = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    collection = relationship("Collection", back_populates="questions")


class Collection(Base):
    __tablename__ = "collections"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    parent_id = Column(String(36), ForeignKey("collections.id", ondelete="CASCADE"), nullable=True)
    color_label = Column(String(20), nullable=True)
    icon = Column(String(50), nullable=True)
    sort_order = Column(Integer, default=0)
    is_system = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    questions = relationship("Question", back_populates="collection")
    parent = relationship("Collection", remote_side=[id], back_populates="children")
    children = relationship("Collection", back_populates="parent", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
