import copy
import csv
import io
import json
import math
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update, delete, or_, and_

from app.database import get_db
from app.models.models import Question, Collection
from app.schemas.schemas import (
    QuestionCreate, QuestionUpdate, QuestionResponse, QuestionListResponse,
    BulkActionRequest, BulkTagRequest, ImportRequest, QuestionStats, CollectionResponse,
)

router = APIRouter(prefix="/questions", tags=["Questions"])


def _question_to_response(q: Question, collection_name: str | None = None) -> QuestionResponse:
    return QuestionResponse(
        id=q.id,
        question=q.question,
        question_latex=q.question_latex,
        answer=q.answer,
        answer_latex=q.answer_latex,
        steps=q.steps or [],
        topic=q.topic,
        difficulty=q.difficulty,
        tags=q.tags or [],
        alternative_methods=q.alternative_methods or [],
        formulas_used=q.formulas_used or [],
        ai_confidence=q.ai_confidence,
        graph_data=q.graph_data,
        is_favorite=q.is_favorite,
        is_pinned=q.is_pinned,
        is_archived=q.is_archived,
        view_count=q.view_count,
        notes=q.notes,
        collection_id=q.collection_id,
        collection_name=collection_name,
        created_at=q.created_at,
        updated_at=q.updated_at,
    )


@router.get("", response_model=QuestionListResponse)
async def list_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str = Query(None),
    topic: str = Query(None),
    difficulty: str = Query(None),
    tags: str = Query(None),
    sort_by: str = Query("newest"),
    collection_id: str = Query(None),
    favorite: bool = Query(None),
    pinned: bool = Query(None),
    archived: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    query = select(Question).outerjoin(Collection, Question.collection_id == Collection.id)

    filters = [Question.is_archived == archived]

    if search:
        filters.append(
            or_(
                Question.question.ilike(f"%{search}%"),
                Question.answer.ilike(f"%{search}%"),
                Question.topic.ilike(f"%{search}%"),
            )
        )
    if topic:
        filters.append(Question.topic == topic)
    if difficulty:
        filters.append(Question.difficulty == difficulty)
    if tags:
        tag_list = [t.strip() for t in tags.split(",")]
        for tag in tag_list:
            filters.append(Question.tags.contains([tag]))
    if collection_id:
        filters.append(Question.collection_id == collection_id)
    if favorite is not None:
        filters.append(Question.is_favorite == favorite)
    if pinned is not None:
        filters.append(Question.is_pinned == pinned)

    query = query.where(and_(*filters))

    # Sorting
    sort_map = {
        "newest": Question.created_at.desc(),
        "oldest": Question.created_at.asc(),
        "alphabetical": Question.question.asc(),
        "most_viewed": Question.view_count.desc(),
    }
    query = query.order_by(Question.is_pinned.desc(), sort_map.get(sort_by, Question.created_at.desc()))

    # Count
    count_query = select(func.count()).select_from(Question).where(and_(*filters))
    total = (await db.execute(count_query)).scalar() or 0

    # Paginate
    offset = (page - 1) * page_size
    result = await db.execute(query.offset(offset).limit(page_size))
    questions = result.scalars().all()

    items = []
    for q in questions:
        col_name = q.collection.name if q.collection else None
        items.append(_question_to_response(q, col_name))

    return QuestionListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=math.ceil(total / page_size) if total > 0 else 0,
    )


@router.get("/stats", response_model=QuestionStats)
async def get_stats(db: AsyncSession = Depends(get_db)):
    # Total count
    total = (await db.execute(select(func.count()).select_from(Question))).scalar() or 0

    # Per topic
    topic_query = await db.execute(
        select(Question.topic, func.count()).where(Question.topic != None).group_by(Question.topic).order_by(func.count().desc())
    )
    per_topic = {row[0]: row[1] for row in topic_query.all()}

    # Difficulty distribution
    diff_query = await db.execute(
        select(Question.difficulty, func.count()).where(Question.difficulty != None).group_by(Question.difficulty)
    )
    difficulty_dist = {row[0]: row[1] for row in diff_query.all()}

    # Recently added
    recent_added = await db.execute(
        select(Question).order_by(Question.created_at.desc()).limit(5)
    )
    recently_added = [_question_to_response(q) for q in recent_added.scalars().all()]

    # Recently viewed
    recent_viewed = await db.execute(
        select(Question).order_by(Question.view_count.desc()).limit(5)
    )
    recently_viewed = [_question_to_response(q) for q in recent_viewed.scalars().all()]

    # Most solved topics
    most_solved = sorted(per_topic.items(), key=lambda x: x[1], reverse=True)[:10]

    return QuestionStats(
        total_questions=total,
        questions_per_topic=per_topic,
        difficulty_distribution=difficulty_dist,
        recently_added=recently_added,
        recently_viewed=recently_viewed,
        most_solved_topics=most_solved,
    )


@router.get("/{question_id}", response_model=QuestionResponse)
async def get_question(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")

    q.view_count += 1
    col_name = q.collection.name if q.collection else None
    return _question_to_response(q, col_name)


@router.post("", response_model=QuestionResponse, status_code=201)
async def create_question(data: QuestionCreate, db: AsyncSession = Depends(get_db)):
    q = Question(
        question=data.question,
        question_latex=data.question_latex,
        answer=data.answer,
        answer_latex=data.answer_latex,
        steps=[s.model_dump() for s in data.steps],
        topic=data.topic,
        difficulty=data.difficulty,
        tags=data.tags,
        alternative_methods=[m.model_dump() for m in data.alternative_methods],
        formulas_used=[f.model_dump() for f in data.formulas_used],
        ai_confidence=data.ai_confidence,
        graph_data=data.graph_data.model_dump() if data.graph_data else None,
        notes=data.notes,
        collection_id=data.collection_id,
    )
    db.add(q)
    await db.flush()
    await db.refresh(q)
    return _question_to_response(q)


@router.put("/{question_id}", response_model=QuestionResponse)
async def update_question(question_id: str, data: QuestionUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")

    update_data = data.model_dump(exclude_unset=True)

    if "steps" in update_data and update_data["steps"] is not None:
        update_data["steps"] = [s if isinstance(s, dict) else s.model_dump() for s in update_data["steps"]]
    if "alternative_methods" in update_data and update_data["alternative_methods"] is not None:
        update_data["alternative_methods"] = [m if isinstance(m, dict) else m.model_dump() for m in update_data["alternative_methods"]]
    if "formulas_used" in update_data and update_data["formulas_used"] is not None:
        update_data["formulas_used"] = [f if isinstance(f, dict) else f.model_dump() for f in update_data["formulas_used"]]
    if "graph_data" in update_data and update_data["graph_data"] is not None:
        if isinstance(update_data["graph_data"], dict):
            pass
        else:
            update_data["graph_data"] = update_data["graph_data"].model_dump()

    for key, value in update_data.items():
        setattr(q, key, value)

    q.updated_at = datetime.utcnow()
    await db.flush()
    await db.refresh(q)
    return _question_to_response(q)


@router.delete("/{question_id}")
async def delete_question(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    await db.delete(q)
    return {"message": "Question deleted"}


@router.post("/{question_id}/duplicate", response_model=QuestionResponse, status_code=201)
async def duplicate_question(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")

    new_q = Question(
        question=f"{q.question} (copy)",
        question_latex=q.question_latex,
        answer=q.answer,
        answer_latex=q.answer_latex,
        steps=copy.deepcopy(q.steps),
        topic=q.topic,
        difficulty=q.difficulty,
        tags=copy.deepcopy(q.tags),
        alternative_methods=copy.deepcopy(q.alternative_methods),
        formulas_used=copy.deepcopy(q.formulas_used),
        ai_confidence=q.ai_confidence,
        graph_data=copy.deepcopy(q.graph_data),
        notes=q.notes,
        collection_id=q.collection_id,
    )
    db.add(new_q)
    await db.flush()
    await db.refresh(new_q)
    return _question_to_response(new_q)


@router.post("/{question_id}/favorite")
async def toggle_favorite(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    q.is_favorite = not q.is_favorite
    return {"is_favorite": q.is_favorite}


@router.post("/{question_id}/pin")
async def toggle_pin(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    q.is_pinned = not q.is_pinned
    return {"is_pinned": q.is_pinned}


@router.post("/{question_id}/archive")
async def toggle_archive(question_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")
    q.is_archived = not q.is_archived
    return {"is_archived": q.is_archived}


@router.post("/bulk/delete")
async def bulk_delete(data: BulkActionRequest, db: AsyncSession = Depends(get_db)):
    await db.execute(delete(Question).where(Question.id.in_(data.question_ids)))
    return {"message": f"Deleted {len(data.question_ids)} questions"}


@router.post("/bulk/archive")
async def bulk_archive(data: BulkActionRequest, db: AsyncSession = Depends(get_db)):
    await db.execute(
        update(Question).where(Question.id.in_(data.question_ids)).values(is_archived=True)
    )
    return {"message": f"Archived {len(data.question_ids)} questions"}


@router.post("/bulk/tag")
async def bulk_tag(data: BulkTagRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.id.in_(data.question_ids)))
    questions = result.scalars().all()
    for q in questions:
        existing = list(q.tags or [])
        for tag in data.tags:
            if tag not in existing:
                existing.append(tag)
        q.tags = existing
    return {"message": f"Tagged {len(data.question_ids)} questions"}


@router.post("/import")
async def import_questions(data: ImportRequest, db: AsyncSession = Depends(get_db)):
    imported = 0
    for q_data in data.questions:
        q = Question(
            question=q_data.question,
            question_latex=q_data.question_latex,
            answer=q_data.answer,
            answer_latex=q_data.answer_latex,
            steps=[s.model_dump() for s in q_data.steps],
            topic=q_data.topic,
            difficulty=q_data.difficulty,
            tags=q_data.tags,
            alternative_methods=[m.model_dump() for m in q_data.alternative_methods],
            formulas_used=[f.model_dump() for f in q_data.formulas_used],
            ai_confidence=q_data.ai_confidence,
            graph_data=q_data.graph_data.model_dump() if q_data.graph_data else None,
            notes=q_data.notes,
            collection_id=data.collection_id or q_data.collection_id,
        )
        db.add(q)
        imported += 1
    return {"message": f"Imported {imported} questions"}


@router.post("/import/file")
async def import_file(
    file: UploadFile = File(...),
    collection_id: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    content = await file.read()
    imported = 0

    if file.filename.endswith(".json"):
        data = json.loads(content)
        questions_data = data if isinstance(data, list) else data.get("questions", [])
        for item in questions_data:
            q = Question(
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                topic=item.get("topic"),
                difficulty=item.get("difficulty"),
                tags=item.get("tags", []),
                steps=item.get("steps", []),
                notes=item.get("notes"),
                collection_id=collection_id or item.get("collection_id"),
            )
            db.add(q)
            imported += 1
    elif file.filename.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(content.decode()))
        for row in reader:
            q = Question(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                topic=row.get("topic"),
                difficulty=row.get("difficulty"),
                tags=row.get("tags", "").split(",") if row.get("tags") else [],
                collection_id=collection_id,
            )
            db.add(q)
            imported += 1

    return {"message": f"Imported {imported} questions"}


@router.get("/export/format")
async def export_questions(
    format: str = Query("json"),
    question_ids: str = Query(None),
    db: AsyncSession = Depends(get_db),
):
    query = select(Question)
    if question_ids:
        ids = question_ids.split(",")
        query = query.where(Question.id.in_(ids))

    result = await db.execute(query)
    questions = result.scalars().all()
    items = [_question_to_response(q).model_dump(mode="json") for q in questions]

    if format == "csv":
        output = io.StringIO()
        if items:
            fieldnames = ["question", "answer", "topic", "difficulty", "tags"]
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in items:
                item["tags"] = ",".join(item.get("tags", []))
                writer.writerow(item)
        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=questions.csv"},
        )

    response_data = json.dumps(items, default=str)
    return StreamingResponse(
        io.BytesIO(response_data.encode()),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=questions.json"},
    )
