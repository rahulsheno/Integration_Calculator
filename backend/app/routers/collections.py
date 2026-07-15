from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update

from app.database import get_db
from app.models.models import Collection, Question
from app.schemas.schemas import CollectionCreate, CollectionUpdate, CollectionResponse

router = APIRouter(prefix="/collections", tags=["Collections"])


def _collection_to_response(c: Collection, count: int = 0) -> CollectionResponse:
    return CollectionResponse(
        id=c.id,
        name=c.name,
        description=c.description,
        parent_id=c.parent_id,
        color_label=c.color_label,
        icon=c.icon,
        sort_order=c.sort_order,
        question_count=count,
        created_at=c.created_at,
        updated_at=c.updated_at,
    )


@router.get("", response_model=list[CollectionResponse])
async def list_collections(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Collection).order_by(Collection.sort_order, Collection.name)
    )
    collections = result.scalars().all()

    response = []
    for c in collections:
        count_result = await db.execute(
            select(func.count()).select_from(Question).where(
                Question.collection_id == c.id, Question.is_archived == False
            )
        )
        count = count_result.scalar() or 0
        response.append(_collection_to_response(c, count))

    return response


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Collection).where(Collection.id == collection_id))
    c = result.scalar_one_or_none()
    if c is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    count_result = await db.execute(
        select(func.count()).select_from(Question).where(Question.collection_id == c.id)
    )
    count = count_result.scalar() or 0
    return _collection_to_response(c, count)


@router.post("", response_model=CollectionResponse, status_code=201)
async def create_collection(data: CollectionCreate, db: AsyncSession = Depends(get_db)):
    c = Collection(
        name=data.name,
        description=data.description,
        parent_id=data.parent_id,
        color_label=data.color_label,
        icon=data.icon,
    )
    db.add(c)
    await db.flush()
    await db.refresh(c)
    return _collection_to_response(c, 0)


@router.put("/{collection_id}", response_model=CollectionResponse)
async def update_collection(collection_id: str, data: CollectionUpdate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Collection).where(Collection.id == collection_id))
    c = result.scalar_one_or_none()
    if c is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(c, key, value)
    await db.flush()
    await db.refresh(c)

    count_result = await db.execute(
        select(func.count()).select_from(Question).where(Question.collection_id == c.id)
    )
    return _collection_to_response(c, count_result.scalar() or 0)


@router.delete("/{collection_id}")
async def delete_collection(collection_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Collection).where(Collection.id == collection_id))
    c = result.scalar_one_or_none()
    if c is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    await db.execute(
        update(Question).where(Question.collection_id == collection_id).values(collection_id=None)
    )
    await db.delete(c)
    return {"message": "Collection deleted"}


@router.post("/{collection_id}/move/{question_id}")
async def move_question(collection_id: str, question_id: str, db: AsyncSession = Depends(get_db)):
    collection_result = await db.execute(select(Collection).where(Collection.id == collection_id))
    if collection_result.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    result = await db.execute(select(Question).where(Question.id == question_id))
    q = result.scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail="Question not found")

    q.collection_id = collection_id
    return {"message": "Question moved"}


@router.post("/reorder")
async def reorder_collections(data: list[dict], db: AsyncSession = Depends(get_db)):
    ids = [item["id"] for item in data]
    existing_result = await db.execute(select(Collection.id).where(Collection.id.in_(ids)))
    existing_ids = {row[0] for row in existing_result.all()}

    missing = [item["id"] for item in data if item["id"] not in existing_ids]
    if missing:
        raise HTTPException(status_code=404, detail=f"Collection(s) not found: {', '.join(missing)}")

    parent_ids = {item.get("parent_id") for item in data if item.get("parent_id")}
    if parent_ids:
        parent_result = await db.execute(select(Collection.id).where(Collection.id.in_(parent_ids)))
        found_parent_ids = {row[0] for row in parent_result.all()}
        missing_parents = parent_ids - found_parent_ids
        if missing_parents:
            raise HTTPException(status_code=404, detail=f"Parent collection(s) not found: {', '.join(missing_parents)}")

    for item in data:
        await db.execute(
            update(Collection).where(Collection.id == item["id"]).values(
                sort_order=item.get("sort_order", 0),
                parent_id=item.get("parent_id"),
            )
        )
    return {"message": "Collections reordered"}
