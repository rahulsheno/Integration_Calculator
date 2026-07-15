import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.services.math_ocr import extract_math_expression_from_image, is_ocr_engine_available
from app.services.solver import solve_calculus
from app.schemas.schemas import SolveRequest, SolveResponse

router = APIRouter(prefix="/upload", tags=["Upload"])

settings = get_settings()


@router.post("", response_model=SolveResponse)
async def upload_image(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail="File too large")

    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    ext = os.path.splitext(file.filename or "image.png")[1]
    filename = f"{uuid.uuid4()}{ext}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)

    with open(filepath, "wb") as f:
        f.write(content)

    if not is_ocr_engine_available():
        return SolveResponse(
            question="OCR engine is not installed.",
            extracted_text="",
            ocr_confidence=0.0,
            topic="Unknown",
            answer="Math OCR is not installed. Install Pix2Text or pix2tex, then upload the image again.",
            difficulty="Unknown",
            ai_confidence=0.0,
        )

    extracted_text = ""
    raw_ocr_text = ""
    ocr_confidence = 0.0
    try:
        ocr_result = extract_math_expression_from_image(filepath)
        extracted_text = ocr_result.expression.strip()
        raw_ocr_text = ocr_result.raw_text.strip()
        ocr_confidence = ocr_result.confidence
    except Exception as exc:
        extracted_text = ""
        return SolveResponse(
            question="Image OCR failed.",
            extracted_text="",
            ocr_confidence=0.0,
            topic="Unknown",
            answer=f"Math OCR failed: {exc}",
            difficulty="Unknown",
            ai_confidence=0.0,
        )

    if not extracted_text:
        return SolveResponse(
            question="Image uploaded but no text could be extracted.",
            extracted_text=raw_ocr_text,
            ocr_confidence=0.0,
            topic="Unknown",
            answer="Unable to convert the image into a solvable expression. Check the extracted text above or type the expression manually.",
            difficulty="Unknown",
            ai_confidence=0.0,
        )

    request = SolveRequest(expression=extracted_text, session_id="default")
    response = solve_calculus(request)
    return response.model_copy(
        update={
            "extracted_text": extracted_text,
            "ocr_confidence": ocr_confidence,
        }
    )
