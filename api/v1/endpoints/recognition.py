from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import io
import base64
from PIL import Image
import time
import logging
from pathlib import Path

from config import settings
from utils.image import validate_image, preprocess_image
from utils.face_db import FaceDB
try:
    from backends.face.engine import FaceEngine, INSIGHTFACE_AVAILABLE
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

router = APIRouter()

# Global instances (lazy loaded)
_face_engine = None
_face_db = None

def get_face_engine() -> "FaceEngine":
    global _face_engine
    if not INSIGHTFACE_AVAILABLE:
        raise HTTPException(status_code=500, detail="InsightFace is not installed.")
    if _face_engine is None:
        _face_engine = FaceEngine(use_gpu=False, det_size=settings.FACE_DET_SIZE)
    return _face_engine

def get_face_db() -> FaceDB:
    global _face_db
    if _face_db is None:
        _face_db = FaceDB(settings.FACE_DB_PATH)
    return _face_db

class TrainResponse(BaseModel):
    success: bool
    face_id: int
    name: str
    message: str

class RecognizeResponse(BaseModel):
    success: bool
    name: str
    confidence: float
    process_time_ms: int

class FaceBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class DetectedFace(BaseModel):
    index: int
    bbox: FaceBox
    confidence: float
    crop_width: int
    crop_height: int
    crop_jpeg_base64: str

class FaceDetectResponse(BaseModel):
    success: bool
    face_count: int
    faces: List[DetectedFace]
    process_time_ms: int

def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))

def _square_crop_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    face_w = max(1.0, x2 - x1)
    face_h = max(1.0, y2 - y1)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(face_w, face_h) * (1.0 + 2.0 * max(0.0, padding_ratio))

    max_side = float(max(1, min(image_width, image_height)))
    side = min(side, max_side)

    side_i = max(1, int(round(side)))
    left = int(round(center_x - side_i / 2.0))
    top = int(round(center_y - side_i / 2.0))
    right = left + side_i
    bottom = top + side_i

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > image_width:
        left -= right - image_width
        right = image_width
    if bottom > image_height:
        top -= bottom - image_height
        bottom = image_height

    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    if right <= left:
        right = min(image_width, left + 1)
    if bottom <= top:
        bottom = min(image_height, top + 1)

    return left, top, right, bottom

def _resize_crop_for_review(crop: Image.Image) -> Image.Image:
    min_size = _clamp_int(settings.FACE_CROP_MIN_SIZE, 64, 2048)
    max_size = _clamp_int(settings.FACE_CROP_MAX_SIZE, min_size, 2048)
    longest = max(crop.size)

    if longest < min_size:
        target = min_size
    elif longest > max_size:
        target = max_size
    else:
        return crop

    scale = target / float(longest)
    new_size = (
        max(1, int(round(crop.width * scale))),
        max(1, int(round(crop.height * scale))),
    )
    return crop.resize(new_size, Image.Resampling.LANCZOS)

def _face_sample_dir() -> Path:
    db_path = Path(settings.FACE_DB_PATH)
    sample_dir = db_path.parent / "face_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir

def _face_sample_path(face_id: int) -> Path:
    return _face_sample_dir() / f"{face_id}.jpg"

def _save_face_sample(face_id: int, image: Image.Image) -> None:
    sample = image.copy()
    if sample.mode != "RGB":
        sample = sample.convert("RGB")

    max_size = _clamp_int(settings.FACE_CROP_MAX_SIZE, 160, 2048)
    sample.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    quality = _clamp_int(settings.FACE_CROP_JPEG_QUALITY, 60, 100)
    sample.save(_face_sample_path(face_id), format="JPEG", quality=quality, optimize=True)

@router.post("/faces/train", response_model=TrainResponse)
async def train_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Train a new face for the given name.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        # Avoid preprocessing that changes aspect ratio/size too much for face rec
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    engine = get_face_engine()
    db = get_face_db()

    embedding = engine.get_embedding(image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    face_id = db.add_face(name, embedding)
    try:
        _save_face_sample(face_id, image)
    except Exception as e:
        logger.warning(f"Failed to save face sample for ID {face_id}: {str(e)}")

    return TrainResponse(
        success=True,
        face_id=face_id,
        name=name,
        message=f"Successfully trained face for '{name}'."
    )

@router.post("/faces/train-box", response_model=TrainResponse)
async def train_face_from_box(
    name: str = Form(...),
    bbox_x: float = Form(...),
    bbox_y: float = Form(...),
    bbox_w: float = Form(...),
    bbox_h: float = Form(...),
    file: UploadFile = File(...),
):
    """
    Train a face from a full image, selecting the detected face nearest a
    normalized target box. This is useful when a saved review crop is too
    degraded for a second detector pass.
    """
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    width, height = image.size
    x1 = _clamp_int(int(round(bbox_x * width)), 0, width)
    y1 = _clamp_int(int(round(bbox_y * height)), 0, height)
    x2 = _clamp_int(int(round((bbox_x + bbox_w) * width)), 0, width)
    y2 = _clamp_int(int(round((bbox_y + bbox_h) * height)), 0, height)
    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=400, detail="Invalid face box.")

    engine = get_face_engine()
    db = get_face_db()

    selected = engine.get_embedding_for_box(image, (x1, y1, x2, y2))
    if selected is None:
        raise HTTPException(status_code=400, detail="No face detected in the image.")

    embedding, selected_box = selected
    face_id = db.add_face(name, embedding)

    try:
        crop_box = _square_crop_box(
            selected_box["x1"],
            selected_box["y1"],
            selected_box["x2"],
            selected_box["y2"],
            width,
            height,
            settings.FACE_CROP_PADDING_RATIO,
        )
        sample = image.crop(crop_box)
        sample = _resize_crop_for_review(sample)
        _save_face_sample(face_id, sample)
    except Exception as e:
        logger.warning(f"Failed to save face sample for ID {face_id}: {str(e)}")

    return TrainResponse(
        success=True,
        face_id=face_id,
        name=name,
        message=f"Successfully trained face for '{name}'."
    )

@router.post("/faces/recognize", response_model=RecognizeResponse)
async def recognize_face(
    file: UploadFile = File(...)
):
    """
    Recognize a face in the uploaded image.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    start_time = time.time()
    
    engine = get_face_engine()
    db = get_face_db()

    # Extract embedding
    embedding = engine.get_embedding(image)
    if embedding is None:
        return RecognizeResponse(
            success=False,
            name="Unknown",
            confidence=0.0,
            process_time_ms=int((time.time() - start_time) * 1000)
        )

    # Compare against known faces
    best_match_name = "Unknown"
    highest_score = 0.0

    known_faces = db.get_all_faces()
    for face_id, name, known_emb in known_faces:
        score = engine.compare_embeddings(embedding, known_emb)
        if score > highest_score:
            highest_score = score
            best_match_name = name

    # Check threshold
    if highest_score < settings.FACE_MATCH_THRESHOLD:
        best_match_name = "Unknown"

    process_time = time.time() - start_time

    return RecognizeResponse(
        success=best_match_name != "Unknown",
        name=best_match_name,
        confidence=highest_score,
        process_time_ms=int(process_time * 1000)
    )

@router.post("/faces/detect", response_model=FaceDetectResponse)
async def detect_faces(
    file: UploadFile = File(...)
):
    """
    Detect faces and return normalized boxes plus JPEG crops.
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        validate_image(image, file.filename)
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        logger.error(f"Image validation failed: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

    start_time = time.time()
    engine = get_face_engine()
    detected = engine.detect_faces(image)
    width, height = image.size
    faces = []

    for index, item in enumerate(detected):
        bbox = item["bbox"]
        x1 = max(0.0, min(float(width), bbox["x1"]))
        y1 = max(0.0, min(float(height), bbox["y1"]))
        x2 = max(0.0, min(float(width), bbox["x2"]))
        y2 = max(0.0, min(float(height), bbox["y2"]))
        if x2 <= x1 or y2 <= y1:
            continue

        face_w = x2 - x1
        face_h = y2 - y1
        crop_box = _square_crop_box(
            x1,
            y1,
            x2,
            y2,
            width,
            height,
            settings.FACE_CROP_PADDING_RATIO,
        )
        crop = image.crop(crop_box)
        crop = _resize_crop_for_review(crop)
        buf = io.BytesIO()
        quality = _clamp_int(settings.FACE_CROP_JPEG_QUALITY, 60, 100)
        crop.save(buf, format="JPEG", quality=quality, optimize=True)
        crop_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        faces.append(DetectedFace(
            index=len(faces),
            bbox=FaceBox(
                x=x1 / width,
                y=y1 / height,
                width=face_w / width,
                height=face_h / height,
            ),
            confidence=float(item.get("confidence", 0.0)),
            crop_width=crop.width,
            crop_height=crop.height,
            crop_jpeg_base64=crop_b64,
        ))

    process_time = time.time() - start_time
    return FaceDetectResponse(
        success=True,
        face_count=len(faces),
        faces=faces,
        process_time_ms=int(process_time * 1000),
    )

class FaceListItem(BaseModel):
    id: int
    name: str
    sample_count: int = 1
    thumbnail_url: Optional[str] = None

class FaceListResponse(BaseModel):
    success: bool
    faces: List[FaceListItem]

class DeleteFaceResponse(BaseModel):
    success: bool
    message: str

@router.get("/faces/list", response_model=FaceListResponse)
async def list_faces():
    """
    List trained people, grouping multiple samples with the same name.
    """
    db = get_face_db()
    known_faces = db.get_all_faces()
    
    grouped = {}
    for face_id, name, _ in known_faces:
        display_name = (name or "").strip() or "Unknown"
        key = display_name.lower()
        group = grouped.setdefault(key, {
            "id": face_id,
            "name": display_name,
            "sample_count": 0,
            "thumbnail_id": None,
        })
        group["sample_count"] += 1
        group["id"] = min(group["id"], face_id)
        if _face_sample_path(face_id).exists():
            group["thumbnail_id"] = face_id

    faces_list = []
    for group in sorted(grouped.values(), key=lambda item: item["name"].lower()):
        thumbnail_url = None
        if group["thumbnail_id"] is not None:
            thumbnail_url = f"/api/v1/faces/{group['thumbnail_id']}/image"
        faces_list.append(FaceListItem(
            id=group["id"],
            name=group["name"],
            sample_count=group["sample_count"],
            thumbnail_url=thumbnail_url,
        ))
        
    return FaceListResponse(
        success=True,
        faces=faces_list
    )

@router.get("/faces/{face_id}/image")
async def get_face_image(face_id: int):
    """
    Return the stored training sample image for a face profile.
    """
    image_path = _face_sample_path(face_id)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image for face ID {face_id} not found.")
    return FileResponse(
        path=str(image_path),
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"}
    )

@router.delete("/faces/{face_id}", response_model=DeleteFaceResponse)
async def delete_face(face_id: int):
    """
    Delete a visible face profile by ID. If multiple samples have the same
    name, all of those samples are removed together.
    """
    db = get_face_db()
    face_name = db.get_face_name(face_id)
    if face_name is None:
        raise HTTPException(status_code=404, detail=f"Face with ID {face_id} not found.")
        
    deleted_ids = db.delete_faces_by_name(face_name)
    for deleted_id in deleted_ids:
        image_path = _face_sample_path(deleted_id)
        try:
            if image_path.exists():
                image_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete face sample for ID {deleted_id}: {str(e)}")

    return DeleteFaceResponse(
        success=True,
        message=f"Successfully deleted {len(deleted_ids)} sample(s) for '{face_name}'."
    )
