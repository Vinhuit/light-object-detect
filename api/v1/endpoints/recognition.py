from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import io
import base64
from PIL import Image
import time
import logging

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
        _face_engine = FaceEngine(use_gpu=False)
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
    crop_jpeg_base64: str

class FaceDetectResponse(BaseModel):
    success: bool
    face_count: int
    faces: List[DetectedFace]
    process_time_ms: int

@router.post("/faces/train", response_model=TrainResponse)
async def train_face(
    name: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Train a new face for the given name.
    """
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
        pad_x = face_w * 0.30
        pad_y = face_h * 0.30
        crop_box = (
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x)),
            min(height, int(y2 + pad_y)),
        )
        crop = image.crop(crop_box)
        buf = io.BytesIO()
        crop.save(buf, format="JPEG", quality=90)
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

class FaceListResponse(BaseModel):
    success: bool
    faces: List[FaceListItem]

class DeleteFaceResponse(BaseModel):
    success: bool
    message: str

@router.get("/faces/list", response_model=FaceListResponse)
async def list_faces():
    """
    List all trained faces.
    """
    db = get_face_db()
    known_faces = db.get_all_faces()
    
    faces_list = []
    for face_id, name, _ in known_faces:
        faces_list.append(FaceListItem(id=face_id, name=name))
        
    return FaceListResponse(
        success=True,
        faces=faces_list
    )

@router.delete("/faces/{face_id}", response_model=DeleteFaceResponse)
async def delete_face(face_id: int):
    """
    Delete a trained face by ID.
    """
    db = get_face_db()
    # Check if face exists in cache
    face_exists = any(f[0] == face_id for f in db.get_all_faces())
    if not face_exists:
        raise HTTPException(status_code=404, detail=f"Face with ID {face_id} not found.")
        
    db.delete_face(face_id)
    return DeleteFaceResponse(
        success=True,
        message=f"Successfully deleted face with ID {face_id}."
    )
