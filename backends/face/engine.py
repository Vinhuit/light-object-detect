import numpy as np
import logging
from PIL import Image
import cv2
from typing import List, Optional

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self, use_gpu: bool = False):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface is not installed. Please install it to use Face Recognition.")
        
        self.use_gpu = use_gpu
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        
        logger.info("Loading InsightFace model...")
        # Initialize FaceAnalysis (this downloads models if not present, typically buffalo_l)
        # We specify name='buffalo_l' which includes retinaface (detection) and arcface (recognition)
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(640, 640))
        logger.info("InsightFace model loaded successfully.")

    def get_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Extract the face embedding from an image. 
        If multiple faces are found, it returns the largest/most prominent one.
        """
        # Convert PIL to CV2 BGR
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces and extract features
        faces = self.app.get(img_cv)
        
        if not faces:
            return None
            
        # Get the largest face by bounding box area
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Normed embedding is usually a 512-d array
        return largest_face.normed_embedding

    def detect_faces(self, image: Image.Image) -> List[dict]:
        """
        Detect faces and return bounding boxes plus detector confidence.
        Coordinates are returned in source-image pixels.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = self.app.get(img_cv)
        detected = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.tolist()
            confidence = float(getattr(face, "det_score", 0.0) or 0.0)
            detected.append({
                "bbox": {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                },
                "confidence": confidence,
            })

        detected.sort(
            key=lambda f: (f["bbox"]["x2"] - f["bbox"]["x1"]) * (f["bbox"]["y2"] - f["bbox"]["y1"]),
            reverse=True,
        )
        return detected

    def compare_embeddings(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        Values range from -1.0 to 1.0. Higher is more similar.
        """
        # Since embeddings are normed, dot product is cosine similarity
        sim = np.dot(emb1, emb2)
        return float(sim)
