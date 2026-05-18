import numpy as np
import logging
from PIL import Image
import cv2
from typing import List, Optional, Tuple

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

logger = logging.getLogger(__name__)

class FaceEngine:
    def __init__(self, use_gpu: bool = False, det_size: int = 640):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("insightface is not installed. Please install it to use Face Recognition.")
        
        self.use_gpu = use_gpu
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        det_size = max(320, min(int(det_size or 640), 1280))
        
        logger.info("Loading InsightFace model with detector size %dx%d...", det_size, det_size)
        # Initialize FaceAnalysis (this downloads models if not present, typically buffalo_l)
        # We specify name='buffalo_l' which includes retinaface (detection) and arcface (recognition)
        self.app = FaceAnalysis(name='buffalo_l', providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_size=(det_size, det_size))
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

    def get_embedding_for_box(
        self,
        image: Image.Image,
        target_box: Tuple[float, float, float, float],
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Extract the embedding for the detected face that best matches a target box.
        target_box uses source-image pixel coordinates: x1, y1, x2, y2.
        """
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        faces = self.app.get(img_cv)

        if not faces:
            return None

        tx1, ty1, tx2, ty2 = target_box
        target_w = max(1.0, tx2 - tx1)
        target_h = max(1.0, ty2 - ty1)
        target_cx = (tx1 + tx2) / 2.0
        target_cy = (ty1 + ty2) / 2.0
        target_area = target_w * target_h

        def score(face) -> float:
            fx1, fy1, fx2, fy2 = [float(v) for v in face.bbox.tolist()]
            inter_x1 = max(tx1, fx1)
            inter_y1 = max(ty1, fy1)
            inter_x2 = min(tx2, fx2)
            inter_y2 = min(ty2, fy2)
            inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
            face_area = max(1.0, fx2 - fx1) * max(1.0, fy2 - fy1)
            union = max(1.0, target_area + face_area - inter_area)
            iou = inter_area / union

            face_cx = (fx1 + fx2) / 2.0
            face_cy = (fy1 + fy2) / 2.0
            norm_dx = (face_cx - target_cx) / target_w
            norm_dy = (face_cy - target_cy) / target_h
            center_score = 1.0 / (1.0 + norm_dx * norm_dx + norm_dy * norm_dy)
            return iou * 2.0 + center_score

        best_face = max(faces, key=score)
        x1, y1, x2, y2 = [float(v) for v in best_face.bbox.tolist()]
        confidence = float(getattr(best_face, "det_score", 0.0) or 0.0)
        return best_face.normed_embedding, {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "confidence": confidence,
        }

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
