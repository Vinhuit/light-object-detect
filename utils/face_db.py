import sqlite3
import numpy as np
import os
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
        self.faces_cache = self._load_all_faces()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS known_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def _load_all_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, embedding FROM known_faces')
        rows = cursor.fetchall()
        conn.close()
        
        faces = []
        for row in rows:
            face_id, name, embedding_blob = row
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            faces.append((face_id, name, embedding))
        return faces

    def add_face(self, name: str, embedding: np.ndarray) -> int:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Ensure it's float32 for consistency
        embedding = embedding.astype(np.float32)
        cursor.execute(
            'INSERT INTO known_faces (name, embedding) VALUES (?, ?)',
            (name, embedding.tobytes())
        )
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update cache
        self.faces_cache.append((face_id, name, embedding))
        logger.info(f"Added face '{name}' to database with ID {face_id}")
        return face_id

    def delete_face(self, face_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM known_faces WHERE id = ?', (face_id,))
        conn.commit()
        conn.close()
        
        # Update cache
        self.faces_cache = [f for f in self.faces_cache if f[0] != face_id]

    def get_face_name(self, face_id: int) -> Optional[str]:
        for cached_id, name, _ in self.faces_cache:
            if cached_id == face_id:
                return name
        return None

    def delete_faces_by_name(self, name: str) -> List[int]:
        normalized = (name or "").strip().lower()
        if not normalized:
            return []

        deleted_ids = [
            face_id
            for face_id, face_name, _ in self.faces_cache
            if face_name.strip().lower() == normalized
        ]
        if not deleted_ids:
            return []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM known_faces WHERE lower(trim(name)) = ?', (normalized,))
        conn.commit()
        conn.close()

        self.faces_cache = [
            item
            for item in self.faces_cache
            if item[0] not in deleted_ids
        ]
        logger.info(f"Deleted {len(deleted_ids)} face sample(s) for '{name}'")
        return deleted_ids

    def get_all_faces(self) -> List[Tuple[int, str, np.ndarray]]:
        return self.faces_cache

    def get_all_face_summaries(self) -> List[Tuple[int, str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, created_at FROM known_faces ORDER BY created_at DESC, id DESC')
        rows = cursor.fetchall()
        conn.close()
        return [(int(face_id), name, created_at) for face_id, name, created_at in rows]
