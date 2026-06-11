import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SpeakerProfile:
    id: int
    name: str
    embedding: np.ndarray
    sample_count: int


class SpeakerEmbeddingStore:
    def __init__(self, db_path: str = "~/.whisper-diarization/speakers.db"):
        self._embedding_dim: Optional[int] = None
        if db_path == ":memory:":
            resolved = db_path
        else:
            resolved = str(Path(db_path).expanduser())
            Path(resolved).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(resolved)
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        self._load_embedding_dim()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def _create_table(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS speakers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                sample_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self._conn.commit()

    def _load_embedding_dim(self):
        row = self._conn.execute("SELECT embedding FROM speakers LIMIT 1").fetchone()
        if row is not None:
            self._embedding_dim = len(self._deserialize_embedding(row["embedding"]))

    def _validate_embedding(self, embedding: np.ndarray):
        if embedding.ndim != 1:
            raise ValueError(
                f"Embedding must be a 1D array, got {embedding.ndim}D"
            )
        if self._embedding_dim is not None and embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Embedding dimension must be {self._embedding_dim}, got {embedding.shape[0]}"
            )

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    def add_speaker(self, name: str, embedding: np.ndarray):
        self._validate_embedding(embedding)
        blob = self._serialize_embedding(embedding)
        try:
            self._conn.execute(
                "INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                (name, blob),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Speaker '{name}' already exists")
        if self._embedding_dim is None:
            self._embedding_dim = embedding.shape[0]

    def get_all_profiles(self) -> List[SpeakerProfile]:
        rows = self._conn.execute("SELECT * FROM speakers").fetchall()
        return [
            SpeakerProfile(
                id=row["id"],
                name=row["name"],
                embedding=self._deserialize_embedding(row["embedding"]),
                sample_count=row["sample_count"],
            )
            for row in rows
        ]

    def _get_profile_by_name(self, name: str) -> Optional[SpeakerProfile]:
        row = self._conn.execute(
            "SELECT * FROM speakers WHERE name = ?", (name,)
        ).fetchone()
        if row is None:
            return None
        return SpeakerProfile(
            id=row["id"],
            name=row["name"],
            embedding=self._deserialize_embedding(row["embedding"]),
            sample_count=row["sample_count"],
        )

    def find_best_match(
        self, embedding: np.ndarray, min_threshold: float = 0.6
    ) -> Tuple[Optional[str], float]:
        profiles = self.get_all_profiles()
        if not profiles:
            return None, 0.0
        best_name = None
        best_score = -1.0
        query_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        for profile in profiles:
            stored_norm = profile.embedding / (np.linalg.norm(profile.embedding) + 1e-8)
            score = float(np.dot(query_norm, stored_norm))
            if score > best_score:
                best_score = score
                best_name = profile.name
        if best_score < min_threshold:
            return None, best_score
        return best_name, best_score

    def update_embedding(self, name: str, new_embedding: np.ndarray):
        self._validate_embedding(new_embedding)
        profile = self._get_profile_by_name(name)
        if profile is None:
            raise ValueError(f"Speaker '{name}' not found")
        updated = (profile.embedding * profile.sample_count + new_embedding) / (
            profile.sample_count + 1
        )
        blob = self._serialize_embedding(updated)
        self._conn.execute(
            "UPDATE speakers SET embedding = ?, sample_count = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
            (blob, profile.sample_count + 1, name),
        )
        self._conn.commit()

    def rename_speaker(self, old_name: str, new_name: str):
        try:
            cursor = self._conn.execute(
                "UPDATE speakers SET name = ? WHERE name = ?", (new_name, old_name)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"Speaker '{old_name}' not found")
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Speaker '{new_name}' already exists")

    def delete_speaker(self, name: str):
        cursor = self._conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
        if cursor.rowcount == 0:
            raise ValueError(f"Speaker '{name}' not found")
        self._conn.commit()

    def list_speakers(self) -> List[dict]:
        rows = self._conn.execute(
            "SELECT name, sample_count, created_at, updated_at FROM speakers"
        ).fetchall()
        return [
            {
                "name": row["name"],
                "sample_count": row["sample_count"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def close(self):
        self._conn.close()
