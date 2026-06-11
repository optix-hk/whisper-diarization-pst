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
        resolved = Path(db_path).expanduser()
        if str(resolved) != ":memory:":
            resolved.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(resolved))
        self._conn.row_factory = sqlite3.Row
        self._create_table()

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

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        return embedding.astype(np.float32).tobytes()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).copy()

    def add_speaker(self, name: str, embedding: np.ndarray):
        blob = self._serialize_embedding(embedding)
        try:
            self._conn.execute(
                "INSERT INTO speakers (name, embedding) VALUES (?, ?)",
                (name, blob),
            )
            self._conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Speaker '{name}' already exists")

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
        profiles = self.get_all_profiles()
        profile = next((p for p in profiles if p.name == name), None)
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
        cursor = self._conn.execute(
            "UPDATE speakers SET name = ? WHERE name = ?", (new_name, old_name)
        )
        if cursor.rowcount == 0:
            raise ValueError(f"Speaker '{old_name}' not found")
        self._conn.commit()

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
