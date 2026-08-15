"""PostgreSQL/pgvector persistence for AdaFace embeddings."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import numpy as np

LOGGER = logging.getLogger(__name__)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class FaceVectorRepositoryError(RuntimeError):
    """Base database repository error."""


class DatabaseRuntimeUnavailable(FaceVectorRepositoryError):
    """The optional psycopg runtime is unavailable."""


class FaceVectorRepositoryOperationError(FaceVectorRepositoryError):
    """A PostgreSQL operation failed."""


@dataclass(frozen=True, slots=True)
class FaceVectorMatch:
    worker_id: str
    similarity: float
    record_id: int | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id cannot be empty")
        value = float(self.similarity)
        if not np.isfinite(value) or not -1.0 <= value <= 1.0:
            raise ValueError("similarity must be between -1 and 1")
        object.__setattr__(self, "similarity", value)


@dataclass(frozen=True, slots=True)
class StoredFaceVector:
    worker_id: str
    embedding: np.ndarray = field(repr=False, compare=False)
    record_id: int | None = None
    created_at: datetime | None = None
    sample_type: str | None = None
    pose: str | None = None
    quality: float | None = None
    image_sha256: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id cannot be empty")
        object.__setattr__(self, "embedding", _normalize_embedding(self.embedding))
        if self.quality is not None:
            value = float(self.quality)
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("quality must be between 0 and 1")
            object.__setattr__(self, "quality", value)


@runtime_checkable
class FaceVectorRepository(Protocol):
    def add(self, worker_id: str, embedding: np.ndarray | Sequence[float]) -> StoredFaceVector: ...

    def find_nearest(
        self,
        embedding: np.ndarray | Sequence[float],
        *,
        min_similarity: float | None = None,
    ) -> FaceVectorMatch | None: ...


ConnectionFactory = Callable[[], Any]


class PgVectorFaceRepository:
    """A psycopg 3 repository using pgvector cosine distance.

    Multiple rows may belong to one worker. Runtime recognition searches all
    rows, which allows one clear frontal template plus later left/right pose
    supplements without averaging away useful pose-specific information.
    """

    DEFAULT_TABLE = "t_worker_face_vector"
    VECTOR_DIMENSIONS = 512

    def __init__(
        self,
        dsn: str | None = None,
        *,
        connection_factory: ConnectionFactory | None = None,
        schema: str = "public",
        table: str = DEFAULT_TABLE,
    ) -> None:
        if connection_factory is None and not dsn:
            raise ValueError("dsn or connection_factory is required")
        for name, value in (("schema", schema), ("table", table)):
            if not _IDENTIFIER.fullmatch(value):
                raise ValueError(f"invalid PostgreSQL {name}: {value!r}")
        self.dsn = dsn
        self.connection_factory = connection_factory
        self.schema = schema
        self.table = table
        self._qualified_table = f'"{schema}"."{table}"'
        self._index_name = f'"{table}_vector_hnsw_idx"'

    @property
    def schema_statements(self) -> tuple[str, ...]:
        return (
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"""CREATE TABLE IF NOT EXISTS {self._qualified_table} (
                id BIGSERIAL PRIMARY KEY,
                worker_id TEXT NOT NULL,
                vector vector({self.VECTOR_DIMENSIONS}) NOT NULL,
                sample_type TEXT NOT NULL DEFAULT 'legacy',
                pose TEXT,
                quality REAL,
                image_sha256 TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            f"ALTER TABLE {self._qualified_table} ADD COLUMN IF NOT EXISTS sample_type TEXT NOT NULL DEFAULT 'legacy'",
            f"ALTER TABLE {self._qualified_table} ADD COLUMN IF NOT EXISTS pose TEXT",
            f"ALTER TABLE {self._qualified_table} ADD COLUMN IF NOT EXISTS quality REAL",
            f"ALTER TABLE {self._qualified_table} ADD COLUMN IF NOT EXISTS image_sha256 TEXT",
            f"""CREATE INDEX IF NOT EXISTS {self._index_name}
                ON {self._qualified_table}
                USING hnsw (vector vector_cosine_ops)""",
            f"""CREATE INDEX IF NOT EXISTS \"{self.table}_worker_id_idx\"
                ON {self._qualified_table} (worker_id)""",
            f"""CREATE INDEX IF NOT EXISTS \"{self.table}_worker_pose_idx\"
                ON {self._qualified_table} (worker_id, pose)""",
        )

    def ensure_schema(self) -> None:
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                for statement in self.schema_statements:
                    cursor.execute(statement)
        except Exception as exc:
            LOGGER.exception("Failed to initialize pgvector face schema")
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError(
                "failed to initialize face vector schema"
            ) from exc

    def add(
        self,
        worker_id: str,
        embedding: np.ndarray | Sequence[float],
        *,
        sample_type: str = "legacy",
        pose: str | None = None,
        quality: float | None = None,
        image_sha256: str | None = None,
    ) -> StoredFaceVector:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id cannot be blank")
        vector = _normalize_embedding(embedding)
        sample_type = str(sample_type or "legacy").strip()[:32]
        pose = str(pose).strip()[:32] if pose else None
        quality_value = None if quality is None else float(quality)
        if quality_value is not None and not 0.0 <= quality_value <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        digest = str(image_sha256).strip().lower() if image_sha256 else None
        if digest and not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("image_sha256 must be a 64-character hex digest")
        query = f"""INSERT INTO {self._qualified_table}
                    (worker_id, vector, sample_type, pose, quality, image_sha256)
                    VALUES (%s, %s::vector, %s, %s, %s, %s)
                    RETURNING id, created_at"""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(
                    query,
                    (worker_id, _vector_literal(vector), sample_type, pose, quality_value, digest),
                )
                row = cursor.fetchone()
        except Exception as exc:
            LOGGER.exception("Failed to store face vector for worker=%s", worker_id)
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError("failed to store face vector") from exc
        if not row:
            raise FaceVectorRepositoryOperationError("INSERT returned no record")
        return StoredFaceVector(
            worker_id=worker_id,
            embedding=vector,
            record_id=int(row[0]),
            created_at=row[1],
            sample_type=sample_type,
            pose=pose,
            quality=quality_value,
            image_sha256=digest,
        )

    def replace_worker(
        self,
        worker_id: str,
        embedding: np.ndarray | Sequence[float],
    ) -> StoredFaceVector:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id cannot be blank")
        vector = _normalize_embedding(embedding)
        delete_query = f"DELETE FROM {self._qualified_table} WHERE worker_id = %s"
        insert_query = f"""INSERT INTO {self._qualified_table} (worker_id, vector, sample_type)
                           VALUES (%s, %s::vector, 'primary')
                           RETURNING id, created_at"""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(delete_query, (worker_id,))
                cursor.execute(insert_query, (worker_id, _vector_literal(vector)))
                row = cursor.fetchone()
        except Exception as exc:
            LOGGER.exception("Failed to replace face vector for worker=%s", worker_id)
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError("failed to replace face vector") from exc
        if not row:
            raise FaceVectorRepositoryOperationError("INSERT returned no record")
        return StoredFaceVector(
            worker_id,
            vector,
            int(row[0]),
            row[1],
            sample_type="primary",
        )

    def list_worker(self, worker_id: str) -> tuple[StoredFaceVector, ...]:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id cannot be blank")
        query = f"""SELECT id, worker_id, vector::text, created_at,
                           sample_type, pose, quality, image_sha256
                    FROM {self._qualified_table}
                    WHERE worker_id = %s
                    ORDER BY created_at ASC, id ASC"""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, (worker_id,))
                rows = cursor.fetchall()
        except Exception as exc:
            LOGGER.exception("Failed to list face vectors for worker=%s", worker_id)
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError("failed to list worker face vectors") from exc
        return tuple(
            StoredFaceVector(
                worker_id=str(row[1]),
                embedding=_parse_vector(row[2]),
                record_id=int(row[0]),
                created_at=row[3],
                sample_type=str(row[4]) if row[4] is not None else None,
                pose=str(row[5]) if row[5] is not None else None,
                quality=float(row[6]) if row[6] is not None else None,
                image_sha256=str(row[7]) if row[7] is not None else None,
            )
            for row in rows
        )

    def find_nearest(
        self,
        embedding: np.ndarray | Sequence[float],
        *,
        min_similarity: float | None = None,
    ) -> FaceVectorMatch | None:
        return self._find_nearest(embedding, min_similarity=min_similarity, exclude_worker_id=None)

    def find_nearest_other(
        self,
        embedding: np.ndarray | Sequence[float],
        worker_id: str,
    ) -> FaceVectorMatch | None:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be blank")
        return self._find_nearest(embedding, min_similarity=None, exclude_worker_id=worker_id.strip())

    def _find_nearest(
        self,
        embedding: np.ndarray | Sequence[float],
        *,
        min_similarity: float | None,
        exclude_worker_id: str | None,
    ) -> FaceVectorMatch | None:
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        vector_literal = _vector_literal(_normalize_embedding(embedding))
        where = "WHERE worker_id <> %s" if exclude_worker_id is not None else ""
        query = f"""SELECT id, worker_id,
                           1 - (vector <=> %s::vector) AS similarity,
                           created_at
                    FROM {self._qualified_table}
                    {where}
                    ORDER BY vector <=> %s::vector
                    LIMIT 1"""
        params: tuple[Any, ...]
        if exclude_worker_id is None:
            params = (vector_literal, vector_literal)
        else:
            params = (vector_literal, exclude_worker_id, vector_literal)
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
        except Exception as exc:
            LOGGER.exception("Failed to query nearest face vector")
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError("failed to query nearest face vector") from exc
        if not row:
            return None
        match = FaceVectorMatch(
            record_id=int(row[0]),
            worker_id=str(row[1]),
            similarity=float(row[2]),
            created_at=row[3],
        )
        if min_similarity is not None and match.similarity < min_similarity:
            return None
        return match

    def delete_worker(self, worker_id: str) -> int:
        if not worker_id.strip():
            raise ValueError("worker_id cannot be blank")
        query = f"DELETE FROM {self._qualified_table} WHERE worker_id = %s"
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, (worker_id,))
                return int(cursor.rowcount)
        except Exception as exc:
            LOGGER.exception("Failed to delete vectors for worker=%s", worker_id)
            if isinstance(exc, FaceVectorRepositoryError):
                raise
            raise FaceVectorRepositoryOperationError(
                "failed to delete worker face vectors"
            ) from exc

    def _connect(self) -> Any:
        if self.connection_factory is not None:
            try:
                return self.connection_factory()
            except Exception as exc:
                raise FaceVectorRepositoryOperationError("connection factory failed") from exc
        try:
            import psycopg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DatabaseRuntimeUnavailable(
                "psycopg is required for PostgreSQL face-vector persistence"
            ) from exc
        try:
            return psycopg.connect(self.dsn)
        except Exception as exc:
            raise FaceVectorRepositoryOperationError("failed to connect to PostgreSQL") from exc


def _parse_vector(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return _normalize_embedding(value)
    if isinstance(value, (list, tuple)):
        return _normalize_embedding(value)
    text = str(value).strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise FaceVectorRepositoryOperationError("database returned an invalid pgvector value")
    try:
        return _normalize_embedding([float(item) for item in text[1:-1].split(",") if item])
    except ValueError as exc:
        raise FaceVectorRepositoryOperationError("database returned an invalid pgvector value") from exc


def _vector_literal(embedding: np.ndarray | Sequence[float]) -> str:
    vector = _normalize_embedding(embedding)
    return "[" + ",".join(format(float(value), ".9g") for value in vector) + "]"


def _normalize_embedding(embedding: np.ndarray | Sequence[float]) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if vector.size != PgVectorFaceRepository.VECTOR_DIMENSIONS:
        raise ValueError(f"face embedding must contain 512 values, got {vector.size}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("face embedding contains NaN or infinity")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("face embedding has zero norm")
    normalized = vector / norm
    normalized.setflags(write=False)
    return normalized


PostgresFaceRepository = PgVectorFaceRepository
PgVectorRepository = PgVectorFaceRepository


__all__ = [
    "DatabaseRuntimeUnavailable",
    "FaceVectorMatch",
    "FaceVectorRepository",
    "FaceVectorRepositoryError",
    "FaceVectorRepositoryOperationError",
    "PgVectorFaceRepository",
    "PgVectorRepository",
    "PostgresFaceRepository",
    "StoredFaceVector",
]
