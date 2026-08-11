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

    def __post_init__(self) -> None:
        if not self.worker_id:
            raise ValueError("worker_id cannot be empty")
        object.__setattr__(self, "embedding", _normalize_embedding(self.embedding))


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

    ``connection_factory`` is intentionally injectable. A production instance
    normally supplies a DSN; unit tests and connection pools can supply a
    zero-argument callable returning a psycopg-compatible connection.
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
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )""",
            f"""CREATE INDEX IF NOT EXISTS {self._index_name}
                ON {self._qualified_table}
                USING hnsw (vector vector_cosine_ops)""",
            f"""CREATE INDEX IF NOT EXISTS \"{self.table}_worker_id_idx\"
                ON {self._qualified_table} (worker_id)""",
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

    def add(self, worker_id: str, embedding: np.ndarray | Sequence[float]) -> StoredFaceVector:
        worker_id = worker_id.strip()
        if not worker_id:
            raise ValueError("worker_id cannot be blank")
        vector = _normalize_embedding(embedding)
        query = f"""INSERT INTO {self._qualified_table} (worker_id, vector)
                    VALUES (%s, %s::vector)
                    RETURNING id, created_at"""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, (worker_id, _vector_literal(vector)))
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
        insert_query = f"""INSERT INTO {self._qualified_table} (worker_id, vector)
                           VALUES (%s, %s::vector)
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
        return StoredFaceVector(worker_id, vector, int(row[0]), row[1])

    def find_nearest(
        self,
        embedding: np.ndarray | Sequence[float],
        *,
        min_similarity: float | None = None,
    ) -> FaceVectorMatch | None:
        if min_similarity is not None and not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between -1 and 1")
        vector_literal = _vector_literal(_normalize_embedding(embedding))
        # Parameters appear twice because PostgreSQL does not permit referring
        # to the SELECT alias from ORDER BY in all query compositions.
        query = f"""SELECT id, worker_id,
                           1 - (vector <=> %s::vector) AS similarity,
                           created_at
                    FROM {self._qualified_table}
                    ORDER BY vector <=> %s::vector
                    LIMIT 1"""
        try:
            with self._connect() as connection, connection.cursor() as cursor:
                cursor.execute(query, (vector_literal, vector_literal))
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
