"""Database adapters."""

from deepstream_ai.database.repository import (
    DatabaseRuntimeUnavailable,
    FaceVectorMatch,
    FaceVectorRepository,
    FaceVectorRepositoryError,
    FaceVectorRepositoryOperationError,
    PgVectorFaceRepository,
    PgVectorRepository,
    PostgresFaceRepository,
    StoredFaceVector,
)

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
