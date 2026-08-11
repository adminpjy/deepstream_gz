from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from deepstream_ai.database import PgVectorFaceRepository


class FakeCursor:
    def __init__(self, rows):
        self.rows = list(rows)
        self.executions = []
        self.rowcount = 2

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, parameters=None):
        self.executions.append((query, parameters))

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def cursor(self):
        return self._cursor


def repository(cursor: FakeCursor) -> PgVectorFaceRepository:
    connection = FakeConnection(cursor)
    return PgVectorFaceRepository(connection_factory=lambda: connection)


def test_schema_creates_pgvector_dimension_and_cosine_index() -> None:
    cursor = FakeCursor([])
    repo = repository(cursor)
    repo.ensure_schema()
    sql = "\n".join(query for query, _ in cursor.executions)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "vector vector(512)" in sql
    assert "vector_cosine_ops" in sql
    assert '"public"."t_worker_face_vector"' in sql


def test_add_normalizes_and_parameterizes_vector() -> None:
    created = datetime(2026, 8, 10, tzinfo=UTC)
    cursor = FakeCursor([(9, created)])
    stored = repository(cursor).add(" worker-1 ", np.full(512, 2.0))
    query, parameters = cursor.executions[-1]

    assert "VALUES (%s, %s::vector)" in query
    assert parameters[0] == "worker-1"
    values = np.fromstring(parameters[1].strip("[]"), sep=",")
    assert values.size == 512
    assert np.linalg.norm(values) == pytest.approx(1.0, rel=1e-6)
    assert stored.record_id == 9
    assert stored.created_at == created


def test_nearest_neighbor_uses_cosine_operator_and_applies_threshold() -> None:
    created = datetime(2026, 8, 10, tzinfo=UTC)
    cursor = FakeCursor([(3, "worker-3", 0.71, created)])
    match = repository(cursor).find_nearest(np.arange(1, 513), min_similarity=0.7)
    query, parameters = cursor.executions[-1]

    assert "1 - (vector <=> %s::vector)" in query
    assert "ORDER BY vector <=> %s::vector" in query
    assert parameters[0] == parameters[1]
    assert match.worker_id == "worker-3"
    assert match.similarity == pytest.approx(0.71)

    below = repository(FakeCursor([(3, "worker-3", 0.69, created)])).find_nearest(
        np.ones(512), min_similarity=0.7
    )
    assert below is None


def test_identifiers_are_validated_before_sql_composition() -> None:
    with pytest.raises(ValueError, match="invalid PostgreSQL table"):
        PgVectorFaceRepository(
            connection_factory=lambda: None,
            table="vectors; DROP TABLE workers; --",
        )
