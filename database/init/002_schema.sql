CREATE TABLE IF NOT EXISTS t_worker_face_vector
(
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    worker_id   text NOT NULL,
    vector      vector(512) NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at  timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS t_worker_face_vector_worker_id_idx
    ON t_worker_face_vector (worker_id);

CREATE INDEX IF NOT EXISTS t_worker_face_vector_vector_hnsw_idx
    ON t_worker_face_vector
    USING hnsw (vector vector_cosine_ops);

CREATE TABLE IF NOT EXISTS t_video_analytics_event
(
    id              bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id        uuid NOT NULL UNIQUE,
    camera_id       text NOT NULL,
    track_id        text,
    event_type      text NOT NULL,
    worker_id       text,
    confidence      real,
    payload         jsonb NOT NULL DEFAULT '{}'::jsonb,
    snapshot_path   text,
    occurred_at     timestamptz NOT NULL,
    created_at      timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_video_event_camera_time
    ON t_video_analytics_event (camera_id, occurred_at DESC);
