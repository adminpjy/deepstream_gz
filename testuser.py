#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def resolve_dsn() -> str:
    dsn = os.getenv("DATABASE_DSN")
    if dsn:
        return dsn
    return "postgresql://deepstream:change-this-local-password@localhost:5432/deepstream"


def load_image(image_path: Path):
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("OpenCV is required. Install with: pip install opencv-python") from exc

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def select_face_crop(image):
    try:
        import cv2
    except ImportError:  # pragma: no cover - environment guard
        return image

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        return image

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    padding = int(0.15 * max(w, h))
    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image.shape[1], x + w + padding)
    y1 = min(image.shape[0], y + h + padding)
    return image[y0:y1, x0:x1]


def build_embedder():
    from deepstream_ai.face import AdaFaceONNXAdapter, AdaFacePreprocessor

    model_path = ROOT / "models" / "face.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"AdaFace ONNX model not found: {model_path}")

    return AdaFaceONNXAdapter(
        model_path=str(model_path),
        preprocessor=AdaFacePreprocessor((112, 112), input_color="bgr"),
    )


def register_user(worker_id: str, image_path: str) -> str:
    if not worker_id or not worker_id.strip():
        raise ValueError("worker_id cannot be empty")

    image_file = Path(image_path).expanduser()
    if not image_file.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_file}")

    image = load_image(image_file)
    face_crop = select_face_crop(image)
    embedding = build_embedder().embed(face_crop)

    from deepstream_ai.database import PgVectorFaceRepository

    repo = PgVectorFaceRepository(resolve_dsn())
    repo.ensure_schema()
    stored = repo.replace_worker(worker_id.strip(), embedding)
    return (
        f"registered worker_id={stored.worker_id}, "
        f"record_id={stored.record_id}, "
        f"created_at={stored.created_at}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Register a worker/user face embedding.")
    parser.add_argument("worker_id", help="User/worker identifier to register")
    parser.add_argument("image_path", help="Path to the face photo to register")
    args = parser.parse_args()

    try:
        result = register_user(args.worker_id, args.image_path)
        print(result)
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
