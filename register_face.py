#!/usr/bin/env python3
"""Register or supplement one worker face using the same web-enrollment policy.

Examples:
  python register_face.py --workid 100086 --image ./front.jpg
  python register_face.py --workid 100086 --image ./left-30.jpg --mode supplement

The command is intentionally a thin wrapper around FaceRegistrationService so
CLI and Web registrations share exactly the same quality gate and pgvector data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepstream_ai.config import load_config
from deepstream_ai.face_registration_factory import build_face_registration_service


def register_face(
    *,
    config_path: str | Path,
    workid: str,
    image_path: str | Path,
    mode: str = "primary",
) -> dict[str, object]:
    config = load_config(Path(config_path))
    service = build_face_registration_service(config)
    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return service.register(
        workid,
        path.read_bytes(),
        mode=mode,
        filename=path.name,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quality-gated AdaFace enrollment")
    parser.add_argument("--config", default="configs/config.yaml", type=Path)
    parser.add_argument("--workid", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--mode", choices=("primary", "supplement"), default="primary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = register_face(
        config_path=args.config,
        workid=args.workid,
        image_path=args.image,
        mode=args.mode,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("accepted") else 2


if __name__ == "__main__":
    raise SystemExit(main())
