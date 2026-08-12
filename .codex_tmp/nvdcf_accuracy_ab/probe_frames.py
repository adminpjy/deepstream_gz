from __future__ import annotations

import json
import sys
from collections import Counter

from deepstream_ai.config import load_config
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.runtime import load_runtime
from deepstream_ai.preflight import validate_assets


class Collector:
    def __init__(self) -> None:
        self.frames = 0
        self.track_frames = 0
        self.ids: set[str] = set()
        self.by_id: dict[str, dict[str, object]] = {}
        self.duplicate_frames = 0
        self.observations = []

    def submit(self, packet) -> bool:
        self.frames += 1
        if packet.tracks:
            self.track_frames += 1
            seen: Counter[str] = Counter()
            for track in packet.tracks:
                track_id = str(track.track_id)
                seen[track_id] += 1
                self.ids.add(track_id)
                stats = self.by_id.setdefault(
                    track_id,
                    {
                        "first_frame": packet.frame_number,
                        "last_frame": packet.frame_number,
                        "frames": 0,
                        "first_confidence": track.confidence,
                        "first_bbox": list(track.bbox.as_tuple()),
                        "last_bbox": list(track.bbox.as_tuple()),
                        "min_x1": track.bbox.x1,
                        "max_x2": track.bbox.x2,
                        "min_y1": track.bbox.y1,
                        "max_y2": track.bbox.y2,
                    },
                )
                stats["last_frame"] = packet.frame_number
                stats["frames"] = int(stats["frames"]) + 1
                stats["last_bbox"] = list(track.bbox.as_tuple())
                stats["min_x1"] = min(float(stats["min_x1"]), track.bbox.x1)
                stats["max_x2"] = max(float(stats["max_x2"]), track.bbox.x2)
                stats["min_y1"] = min(float(stats["min_y1"]), track.bbox.y1)
                stats["max_y2"] = max(float(stats["max_y2"]), track.bbox.y2)
                self.observations.append(
                    [
                        packet.frame_number,
                        track_id,
                        round(track.confidence, 6),
                        *[round(value, 3) for value in track.bbox.as_tuple()],
                    ]
                )
            if any(count > 1 for count in seen.values()):
                self.duplicate_frames += 1
        return True

    def identity_label(self, camera_id, track_id):
        return None


config = load_config(sys.argv[1])
validate_assets(config)
runtime = load_runtime()
collector = Collector()
graph = DeepStreamPipelineBuilder(runtime, config, collector).build()
PipelineRunner(runtime, config, graph).run()
print(
    "SMOKE_AUDIT="
    + json.dumps(
        {
            "frames": collector.frames,
            "track_frames": collector.track_frames,
            "track_ids": sorted(collector.ids),
            "by_id": collector.by_id,
            "same_id_duplicate_frames": collector.duplicate_frames,
        },
        separators=(",", ":"),
    )
)
with open(sys.argv[2], "w", encoding="utf-8") as stream:
    json.dump(collector.observations, stream, separators=(",", ":"))
