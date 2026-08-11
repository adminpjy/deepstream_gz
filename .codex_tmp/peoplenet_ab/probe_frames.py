from __future__ import annotations

import json
import sys

from deepstream_ai.config import load_config
from deepstream_ai.pipeline.builder import DeepStreamPipelineBuilder
from deepstream_ai.pipeline.runner import PipelineRunner
from deepstream_ai.pipeline.runtime import load_runtime
from deepstream_ai.preflight import validate_assets


class Collector:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []

    def submit(self, packet) -> bool:
        if packet.tracks:
            self.frames.append(
                {
                    "frame": packet.frame_number,
                    "stream_time_ns": packet.stream_time_ns,
                    "tracks": [
                        {
                            "id": str(track.track_id),
                            "confidence": track.confidence,
                            "bbox": track.bbox.as_tuple(),
                        }
                        for track in packet.tracks
                    ],
                }
            )
        return True

    def identity_label(self, camera_id, track_id):
        return None


config = load_config(sys.argv[1])
validate_assets(config)
runtime = load_runtime()
collector = Collector()
graph = DeepStreamPipelineBuilder(runtime, config, collector).build()
PipelineRunner(runtime, config, graph).run()
print("FRAME_AUDIT=" + json.dumps(collector.frames, separators=(",", ":")))
