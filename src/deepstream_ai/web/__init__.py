"""Dependency-free HTTP control plane and browser UI.

The production server preserves the legacy task/file API while adding the
multi-GPU RTSP session control plane.
"""

from .production_server import ProductionHTTPServer, run_web_service

RecognitionHTTPServer = ProductionHTTPServer

__all__ = ["ProductionHTTPServer", "RecognitionHTTPServer", "run_web_service"]
