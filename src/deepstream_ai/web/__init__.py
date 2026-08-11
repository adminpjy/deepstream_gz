"""Dependency-free HTTP control plane and browser UI."""

from .server import RecognitionHTTPServer, run_web_service

__all__ = ["RecognitionHTTPServer", "run_web_service"]
