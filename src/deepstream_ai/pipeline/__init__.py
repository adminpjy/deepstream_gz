"""DeepStream/GStreamer adapters.

Imports are intentionally lazy so configuration and business tests can run on
developer machines that do not have the DeepStream runtime installed.
"""

from .metadata import FramePacket, FramePacketConsumer

__all__ = ["FramePacket", "FramePacketConsumer"]
