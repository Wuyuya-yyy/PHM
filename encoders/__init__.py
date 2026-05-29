"""Encoder modules for cross-modal degradation representation learning."""

from .bearing_encoder import BearingEncoder
from .flywheel_encoder import FlywheelEncoder

__all__ = ["BearingEncoder", "FlywheelEncoder"]
