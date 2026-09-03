"""Degradation pipeline: composable, seeded transforms with named profiles."""

from balloonbench.degrade.base import (
    Sample,
    SampleDestroyed,
    Transform,
    apply_transforms,
)
from balloonbench.degrade.profiles import PROFILES, degrade, profile_names

__all__ = [
    "PROFILES",
    "Sample",
    "SampleDestroyed",
    "Transform",
    "apply_transforms",
    "degrade",
    "profile_names",
]
