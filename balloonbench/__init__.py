"""BalloonBench: a benchmark, harness and verifier for GD&T extraction from 2D drawings."""

__version__ = "0.1.0"

from balloonbench.schema import SCHEMA_VERSION, Characteristic, Drawing

__all__ = ["SCHEMA_VERSION", "Characteristic", "Drawing", "__version__"]
