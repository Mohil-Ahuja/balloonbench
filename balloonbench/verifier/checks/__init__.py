"""The five checks of SPEC.md section 12.2.

Each module exposes ``NAME`` and ``run(context)``, and reaches its conclusions through the
context rather than by returning them, so a check can record several findings about one
characteristic without every check having to agree on a return shape.
"""

from balloonbench.verifier.checks import (
    datum_dof,
    mmc_consistency,
    size_exists,
    tolerance_stack,
    unit_sanity,
)

__all__ = [
    "datum_dof",
    "mmc_consistency",
    "size_exists",
    "tolerance_stack",
    "unit_sanity",
]
