"""Does this tolerance's reference frame constrain enough to mean anything?

SPEC.md section 12.2 asks for the degrees-of-freedom table, and notes that real drawings
contain this error constantly -- which makes the check a genuine feature rather than an
extraction guard. A positional tolerance referencing one planar datum leaves the part free
to slide and spin in that plane, so the callout does not say where the feature is. Nobody
notices until an inspector tries to set the part up.

A rigid body has six degrees of freedom, three translations and three rotations. Each datum
in the frame removes some of them, and only the ones still remaining: a secondary datum
cannot re-remove what the primary already did. The numbers below are the standard
consequences of what each kind of feature *is* -- a plane can stop motion through itself and
tipping about two axes in it; an axis can stop two translations and two tilts -- written
here in our own words and derived from the geometry rather than quoted from a standard, per
the licensing rules in CLAUDE.md.

**The deliberately lenient case.** A pattern of holes located by a plane and a coaxial bore
constrains five of six: rotation about the bore's axis is left free. That is not a defect --
a circular bolt pattern positioned by basic angles from its own bore is exactly how flanges
are drawn, and the free rotation is a property of the part's symmetry rather than an
omission. Flagging it would fire on almost every flange ever drawn, which is how a verifier
teaches people to ignore it. So a frame that includes an axis-bearing datum is allowed to
leave one rotation free.
"""

from __future__ import annotations

from balloonbench.schema import Characteristic, Drawing
from balloonbench.verifier.base import CheckContext

__all__ = ["NAME", "constrained_dof", "is_sufficient", "required_dof", "run"]

NAME = "datum_dof"

#: What each kind of datum feature can remove, by position in the frame. A datum removes
#: only what is left, so the later columns are smaller: the primary has already taken the
#: motions it could.
_DOF_TABLE: dict[str, tuple[int, int, int]] = {
    # feature type:      primary, secondary, tertiary
    "planar_face": (3, 2, 1),
    "cylindrical_feature": (4, 2, 1),
    "axis": (4, 2, 1),
    "width": (3, 2, 1),
    "spherical": (3, 1, 1),
}

#: Datum features that carry an axis, and so leave a rotation about it genuinely free.
_AXIAL = {"cylindrical_feature", "axis"}

#: Tolerances that locate a feature, and therefore need the frame to fix a position.
_LOCATING = {"position", "concentricity", "symmetry", "profile_surface"}

#: Tolerances that only orient a feature. Orientation needs a direction, not a position, so
#: a single datum is enough and demanding more would be a false alarm.
_ORIENTING = {"perpendicularity", "parallelism", "angularity"}

#: Runout is measured by spinning the part, so its datum has to be something to spin about.
_RUNOUT = {"circular_runout", "total_runout"}

FULL_DOF = 6


def constrained_dof(feature_types: list[str]) -> int:
    """How many degrees of freedom this ordered sequence of datums removes."""
    total = 0
    for position, feature_type in enumerate(feature_types[:3]):
        row = _DOF_TABLE.get(feature_type)
        if row is None:
            continue
        total += row[position]
    return min(FULL_DOF, total)


def required_dof(gtol_symbol: str, feature_types: list[str]) -> int | None:
    """How many degrees of freedom this tolerance needs its frame to remove.

    ``None`` for tolerances where the count is not the question -- runout needs an axis
    rather than a number of constraints.
    """
    if gtol_symbol in _RUNOUT:
        return None
    if gtol_symbol in _ORIENTING:
        return 3
    if gtol_symbol in _LOCATING:
        return 5 if any(t in _AXIAL for t in feature_types) else FULL_DOF
    return None


def is_sufficient(gtol_symbol: str, feature_types: list[str]) -> bool:
    """Whether this reference frame constrains enough for this tolerance to mean something.

    Exposed so the error-injection harness can ask the same question the check asks. Two
    frames that are both sufficient are indistinguishable to a check that works from
    geometry alone, and labelling such an injection "missed" would be measuring the
    verifier against something no geometry can decide.
    """
    if gtol_symbol in _RUNOUT:
        return any(t in _AXIAL for t in feature_types)
    needed = required_dof(gtol_symbol, feature_types)
    if needed is None:
        return True
    return constrained_dof(feature_types) >= needed


def run(context: CheckContext) -> None:
    types = {d.label: d.feature_type for d in context.drawing.datums}

    for c in context.drawing.characteristics:
        if c.kind != "geometric_tolerance" or c.gtol_symbol is None:
            continue
        if not c.datum_refs:
            # Form tolerances reference nothing by design; the schema already enforces that.
            continue

        referenced = [types.get(ref.label) for ref in c.datum_refs]
        if any(feature_type is None for feature_type in referenced):
            missing = [
                ref.label for ref, t in zip(c.datum_refs, referenced, strict=True) if t is None
            ]
            context.defect(
                "undeclared_datum",
                f"references datum {', '.join(missing)}, which the drawing never establishes",
                c.id,
            )
            continue

        known = [feature_type for feature_type in referenced if feature_type]
        removed = constrained_dof(known)

        if c.gtol_symbol in _RUNOUT:
            if not any(feature_type in _AXIAL for feature_type in known):
                context.defect(
                    "runout_without_an_axis",
                    f"{c.gtol_symbol} is measured by rotating the part, but its frame "
                    f"({_describe(c, types)}) establishes no axis to rotate about",
                    c.id,
                )
            continue

        if c.gtol_symbol in _ORIENTING:
            if not is_sufficient(c.gtol_symbol, known):
                context.defect(
                    "underconstrained_drf",
                    f"{c.gtol_symbol} needs a direction to orient against; "
                    f"{_describe(c, types)} removes only {removed} of 6 degrees of freedom",
                    c.id,
                )
            continue

        # An axial frame may leave the rotation about its own axis free; see the module
        # docstring for why that is a property of the part rather than a defect.
        if c.gtol_symbol in _LOCATING and not is_sufficient(c.gtol_symbol, known):
            context.defect(
                "underconstrained_drf",
                f"{c.gtol_symbol} locates the feature, but {_describe(c, types)} "
                f"removes {removed} of 6 degrees of freedom, leaving "
                f"{FULL_DOF - removed} free; the callout does not say where the feature is",
                c.id,
            )


def _describe(c: Characteristic, types: dict[str, str]) -> str:
    parts = [
        f"{ref.label} ({types.get(ref.label, 'undeclared')})" for ref in c.datum_refs
    ]
    return "|".join(parts) or "no datum"


def unused_datums(drawing: Drawing) -> list[str]:
    """Datum labels the drawing establishes and then never references.

    A datum symbol costs nothing to draw and means something to whoever sets the part up, so
    an unused one is a mild defect rather than a serious one -- usually the remains of an
    earlier revision. Reported for the same reason: it is a real observation about the
    drawing that no extraction metric would surface.
    """
    referenced = {
        ref.label
        for c in drawing.characteristics
        for ref in c.datum_refs
    }
    return sorted({d.label for d in drawing.datums} - referenced)
