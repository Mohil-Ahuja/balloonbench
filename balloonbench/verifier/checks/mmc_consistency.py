"""Is this material condition modifier backed by a size tolerance, and does it fit?

A material condition modifier says the tolerance zone grows as the feature departs from its
maximum material size. That sentence contains a size tolerance: without one there is no
departure, no bonus, and the modifier is decoration. SPEC.md section 12.2 asks for exactly
that check, plus the bonus arithmetic and the virtual condition boundary.

Two things this check computes and reports even when everything is fine, because they are
what a quality engineer would otherwise work out by hand:

* the **bonus tolerance** available -- the size tolerance span, which is how much wider the
  positional zone becomes at the least material end;
* the **virtual condition** -- the worst-case boundary the feature can occupy. For an
  internal feature it is the smallest hole minus the positional zone; for an external one
  the largest shaft plus it. That boundary is what has to clear its neighbours, and when it
  does not, the tolerancing scheme is unmanufacturable as drawn.

**Linking a tolerance to its size.** The schema has no field joining a feature control frame
to the dimension of the same feature -- on a drawing they are joined by being drawn on the
same leader. So the link is recovered geometrically, from the boxes: the frame's leader
target and the dimension's must point at the same place. When no such dimension is found the
verdict is ``unverifiable``, never a defect. A missing link is far more likely to mean the
extraction lost a leader than that the draughtsman forgot the size.
"""

from __future__ import annotations

import math

from balloonbench.evalkit.matching import bbox_iou
from balloonbench.schema import Characteristic
from balloonbench.verifier.base import CheckContext

__all__ = ["NAME", "run"]

NAME = "mmc_consistency"

#: How near two leader targets must be to count as pointing at the same feature, as a
#: fraction of the sheet's diagonal.
#:
#: Proximity rather than overlap. An earlier version required the boxes to intersect, and
#: linked nothing at all on 88 frames out of 88: a feature control frame and the size
#: dimension of the same feature point at the same *feature* but at different parts of it --
#: the frame at the face, the dimension across the bore -- so their targets sit near each
#: other and rarely touch. The check was inert and looked like it was working.
LINK_DISTANCE = 0.06

#: Features whose size grows into their surroundings. A hole is internal: its material
#: condition boundary shrinks as the hole grows.
_INTERNAL_HINTS = ("⌀", "HOLE", "BORE", "THRU")


def run(context: CheckContext) -> None:
    dimensions = [
        c
        for c in context.drawing.characteristics
        if c.kind == "dimension" and c.nominal is not None
    ]

    width, height = context.drawing.image_size
    diagonal = math.hypot(width, height)

    for c in context.drawing.characteristics:
        if c.kind != "geometric_tolerance":
            continue
        modified = c.material_modifier is not None or any(
            ref.modifier for ref in c.datum_refs
        )
        if not modified:
            continue

        if c.material_modifier is None:
            # Only a datum reference carries the modifier. The bonus then applies to the
            # datum feature's size, which this check cannot locate from the frame alone.
            context.unverifiable(
                c, NAME,
                "modifier applies to a datum reference; the datum feature's size tolerance "
                "is not linked to this frame",
            )
            continue

        size = _linked_size(c, dimensions, diagonal)
        if size is None:
            context.unverifiable(
                c, NAME,
                f"{c.material_modifier} needs the size tolerance of the same feature, and "
                f"no dimension on this drawing points at it",
            )
            continue

        upper = context.mm(size.upper_tol) or 0.0
        lower = context.mm(size.lower_tol) or 0.0
        span = abs(upper - lower)
        if span <= 1e-9:
            stated = size.raw_text or f"{size.nominal}"
            general = context.drawing.title_block.general_tolerance
            if general:
                # The size is governed by the title block's general tolerance note, which is
                # a real size tolerance even though this characteristic does not state one.
                # Calling that a defect flagged five flanges out of eight, all of them
                # correctly drawn -- exactly the crying wolf SPEC.md section 12.4 warns of.
                context.unverifiable(
                    c, NAME,
                    f"{c.material_modifier} draws its bonus from the size of {stated}, "
                    f"which carries no stated tolerance and is governed by the general "
                    f"note ({general}); the bonus cannot be computed from the sheet alone",
                )
                continue
            context.defect(
                "modifier_without_size_tolerance",
                f"{c.material_modifier} on this frame promises a bonus that grows with the "
                f"feature's departure from maximum material, but its size ({stated}) "
                f"carries no tolerance, so the bonus is always zero",
                c.id,
            )
            continue

        nominal = context.mm(size.nominal) or 0.0
        zone = context.mm(c.gtol_value) or 0.0
        internal = _is_internal(size)
        mmc_size = nominal + (lower if internal else upper)
        virtual = mmc_size - zone if internal else mmc_size + zone

        detail = (
            f"{c.material_modifier} backed by a size tolerance of {span:.3f} mm, so the "
            f"zone may grow from {zone:.3f} to {zone + span:.3f} mm. Virtual condition "
            f"boundary {virtual:.3f} mm"
        )

        breach = _breaches_neighbour(context, size, virtual, internal)
        if breach:
            context.defect("virtual_condition_interference", breach, c.id)
            context.unverifiable(c, NAME, detail + "; see drawing defects")
        else:
            context.verify(c, NAME, detail, confidence=0.85)


def _linked_size(
    c: Characteristic, dimensions: list[Characteristic], diagonal: float = 1.0
) -> Characteristic | None:
    """The size dimension that points at the same feature as this frame.

    Nearest leader target wins, and only within a short distance. A frame that overlaps a
    dimension's target outright is taken first, since touching is stronger evidence than
    being close.
    """
    target = c.leader_target_bbox or c.bbox
    best: tuple[float, float, Characteristic] | None = None
    for dimension in dimensions:
        if dimension.view != c.view:
            continue
        other = dimension.leader_target_bbox or dimension.bbox
        overlap = bbox_iou(target, other)
        distance = _centre_distance(target, other) / max(diagonal, 1e-9)
        if overlap <= 0 and distance > LINK_DISTANCE:
            continue
        key = (-overlap, distance)
        if best is None or key < (-best[0], best[1]):
            best = (overlap, distance, dimension)
    return None if best is None else best[2]


def _centre_distance(a, b) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return math.hypot(ax - bx, ay - by)


def _is_internal(size: Characteristic) -> bool:
    text = (size.raw_text or "").upper()
    return size.dim_type == "diameter" and any(hint in text for hint in _INTERNAL_HINTS)


def _breaches_neighbour(
    context: CheckContext, size: Characteristic, virtual: float, internal: bool
) -> str | None:
    """Whether the worst-case boundary runs into the geometry next to it.

    Only holes in a recovered pattern are checked, because those are the ones where a
    boundary has an unambiguous neighbour: the hole beside it on the same circle. Reaching
    further -- to a wall, a rim, an adjacent pocket -- needs a notion of "the material
    between" that this index does not have, and guessing at it would produce the confident
    wrong answer this verifier is built to avoid.
    """
    if not internal:
        return None
    nominal = context.mm(size.nominal) or 0.0
    for pattern in context.index.hole_patterns():
        if abs(2 * pattern.radius - nominal) > 0.5 or pattern.count < 2:
            continue
        spacing = pattern.bolt_circle * math.sin(math.pi / pattern.count)
        if spacing <= virtual:
            return (
                f"the virtual condition boundary of {virtual:.3f} mm on a pattern of "
                f"{pattern.count} holes spaced {spacing:.3f} mm apart leaves no material "
                f"between neighbouring holes; the scheme is unmanufacturable as drawn"
            )
    return None
