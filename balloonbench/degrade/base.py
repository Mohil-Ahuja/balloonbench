"""The degradation pipeline's vocabulary: a sample, a transform, and a seeded run.

SPEC.md section 8 asks for composable, seeded transforms with named profiles, because the
synthetic-to-real gap is the thing this module exists to close and it has to be closed
*measurably*. A profile that cannot be reproduced from a seed is not a benchmark condition,
it is a one-off image.

A :class:`Sample` carries the image and its ground truth together, and every transform takes
and returns both. That coupling is the whole design. A transform that could touch the image
without being handed the boxes would be a transform that can silently invalidate them, and
the failure would be invisible -- the JSON still validates, the image still looks degraded,
and only an overlay reveals that the boxes now point at blank paper.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image

from balloonbench.schema import Drawing

__all__ = ["Sample", "SampleDestroyed", "Transform", "apply_transforms", "boxed_fields"]


class SampleDestroyed(RuntimeError):
    """A transform left no labelled callout on the sheet, so there is nothing to evaluate."""


#: Where boxes live in the ground truth: ``(owner kind, attribute)``. Kept as data rather
#: than spelled out at each call site so that a schema change adding a boxed field is a
#: one-line edit here instead of a bug in whichever transform was not updated.
BOXED_FIELDS: tuple[tuple[str, str], ...] = (
    ("datum", "bbox"),
    ("characteristic", "bbox"),
    ("characteristic", "leader_target_bbox"),
)

#: A box that keeps less than this fraction of its area after a geometric transform has been
#: cropped away rather than moved, so what remains no longer delimits the callout. The
#: characteristic is dropped instead of being recorded with a box a labeller would reject.
MIN_RETAINED_AREA = 0.45


@dataclass
class Sample:
    """One drawing: the raster, its ground truth, and what has been done to it.

    ``drawing`` is always a deep copy owned by this sample, so a transform may mutate boxes
    in place without reaching back into the caller's object. ``applied`` is the record that
    ends up in ``provenance.degradation_profile``.
    """

    image: Image.Image
    drawing: Drawing
    applied: tuple[str, ...] = ()
    #: Free-form notes a transform can leave for tests and debugging.
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, image_path, drawing: Drawing) -> Sample:
        image = Image.open(image_path).convert("RGB")
        sample = cls(image=image, drawing=drawing.model_copy(deep=True))
        sample._sync_image_size()
        return sample

    # -- image ---------------------------------------------------------------------------

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    def with_image(self, image: Image.Image, name: str) -> Sample:
        """Replace the raster, recording the transform that did it.

        Used by photometric transforms, which change pixels without moving anything. A
        geometric transform must go through :meth:`with_warp` instead, so that it cannot
        change the image while leaving the boxes behind.
        """
        out = replace(self, image=image, applied=(*self.applied, name))
        out.drawing = self.drawing
        out._sync_image_size()
        return out

    def _sync_image_size(self) -> None:
        self.drawing.image_size = list(self.image.size)  # type: ignore[assignment]

    # -- boxes ---------------------------------------------------------------------------

    def boxes(self) -> list[tuple[str, int | str, str, list[float]]]:
        """Every box in the ground truth, as ``(kind, owner id, attribute, box)``."""
        out: list[tuple[str, int | str, str, list[float]]] = []
        for datum in self.drawing.datums:
            out.append(("datum", datum.label, "bbox", list(datum.bbox)))
        for c in self.drawing.characteristics:
            out.append(("characteristic", c.id, "bbox", list(c.bbox)))
            if c.leader_target_bbox is not None:
                out.append(
                    ("characteristic", c.id, "leader_target_bbox", list(c.leader_target_bbox))
                )
        return out

    def map_boxes(
        self,
        transform: Callable[[np.ndarray], np.ndarray],
        *,
        name: str,
        size: tuple[int, int] | None = None,
    ) -> Sample:
        """Apply a point map to every box, then clip, then drop what did not survive.

        ``transform`` takes an ``(n, 2)`` array of pixel coordinates and returns where they
        land. The box handed back is the axis-aligned bound of the mapped outline, which is
        the honest answer for axis-aligned ground truth: a rotated callout genuinely does
        occupy a larger upright rectangle, and reporting the original size would understate
        the region a detector has to find.

        Dropping rather than clamping follows the same rule the sampler uses in ``partgen``.
        A box clamped back inside the image describes a region that is not where the callout
        is; a dropped characteristic is simply one the degraded sheet no longer shows, which
        is true of a real scan with a torn corner.
        """
        width, height = size or self.image.size
        drawing = self.drawing

        surviving_datums = []
        for datum in drawing.datums:
            box = _map_box(np.asarray(datum.bbox, dtype=float), transform)
            clipped = _clip(box, width, height)
            if clipped is None or _area(clipped) / max(_area(box), 1e-9) < MIN_RETAINED_AREA:
                continue
            datum.bbox = [float(v) for v in clipped]
            surviving_datums.append(datum)

        surviving_labels = {d.label for d in surviving_datums}
        surviving: list = []
        for c in drawing.characteristics:
            box = _map_box(np.asarray(c.bbox, dtype=float), transform)
            clipped = _clip(box, width, height)
            if clipped is None or _area(clipped) / max(_area(box), 1e-9) < MIN_RETAINED_AREA:
                continue
            # A characteristic whose datum was cropped off the sheet keeps its reference:
            # the frame is still legible and still says "|A|". Dropping it would misreport
            # the image. Instead the datum's absence is the drawing's problem to state, so
            # the reference is only removed when the schema would otherwise reject it.
            if any(ref.label not in surviving_labels for ref in c.datum_refs):
                continue
            c.bbox = [float(v) for v in clipped]
            if c.leader_target_bbox is not None:
                target = _clip(
                    _map_box(np.asarray(c.leader_target_bbox, dtype=float), transform),
                    width,
                    height,
                )
                c.leader_target_bbox = None if target is None else [float(v) for v in target]
            surviving.append(c)

        # R1 requires contiguous ids from 1, so dropping forces a renumber. Ids are a
        # per-sheet label, not an identity that survives across images, and evaluation
        # matches on content and position rather than on id.
        for new_id, c in enumerate(surviving, start=1):
            c.id = new_id

        if not surviving:
            # The schema requires at least one characteristic, and rightly so: a sheet with
            # no callouts is not a drawing. Rather than let pydantic reject the assignment
            # halfway through the mutation -- leaving a half-updated object behind -- say
            # what happened. No shipped profile does this; a warp that does is a bug in the
            # profile, not a sample to be salvaged.
            raise SampleDestroyed(
                f"{name!r} moved every characteristic off the sheet; nothing left to label"
            )

        # Characteristics first, then datums. ``Drawing`` validates on assignment, so each
        # of these two statements is checked against the *other* list as it stands at that
        # moment. Assigning the shortened datum list while the old characteristics are still
        # in place fails R4 -- a callout referencing a datum that has just been dropped --
        # even though the pair about to be installed is consistent. The survivors only ever
        # reference surviving datums, so this order passes both checks.
        drawing.characteristics = surviving
        drawing.datums = surviving_datums
        out = replace(self, applied=(*self.applied, name))
        out.drawing = drawing
        return out


def _map_box(box: np.ndarray, transform) -> tuple[float, float, float, float]:
    """AABB of a box's mapped outline.

    The outline is sampled rather than just its four corners. Under a homography a straight
    edge stays straight and corners would be enough, but a radial distortion bows it, and a
    corners-only bound then cuts through the middle of the bulge -- clipping ink out of the
    very box that is supposed to contain it.
    """
    x0, y0, x1, y1 = box
    steps = 8
    ts = np.linspace(0.0, 1.0, steps + 1)
    edges = np.concatenate(
        [
            np.column_stack([x0 + (x1 - x0) * ts, np.full(steps + 1, y0)]),
            np.column_stack([x0 + (x1 - x0) * ts, np.full(steps + 1, y1)]),
            np.column_stack([np.full(steps + 1, x0), y0 + (y1 - y0) * ts]),
            np.column_stack([np.full(steps + 1, x1), y0 + (y1 - y0) * ts]),
        ]
    )
    mapped = np.asarray(transform(edges), dtype=float)
    return (
        float(mapped[:, 0].min()),
        float(mapped[:, 1].min()),
        float(mapped[:, 0].max()),
        float(mapped[:, 1].max()),
    )


def _clip(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float] | None:
    x0 = max(0.0, min(box[0], width))
    y0 = max(0.0, min(box[1], height))
    x1 = max(0.0, min(box[2], width))
    y1 = max(0.0, min(box[3], height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return (x0, y0, x1, y1)


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def boxed_fields() -> tuple[tuple[str, str], ...]:
    return BOXED_FIELDS


@dataclass(frozen=True)
class Transform:
    """One named, optional, seeded degradation step."""

    name: str
    fn: Callable[[Sample, np.random.Generator], Sample]
    #: Probability the step runs at all. Real scans are not uniformly afflicted, and a
    #: profile in which every listed defect always appears is its own kind of unrealistic.
    probability: float = 1.0

    def __call__(self, sample: Sample, rng: np.random.Generator) -> Sample:
        if self.probability < 1.0 and rng.random() > self.probability:
            return sample
        return self.fn(sample, rng)


def apply_transforms(
    sample: Sample, transforms: Sequence[Transform], rng: np.random.Generator
) -> Sample:
    """Run a profile's transforms in order.

    Order is part of the profile, not an implementation detail: creasing paper and then
    photocopying it looks nothing like photocopying it and then creasing it, and only the
    first is a thing that happens.
    """
    for transform in transforms:
        sample = transform(sample, rng)
    return sample
