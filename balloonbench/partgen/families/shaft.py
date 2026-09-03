"""The shaft family.

A stepped cylindrical shaft turned on a lathe: a run of coaxial diameters, largest in the
middle or descending toward one end, with chamfered ends and optionally a keyway, a
retaining ring groove and a threaded end.

What makes a shaft worth generating is that it is the natural home of the *runout*
tolerances, and of the two-datum axis. A shaft is inspected between centres or in vee
blocks on its two bearing journals, so its drawing almost always carries a datum axis
written ``A-B`` -- two coaxial features acting jointly as one datum. That construction is
common in practice, easy for a model to misread, and impossible to represent if the part
has only one candidate datum. Per SPEC.md section 6.2 the natural GD&T is circular and
total runout to that axis, cylindricity of the journals, and perpendicularity of the
shoulder faces.

Geometry: steps stacked along +Z from z = 0, so the axis is the Z axis throughout.
"""

from __future__ import annotations

from typing import Any

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen import registry
from balloonbench.partgen.fits import COMMON_SHAFT_FITS, fit_limits
from balloonbench.partgen.preferred import (
    METRIC_THREADS,
    R10,
    R20,
    sample_renard,
    snap_to_series,
)
from balloonbench.partgen.types import FaceIndex, SemanticFeature, UnbuildableParams

__all__ = ["Shaft"]

#: A shoulder shorter than this is a fillet, not a step, and will not survive chamfering.
MIN_STEP_LENGTH = 4.0

#: Smallest usable diameter difference between adjacent steps. Below this the shoulder is
#: not machinable as a distinct face and the drawing would dimension a step nobody can cut.
MIN_STEP_DROP = 2.0

#: Keyway proportions by shaft diameter (DIN 6885 A, the common parallel key): the key
#: width and the depth cut into the shaft. Only the sizes a generated shaft can reach.
KEYWAY_BY_DIAMETER: tuple[tuple[float, float, float], ...] = (
    # (max shaft diameter, key width, shaft depth)
    (10.0, 3.0, 1.8),
    (12.0, 4.0, 2.5),
    (17.0, 5.0, 3.0),
    (22.0, 6.0, 3.5),
    (30.0, 8.0, 4.0),
    (38.0, 10.0, 5.0),
    (44.0, 12.0, 5.0),
    (50.0, 14.0, 5.5),
    (58.0, 16.0, 6.0),
    (65.0, 18.0, 7.0),
    (75.0, 20.0, 7.5),
    (85.0, 22.0, 9.0),
    (95.0, 25.0, 9.0),
    (110.0, 28.0, 10.0),
)


def _keyway_for(diameter: float) -> tuple[float, float] | None:
    for limit, width, depth in KEYWAY_BY_DIAMETER:
        if diameter <= limit:
            return width, depth
    return None


class Shaft:
    """A stepped shaft with chamfered ends and optional keyway and groove."""

    name = "shaft"

    # -- sampling -------------------------------------------------------------------

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw a shaft as a descending run of preferred diameters.

        Diameters descend monotonically from one end rather than being drawn
        independently. That is how a shaft is actually turned -- each step is cut without
        reversing the part -- and it also guarantees a machinable shoulder at every
        transition instead of leaving it to a rejection check.
        """
        n_steps = int(rng.integers(2, 6))
        largest = sample_renard(rng, 25.0, 120.0, series=10)

        diameters: list[float] = [largest]
        for _ in range(n_steps - 1):
            target = diameters[-1] * rng.uniform(0.72, 0.90)
            options = [
                d for d in R10 if 8.0 <= d <= diameters[-1] - MIN_STEP_DROP
            ]
            if not options:
                break
            diameters.append(min(options, key=lambda d: abs(d - target)))

        if len(diameters) < 2:
            raise UnbuildableParams(
                f"a {largest:g} shaft has no room for a second step above 8 mm"
            )

        lengths = [
            snap_to_series(d * rng.uniform(0.8, 2.2), R20) for d in diameters
        ]
        lengths = [max(length, MIN_STEP_LENGTH) for length in lengths]

        # The two journals are the largest and the smallest steps: the features a shaft is
        # supported on, and so the pair that becomes datum A-B.
        journal_a = 0
        journal_b = len(diameters) - 1

        keyway = _keyway_for(diameters[0]) if rng.random() < 0.5 else None
        keyway_step = 0 if keyway else -1
        keyway_length = (
            snap_to_series(lengths[0] * 0.6, R20) if keyway else 0.0
        )

        has_groove = bool(rng.random() < 0.35)
        groove_step = len(diameters) - 1
        groove_width = 2.0 if has_groove else 0.0
        groove_depth = (
            round(min(1.5, diameters[groove_step] * 0.04), 2) if has_groove else 0.0
        )

        thread_candidates = [
            d for d in METRIC_THREADS if abs(d - diameters[-1]) < 0.51 and d >= 6.0
        ]
        wants_thread = bool(thread_candidates) and rng.random() < 0.4
        thread = float(thread_candidates[0]) if wants_thread else 0.0

        chamfer = min(1.0 if diameters[-1] >= 20 else 0.5, min(lengths) / 3.0)

        return {
            "diameters": diameters,
            "lengths": lengths,
            "journal_a": journal_a,
            "journal_b": journal_b,
            "keyway_step": keyway_step,
            "keyway_width": keyway[0] if keyway else 0.0,
            "keyway_depth": keyway[1] if keyway else 0.0,
            "keyway_length": keyway_length,
            "groove_step": groove_step if has_groove else -1,
            "groove_width": groove_width,
            "groove_depth": groove_depth,
            "thread": thread,
            "thread_pitch": METRIC_THREADS.get(thread, 0.0),
            "chamfer": chamfer,
            "journal_fit": str(rng.choice([f for f in COMMON_SHAFT_FITS if f[0] in "hkmg"])),
        }

    # -- manufacturability ----------------------------------------------------------

    @staticmethod
    def _check(p: dict[str, Any]) -> None:
        diameters, lengths = p["diameters"], p["lengths"]
        if len(diameters) != len(lengths):
            raise UnbuildableParams("every step needs a length")
        if len(diameters) < 2:
            raise UnbuildableParams("a shaft needs at least two steps to be stepped")

        for i in range(1, len(diameters)):
            drop = diameters[i - 1] - diameters[i]
            if drop < MIN_STEP_DROP:
                raise UnbuildableParams(
                    f"step {i} drops only {drop:.1f} from {diameters[i - 1]:g}, under the "
                    f"{MIN_STEP_DROP} minimum for a machinable shoulder"
                )

        if min(lengths) < MIN_STEP_LENGTH:
            raise UnbuildableParams(
                f"step length {min(lengths):.1f} is under {MIN_STEP_LENGTH}"
            )

        if p["chamfer"] * 2 >= min(lengths):
            raise UnbuildableParams("chamfers consume the shortest step")

        k = p["keyway_step"]
        if k >= 0:
            if p["keyway_depth"] >= diameters[k] * 0.25:
                raise UnbuildableParams(
                    f"keyway depth {p['keyway_depth']} is over a quarter of the "
                    f"{diameters[k]:g} step it is cut into"
                )
            if p["keyway_length"] >= lengths[k] - 2 * p["chamfer"]:
                raise UnbuildableParams("keyway runs past the end of its step")

        g = p["groove_step"]
        if g >= 0:
            if p["groove_depth"] >= diameters[g] * 0.15:
                raise UnbuildableParams("retaining ring groove is too deep for its step")
            if p["groove_width"] + 2 * p["chamfer"] >= lengths[g]:
                raise UnbuildableParams("groove does not fit within its step")

    # -- construction ---------------------------------------------------------------

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        self._check(params)
        p = params
        diameters, lengths = p["diameters"], p["lengths"]

        # Each step is built on its own workplane at an absolute height and then fused.
        # Chaining .workplane(offset=...) off the running solid instead would offset from
        # the *previous* workplane rather than from the origin, so the offsets compound and
        # the part comes out longer than the sum of its steps -- silently, because it is
        # still a valid solid.
        z = 0.0
        body = None
        for diameter, length in zip(diameters, lengths, strict=True):
            step = (
                cq.Workplane("XY").workplane(offset=z).circle(diameter / 2).extrude(length)
            )
            body = step if body is None else body.union(step)
            z += length
        total_length = z
        assert body is not None  # _check guarantees at least two steps

        # Keyway: a flat-bottomed slot milled into one step, open at neither end (a closed
        # keyway), which is what a parallel key sits in.
        k = p["keyway_step"]
        if k >= 0:
            z0 = sum(lengths[:k]) + (lengths[k] - p["keyway_length"]) / 2
            radius = diameters[k] / 2
            cutter = (
                cq.Workplane("XY")
                .workplane(offset=z0 + p["keyway_length"] / 2)
                .box(
                    p["keyway_length"],
                    p["keyway_width"],
                    p["keyway_depth"] * 2,
                    centered=(True, True, True),
                )
                .translate((0, 0, radius))
            )
            body = body.cut(cutter)

        # Retaining ring groove: a square-bottomed annular channel.
        g = p["groove_step"]
        if g >= 0:
            z0 = sum(lengths[:g]) + (lengths[g] - p["groove_width"]) / 2
            outer = diameters[g] / 2 + 1.0
            inner = diameters[g] / 2 - p["groove_depth"]
            ring = (
                cq.Workplane("XY")
                .workplane(offset=z0)
                .circle(outer)
                .circle(inner)
                .extrude(p["groove_width"])
            )
            body = body.cut(ring)

        # Chamfer only the two end faces. Chamfering every edge would round the shoulders
        # into fillets and destroy the perpendicularity features the family exists to
        # exercise.
        ends = [
            e
            for e in body.edges().vals()
            if e.geomType() == "CIRCLE"
            and (
                abs(e.BoundingBox().zmin) < 1e-6
                or abs(e.BoundingBox().zmax - total_length) < 1e-6
            )
        ]
        if ends:
            try:
                body = body.newObject(ends).chamfer(p["chamfer"])
            except Exception as exc:  # noqa: BLE001 - OCCT raises many types here
                raise UnbuildableParams(f"end chamfer failed: {exc}") from exc

        return body.val().wrapped

    # -- description ----------------------------------------------------------------

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        p = params
        diameters, lengths = p["diameters"], p["lengths"]
        features: list[SemanticFeature] = []

        z0 = 0.0
        for i, (diameter, length) in enumerate(zip(diameters, lengths, strict=True)):
            mid = z0 + length / 2
            step_faces = [
                f
                for f in faces.cylinders_of_radius(diameter / 2, tol=1e-3)
                if z0 - 1e-3 <= f.centroid[2] <= z0 + length + 1e-3
            ]
            if not step_faces:
                raise UnbuildableParams(
                    f"step {i} of diameter {diameter:g} left no cylindrical face; nearest "
                    f"radii are {faces.nearest_cylinder_radii(diameter / 2)}"
                )

            is_journal = i in (p["journal_a"], p["journal_b"])
            nominal: dict[str, float] = {"diameter": diameter, "length": length}
            if is_journal:
                limits = fit_limits(diameter, p["journal_fit"])
                nominal["upper_tol"] = limits.upper
                nominal["lower_tol"] = limits.lower

            features.append(
                SemanticFeature(
                    fid=f"step_{i}",
                    kind="cylindrical_face",
                    faces=tuple(f.fid for f in step_faces),
                    nominal=nominal,
                    # The anchor carries the step's own z, so two journals of equal
                    # diameter remain distinguishable features rather than reading as one
                    # feature described twice.
                    anchor=(diameter / 2, 0.0, mid),
                    axis=((0.0, 0.0, mid), (0.0, 0.0, 1.0)),
                    meta=(
                        {"fit": p["journal_fit"], "datum": "A" if i == p["journal_a"] else "B"}
                        if is_journal
                        else {}
                    ),
                )
            )
            z0 += length

        # Shoulders: the annular planar faces between steps. These carry the
        # perpendicularity callouts, so they are named rather than left anonymous.
        z0 = 0.0
        for i in range(len(diameters) - 1):
            z0 += lengths[i]
            shoulder = [
                f
                for f in faces.planes_with_normal((0, 0, 1))
                if abs(f.centroid[2] - z0) < 1e-3
            ]
            if shoulder:
                features.append(
                    SemanticFeature(
                        fid=f"shoulder_{i}",
                        kind="planar_face",
                        faces=tuple(f.fid for f in shoulder),
                        nominal={
                            "outer_diameter": diameters[i],
                            "inner_diameter": diameters[i + 1],
                            "z": z0,
                        },
                        anchor=(diameters[i] / 2 * 0.8, 0.0, z0),
                        axis=((0.0, 0.0, z0), (0.0, 0.0, 1.0)),
                    )
                )

        k = p["keyway_step"]
        if k >= 0:
            key_z = sum(lengths[:k]) + lengths[k] / 2
            key_faces = [
                f
                for f in faces.of_type("plane")
                if abs(f.centroid[2] - key_z) < lengths[k]
                and abs(f.centroid[1]) < p["keyway_width"]
                and f.normal is not None
                and abs(f.normal[2]) < 0.5
            ]
            if key_faces:
                features.append(
                    SemanticFeature(
                        fid="keyway",
                        kind="keyway",
                        faces=tuple(f.fid for f in key_faces),
                        nominal={
                            "width": p["keyway_width"],
                            "depth": p["keyway_depth"],
                            "length": p["keyway_length"],
                        },
                        anchor=(0.0, 0.0, key_z + diameters[k] / 2),
                        axis=((0.0, 0.0, key_z), (0.0, 1.0, 0.0)),
                        parent=f"step_{k}",
                    )
                )

        g = p["groove_step"]
        if g >= 0:
            groove_r = diameters[g] / 2 - p["groove_depth"]
            groove_faces = faces.cylinders_of_radius(groove_r, tol=1e-3)
            if groove_faces:
                groove_z = sum(lengths[:g]) + lengths[g] / 2
                features.append(
                    SemanticFeature(
                        fid="ring_groove",
                        kind="groove",
                        faces=tuple(f.fid for f in groove_faces),
                        nominal={
                            "diameter": groove_r * 2,
                            "width": p["groove_width"],
                            "depth": p["groove_depth"],
                        },
                        anchor=(groove_r, 0.0, groove_z),
                        axis=((0.0, 0.0, groove_z), (0.0, 0.0, 1.0)),
                        parent=f"step_{g}",
                    )
                )

        if p["thread"]:
            # The thread is a callout on the end step, not separate geometry: BalloonBench
            # generates the drawing annotation, and a modelled helix would add hundreds of
            # faces to every STEP file for no benefit to any downstream module.
            features.append(
                SemanticFeature(
                    fid="end_thread",
                    kind="thread",
                    faces=features[len(diameters) - 1].faces,
                    nominal={
                        "diameter": p["thread"],
                        "pitch": p["thread_pitch"],
                        "length": lengths[-1],
                    },
                    anchor=(p["thread"] / 2, 0.0, sum(lengths) - lengths[-1] / 2),
                    axis=((0.0, 0.0, sum(lengths)), (0.0, 0.0, 1.0)),
                    parent=f"step_{len(diameters) - 1}",
                    meta={"callout": f"M{p['thread']:g}x{p['thread_pitch']:g} - 6g"},
                )
            )

        return tuple(features)


registry.register(Shaft())
