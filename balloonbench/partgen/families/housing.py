"""The housing family.

A prismatic machined block with a main bore, a cross bore, counterbores, tapped holes and
a pocket. This is the family that exercises the **three-datum reference frame**, which is
the construction the whole benchmark is ultimately about.

A rotational part has an axis, and one axis plus one face is usually enough to say where
everything is. A block has no axis. To locate a hole in it you must nominate three
mutually perpendicular datums -- a primary face that arrests three degrees of freedom, a
secondary face that arrests two more, and a tertiary that arrests the last -- and the
*order* of those three in the feature control frame changes the meaning of the tolerance.
SPEC.md section 10.2 makes datum reference ordering its own metric for that reason, and
section 12.2's ``datum_dof`` check exists to catch frames that leave the part free to
move. Neither is testable without parts that legitimately need three datums.

Geometry: a block spanning the origin to (length, width, height). The main bore runs
through in Z; the cross bore runs through in Y and intersects it.
"""

from __future__ import annotations

from typing import Any

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen import registry
from balloonbench.partgen.fits import COMMON_HOLE_FITS, fit_limits
from balloonbench.partgen.preferred import (
    CLEARANCE_HOLES,
    METRIC_THREADS,
    R10,
    R20,
    sample_renard,
    snap_to_series,
)
from balloonbench.partgen.types import FaceIndex, SemanticFeature, UnbuildableParams

__all__ = ["Housing"]

#: Minimum wall between a bore and an outside face, as a fraction of the bore diameter.
#: A machined housing thinner than this distorts when clamped.
MIN_WALL_RATIO = 0.35

#: Minimum metal between the main bore and the cross bore where they meet.
MIN_BORE_SEPARATION = 3.0


class Housing:
    """A prismatic housing with intersecting bores and a tapped hole pattern."""

    name = "housing"

    # -- sampling -------------------------------------------------------------------

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        length = sample_renard(rng, 63.0, 200.0, series=10)
        width = snap_to_series(length * float(rng.uniform(0.6, 1.0)), R20)
        height = snap_to_series(length * float(rng.uniform(0.5, 0.9)), R20)

        # The main bore is sized from the wall it must leave on the tightest axis, so the
        # wall constraint is satisfied by construction rather than by rejection.
        smallest_side = min(length, width)
        max_bore = smallest_side / (1 + 2 * MIN_WALL_RATIO)
        bore = snap_to_series(max_bore * float(rng.uniform(0.55, 0.85)), R10)
        if bore < 10.0:
            raise UnbuildableParams(
                f"a {length:g} x {width:g} housing leaves room for only a {bore:g} bore"
            )

        # The cross bore must fit between the main bore and the outside on both sides.
        cross_room = (width - bore) / 2 - MIN_BORE_SEPARATION
        cross_bore = snap_to_series(min(bore * 0.5, cross_room * 0.8), R10)
        has_cross = bool(cross_bore >= 6.0 and rng.random() < 0.7)

        has_counterbore = bool(rng.random() < 0.5)
        cbore_dia = snap_to_series(bore * 1.35, R20) if has_counterbore else 0.0
        cbore_depth = snap_to_series(height * 0.2, R20) if has_counterbore else 0.0

        tap_options = [d for d in METRIC_THREADS if 5.0 <= d <= max(6.0, bore / 3)]
        tap = float(rng.choice(tap_options)) if tap_options else 0.0
        tap_drill = round(tap - METRIC_THREADS[tap], 2) if tap else 0.0
        tap_depth = snap_to_series(tap * 2.0, R20) if tap else 0.0
        tap_inset = snap_to_series(max(tap * 2.0, (length - bore) / 4), R20) if tap else 0.0

        has_pocket = bool(rng.random() < 0.4)
        pocket_w = snap_to_series(width * 0.4, R20) if has_pocket else 0.0
        pocket_l = snap_to_series(length * 0.3, R20) if has_pocket else 0.0
        pocket_depth = snap_to_series(height * 0.15, R20) if has_pocket else 0.0

        return {
            "length": length,
            "width": width,
            "height": height,
            "bore": bore,
            "bore_fit": str(rng.choice([f for f in COMMON_HOLE_FITS if f.startswith("H")])),
            "cross_bore": cross_bore if has_cross else 0.0,
            "cbore_dia": cbore_dia,
            "cbore_depth": cbore_depth,
            "tap": tap,
            "tap_drill": tap_drill,
            "tap_depth": tap_depth,
            "tap_inset": tap_inset,
            "pocket_w": pocket_w,
            "pocket_l": pocket_l,
            "pocket_depth": pocket_depth,
        }

    # -- manufacturability ----------------------------------------------------------

    @staticmethod
    def _check(p: dict[str, Any]) -> None:
        wall_x = (p["length"] - p["bore"]) / 2
        wall_y = (p["width"] - p["bore"]) / 2
        for axis, wall in (("length", wall_x), ("width", wall_y)):
            if wall < p["bore"] * MIN_WALL_RATIO:
                raise UnbuildableParams(
                    f"a {p['bore']:g} bore leaves a {wall:.1f} wall in {axis}, under "
                    f"{MIN_WALL_RATIO} x bore"
                )

        if p["cross_bore"]:
            gap = (p["width"] - p["bore"]) / 2 - p["cross_bore"] / 2
            if gap < MIN_BORE_SEPARATION:
                raise UnbuildableParams(
                    f"the cross bore leaves {gap:.1f} to the outside, under "
                    f"{MIN_BORE_SEPARATION}"
                )

        if p["cbore_dia"]:
            if p["cbore_dia"] >= min(p["length"], p["width"]) - 2 * MIN_BORE_SEPARATION:
                raise UnbuildableParams("counterbore breaks out of the housing wall")
            if p["cbore_depth"] >= p["height"] * 0.5:
                raise UnbuildableParams("counterbore is over half the housing height")

        if p["tap"]:
            if p["tap_inset"] * 2 >= min(p["length"], p["width"]) - p["bore"]:
                raise UnbuildableParams("tapped holes fall inside the bore")
            if p["tap_depth"] >= p["height"] * 0.6:
                raise UnbuildableParams("tapped holes are too deep for the housing")

        if p["pocket_depth"] and p["pocket_depth"] >= p["height"] * 0.4:
            raise UnbuildableParams("pocket is too deep")

    # -- construction ---------------------------------------------------------------

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        self._check(params)
        p = params
        half_h = p["height"] / 2

        body = cq.Workplane("XY").box(p["length"], p["width"], p["height"])

        body = body.faces(">Z").workplane().circle(p["bore"] / 2).cutThruAll()

        if p["cbore_dia"]:
            cbore = (
                cq.Workplane("XY")
                .workplane(offset=half_h - p["cbore_depth"])
                .circle(p["cbore_dia"] / 2)
                .extrude(p["cbore_depth"] + 1)
            )
            body = body.cut(cbore)

        if p["cross_bore"]:
            cross = (
                cq.Workplane("XZ")
                .workplane(offset=-p["width"])
                .circle(p["cross_bore"] / 2)
                .extrude(p["width"] * 2)
            )
            body = body.cut(cross)

        if p["tap"]:
            corners = [
                (x, y)
                for x in (-p["length"] / 2 + p["tap_inset"], p["length"] / 2 - p["tap_inset"])
                for y in (-p["width"] / 2 + p["tap_inset"], p["width"] / 2 - p["tap_inset"])
            ]
            taps = (
                cq.Workplane("XY")
                .workplane(offset=half_h - p["tap_depth"])
                .pushPoints(corners)
                .circle(p["tap_drill"] / 2)
                .extrude(p["tap_depth"] + 1)
            )
            body = body.cut(taps)

        if p["pocket_depth"]:
            pocket = (
                cq.Workplane("XY")
                .workplane(offset=-half_h - 1)
                .rect(p["pocket_l"], p["pocket_w"])
                .extrude(p["pocket_depth"] + 1)
            )
            body = body.cut(pocket)

        return body.val().wrapped

    # -- description ----------------------------------------------------------------

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        p = params
        half_h = p["height"] / 2
        features: list[SemanticFeature] = []

        # The three-datum reference frame, in the order a drawing would nominate them:
        # the largest face first, then the longest side, then the end.
        frame = (
            ("A", (0.0, 0.0, 1.0), 2, -half_h, "datum_face_a"),
            ("B", (0.0, 1.0, 0.0), 1, -p["width"] / 2, "datum_face_b"),
            ("C", (1.0, 0.0, 0.0), 0, -p["length"] / 2, "datum_face_c"),
        )
        for label, direction, index, offset, fid in frame:
            candidates = [
                f
                for f in faces.planes_with_normal(direction)
                if abs(f.centroid[index] - offset) < 1e-6
            ]
            if not candidates:
                raise UnbuildableParams(f"no planar face to serve as datum {label}")
            face = max(candidates, key=lambda f: f.area)
            features.append(
                SemanticFeature(
                    fid=fid,
                    kind="planar_face",
                    faces=(face.fid,),
                    nominal={"offset": abs(offset)},
                    anchor=face.centroid,
                    axis=(face.centroid, direction),
                    meta={"datum": label},
                )
            )

        bore_faces = faces.cylinders_of_radius(p["bore"] / 2, tol=1e-3)
        if not bore_faces:
            raise UnbuildableParams(
                f"the {p['bore']:g} main bore left no cylindrical face; nearest radii "
                f"are {faces.nearest_cylinder_radii(p['bore'] / 2)}"
            )
        limits = fit_limits(p["bore"], p["bore_fit"])
        features.append(
            SemanticFeature(
                fid="main_bore",
                kind="through_hole",
                faces=tuple(f.fid for f in bore_faces),
                nominal={
                    "diameter": p["bore"],
                    "depth": p["height"],
                    "upper_tol": limits.upper,
                    "lower_tol": limits.lower,
                },
                anchor=(0.0, 0.0, 0.0),
                axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                meta={"fit": p["bore_fit"], "datum": "D"},
            )
        )

        if p["cross_bore"]:
            cross_faces = faces.cylinders_of_radius(p["cross_bore"] / 2, tol=1e-3)
            if cross_faces:
                features.append(
                    SemanticFeature(
                        fid="cross_bore",
                        kind="through_hole",
                        faces=tuple(f.fid for f in cross_faces),
                        nominal={"diameter": p["cross_bore"], "depth": p["width"]},
                        anchor=(0.0, 0.0, 0.0),
                        axis=((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                    )
                )

        if p["cbore_dia"]:
            cb_faces = faces.cylinders_of_radius(p["cbore_dia"] / 2, tol=1e-3)
            if cb_faces:
                features.append(
                    SemanticFeature(
                        fid="bore_counterbore",
                        kind="counterbore",
                        faces=tuple(f.fid for f in cb_faces),
                        nominal={"diameter": p["cbore_dia"], "depth": p["cbore_depth"]},
                        anchor=(0.0, 0.0, half_h - p["cbore_depth"] / 2),
                        axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                        parent="main_bore",
                    )
                )

        if p["tap"]:
            tap_faces = faces.cylinders_of_radius(p["tap_drill"] / 2, tol=1e-3)
            if tap_faces:
                features.append(
                    SemanticFeature(
                        fid="tapped_holes",
                        kind="thread",
                        faces=tuple(f.fid for f in tap_faces),
                        nominal={
                            "diameter": p["tap"],
                            "pitch": METRIC_THREADS[p["tap"]],
                            "depth": p["tap_depth"],
                            "count": float(len(tap_faces)),
                        },
                        anchor=(
                            p["length"] / 2 - p["tap_inset"],
                            p["width"] / 2 - p["tap_inset"],
                            half_h - p["tap_depth"] / 2,
                        ),
                        axis=(
                            (
                                p["length"] / 2 - p["tap_inset"],
                                p["width"] / 2 - p["tap_inset"],
                                0.0,
                            ),
                            (0.0, 0.0, 1.0),
                        ),
                        meta={
                            "callout": f"M{p['tap']:g}x{METRIC_THREADS[p['tap']]:g} - 6H",
                            "pattern": "rectangular",
                        },
                    )
                )

        if p["pocket_depth"]:
            floors = [
                f
                for f in faces.planes_with_normal((0.0, 0.0, 1.0))
                if abs(f.centroid[2] - (-half_h + p["pocket_depth"])) < 1e-6
            ]
            if floors:
                features.append(
                    SemanticFeature(
                        fid="pocket",
                        kind="slot",
                        faces=tuple(f.fid for f in floors),
                        nominal={
                            "width": p["pocket_w"],
                            "length": p["pocket_l"],
                            "depth": p["pocket_depth"],
                        },
                        anchor=(0.0, 0.0, -half_h + p["pocket_depth"] / 2),
                        axis=((0.0, 0.0, -half_h), (0.0, 0.0, 1.0)),
                    )
                )

        return tuple(features)


registry.register(Housing())


# Note on CLEARANCE_HOLES: imported for symmetry with the other families but not used --
# a housing's holes are tapped, not clearance. Kept out of the import list rather than
# left dangling.
_ = CLEARANCE_HOLES
