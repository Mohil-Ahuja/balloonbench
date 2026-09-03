"""The plate and bracket family.

A flat plate, optionally folded into an L, with mounting holes, slots and a cutout. It is
the cheapest part in the catalogue to make and the most demanding to tolerance, which is
exactly why it earns a place: a plate is where *position at MMC* and *profile of a
surface* live.

Two things this family exercises that the rotational families cannot:

**Bonus tolerance.** A clearance hole located with position at maximum material condition
gains tolerance as it is drilled larger. That relationship between a size tolerance and a
location tolerance is the single most misread construction in GD&T, and SPEC.md section
10.2 predicts modifier accuracy will be where models fail most. A plate with a bolt
pattern at MMC is the natural place to generate it.

**Slots.** A slot is a feature of size whose size is a *width*, not a diameter, so it is
located by position to a centre plane rather than a centre axis. Models routinely
misclassify slot dimensions as hole diameters.

Geometry: the plate lies in the XY plane from z = 0 to z = thickness. An optional flange
folds up in +Z along the +X edge.
"""

from __future__ import annotations

from typing import Any

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen import registry
from balloonbench.partgen.preferred import (
    CLEARANCE_HOLES,
    PLATE_THICKNESSES,
    R20,
    snap_to_series,
)
from balloonbench.partgen.types import FaceIndex, SemanticFeature, UnbuildableParams

__all__ = ["PlateBracket"]

#: Minimum metal from a hole or slot to any edge, in hole diameters.
MIN_EDGE_DISTANCE_RATIO = 1.5

#: Minimum metal between two adjacent features, in hole diameters.
MIN_FEATURE_GAP_RATIO = 1.0


class PlateBracket:
    """A rectangular plate or L-bracket with holes, slots and a cutout."""

    name = "plate_bracket"

    # -- sampling -------------------------------------------------------------------

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        width = snap_to_series(float(rng.uniform(60.0, 250.0)), R20)
        height = snap_to_series(width * float(rng.uniform(0.5, 1.1)), R20)
        thickness = float(
            rng.choice([t for t in PLATE_THICKNESSES if t <= max(6.0, width / 20)])
        )

        # Hole size is bounded by what the plate can carry with legal edge distance on
        # both axes, so the choice cannot strand the draw.
        smaller_side = min(width, height)
        max_hole = smaller_side / (2 * (MIN_EDGE_DISTANCE_RATIO + 0.5) + 2)
        options = [d for d in sorted(CLEARANCE_HOLES) if CLEARANCE_HOLES[d] <= max_hole]
        if not options:
            raise UnbuildableParams(
                f"a {width:g} x {height:g} plate has no room for even the smallest "
                f"clearance hole with legal edge distance"
            )
        bolt = float(rng.choice(options[: max(1, len(options) // 2)]))
        hole = CLEARANCE_HOLES[bolt]

        # Four corner holes on a rectangular pattern, inset by the edge distance.
        inset = snap_to_series(hole * (MIN_EDGE_DISTANCE_RATIO + 0.5), R20)
        pattern_w = width - 2 * inset
        pattern_h = height - 2 * inset
        if pattern_w <= hole * 2 or pattern_h <= hole * 2:
            raise UnbuildableParams(
                f"an inset of {inset:g} leaves no room for a hole pattern on a "
                f"{width:g} x {height:g} plate"
            )

        # A slot replaces one pair of holes often enough on real brackets to be worth
        # generating: it is how a designer buys adjustment in one axis.
        has_slot = bool(rng.random() < 0.5)
        slot_width = hole if has_slot else 0.0
        slot_length = (
            snap_to_series(min(pattern_w * 0.35, hole * 4), R20) if has_slot else 0.0
        )

        has_cutout = bool(rng.random() < 0.45)
        cutout_dia = (
            snap_to_series(min(pattern_w, pattern_h) * 0.4, R20) if has_cutout else 0.0
        )

        is_bracket = bool(rng.random() < 0.4)
        flange_height = (
            snap_to_series(height * float(rng.uniform(0.3, 0.6)), R20) if is_bracket else 0.0
        )

        corner_radius = snap_to_series(max(hole, thickness * 1.5), R20)

        return {
            "width": width,
            "height": height,
            "thickness": thickness,
            "bolt": bolt,
            "hole": hole,
            "inset": inset,
            "pattern_w": pattern_w,
            "pattern_h": pattern_h,
            "slot_width": slot_width,
            "slot_length": slot_length,
            "cutout_dia": cutout_dia,
            "flange_height": flange_height,
            "corner_radius": corner_radius,
        }

    # -- manufacturability ----------------------------------------------------------

    @staticmethod
    def _check(p: dict[str, Any]) -> None:
        hole = p["hole"]
        edge_x = (p["width"] - p["pattern_w"]) / 2 - hole / 2
        edge_y = (p["height"] - p["pattern_h"]) / 2 - hole / 2
        for name, edge in (("x", edge_x), ("y", edge_y)):
            if edge < hole * MIN_EDGE_DISTANCE_RATIO:
                raise UnbuildableParams(
                    f"edge distance in {name} is {edge:.1f}, under "
                    f"{MIN_EDGE_DISTANCE_RATIO} x hole diameter {hole:g}"
                )

        if p["cutout_dia"]:
            gap = (p["pattern_w"] - p["cutout_dia"]) / 2 - hole / 2
            if gap < hole * MIN_FEATURE_GAP_RATIO:
                raise UnbuildableParams(
                    f"the {p['cutout_dia']:g} cutout leaves {gap:.1f} to the hole "
                    f"pattern, under {MIN_FEATURE_GAP_RATIO} x hole diameter"
                )

        if p["slot_length"] and p["slot_length"] >= p["pattern_w"] - 2 * hole:
            raise UnbuildableParams("slot is too long for the pattern it sits in")

        if p["corner_radius"] * 2 >= min(p["width"], p["height"]):
            raise UnbuildableParams("corner radii consume the plate")

        if p["flange_height"] and p["flange_height"] <= p["thickness"] * 2:
            raise UnbuildableParams("the bent flange is shorter than it is thick")

    # -- construction ---------------------------------------------------------------

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        self._check(params)
        p = params

        body = (
            cq.Workplane("XY")
            .rect(p["width"], p["height"])
            .extrude(p["thickness"])
            .edges("|Z")
            .fillet(p["corner_radius"])
        )

        # Mounting holes. When the part has a slot, one pair of holes becomes the slot, so
        # the remaining holes are the other pair -- that is what a designer does, rather
        # than adding a slot alongside a full set of holes.
        xs = (-p["pattern_w"] / 2, p["pattern_w"] / 2)
        ys = (-p["pattern_h"] / 2, p["pattern_h"] / 2)
        hole_points = (
            [(xs[0], y) for y in ys] if p["slot_length"] else [(x, y) for x in xs for y in ys]
        )
        body = (
            body.faces(">Z")
            .workplane()
            .pushPoints(hole_points)
            .circle(p["hole"] / 2)
            .cutThruAll()
        )

        if p["slot_length"]:
            slot = (
                cq.Workplane("XY")
                .pushPoints([(xs[1], y) for y in ys])
                .slot2D(p["slot_length"], p["slot_width"], angle=0)
                .extrude(p["thickness"] * 3)
                .translate((0, 0, -p["thickness"]))
            )
            body = body.cut(slot)

        if p["cutout_dia"]:
            body = (
                body.faces(">Z")
                .workplane()
                .circle(p["cutout_dia"] / 2)
                .cutThruAll()
            )

        if p["flange_height"]:
            flange = (
                cq.Workplane("XZ")
                .workplane(offset=-p["height"] / 2)
                .center(p["width"] / 2 - p["thickness"] / 2, p["flange_height"] / 2)
                .rect(p["thickness"], p["flange_height"])
                .extrude(p["height"])
            )
            body = body.union(flange)

        return body.val().wrapped

    # -- description ----------------------------------------------------------------

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        p = params
        features: list[SemanticFeature] = []

        # The top face is datum A: a plate is inspected lying on a surface plate, and the
        # face it rests on is what constrains its three primary degrees of freedom.
        tops = [
            f
            for f in faces.planes_with_normal((0, 0, 1))
            if abs(f.centroid[2] - p["thickness"]) < 1e-6
        ]
        if not tops:
            raise UnbuildableParams("no planar face at the plate's top surface")
        primary = max(tops, key=lambda f: f.area)
        features.append(
            SemanticFeature(
                fid="primary_face",
                kind="planar_face",
                faces=(primary.fid,),
                nominal={"width": p["width"], "height": p["height"], "z": p["thickness"]},
                anchor=(0.0, 0.0, p["thickness"]),
                axis=((0.0, 0.0, p["thickness"]), (0.0, 0.0, 1.0)),
                meta={"datum": "A"},
            )
        )

        # The two side faces complete the reference frame. A plate needs all three: datum
        # A alone leaves it free to slide and spin on the surface plate, which is exactly
        # the under-constrained reference frame the verifier's datum_dof check looks for.
        for label, direction, offset, fid in (
            ("B", (0.0, 1.0, 0.0), -p["height"] / 2, "datum_edge_b"),
            ("C", (1.0, 0.0, 0.0), -p["width"] / 2, "datum_edge_c"),
        ):
            axis_index = 1 if label == "B" else 0
            sides = [
                f
                for f in faces.planes_with_normal(direction)
                if abs(f.centroid[axis_index] - offset) < 1e-6
            ]
            if sides:
                side = max(sides, key=lambda f: f.area)
                features.append(
                    SemanticFeature(
                        fid=fid,
                        kind="planar_face",
                        faces=(side.fid,),
                        nominal={"offset": abs(offset)},
                        anchor=side.centroid,
                        axis=(side.centroid, direction),
                        meta={"datum": label},
                    )
                )

        # A slot is a widened hole, so its end radii equal the hole radius and a radius
        # query alone returns both. The holes sit on the -X side and the slots on +X, so
        # split them there; without this the hole feature claims the slot ends as its own
        # and reports a count that no drawing would show.
        hole_faces = [
            f
            for f in faces.cylinders_of_radius(p["hole"] / 2, tol=1e-3)
            if not p["slot_length"] or f.centroid[0] < 0
        ]
        if hole_faces:
            features.append(
                SemanticFeature(
                    fid="mounting_holes",
                    kind="through_hole",
                    faces=tuple(f.fid for f in hole_faces),
                    nominal={
                        "diameter": p["hole"],
                        "count": float(len(hole_faces)),
                        "pitch_x": p["pattern_w"],
                        "pitch_y": p["pattern_h"],
                        "depth": p["thickness"],
                    },
                    anchor=(-p["pattern_w"] / 2, p["pattern_h"] / 2, p["thickness"] / 2),
                    axis=((-p["pattern_w"] / 2, p["pattern_h"] / 2, 0.0), (0.0, 0.0, 1.0)),
                    meta={"pattern": "rectangular", "for_bolt": f"M{p['bolt']:g}"},
                )
            )

        if p["slot_length"]:
            # A slot's size is its width, and its ends are half-cylinders of that radius.
            slot_faces = [
                f
                for f in faces.cylinders_of_radius(p["slot_width"] / 2, tol=1e-3)
                if f.centroid[0] > 0
            ]
            if slot_faces:
                features.append(
                    SemanticFeature(
                        fid="adjustment_slots",
                        kind="slot",
                        faces=tuple(f.fid for f in slot_faces),
                        nominal={
                            "width": p["slot_width"],
                            "length": p["slot_length"],
                            "count": 2.0,
                            "depth": p["thickness"],
                        },
                        anchor=(p["pattern_w"] / 2, p["pattern_h"] / 2, p["thickness"] / 2),
                        axis=((p["pattern_w"] / 2, p["pattern_h"] / 2, 0.0), (0.0, 0.0, 1.0)),
                        meta={"located_by": "centre_plane"},
                    )
                )

        if p["cutout_dia"]:
            cutout_faces = faces.cylinders_of_radius(p["cutout_dia"] / 2, tol=1e-3)
            if cutout_faces:
                features.append(
                    SemanticFeature(
                        fid="central_cutout",
                        kind="through_hole",
                        faces=tuple(f.fid for f in cutout_faces),
                        nominal={"diameter": p["cutout_dia"], "depth": p["thickness"]},
                        anchor=(0.0, 0.0, p["thickness"] / 2),
                        axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    )
                )

        if p["flange_height"]:
            flange_faces = [
                f
                for f in faces.planes_with_normal((1.0, 0.0, 0.0))
                if f.centroid[2] > p["thickness"] + 1e-6
            ]
            if flange_faces:
                outer = max(flange_faces, key=lambda f: f.centroid[0])
                features.append(
                    SemanticFeature(
                        fid="bent_flange",
                        kind="planar_face",
                        faces=(outer.fid,),
                        nominal={
                            "height": p["flange_height"],
                            "thickness": p["thickness"],
                        },
                        anchor=outer.centroid,
                        axis=(outer.centroid, (1.0, 0.0, 0.0)),
                    )
                )

        return tuple(features)


registry.register(PlateBracket())
