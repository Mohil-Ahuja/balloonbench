"""The valve body family (simplified).

A cast body with an inlet and an outlet on a common axis, a flange at each end, a bonnet
opening on top, and an internal seat. SPEC.md section 6.2 keeps this family deliberately
simplified -- a real valve body is a lofted casting -- because what BalloonBench needs
from it is not the shape but the *drawing conventions a casting brings with it*:

- **Two tolerance regimes on one part.** Cast surfaces carry a general tolerance note
  (ISO 2768-m or a casting standard); machined surfaces carry specific tolerances. A
  drawing where most dimensions are governed by a note in the title block, and only a few
  by explicit callouts, is a genuinely different extraction problem -- a model that
  reports "no tolerance" for the cast dimensions is wrong, because the note applies.
- **Machined-surface callouts.** Only the flange faces, the seat and the bonnet spigot are
  machined, and the drawing says so.
- **Draft angles.** A cast wall is not parallel to itself; it tapers so the pattern can be
  drawn from the mould. Draft produces conical faces where a machinist would expect
  cylindrical ones, which is a useful stress test for the verifier's surface
  classification.

Geometry: the run bore lies on the X axis with a flange at each end; the bonnet rises in
+Z with its own flange.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen import registry
from balloonbench.partgen.preferred import (
    BOLT_CIRCLE_COUNTS,
    CLEARANCE_HOLES,
    R10,
    R20,
    sample_renard,
    snap_to_series,
)
from balloonbench.partgen.types import FaceIndex, SemanticFeature, UnbuildableParams

__all__ = ["ValveBody"]

#: Draft applied to the cast body wall, in degrees. Real castings use 1 to 3 degrees; the
#: shallower end of that is normal for a machined-after-cast body.
DRAFT_DEGREES = 2.0

#: Minimum cast wall thickness. Thinner than this will not fill reliably.
MIN_CAST_WALL = 4.0

#: The general tolerance note that governs every unmachined dimension on the drawing.
CAST_GENERAL_TOLERANCE = "ISO 2768-mK"


class ValveBody:
    """A simplified cast valve body with flanged ends and a bonnet opening."""

    name = "valve_body"

    # -- sampling -------------------------------------------------------------------

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        bore = sample_renard(rng, 25.0, 80.0, series=10)
        wall = snap_to_series(max(MIN_CAST_WALL, bore * 0.12), R20)
        body_od = snap_to_series(bore + 2 * wall, R10)

        flange_od = snap_to_series(body_od * float(rng.uniform(1.6, 2.2)), R10)
        flange_thickness = snap_to_series(max(8.0, bore * 0.25), R20)

        # The end-to-end length must clear both flanges and leave a cast body between.
        face_to_face = snap_to_series(
            max(flange_od * 1.4, bore * 4 + 2 * flange_thickness), R20
        )

        bolt_options = [
            d
            for d in sorted(CLEARANCE_HOLES)
            if 8.0 <= d <= 20.0
            and CLEARANCE_HOLES[d] * 3 <= (flange_od - body_od) / 2 + CLEARANCE_HOLES[d]
        ]
        if not bolt_options:
            raise UnbuildableParams(
                f"a {flange_od:g} flange over a {body_od:g} body has no rim for bolts"
            )
        bolt = float(rng.choice(bolt_options))
        hole = CLEARANCE_HOLES[bolt]
        bcd = snap_to_series((body_od + flange_od) / 2, R20)

        max_count = int(math.pi * bcd / (hole * 2.5))
        allowed = [c for c in BOLT_CIRCLE_COUNTS if c <= max_count]
        if not allowed:
            raise UnbuildableParams(f"a {bcd:g} bolt circle cannot carry four bolts")
        bolt_count = int(rng.choice(allowed))

        bonnet_bore = snap_to_series(bore * float(rng.uniform(0.7, 1.0)), R10)
        bonnet_od = snap_to_series(bonnet_bore + 2 * wall, R10)
        bonnet_height = snap_to_series(bore * float(rng.uniform(0.8, 1.4)), R20)
        bonnet_flange_od = snap_to_series(bonnet_od * 1.7, R10)

        seat_dia = snap_to_series(bore * 0.85, R10)

        return {
            "bore": bore,
            "wall": wall,
            "body_od": body_od,
            "flange_od": flange_od,
            "flange_thickness": flange_thickness,
            "face_to_face": face_to_face,
            "bolt": bolt,
            "bolt_hole": hole,
            "bolt_count": bolt_count,
            "bcd": bcd,
            "bonnet_bore": bonnet_bore,
            "bonnet_od": bonnet_od,
            "bonnet_height": bonnet_height,
            "bonnet_flange_od": bonnet_flange_od,
            "bonnet_flange_thickness": snap_to_series(flange_thickness * 0.8, R20),
            "seat_dia": seat_dia,
            "draft": DRAFT_DEGREES,
            "general_tolerance": CAST_GENERAL_TOLERANCE,
        }

    # -- manufacturability ----------------------------------------------------------

    @staticmethod
    def _check(p: dict[str, Any]) -> None:
        wall = (p["body_od"] - p["bore"]) / 2
        if wall < MIN_CAST_WALL:
            raise UnbuildableParams(
                f"cast wall is {wall:.1f}, under the {MIN_CAST_WALL} minimum"
            )

        if p["flange_od"] <= p["body_od"]:
            raise UnbuildableParams("the flange does not stand proud of the body")

        rim = (p["flange_od"] - p["bcd"]) / 2 - p["bolt_hole"] / 2
        if rim < p["bolt_hole"]:
            raise UnbuildableParams(f"flange rim {rim:.1f} is under one hole diameter")

        inner = (p["bcd"] - p["body_od"]) / 2 - p["bolt_hole"] / 2
        if inner < p["bolt_hole"] * 0.5:
            raise UnbuildableParams(f"bolt circle sits {inner:.1f} from the body wall")

        body_length = p["face_to_face"] - 2 * p["flange_thickness"]
        if body_length < p["bonnet_od"] + 2 * MIN_CAST_WALL:
            raise UnbuildableParams(
                f"a {body_length:.1f} body between the flanges cannot carry a "
                f"{p['bonnet_od']:g} bonnet"
            )

        if p["bonnet_bore"] >= p["bore"] + 2 * MIN_CAST_WALL:
            raise UnbuildableParams("the bonnet bore is wider than the run it opens into")

        if p["seat_dia"] >= p["bore"]:
            raise UnbuildableParams("the seat is not smaller than the bore it seats in")

        # Draft over the body length must not consume the wall.
        taper = math.tan(math.radians(p["draft"])) * body_length / 2
        if taper >= (p["body_od"] - p["bore"]) / 2 - MIN_CAST_WALL / 2:
            raise UnbuildableParams(
                f"{p['draft']} degrees of draft over {body_length:.0f} eats the cast wall"
            )

    # -- construction ---------------------------------------------------------------

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        self._check(params)
        p = params
        half_len = p["face_to_face"] / 2
        body_length = p["face_to_face"] - 2 * p["flange_thickness"]

        # The run body, drafted: modelled as a truncated cone in each direction from the
        # mid-plane, which is what a two-part mould with a central parting line produces.
        taper = math.tan(math.radians(p["draft"])) * body_length / 2
        mid_r = p["body_od"] / 2
        end_r = mid_r - taper

        body = (
            cq.Workplane("YZ")
            .workplane(offset=-body_length / 2)
            .circle(end_r)
            .workplane(offset=body_length / 2)
            .circle(mid_r)
            .loft()
        )
        body = body.union(
            cq.Workplane("YZ")
            .circle(mid_r)
            .workplane(offset=body_length / 2)
            .circle(end_r)
            .loft()
        )

        for sign in (-1, 1):
            flange = (
                cq.Workplane("YZ")
                .workplane(offset=sign * (half_len - p["flange_thickness"]))
                .circle(p["flange_od"] / 2)
                .extrude(sign * p["flange_thickness"])
            )
            body = body.union(flange)

        bonnet = (
            cq.Workplane("XY")
            .circle(p["bonnet_od"] / 2)
            .extrude(p["bonnet_height"])
        )
        bonnet_flange = (
            cq.Workplane("XY")
            .workplane(offset=p["bonnet_height"])
            .circle(p["bonnet_flange_od"] / 2)
            .extrude(p["bonnet_flange_thickness"])
        )
        body = body.union(bonnet).union(bonnet_flange)

        # The run bore, through everything on the X axis.
        body = body.cut(
            cq.Workplane("YZ")
            .workplane(offset=-half_len - 1)
            .circle(p["bore"] / 2)
            .extrude(p["face_to_face"] + 2)
        )

        # The bonnet bore, down into the run.
        body = body.cut(
            cq.Workplane("XY")
            .circle(p["bonnet_bore"] / 2)
            .extrude(p["bonnet_height"] + p["bonnet_flange_thickness"] + 1)
        )

        flange_holes = None
        for sign in (-1, 1):
            holes = (
                cq.Workplane("YZ")
                .workplane(offset=sign * (half_len + 1))
                .polarArray(p["bcd"] / 2, 0, 360, p["bolt_count"])
                .circle(p["bolt_hole"] / 2)
                .extrude(-sign * (p["flange_thickness"] + 2))
            )
            flange_holes = holes if flange_holes is None else flange_holes.union(holes)
        if flange_holes is not None:
            body = body.cut(flange_holes)

        return body.val().wrapped

    # -- description ----------------------------------------------------------------

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        p = params
        half_len = p["face_to_face"] / 2
        features: list[SemanticFeature] = []

        # The two flange faces are the machined datums: a valve is inspected on the faces
        # it seals against, not on the cast body.
        for label, sign, fid in (("A", -1, "inlet_face"), ("B", 1, "outlet_face")):
            candidates = [
                f
                for f in faces.planes_with_normal((1.0, 0.0, 0.0))
                if abs(abs(f.centroid[0]) - half_len) < 1e-6
                and f.centroid[0] * sign > 0
            ]
            if not candidates:
                raise UnbuildableParams(f"no machined flange face for datum {label}")
            face = max(candidates, key=lambda f: f.area)
            features.append(
                SemanticFeature(
                    fid=fid,
                    kind="planar_face",
                    faces=(face.fid,),
                    nominal={"diameter": p["flange_od"], "x": sign * half_len},
                    anchor=face.centroid,
                    axis=(face.centroid, (1.0, 0.0, 0.0)),
                    meta={"datum": label, "machined": True, "finish": "Ra 3.2"},
                )
            )

        run_faces = faces.cylinders_of_radius(p["bore"] / 2, tol=1e-3)
        if not run_faces:
            raise UnbuildableParams(
                f"the {p['bore']:g} run bore left no cylindrical face; nearest radii are "
                f"{faces.nearest_cylinder_radii(p['bore'] / 2)}"
            )
        features.append(
            SemanticFeature(
                fid="run_bore",
                kind="through_hole",
                faces=tuple(f.fid for f in run_faces),
                nominal={"diameter": p["bore"], "depth": p["face_to_face"]},
                anchor=(0.0, 0.0, 0.0),
                axis=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                meta={"machined": True, "datum": "C"},
            )
        )

        bonnet_faces = faces.cylinders_of_radius(p["bonnet_bore"] / 2, tol=1e-3)
        if bonnet_faces:
            features.append(
                SemanticFeature(
                    fid="bonnet_bore",
                    kind="blind_hole",
                    faces=tuple(f.fid for f in bonnet_faces),
                    nominal={
                        "diameter": p["bonnet_bore"],
                        "depth": p["bonnet_height"] + p["bonnet_flange_thickness"],
                    },
                    anchor=(0.0, 0.0, p["bonnet_height"] / 2),
                    axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    meta={"machined": True},
                )
            )

        hole_faces = faces.cylinders_of_radius(p["bolt_hole"] / 2, tol=1e-3)
        if hole_faces:
            features.append(
                SemanticFeature(
                    fid="flange_bolt_holes",
                    kind="through_hole",
                    faces=tuple(f.fid for f in hole_faces),
                    nominal={
                        "diameter": p["bolt_hole"],
                        "bolt_circle": p["bcd"],
                        "count": float(len(hole_faces)),
                        "depth": p["flange_thickness"],
                    },
                    anchor=(half_len - p["flange_thickness"] / 2, p["bcd"] / 2, 0.0),
                    axis=((half_len, p["bcd"] / 2, 0.0), (1.0, 0.0, 0.0)),
                    meta={"pattern": "polar", "for_bolt": f"M{p['bolt']:g}"},
                )
            )

        # The cast outer wall. It is conical rather than cylindrical because of the draft,
        # and it is governed by the general tolerance note rather than by a callout --
        # which is the whole reason this family exists.
        cast_faces = [f for f in faces if f.surface in ("cone", "cylinder") and f.radius
                      and abs(f.radius - p["body_od"] / 2) < p["body_od"] * 0.15]
        if cast_faces:
            features.append(
                SemanticFeature(
                    fid="cast_body_wall",
                    kind="cylindrical_face",
                    faces=tuple(f.fid for f in cast_faces),
                    nominal={
                        "diameter": p["body_od"],
                        "draft_degrees": p["draft"],
                        "wall": p["wall"],
                    },
                    anchor=(0.0, p["body_od"] / 2, 0.0),
                    axis=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                    meta={
                        "machined": False,
                        "governed_by": p["general_tolerance"],
                        "as_cast": True,
                    },
                )
            )

        return tuple(features)


registry.register(ValveBody())
