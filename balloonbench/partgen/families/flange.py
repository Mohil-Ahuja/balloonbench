"""The flange family.

A circular flange is the densest single test of the whole BalloonBench pipeline, which is
why PLAN.md section 2.1 drives it end to end before the other families. One part
exercises: diameter dimensions, a fitted bore, a bolt pattern positioned to a
three-datum-ish frame with a material condition modifier, flatness of a face,
perpendicularity of a bore to that face, a raised face or hub step, and a section view
(a flange is nearly always drawn as a front view plus a section, because the bolt circle
is only legible face-on while the thicknesses are only legible in section).

Geometry, on the Z axis, hub upward:

    - a disc of diameter ``od`` and thickness ``thickness``
    - a through bore of diameter ``bore`` on the axis
    - optionally a hub of diameter ``hub_od`` rising ``hub_height`` above the disc
    - optionally a raised face of diameter ``rf_od`` standing ``rf_height`` proud
    - ``bolt_count`` clearance holes on a bolt circle of diameter ``bcd``
    - optional counterbores on those holes
    - a chamfer at the bore mouth

The natural GD&T, per SPEC.md section 6.2: position of the bolt pattern to |A|B(M)|,
flatness of the mounting face, perpendicularity of the bore to that face, and
concentricity or runout of the hub.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen import registry
from balloonbench.partgen.fits import COMMON_HOLE_FITS, fit_limits
from balloonbench.partgen.preferred import (
    BOLT_CIRCLE_COUNTS,
    CLEARANCE_HOLES,
    R10,
    R20,
    sample_renard,
    snap_to_series,
)
from balloonbench.partgen.types import (
    FaceIndex,
    SemanticFeature,
    UnbuildableParams,
)

__all__ = ["Flange"]

#: Minimum metal between a bolt hole and any free edge, as a multiple of hole diameter.
#: Below roughly one diameter of edge distance a bolted joint tears out rather than
#: holding, so a flange proportioned tighter than this is not a real part.
MIN_EDGE_DISTANCE_RATIO = 1.0

#: Minimum wall between the bore and the bolt circle, likewise in hole diameters.
MIN_BORE_WALL_RATIO = 1.0

#: Minimum arc between adjacent bolt holes, as a multiple of hole diameter. Two holes
#: closer than this cannot both be spot-faced.
MIN_HOLE_PITCH_RATIO = 2.0


def _snap_within(
    value: float, series: tuple[float, ...], low: float, high: float
) -> float | None:
    """The preferred value nearest ``value`` that also lies within ``[low, high]``.

    Snapping and then checking is not enough: a constraint window narrower than the gap
    between two preferred values has no valid answer, and rounding to the nearest one
    would step outside the window. Returning ``None`` lets the caller reject honestly
    instead of shipping a part that violates its own constraint.
    """
    inside = [v for v in series if low <= v <= high]
    if not inside:
        return None
    return min(inside, key=lambda v: abs(v - value))


class Flange:
    """A circular flange with a bore and a bolt pattern."""

    name = "flange"

    # -- sampling -------------------------------------------------------------------

    def sample_params(self, rng: np.random.Generator) -> dict[str, Any]:
        """Draw a flange from preferred sizes and standard fasteners.

        Every dimension that has a geometric constraint is drawn *inside* that
        constraint, not drawn freely and rejected afterwards. An earlier version sampled
        the bolt circle, hub diameter and bolt count independently and let
        :meth:`_check` reject the result; only 8% of draws survived, which is not bad
        luck but a sampler describing a different distribution from the one the
        constraints allow. Rejection is still the response to a genuinely impossible
        combination -- a bolt too large for the flange it is on -- and
        :meth:`_check` remains the authority.
        """
        od = sample_renard(rng, 63.0, 400.0, series=10)
        bore = snap_to_series(od * rng.uniform(0.25, 0.50), R10)
        thickness = snap_to_series(od * rng.uniform(0.08, 0.18), R20)

        # Bolt size scales with the flange, the way a real one does: a 400 mm flange does
        # not get M6, and a 63 mm flange cannot fit M24. Only sizes for which a legal bolt
        # circle actually exists are offered, so the choice cannot strand the draw -- the
        # bolt circle must clear the bore by a wall and the rim by an edge distance, and
        # a preferred value has to land inside that window.
        viable: list[tuple[float, float, float]] = []
        for candidate in sorted(CLEARANCE_HOLES):
            if not 6.0 <= candidate <= 24.0:
                continue
            h = CLEARANCE_HOLES[candidate]
            low = bore + h * (2 * MIN_BORE_WALL_RATIO + 1)
            high = od - h * (2 * MIN_EDGE_DISTANCE_RATIO + 1)
            circle = _snap_within((low + high) / 2, R20, low, high)
            if circle is not None and math.pi * circle >= h * MIN_HOLE_PITCH_RATIO * min(
                BOLT_CIRCLE_COUNTS
            ):
                viable.append((candidate, h, circle))
        if not viable:
            raise UnbuildableParams(
                f"a {od:g} flange with a {bore:g} bore has no room for any bolt circle "
                f"between M6 and M24"
            )

        want = od / 18.0
        near = sorted(viable, key=lambda t: abs(t[0] - want))[:3]
        bolt, hole, bcd = near[int(rng.integers(len(near)))]

        # Bolt count is bounded by the arc each hole needs, so choose among the counts
        # that actually fit rather than drawing one and hoping.
        max_count = int(math.pi * bcd / (hole * MIN_HOLE_PITCH_RATIO))
        allowed = [c for c in BOLT_CIRCLE_COUNTS if c <= max_count]
        bolt_count = int(rng.choice(allowed))

        has_hub = bool(rng.random() < 0.55)
        hub_od = 0.0
        hub_height = 0.0
        if has_hub:
            hub_od = _snap_within(
                (bore * 1.6 + bcd - hole) / 2, R10, bore * 1.3, bcd - hole - 1.0
            ) or 0.0
            if hub_od:
                hub_height = snap_to_series(thickness * rng.uniform(0.8, 2.0), R20)

        has_raised_face = bool(not hub_od and rng.random() < 0.4)
        rf_od = 0.0
        rf_height = 0.0
        if has_raised_face:
            rf_od = _snap_within(
                (bore + bcd - hole) / 2, R10, bore * 1.2, bcd - hole - 1.0
            ) or 0.0
            if rf_od:
                rf_height = 2.0

        has_counterbore = bool(rng.random() < 0.3)
        cbore_dia = 0.0
        cbore_depth = 0.0
        if has_counterbore:
            cbore_dia = snap_to_series(hole * 1.8, R20)
            cbore_depth = _snap_within(
                thickness * 0.35, R20, 1.0, thickness * 0.65
            ) or 0.0

        # The chamfer must fit in the wall it is cut into as well as in the thickness.
        chamfer = 1.0 if bore >= 20 else 0.5
        chamfer = min(chamfer, thickness / 3.0)

        return {
            "od": od,
            "bore": bore,
            "thickness": thickness,
            "bolt": bolt,
            "bolt_hole": hole,
            "bolt_count": bolt_count,
            "bcd": bcd,
            "hub_od": hub_od,
            "hub_height": hub_height,
            "rf_od": rf_od,
            "rf_height": rf_height,
            "cbore_dia": cbore_dia,
            "cbore_depth": cbore_depth,
            "bore_chamfer": chamfer,
            "bore_fit": str(rng.choice([f for f in COMMON_HOLE_FITS if f.startswith("H")])),
        }

    # -- manufacturability ----------------------------------------------------------

    @staticmethod
    def _check(p: dict[str, Any]) -> None:
        """Reject unmanufacturable proportions rather than clamping them.

        Clamping would keep the build rate up while quietly changing the distribution the
        parts are drawn from, and would put parts on the benchmark that an engineer would
        reject on sight -- which is exactly the credibility problem SPEC.md section 6.3
        warns about.
        """
        od, bore, bcd, hole = p["od"], p["bore"], p["bcd"], p["bolt_hole"]

        if bore >= od * 0.8:
            raise UnbuildableParams(f"bore {bore} leaves no rim inside an OD of {od}")

        edge = (od - bcd) / 2 - hole / 2
        if edge < hole * MIN_EDGE_DISTANCE_RATIO:
            raise UnbuildableParams(
                f"edge distance {edge:.1f} is under {MIN_EDGE_DISTANCE_RATIO} x hole "
                f"diameter {hole}"
            )

        wall = (bcd - bore) / 2 - hole / 2
        if wall < hole * MIN_BORE_WALL_RATIO:
            raise UnbuildableParams(
                f"bore wall {wall:.1f} is under {MIN_BORE_WALL_RATIO} x hole diameter {hole}"
            )

        pitch = math.pi * bcd / p["bolt_count"]
        if pitch < hole * MIN_HOLE_PITCH_RATIO:
            raise UnbuildableParams(
                f"{p['bolt_count']} holes on a {bcd} bolt circle leaves a pitch of "
                f"{pitch:.1f}, under {MIN_HOLE_PITCH_RATIO} x hole diameter {hole}"
            )

        if p["hub_od"] and p["hub_od"] >= bcd - hole:
            raise UnbuildableParams(
                f"hub OD {p['hub_od']} runs into the bolt circle at {bcd}"
            )
        if p["rf_od"] and p["rf_od"] >= bcd - hole:
            raise UnbuildableParams(
                f"raised face OD {p['rf_od']} runs into the bolt circle at {bcd}"
            )
        if p["cbore_depth"] and p["cbore_depth"] >= p["thickness"] * 0.7:
            raise UnbuildableParams(
                f"counterbore depth {p['cbore_depth']} leaves too little material under "
                f"a {p['thickness']} plate"
            )
        if p["bore_chamfer"] * 2 >= p["thickness"]:
            raise UnbuildableParams("bore chamfer consumes the whole thickness")

    # -- construction ---------------------------------------------------------------

    def build(self, params: dict[str, Any]) -> TopoDS_Shape:
        self._check(params)
        p = params

        body = cq.Workplane("XY").circle(p["od"] / 2).extrude(p["thickness"])

        if p["hub_od"]:
            body = (
                body.faces(">Z")
                .workplane()
                .circle(p["hub_od"] / 2)
                .extrude(p["hub_height"])
            )
        elif p["rf_od"]:
            body = (
                body.faces(">Z")
                .workplane()
                .circle(p["rf_od"] / 2)
                .extrude(p["rf_height"])
            )

        # The bore is cut last and through everything, so it is one cylindrical face
        # rather than a stack of coaxial ones -- which keeps the face index honest about
        # how many distinct bore surfaces exist.
        body = body.faces(">Z").workplane().circle(p["bore"] / 2).cutThruAll()

        holes = (
            cq.Workplane("XY")
            .polarArray(p["bcd"] / 2, 0, 360, p["bolt_count"])
            .circle(p["bolt_hole"] / 2)
            .extrude(p["thickness"] + p["hub_height"] + p["rf_height"] + 1)
        )
        body = body.cut(holes)

        if p["cbore_depth"]:
            cbores = (
                cq.Workplane("XY")
                .workplane(offset=p["thickness"] - p["cbore_depth"])
                .polarArray(p["bcd"] / 2, 0, 360, p["bolt_count"])
                .circle(p["cbore_dia"] / 2)
                .extrude(p["cbore_depth"] + 1)
            )
            body = body.cut(cbores)

        # Select the bore's mouth edges by radius rather than by proximity to a point.
        # A point selector picks whichever edge happens to be nearest, which on a flange
        # with a counterbore is often a counterbore rim -- chamfering that one both fails
        # more often and, when it succeeds, silently produces a different part than the
        # feature description claims.
        bore_r = p["bore"] / 2
        mouths = [
            e
            for e in body.edges().vals()
            if e.geomType() == "CIRCLE" and abs(e.radius() - bore_r) < 1e-6
        ]
        if not mouths:
            raise UnbuildableParams(f"no circular edge of radius {bore_r} to chamfer")
        try:
            body = body.newObject(mouths).chamfer(p["bore_chamfer"])
        except Exception as exc:  # noqa: BLE001 - OCCT raises many types for a bad blend
            raise UnbuildableParams(f"bore chamfer failed: {exc}") from exc

        return body.val().wrapped

    # -- description ----------------------------------------------------------------

    def describe(
        self, params: dict[str, Any], faces: FaceIndex
    ) -> tuple[SemanticFeature, ...]:
        """Name the features, locating every face by geometry rather than build order."""
        p = params
        total_h = p["thickness"] + p["hub_height"] + p["rf_height"]
        features: list[SemanticFeature] = []

        bore_faces = faces.cylinders_of_radius(p["bore"] / 2, tol=1e-3)
        if not bore_faces:
            raise UnbuildableParams(
                f"no cylindrical face of radius {p['bore'] / 2} survived the build; "
                f"nearest radii are {faces.nearest_cylinder_radii(p['bore'] / 2)}"
            )
        bore_fit = fit_limits(p["bore"], p["bore_fit"])
        features.append(
            SemanticFeature(
                fid="bore_main",
                kind="through_hole",
                faces=tuple(f.fid for f in bore_faces),
                nominal={
                    "diameter": p["bore"],
                    "depth": total_h,
                    "upper_tol": bore_fit.upper,
                    "lower_tol": bore_fit.lower,
                },
                anchor=(0.0, 0.0, p["thickness"] / 2),
                axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                meta={"fit": p["bore_fit"], "datum": "B"},
            )
        )

        # The mounting face is the flange's underside: the planar face at z == 0 with the
        # largest area. It is datum A on essentially every flange drawing, because it is
        # the face the part is bolted down on and therefore the face it is inspected on.
        bottom = [
            f
            for f in faces.planes_with_normal((0, 0, 1))
            if abs(f.centroid[2]) < 1e-6
        ]
        if not bottom:
            raise UnbuildableParams("no planar face found at z = 0 for the mounting face")
        mounting = max(bottom, key=lambda f: f.area)
        features.append(
            SemanticFeature(
                fid="mounting_face",
                kind="planar_face",
                faces=(mounting.fid,),
                nominal={"diameter": p["od"], "z": 0.0},
                anchor=(p["od"] / 2 * 0.7, 0.0, 0.0),
                axis=((0.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
                meta={"datum": "A"},
            )
        )

        hole_r = p["bolt_hole"] / 2
        hole_faces = faces.cylinders_of_radius(hole_r, tol=1e-3)
        if len(hole_faces) < p["bolt_count"]:
            raise UnbuildableParams(
                f"expected {p['bolt_count']} bolt holes of radius {hole_r}, the face "
                f"index found {len(hole_faces)}"
            )
        features.append(
            SemanticFeature(
                fid="bolt_pattern",
                kind="through_hole",
                faces=tuple(f.fid for f in hole_faces),
                nominal={
                    "diameter": p["bolt_hole"],
                    "bolt_circle": p["bcd"],
                    "count": float(p["bolt_count"]),
                    "depth": p["thickness"],
                },
                anchor=(p["bcd"] / 2, 0.0, p["thickness"] / 2),
                axis=((p["bcd"] / 2, 0.0, 0.0), (0.0, 0.0, 1.0)),
                meta={"pattern": "polar", "for_bolt": f"M{p['bolt']:g}"},
            )
        )

        if p["hub_od"]:
            hub_faces = faces.cylinders_of_radius(p["hub_od"] / 2, tol=1e-3)
            if hub_faces:
                features.append(
                    SemanticFeature(
                        fid="hub",
                        kind="boss",
                        faces=tuple(f.fid for f in hub_faces),
                        nominal={"diameter": p["hub_od"], "height": p["hub_height"]},
                        anchor=(p["hub_od"] / 2, 0.0, p["thickness"] + p["hub_height"] / 2),
                        axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    )
                )
        elif p["rf_od"]:
            rf_faces = faces.cylinders_of_radius(p["rf_od"] / 2, tol=1e-3)
            if rf_faces:
                features.append(
                    SemanticFeature(
                        fid="raised_face",
                        kind="boss",
                        faces=tuple(f.fid for f in rf_faces),
                        nominal={"diameter": p["rf_od"], "height": p["rf_height"]},
                        anchor=(p["rf_od"] / 2, 0.0, p["thickness"] + p["rf_height"] / 2),
                        axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    )
                )

        od_faces = faces.cylinders_of_radius(p["od"] / 2, tol=1e-3)
        if od_faces:
            features.append(
                SemanticFeature(
                    fid="outer_diameter",
                    kind="cylindrical_face",
                    faces=tuple(f.fid for f in od_faces),
                    nominal={"diameter": p["od"], "height": p["thickness"]},
                    anchor=(p["od"] / 2, 0.0, p["thickness"] / 2),
                    axis=((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                )
            )

        if p["cbore_depth"]:
            cb_faces = faces.cylinders_of_radius(p["cbore_dia"] / 2, tol=1e-3)
            if cb_faces:
                features.append(
                    SemanticFeature(
                        fid="bolt_counterbore",
                        kind="counterbore",
                        faces=tuple(f.fid for f in cb_faces),
                        nominal={
                            "diameter": p["cbore_dia"],
                            "depth": p["cbore_depth"],
                            "count": float(p["bolt_count"]),
                        },
                        anchor=(p["bcd"] / 2, 0.0, p["thickness"] - p["cbore_depth"] / 2),
                        axis=((p["bcd"] / 2, 0.0, 0.0), (0.0, 0.0, 1.0)),
                        parent="bolt_pattern",
                    )
                )

        return tuple(features)


registry.register(Flange())
