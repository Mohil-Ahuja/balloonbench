"""A queryable geometric index of a solid: what sizes and relationships actually exist.

SPEC.md section 12.1 calls this reusable and arguably a standalone contribution, and the
reason is that every check above it asks the same shape of question -- *is there really a
cylinder of that radius, a pair of faces that far apart, a bolt circle of that diameter?* --
and none of them should be re-deriving it from raw topology.

The index is built on ``partgen``'s :class:`~balloonbench.partgen.types.FaceIndex`, which
already classifies and measures every face with stable ids. What is added here is the
*relational* half: distances between parallel planes, distances and angles between axes,
coaxial clusters that form hole patterns, and the overall envelope. Those relations are what
a dimension on a drawing refers to. A drawing almost never dimensions a face; it dimensions
the distance between two of them.

Three principles run through this module.

**Measured, never assumed.** The index is built from the solid alone. It does not read
``partgen``'s semantic features, and it must not: the verifier's job is to check a drawing
against geometry, and geometry that came with a label saying what it was meant to be would
make the check circular. The same code therefore works on an imported STEP file nobody in
this repository generated, which is what SPEC.md section 12.5's stress test requires.

**Tolerant of bad input.** Real B-reps carry degenerate faces, tiny slivers and surfaces
OCCT cannot classify. Anything that fails to measure is skipped and counted, not raised: a
verifier that crashes on one malformed face in a thousand is not deployable.

**Distances, not identities.** Queries return candidates with their distances rather than a
yes or no, so a check can decide for itself whether a near miss is a match, an ambiguity, or
a contradiction. Folding that decision into the index would put the verifier's conservatism
in the wrong file.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen.types import FaceIndex, FaceInfo

__all__ = [
    "BrepIndex",
    "Candidate",
    "HolePattern",
    "PlanePair",
    "load_shape",
]

#: Two directions count as parallel when their dot product is this close to one. Loose
#: enough to survive the rounding a STEP round-trip introduces, tight enough that a five
#: degree misalignment is not called parallel.
PARALLEL_TOL = 1e-3

#: Below this area a face is a sliver: a seam artefact, a filleted corner remnant, or a
#: modelling error. They are indexed but excluded from relational queries, where they
#: produce distances that correspond to nothing a drawing would dimension.
MIN_FACE_AREA = 1e-6

#: Coarse pitches of the ISO metric thread series, which is what a drawing calls out unless
#: it says otherwise. Used to work out what a tapped hole looks like in the solid: the tap
#: cuts into a hole roughly one pitch smaller than the thread's stated diameter.
#:
#: A table rather than a ratio, and the difference matters. An earlier version accepted any
#: cylinder between 0.75 and 0.98 of the stated size as "a plausible drill", which let the
#: verifier excuse a genuine error -- a nominal misread as 125 was waved through because the
#: part had a ⌀100 face. Only a standard thread size gets the exemption now, and only for a
#: hole the right distance below it.
_COARSE_PITCH: dict[float, float] = {
    1.6: 0.35, 2.0: 0.4, 2.5: 0.45, 3.0: 0.5, 4.0: 0.7, 5.0: 0.8, 6.0: 1.0, 8.0: 1.25,
    10.0: 1.5, 12.0: 1.75, 14.0: 2.0, 16.0: 2.0, 18.0: 2.5, 20.0: 2.5, 22.0: 2.5,
    24.0: 3.0, 27.0: 3.0, 30.0: 3.5, 33.0: 3.5, 36.0: 4.0, 39.0: 4.0, 42.0: 4.5,
    45.0: 4.5, 48.0: 5.0, 52.0: 5.0, 56.0: 5.5, 60.0: 5.5, 64.0: 6.0,
}

#: How far the drilled hole may sit from the size the pitch table implies. Wide enough to
#: cover a fine pitch and the rounding to a stock drill, narrow enough that an unrelated
#: face cannot pretend to be a tap drill.
TAP_DRILL_SLOP = 0.3


@dataclass(frozen=True)
class Candidate:
    """One geometric feature that could be what a dimension refers to."""

    value: float
    kind: str
    refs: tuple[str, ...] = ()
    detail: str = ""

    def error(self, target: float) -> float:
        return abs(self.value - target)


@dataclass(frozen=True)
class PlanePair:
    """Two parallel planar faces and the perpendicular distance between them."""

    a: str
    b: str
    distance: float
    normal: tuple[float, float, float]


@dataclass(frozen=True)
class HolePattern:
    """Coaxial cylinders of equal radius arranged on a circle.

    Recovered from geometry rather than from the modelling parameters, so it also works on
    an imported solid. ``bolt_circle`` is a diameter, matching how a drawing states it.
    """

    radius: float
    count: int
    centre: tuple[float, float, float]
    axis: tuple[float, float, float]
    bolt_circle: float
    faces: tuple[str, ...]
    #: Whether the holes are equally spaced around the circle. Every rectangle's corners are
    #: equidistant from its centre, so distance alone calls a four-hole rectangular pattern a
    #: bolt circle. Only a uniform pattern has a bolt-circle diameter a drawing would state;
    #: a rectangular one is dimensioned by its pitches instead.
    uniform: bool = True


def load_shape(step_path: str | Path) -> TopoDS_Shape:
    """Read a STEP file into a single shape.

    Wrapped here so every caller gets the same error when a file will not read, rather than
    an OCCT status code that means nothing to a caller three layers up.
    """
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    status = reader.ReadFile(str(step_path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"could not read STEP file {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError(f"STEP file {step_path} contains no shape")
    return shape


@dataclass
class BrepIndex:
    """Every size and relationship the solid actually has."""

    faces: FaceIndex
    #: Faces that could not be measured or are too small to mean anything. Reported rather
    #: than hidden: the count is the robustness statistic SPEC.md section 12.5 asks for.
    skipped: tuple[str, ...] = ()
    _envelope: tuple[float, float, float] | None = field(default=None, repr=False)

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def from_shape(cls, shape: TopoDS_Shape) -> BrepIndex:
        faces = FaceIndex(shape)
        skipped = tuple(
            info.fid
            for info in faces
            if info.area < MIN_FACE_AREA or info.surface == "other"
        )
        index = cls(faces=faces, skipped=skipped)
        index._envelope = _envelope(shape)
        return index

    @classmethod
    def from_step(cls, step_path: str | Path) -> BrepIndex:
        return cls.from_shape(load_shape(step_path))

    # -- primitives -----------------------------------------------------------------------

    @property
    def usable(self) -> tuple[FaceInfo, ...]:
        skipped = set(self.skipped)
        return tuple(f for f in self.faces if f.fid not in skipped)

    @property
    def envelope(self) -> tuple[float, float, float]:
        """Overall bounding-box extents, largest first.

        Sorted because a drawing's overall dimensions are not labelled with axes, and the
        ``unit_sanity`` check compares magnitudes rather than orientations.
        """
        if self._envelope is None:
            return (0.0, 0.0, 0.0)
        return tuple(sorted(self._envelope, reverse=True))  # type: ignore[return-value]

    @property
    def cylinders(self) -> tuple[FaceInfo, ...]:
        return tuple(f for f in self.usable if f.is_cylindrical and f.radius)

    @property
    def planes(self) -> tuple[FaceInfo, ...]:
        return tuple(f for f in self.usable if f.is_planar and f.normal)

    @property
    def cones(self) -> tuple[FaceInfo, ...]:
        return tuple(f for f in self.usable if f.surface == "cone" and f.radius)

    @property
    def round_faces(self) -> tuple[FaceInfo, ...]:
        """Everything a diameter dimension can refer to: cylinders and cones alike.

        A cast wall with a couple of degrees of draft is a cone, and the drawing dimensions
        it by the diameter it was drawn at. Matching only cylinders contradicted every
        drafted feature -- the valve body's ⌀100 body wall among them.
        """
        return self.cylinders + self.cones

    @property
    def radii(self) -> tuple[float, ...]:
        return tuple(sorted({round(f.radius, 6) for f in self.round_faces if f.radius}))

    @property
    def diameters(self) -> tuple[float, ...]:
        return tuple(2 * r for r in self.radii)

    # -- relations ------------------------------------------------------------------------

    def plane_pairs(self) -> tuple[PlanePair, ...]:
        """Parallel planar face pairs and their separations.

        Only pairs that face each other are kept -- their normals antiparallel, or parallel
        with the second plane on the far side. Two parallel steps on the same side of a part
        have a difference of offsets that is a real number and not a thickness, and a
        dimension never refers to it.
        """
        out: list[PlanePair] = []
        planes = self.planes
        for i, a in enumerate(planes):
            na = _unit(a.normal)
            da = float(np.dot(na, a.centroid))
            for b in planes[i + 1 :]:
                nb = _unit(b.normal)
                alignment = float(np.dot(na, nb))
                if abs(abs(alignment) - 1.0) > PARALLEL_TOL:
                    continue
                db = float(np.dot(na, b.centroid))
                distance = abs(da - db)
                if distance < 1e-6:
                    continue
                out.append(
                    PlanePair(a=a.fid, b=b.fid, distance=distance, normal=tuple(na))
                )
        return tuple(out)

    def axis_distances(self) -> tuple[Candidate, ...]:
        """Perpendicular distances between parallel cylinder axes.

        This is what a hole-to-hole dimension refers to. Non-parallel axes are skipped: the
        distance between skew lines is well defined and is not something a drawing states.
        """
        out: list[Candidate] = []
        cylinders = [c for c in self.cylinders if c.axis]
        for i, a in enumerate(cylinders):
            pa, va = np.asarray(a.axis[0]), _unit(a.axis[1])
            for b in cylinders[i + 1 :]:
                pb, vb = np.asarray(b.axis[0]), _unit(b.axis[1])
                if abs(abs(float(np.dot(va, vb))) - 1.0) > PARALLEL_TOL:
                    continue
                delta = pb - pa
                perpendicular = delta - float(np.dot(delta, va)) * va
                distance = float(np.linalg.norm(perpendicular))
                if distance < 1e-6:
                    continue
                out.append(
                    Candidate(
                        value=distance,
                        kind="axis_distance",
                        refs=(a.fid, b.fid),
                        detail=f"axis-to-axis distance {distance:.3f}",
                    )
                )
        return tuple(out)

    def axis_to_plane_distances(self) -> tuple[Candidate, ...]:
        """Distances from a cylinder axis to the planes parallel to it.

        The commonest dimension on a plate: a hole centre to an edge. It is neither a
        plane pair nor an axis pair, so an index carrying only those two contradicts it --
        which it did, on every bracket, before this existed.
        """
        out: list[Candidate] = []
        for cylinder in self.cylinders:
            if not cylinder.axis:
                continue
            point, direction = np.asarray(cylinder.axis[0]), _unit(cylinder.axis[1])
            for plane in self.planes:
                normal = _unit(plane.normal)
                # Only planes the axis runs along: a plane the axis pierces has no single
                # distance to it.
                if abs(float(np.dot(direction, normal))) > PARALLEL_TOL:
                    continue
                distance = abs(float(np.dot(point - np.asarray(plane.centroid), normal)))
                if distance < 1e-6:
                    continue
                out.append(
                    Candidate(
                        value=distance,
                        kind="axis_to_plane",
                        refs=(cylinder.fid, plane.fid),
                        detail=f"from the axis of {cylinder.fid} to {plane.fid}",
                    )
                )
        return tuple(out)

    def hole_patterns(self, *, min_count: int = 3) -> tuple[HolePattern, ...]:
        """Coaxial-direction clusters of equal cylinders lying on a circle.

        Grouped by radius and axis direction, then tested for a common centre: the axes must
        be equidistant from their own centroid, which is what makes a ring of holes a bolt
        circle rather than a scatter. Recovering it geometrically means the check also works
        on a solid this repository did not build.
        """
        groups: dict[tuple, list[FaceInfo]] = defaultdict(list)
        for cylinder in self.cylinders:
            if not cylinder.axis:
                continue
            direction = _unit(cylinder.axis[1])
            # Direction and its negative are the same axis, so the key is canonicalised.
            if direction[np.argmax(np.abs(direction))] < 0:
                direction = -direction
            groups[(round(cylinder.radius or 0.0, 4), *np.round(direction, 4))].append(
                cylinder
            )

        patterns: list[HolePattern] = []
        for key, members in groups.items():
            if len(members) < min_count:
                continue
            direction = np.asarray(key[1:], dtype=float)
            points = np.asarray([_unit_point(m, direction) for m in members])
            centre = points.mean(axis=0)
            distances = np.linalg.norm(points - centre, axis=1)
            spread = float(distances.std())
            mean_radius = float(distances.mean())
            if mean_radius < 1e-6 or spread > 0.02 * mean_radius:
                continue
            patterns.append(
                HolePattern(
                    radius=float(key[0]),
                    count=len(members),
                    centre=tuple(float(v) for v in centre),
                    axis=tuple(float(v) for v in direction),
                    bolt_circle=2 * mean_radius,
                    faces=tuple(sorted(m.fid for m in members)),
                    uniform=_equally_spaced(points - centre, direction),
                )
            )
        return tuple(sorted(patterns, key=lambda p: (-p.count, p.radius)))

    # -- queries --------------------------------------------------------------------------

    def diameter_candidates(self, target: float, band: float) -> tuple[Candidate, ...]:
        """Diameters within ``band`` of ``target``: cylindrical faces and bolt circles.

        Grouped by radius rather than returned per face. Six bolt holes of the same size are
        one answer to "is there a ⌀14 here?", and returning six would make every patterned
        feature look ambiguous when it is not.

        Bolt circles are included because a drawing states one with a diameter symbol like
        any other -- it is the ⌀ of a circle through the hole centres rather than of a face,
        and a check that only looked at faces contradicted every flange's bolt circle.
        """
        by_radius: dict[float, list[str]] = defaultdict(list)
        for cylinder in self.round_faces:
            if abs(2 * (cylinder.radius or 0.0) - target) <= band:
                by_radius[round(cylinder.radius or 0.0, 6)].append(cylinder.fid)
        out = [
            Candidate(
                value=2 * radius,
                kind="diameter",
                refs=tuple(sorted(fids)),
                detail=f"⌀{2 * radius:.3f} on {len(fids)} face(s)",
            )
            for radius, fids in sorted(by_radius.items())
        ]
        for pattern in self.hole_patterns():
            if pattern.uniform and abs(pattern.bolt_circle - target) <= band:
                out.append(
                    Candidate(
                        value=pattern.bolt_circle,
                        kind="bolt_circle",
                        refs=pattern.faces,
                        detail=(
                            f"bolt circle ⌀{pattern.bolt_circle:.3f} through "
                            f"{pattern.count} holes"
                        ),
                    )
                )
        return tuple(out)

    def tap_drill_for(self, thread_diameter: float) -> tuple[Candidate, ...]:
        """Cylinders that could be the drilled hole under a thread of this size.

        A threaded hole is modelled as the hole the tap goes into, so a solid built from a
        drawing that says ⌀14 contains a cylinder somewhere near ⌀12. Without this the
        verifier contradicts every tapped hole on every drawing -- which it did, on the
        housing family, before this existed.
        """
        pitch = _COARSE_PITCH.get(round(thread_diameter, 3))
        if pitch is None:
            return ()
        # A fine pitch cuts a shallower thread, so the hole is larger. The window spans both.
        low = thread_diameter - pitch - TAP_DRILL_SLOP
        high = thread_diameter - 0.5 * pitch + TAP_DRILL_SLOP
        return tuple(
            Candidate(
                value=2 * (c.radius or 0.0),
                kind="tap_drill",
                refs=(c.fid,),
                detail=(
                    f"⌀{2 * (c.radius or 0.0):.3f}, the drill for an M{thread_diameter:g} "
                    f"thread of pitch {pitch:g}"
                ),
            )
            for c in self.round_faces
            if low <= 2 * (c.radius or 0.0) <= high
        )

    def radius_candidates(self, target: float, band: float) -> tuple[Candidate, ...]:
        return tuple(
            Candidate(
                value=candidate.value / 2,
                kind="radius",
                refs=candidate.refs,
                detail=f"R{candidate.value / 2:.3f} on {len(candidate.refs)} face(s)",
            )
            for candidate in self.diameter_candidates(2 * target, 2 * band)
        )

    def length_candidates(self, target: float, band: float) -> tuple[Candidate, ...]:
        """Everything in the solid that is ``target`` long, from any of three sources.

        A linear dimension on a drawing can mean a wall thickness, a hole spacing or an
        overall extent, and the drawing does not say which. All three are searched, and
        distinct values are grouped so that a symmetric part does not look ambiguous merely
        for having the same distance in several places.
        """
        found: dict[float, Candidate] = {}

        def offer(candidate: Candidate) -> None:
            if candidate.error(target) > band:
                return
            key = round(candidate.value, 4)
            existing = found.get(key)
            if existing is None:
                found[key] = candidate
            else:
                found[key] = Candidate(
                    value=existing.value,
                    kind=existing.kind if existing.kind == candidate.kind else "mixed",
                    refs=tuple(sorted(set(existing.refs) | set(candidate.refs))),
                    detail=existing.detail,
                )

        for pair in self.plane_pairs():
            offer(
                Candidate(
                    value=pair.distance,
                    kind="plane_pair",
                    refs=(pair.a, pair.b),
                    detail=f"between {pair.a} and {pair.b}",
                )
            )
        for candidate in self.axis_distances():
            offer(candidate)
        for candidate in self.axis_to_plane_distances():
            offer(candidate)
        for axis, extent in zip("xyz", self._envelope or (), strict=False):
            offer(
                Candidate(
                    value=extent,
                    kind="envelope",
                    refs=(),
                    detail=f"overall extent along {axis}",
                )
            )
        for pattern in self.hole_patterns():
            if not pattern.uniform:
                continue
            offer(
                Candidate(
                    value=pattern.bolt_circle,
                    kind="bolt_circle",
                    refs=pattern.faces,
                    detail=f"bolt circle of {pattern.count} holes",
                )
            )
        return tuple(sorted(found.values(), key=lambda c: c.error(target)))

    def nearest_values(self, target: float, kind: str, count: int = 3) -> tuple[float, ...]:
        """The closest values of a given kind, for explaining a contradiction.

        A verdict of "contradicted" is far more useful with the near misses attached: the
        difference between "no ⌀22 here" and "no ⌀22 here; the part has ⌀20 and ⌀25" is the
        difference between a complaint and a suggested correction.
        """
        if kind == "diameter":
            values = list(self.diameters)
        elif kind == "radius":
            values = list(self.radii)
        else:
            values = sorted(
                {round(p.distance, 4) for p in self.plane_pairs()}
                | {round(c.value, 4) for c in self.axis_distances()}
                | {round(c.value, 4) for c in self.axis_to_plane_distances()}
                | {round(v, 4) for v in (self._envelope or ())}
            )
        return tuple(sorted(values, key=lambda v: abs(v - target))[:count])

    def stats(self) -> dict[str, Any]:
        """Robustness statistics for the stress test of SPEC.md section 12.5."""
        return {
            "faces": len(self.faces),
            "usable": len(self.usable),
            "skipped": len(self.skipped),
            "cylinders": len(self.cylinders),
            "cones": len(self.cones),
            "planes": len(self.planes),
            "plane_pairs": len(self.plane_pairs()),
            "hole_patterns": len(self.hole_patterns()),
            "envelope": self.envelope,
        }


# --- helpers ---------------------------------------------------------------------------


def _unit(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(v))
    return v if norm < 1e-12 else v / norm


def _unit_point(face: FaceInfo, direction: np.ndarray) -> np.ndarray:
    """A cylinder's axis position projected onto the plane perpendicular to ``direction``."""
    point = np.asarray(face.axis[0], dtype=float)
    return point - float(np.dot(point, direction)) * direction


def _equally_spaced(offsets: np.ndarray, direction: np.ndarray) -> bool:
    """Whether points around a centre sit at equal angular intervals."""
    reference = _unit(np.cross(direction, [1.0, 0.0, 0.0]))
    if float(np.linalg.norm(reference)) < 1e-9:
        reference = _unit(np.cross(direction, [0.0, 1.0, 0.0]))
    other = np.cross(direction, reference)
    angles = np.sort(
        np.arctan2(offsets @ other, offsets @ reference) % (2 * math.pi)
    )
    if len(angles) < 2:
        return True
    steps = np.diff(np.append(angles, angles[0] + 2 * math.pi))
    expected = 2 * math.pi / len(angles)
    return bool(np.max(np.abs(steps - expected)) < math.radians(5))


def _envelope(shape: TopoDS_Shape) -> tuple[float, float, float]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box, True)
    if box.IsVoid():
        return (0.0, 0.0, 0.0)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    extents = (xmax - xmin, ymax - ymin, zmax - zmin)
    return tuple(0.0 if math.isinf(e) or math.isnan(e) else e for e in extents)
