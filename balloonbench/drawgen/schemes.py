"""What to say about a part: turning semantic features into callouts.

:mod:`balloonbench.drawgen.annotate` knows how to draw a dimension and where to put it.
This module decides *which* dimensions and geometric tolerances a part should carry, which
is a question about manufacturing rather than about layout, and the two are kept apart so
that a change to either cannot quietly corrupt the other.

The rules are generic wherever a convention is generic -- a datum feature gets a datum
symbol, a hole pattern gets a count-prefixed note and a positional tolerance, a primary
planar datum gets flatness -- and family-specific only where a family genuinely differs. A
shaft's journals carry runout to a common A-B axis; a flange's bolt pattern is located to
its mounting face and bore; a cast valve body carries a general tolerance note over its
unmachined surfaces and individual callouts only where metal was removed. Encoding that in
one generic engine with escapes would have been shorter and would have produced drawings
that are all subtly the same.

Every characteristic emitted here is true of the solid by construction: the numbers come
from the parameters the part was built from, not from measuring the model back. ASME Y14.5
and ISO 1101 conventions are implemented from their public description; no standard text,
table or figure is reproduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from balloonbench.drawgen.annotate import (
    Annotation,
    Primitive,
    datum_symbol,
    diameter_leader,
    feature_frame,
    linear_dimension,
    note_leader,
)
from balloonbench.drawgen.styles import HouseStyle
from balloonbench.drawgen.symbols import (
    DIM_GLYPH,
    feature_control_frame,
    format_value,
    frame_compartments,
    tolerance_text,
)
from balloonbench.drawgen.views import SheetLayout, ViewPlacement
from balloonbench.partgen.types import BuiltPart, SemanticFeature

__all__ = ["DatumRecord", "plan_drawing", "sheet_decorations"]


@dataclass(frozen=True)
class DatumRecord:
    """A datum as the schema records it, plus where its symbol landed."""

    label: str
    feature_type: str
    view: str
    geometry_ref: str | None
    annotation: Annotation


# --- view selection ---------------------------------------------------------------------


def _axis_alignment(placement: ViewPlacement, direction) -> float:
    """|cos| between a model-space direction and the view direction.

    1 means the direction points at the viewer, so a cylinder about it reads as a circle.
    0 means it lies in the view plane, so the cylinder reads as two parallel lines.
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    return abs(float(np.dot(d, placement.view.transform.direction)))


def _circular_view(layout: SheetLayout, axis) -> ViewPlacement | None:
    """The view in which a feature about ``axis`` appears as a circle, if any."""
    best = max(layout.placements, key=lambda p: _axis_alignment(p, axis))
    return best if _axis_alignment(best, axis) > 0.99 else None


def _longitudinal_view(layout: SheetLayout, axis) -> ViewPlacement | None:
    """The view in which a feature about ``axis`` appears edge-on."""
    best = min(layout.placements, key=lambda p: _axis_alignment(p, axis))
    return best if _axis_alignment(best, axis) < 0.01 else None


def _edge_on_view(layout: SheetLayout, normal) -> ViewPlacement | None:
    """The view in which a planar face with ``normal`` is seen as a line.

    A datum symbol attaches to the *edge view* of a surface, never to its face view:
    attaching it to the face would say the datum is the outline rather than the plane.
    """
    return _longitudinal_view(layout, normal)


def _across_axis(
    placement: ViewPlacement,
    point: tuple[float, float, float],
    axis_dir,
    radius: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The two silhouette points of a cylinder, in sheet millimetres.

    Computed in view space rather than by picking two 3D points on the surface: the
    silhouette of a cylinder in an orthographic view is at plus and minus the radius
    perpendicular to the *projected* axis, which is only the same as two surface points
    when the axis happens to lie in the view plane.
    """
    centre = placement.point3d(point)
    a2 = placement.view.transform.direction_2d(tuple(axis_dir))
    norm = math.hypot(*a2)
    # A norm of zero means the axis points at the viewer, so the cylinder is a circle and has
    # no silhouette pair; the sheet x direction is then as good a choice as any.
    n2 = (1.0, 0.0) if norm < 1e-9 else (-a2[1] / norm, a2[0] / norm)
    r = radius * placement.scale
    return (
        (centre[0] - n2[0] * r, centre[1] - n2[1] * r),
        (centre[0] + n2[0] * r, centre[1] + n2[1] * r),
    )


# --- text and payload -------------------------------------------------------------------


def _dim_payload(
    dim_type: str,
    nominal: float,
    *,
    upper: float | None = None,
    lower: float | None = None,
    tol_style: str | None = None,
    fit_class: str | None = None,
    is_basic: bool = False,
    is_critical: bool = False,
    raw_text: str = "",
) -> dict[str, Any]:
    return {
        "kind": "dimension",
        "dim_type": dim_type,
        "nominal": nominal,
        "upper_tol": upper,
        "lower_tol": lower,
        "tol_style": tol_style,
        "fit_class": fit_class,
        "is_basic": is_basic,
        "is_critical": is_critical,
        "raw_text": raw_text,
    }


def _size_text_and_payload(
    feature: SemanticFeature,
    nominal: float,
    style: HouseStyle,
    rng: np.random.Generator,
    *,
    dim_type: str = "diameter",
    prefix: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Choose how this dimension states its tolerance, and build the matching payload.

    The tolerance itself is whatever the part was built with. Only its *presentation* is
    sampled -- so a limit-form and a bilateral rendering of the same bore describe the same
    part, and a model that reads one correctly and the other wrongly is being measured on
    exactly the thing the benchmark is for.
    """
    glyph = (
        DIM_GLYPH["diameter"]
        if prefix is None and dim_type == "diameter"
        else (prefix or "")
    )
    fit = feature.meta.get("fit")
    upper = feature.nominal.get("upper_tol")
    lower = feature.nominal.get("lower_tol")

    # From here on the numbers are in the sheet's units, not the model's millimetres.
    nominal = style.length(nominal)
    upper = None if upper is None else style.length(upper)
    lower = None if lower is None else style.length(lower)

    allowed: list[str] = []
    if fit:
        allowed.append("fit")
    if upper is not None and lower is not None:
        allowed.extend(["bilateral", "unilateral", "limit"])
    if not allowed:
        # Untoleranced: the general note on the title block governs it. That is how most
        # dimensions on a real drawing are toleranced, which is why it is the default
        # rather than an afterthought.
        text = glyph + format_value(
            nominal, style.decimals, trailing_zeros=style.trailing_zeros
        )
        return text, _dim_payload(
            dim_type, nominal, upper=None, lower=None, tol_style="general", raw_text=text
        )

    chosen = style.pick_tolerance_style(rng, tuple(allowed))
    if chosen == "fit":
        text = tolerance_text(
            nominal, upper, lower, "fit", decimals=style.decimals,
            trailing_zeros=style.trailing_zeros, prefix=glyph, fit_class=fit,
        )
        return text, _dim_payload(
            dim_type, nominal, upper=upper, lower=lower, tol_style="fit",
            fit_class=fit, raw_text=text,
        )

    text = tolerance_text(
        nominal, upper, lower, chosen, decimals=style.decimals,
        tol_decimals=style.tol_decimals, trailing_zeros=style.trailing_zeros, prefix=glyph,
    )
    return text, _dim_payload(
        dim_type, nominal, upper=upper, lower=lower, tol_style=chosen, raw_text=text
    )


def _gtol(
    symbol: str,
    value: float,
    style: HouseStyle,
    *,
    diametral: bool = False,
    modifier: str | None = None,
    refs: tuple[tuple[str, str | None], ...] = (),
) -> tuple[tuple[str, ...], str, dict[str, Any]]:
    """Compartments, transcribed text, and payload for one feature control frame."""
    zone_prefix = DIM_GLYPH["diameter"] if (diametral and style.zone_prefix) else ""
    value = style.length(value)
    kwargs = {
        "zone_prefix": zone_prefix,
        "material_modifier": modifier,
        "datum_refs": refs,
        "decimals": style.gtol_decimals,
        "trailing_zeros": style.trailing_zeros,
    }
    compartments = frame_compartments(symbol, value, **kwargs)
    text = feature_control_frame(symbol, value, **kwargs)
    payload = {
        "kind": "geometric_tolerance",
        "gtol_symbol": symbol,
        "gtol_value": value,
        # The zone is a property of the tolerance, not of how the sheet writes it. A
        # legacy style that omits the diameter symbol still specifies a cylindrical zone,
        # and recording it as linear because the glyph is absent would make ground truth
        # agree with a misreading instead of with the part.
        "gtol_zone": "diametral" if diametral else "linear",
        "material_modifier": modifier,
        "datum_refs": [
            {"label": label, "modifier": mod} for label, mod in refs
        ],
        "raw_text": text,
    }
    return compartments, text, payload


# --- shared rules -----------------------------------------------------------------------


def _datum_feature_type(feature: SemanticFeature) -> str:
    if feature.kind in (
        "through_hole", "blind_hole", "cylindrical_face", "boss", "counterbore",
    ):
        return "cylindrical_feature"
    if feature.kind in ("slot", "keyway", "groove"):
        return "width"
    return "planar_face"


def _plan_datums(
    part: BuiltPart, layout: SheetLayout, style: HouseStyle
) -> list[DatumRecord]:
    """A datum symbol for every feature the family marked as a datum."""
    records: list[DatumRecord] = []
    for feature in part.features:
        label = feature.meta.get("datum")
        if not label:
            continue
        feature_type = _datum_feature_type(feature)

        if feature.axis is not None:
            placement = _longitudinal_view(layout, feature.axis[1]) or layout.placements[0]
        else:
            normal = _feature_normal(part, feature)
            placement = (
                (_edge_on_view(layout, normal) if normal is not None else None)
                or layout.placements[0]
            )
        tip = _surface_anchor(placement, feature)
        ann = datum_symbol(
            tip,
            label,
            style,
            view=placement.name,
            payload={"label": label, "feature_type": feature_type},
        )
        records.append(
            DatumRecord(
                label=label,
                feature_type=feature_type,
                view=placement.name,
                geometry_ref=feature.fid,
                annotation=ann,
            )
        )
    return sorted(records, key=lambda r: r.label)


#: Feature kinds that are internal -- material is removed to make them. A hole is clearest
#: in the view that shows it as a circle; an external cylinder is clearest in the view that
#: shows its silhouette, where a diameter can be dimensioned across it.
_INTERNAL_KINDS = frozenset({"through_hole", "blind_hole", "counterbore", "groove"})


def _axis_foot(feature: SemanticFeature) -> tuple[float, float, float]:
    """The point on the feature's axis level with its recorded anchor.

    A cylinder's ``axis`` location is wherever OCCT placed the surface's origin -- for a
    shaft step, the base of the step. Spanning a diameter there rather than at the feature's
    own axial station draws the dimension at the wrong place along the shaft, and on a
    stepped part that is a different step's diameter drawn across the wrong outline. Taking
    the foot of the anchor on the axis puts it back where the feature is.
    """
    if feature.axis is None:
        return feature.anchor
    origin = np.asarray(feature.axis[0], dtype=float)
    d = np.asarray(feature.axis[1], dtype=float)
    d = d / np.linalg.norm(d)
    a = np.asarray(feature.anchor, dtype=float)
    foot = origin + float(np.dot(a - origin, d)) * d
    return (float(foot[0]), float(foot[1]), float(foot[2]))


def _surface_anchor(
    placement: ViewPlacement, feature: SemanticFeature
) -> tuple[float, float]:
    """Where a leader for ``feature`` should touch, in sheet millimetres.

    A cylindrical feature's recorded anchor may sit on its axis, which in a longitudinal
    view is a point *inside solid material* for a boss and inside empty space for a bore.
    Either way a leader drawn to it points at nothing a machinist would recognise. Offsetting
    to the silhouette puts the arrow on the surface the callout is actually about, which is
    what the convention requires and what makes the sheet readable.
    """
    if feature.axis is None or "diameter" not in feature.nominal:
        return placement.point3d(feature.anchor)
    a, b = _across_axis(
        placement, _axis_foot(feature), feature.axis[1], feature.nominal["diameter"] / 2
    )
    base = placement.point3d(feature.anchor)
    # Keep the silhouette point nearest the recorded anchor, so a feature whose anchor
    # already sits on the surface does not jump to the far side of the part.
    return min((a, b), key=lambda p: math.dist(p, base))


def _feature_normal(part: BuiltPart, feature: SemanticFeature):
    for fid in feature.faces:
        info = part.faces.info(fid)
        if info.normal is not None:
            return info.normal
    return None


def _size_annotation(
    part: BuiltPart,
    layout: SheetLayout,
    style: HouseStyle,
    rng: np.random.Generator,
    feature: SemanticFeature,
    *,
    importance: float = 1.0,
    prefer_leader: bool = False,
) -> Annotation | None:
    """A diameter callout for one cylindrical feature, in whichever view suits it."""
    diameter = feature.nominal.get("diameter")
    if diameter is None:
        return None
    axis = feature.axis[1] if feature.axis else (0.0, 0.0, 1.0)
    text, payload = _size_text_and_payload(feature, diameter, style, rng)

    longitudinal = _longitudinal_view(layout, axis)
    circular = _circular_view(layout, axis)

    # A hole is dimensioned where it reads as a circle; an external cylinder where its
    # silhouette can be spanned. Preferring the longitudinal view for everything put a
    # plate's through hole on the plate's 3 mm edge view, where the hole is two hidden
    # lines and the dimension means nothing to a reader.
    order = (
        (circular, longitudinal)
        if feature.kind in _INTERNAL_KINDS
        else (longitudinal, circular)
    )
    for placement in order:
        if placement is None:
            continue
        if placement is longitudinal:
            if prefer_leader:
                # A spanning dimension line is offset perpendicular to what it measures, so
                # on a stepped shaft every diameter has to clear the whole elevation and
                # they stack up in a column at one end, each trailing an extension line the
                # length of the part. A leader to the step's own surface is both what a
                # turned-part drawing actually uses and local to the feature.
                return note_leader(
                    _surface_anchor(placement, feature), text, style,
                    view=placement.name, kind="dimension", payload=payload,
                    importance=importance,
                    target_box=None,
                )
            a, b = _across_axis(placement, _axis_foot(feature), axis, diameter / 2)
            return linear_dimension(
                a, b, text, style, view=placement.name, payload=payload,
                importance=importance,
            )
        centre = placement.point3d(_axis_foot(feature))
        return diameter_leader(
            centre, diameter / 2 * placement.scale, text, style,
            view=placement.name, payload=payload, importance=importance,
        )
    return None


def _pattern_annotations(
    layout: SheetLayout,
    style: HouseStyle,
    feature: SemanticFeature,
    datums: list[DatumRecord],
) -> list[Annotation]:
    """A hole pattern: the count-prefixed size note, its location, and its position tolerance.

    The location dimensions are **basic** -- theoretically exact -- because a positional
    tolerance defines the zone and a toleranced location dimension alongside it would
    doubly constrain the same thing. That is the single most common error in hand-built
    GD&T, and emitting it correctly is part of what makes this a reference set rather than
    just a pile of images.
    """
    out: list[Annotation] = []
    count = int(feature.nominal.get("count", 1))
    diameter = feature.nominal["diameter"]
    axis = feature.axis[1] if feature.axis else (0.0, 0.0, 1.0)
    circular = _circular_view(layout, axis)
    if circular is None:
        return out

    depth_note = " THRU" if feature.kind == "through_hole" else ""
    size = format_value(
        style.length(diameter), style.decimals, trailing_zeros=style.trailing_zeros
    )
    note = f"{count}× {DIM_GLYPH['diameter']}{size}{depth_note}"
    tip = circular.point3d(feature.anchor)
    out.append(
        note_leader(
            tip, note, style, view=circular.name, kind="dimension",
            payload=_dim_payload(
                "diameter", style.length(diameter), tol_style="general", raw_text=note,
            ) | {"notes": f"pattern of {count}"},
            importance=2.5,
        )
    )

    # The pattern's location, as a basic bolt-circle diameter or basic pitches.
    bolt_circle = feature.nominal.get("bolt_circle")
    if bolt_circle:
        # The bolt circle is concentric with the part, not with any one hole. Using the
        # pattern feature's own axis puts the circle's centre on a hole, so the callout
        # describes a circle of the right diameter in the wrong place -- far enough off,
        # on a large flange, to hang past the edge of the sheet.
        centre = circular.point3d((0.0, 0.0, feature.anchor[2]))
        bc_text = DIM_GLYPH["diameter"] + format_value(
            style.length(bolt_circle), style.decimals,
            trailing_zeros=style.trailing_zeros,
        )
        out.append(
            diameter_leader(
                centre, bolt_circle / 2 * circular.scale, bc_text, style,
                view=circular.name,
                payload=_dim_payload(
                    "diameter", style.length(bolt_circle), upper=0.0, lower=0.0,
                    tol_style="basic", is_basic=True, raw_text=bc_text,
                ),
                importance=2.0,
            )
        )

    refs = _position_refs(datums)
    if refs:
        modifier = "MMC" if _refs_allow_modifier(datums, refs) else None
        compartments, text, payload = _gtol(
            "position", _position_tolerance(diameter), style,
            diametral=True, modifier=modifier, refs=refs,
        )
        out.append(
            feature_frame(
                tip, compartments, text, style, view=circular.name,
                payload=payload, importance=3.5,
            )
        )
    return out


def _position_tolerance(diameter: float) -> float:
    """A plausible positional zone for a clearance hole of this size.

    Scaled with the hole because a positional zone that does not grow with the clearance is
    either unachievable on a big hole or meaninglessly loose on a small one. The constant is
    a workshop-typical fraction, not a value from any standard.
    """
    return round(max(0.1, min(0.6, diameter * 0.03)), 2)


def _position_refs(datums: list[DatumRecord]) -> tuple[tuple[str, str | None], ...]:
    """Up to three datums in precedence order, as a positional tolerance references them."""
    return tuple((d.label, None) for d in datums[:3])


def _refs_allow_modifier(
    datums: list[DatumRecord], refs: tuple[tuple[str, str | None], ...]
) -> bool:
    """Whether the *toleranced feature* may carry MMC given the referenced datums.

    A material condition modifier on the feature is always legitimate for a hole. This
    checks the separate question the schema's R3 asks about datum references, so the caller
    can decide once rather than discovering a validation error after the sheet is drawn.
    """
    return bool(refs)


# --- family plans -------------------------------------------------------------------------


def _plan_flange(part, layout, style, rng, datums) -> list[Annotation]:
    out: list[Annotation] = []
    by_id = {f.fid: f for f in part.features}

    for fid, importance in (("bore_main", 3.0), ("hub", 1.5), ("outer_diameter", 2.0)):
        feature = by_id.get(fid)
        if feature is not None:
            ann = _size_annotation(part, layout, style, rng, feature, importance=importance)
            if ann is not None:
                out.append(ann)

    pattern = by_id.get("bolt_pattern")
    if pattern is not None:
        out.extend(_pattern_annotations(layout, style, pattern, datums))

    face = by_id.get("mounting_face")
    if face is not None and datums:
        section = _longitudinal_view(layout, (0.0, 0.0, 1.0)) or layout.placements[-1]
        tip = section.point3d(face.anchor)
        compartments, text, payload = _gtol("flatness", 0.05, style)
        out.append(
            feature_frame(tip, compartments, text, style, view=section.name,
                          payload=payload, importance=2.8)
        )

    bore = by_id.get("bore_main")
    primary = next((d for d in datums if d.feature_type == "planar_face"), None)
    if bore is not None and primary is not None:
        section = _longitudinal_view(layout, (0.0, 0.0, 1.0)) or layout.placements[-1]
        tip = _surface_anchor(section, bore)
        compartments, text, payload = _gtol(
            "perpendicularity", 0.05, style, diametral=True, modifier=None,
            refs=((primary.label, None),),
        )
        out.append(
            feature_frame(tip, compartments, text, style, view=section.name,
                          payload=payload, importance=3.0)
        )

    # Overall thickness, measured in the section where the two end faces are edge-on.
    hub = by_id.get("hub")
    od = by_id.get("outer_diameter")
    if od is not None:
        section = _longitudinal_view(layout, (0.0, 0.0, 1.0))
        height = od.nominal.get("height")
        if section is not None and height:
            r = od.nominal["diameter"] / 2
            a = section.point3d((r, 0.0, 0.0))
            b = section.point3d((r, 0.0, height))
            text, payload = _size_text_and_payload(
                od, height, style, rng, dim_type="linear", prefix=""
            )
            out.append(
                linear_dimension(a, b, text, style, view=section.name,
                                 payload=payload, horizontal=False, importance=1.8)
            )
    if hub is not None:
        section = _longitudinal_view(layout, (0.0, 0.0, 1.0))
        height = hub.nominal.get("height")
        if section is not None and height:
            r = hub.nominal["diameter"] / 2
            base = od.nominal.get("height", 0.0) if od else 0.0
            a = section.point3d((r, 0.0, base))
            b = section.point3d((r, 0.0, base + height))
            text, payload = _size_text_and_payload(
                hub, height, style, rng, dim_type="linear", prefix=""
            )
            out.append(
                linear_dimension(a, b, text, style, view=section.name,
                                 payload=payload, horizontal=False, importance=1.4)
            )
    return out


def _plan_shaft(part, layout, style, rng, datums) -> list[Annotation]:
    out: list[Annotation] = []
    steps = [f for f in part.features if f.fid.startswith("step_")]
    elevation = _longitudinal_view(layout, (0.0, 0.0, 1.0)) or layout.placements[0]

    for feature in steps:
        ann = _size_annotation(
            part, layout, style, rng, feature,
            importance=2.5 if feature.meta.get("datum") else 1.2,
            prefer_leader=True,
        )
        if ann is not None:
            out.append(ann)

    # Step lengths, chained along the axis. A chain rather than a set of dimensions from a
    # common origin because that is how a turned part is actually made -- each step is cut
    # to a length -- and the two forms stack tolerance differently, which the verifier
    # (SPEC.md section 12) later has to reason about.
    z = 0.0
    for feature in steps:
        length = feature.nominal.get("length")
        if not length:
            continue
        r = feature.nominal["diameter"] / 2
        a = elevation.point3d((r, 0.0, z))
        b = elevation.point3d((r, 0.0, z + length))
        shown = style.length(length)
        text = format_value(shown, style.decimals, trailing_zeros=style.trailing_zeros)
        out.append(
            linear_dimension(
                a, b, text, style, view=elevation.name,
                payload=_dim_payload("linear", shown, tol_style="general", raw_text=text),
                importance=1.0,
            )
        )
        z += length

    # Runout on the intermediate steps, referenced to the two journals as a common axis.
    journals = [d for d in datums if d.feature_type == "cylindrical_feature"]
    if len(journals) >= 2:
        refs = tuple((d.label, None) for d in journals[:2])
        for feature in steps:
            if feature.meta.get("datum") or "diameter" not in feature.nominal:
                continue
            tip = _surface_anchor(elevation, feature)
            compartments, text, payload = _gtol(
                "circular_runout", 0.03, style, refs=refs
            )
            out.append(
                feature_frame(tip, compartments, text, style, view=elevation.name,
                              payload=payload, importance=2.6)
            )
            break  # one runout callout is representative; a frame per step is clutter

    keyway = next((f for f in part.features if f.kind == "keyway"), None)
    if keyway is not None:
        width = keyway.nominal.get("width")
        depth = keyway.nominal.get("depth")
        if width:
            trail = style.trailing_zeros
            end = _circular_view(layout, (0.0, 0.0, 1.0))
            target = end or elevation
            tip = target.point3d(keyway.anchor)
            text = format_value(
                style.length(width), style.decimals, trailing_zeros=style.trailing_zeros
            )
            note = f"KEYWAY {text} WIDE"
            if depth:
                shown = format_value(
                    style.length(depth), style.decimals, trailing_zeros=trail
                )
                note += f" × {shown} DEEP"
            out.append(
                note_leader(
                    tip, note, style, view=target.name, kind="note",
                    payload={"kind": "note", "raw_text": note},
                    importance=1.6,
                )
            )
    return out


def _plan_plate(part, layout, style, rng, datums) -> list[Annotation]:
    out: list[Annotation] = []
    by_id = {f.fid: f for f in part.features}
    face_view = _circular_view(layout, (0.0, 0.0, 1.0)) or layout.placements[0]

    holes = by_id.get("mounting_holes")
    if holes is not None:
        out.extend(_pattern_annotations(layout, style, holes, datums))
        # Basic pitches locate the pattern. Both are basic for the same reason the bolt
        # circle is: the positional tolerance already defines the permitted variation.
        for key, horizontal in (("pitch_x", True), ("pitch_y", False)):
            pitch = holes.nominal.get(key)
            if not pitch:
                continue
            half = pitch / 2
            if horizontal:
                a = face_view.point3d((-half, holes.anchor[1], holes.anchor[2]))
                b = face_view.point3d((half, holes.anchor[1], holes.anchor[2]))
            else:
                a = face_view.point3d((holes.anchor[0], -half, holes.anchor[2]))
                b = face_view.point3d((holes.anchor[0], half, holes.anchor[2]))
            shown = style.length(pitch)
            text = format_value(shown, style.decimals, trailing_zeros=style.trailing_zeros)
            out.append(
                linear_dimension(
                    a, b, text, style, view=face_view.name, horizontal=horizontal,
                    payload=_dim_payload(
                        "linear", shown, upper=0.0, lower=0.0, tol_style="basic",
                        is_basic=True, raw_text=text,
                    ),
                    importance=2.2,
                )
            )

    cutout = by_id.get("central_cutout")
    if cutout is not None:
        ann = _size_annotation(part, layout, style, rng, cutout, importance=1.8)
        if ann is not None:
            out.append(ann)

    primary = by_id.get("primary_face")
    if primary is not None:
        width = primary.nominal.get("width")
        height = primary.nominal.get("height")
        thickness = primary.nominal.get("z")
        for value, horizontal in ((width, True), (height, False)):
            if not value:
                continue
            half = value / 2
            if horizontal:
                a = face_view.point3d((-half, -(height or 0) / 2, 0.0))
                b = face_view.point3d((half, -(height or 0) / 2, 0.0))
            else:
                a = face_view.point3d((-(width or 0) / 2, -half, 0.0))
                b = face_view.point3d((-(width or 0) / 2, half, 0.0))
            shown = style.length(value)
            text = format_value(shown, style.decimals, trailing_zeros=style.trailing_zeros)
            out.append(
                linear_dimension(
                    a, b, text, style, view=face_view.name, horizontal=horizontal,
                    payload=_dim_payload("linear", shown, tol_style="general", raw_text=text),
                    importance=1.5,
                )
            )
        side = _longitudinal_view(layout, (0.0, 0.0, 1.0))
        if side is not None and thickness:
            a = side.point3d((0.0, (height or 0) / 2, 0.0))
            b = side.point3d((0.0, (height or 0) / 2, thickness))
            shown = style.length(thickness)
            text = format_value(shown, style.decimals, trailing_zeros=style.trailing_zeros)
            out.append(
                linear_dimension(
                    a, b, text, style, view=side.name, horizontal=False,
                    payload=_dim_payload(
                        "linear", shown, tol_style="general", raw_text=text
                    ),
                    importance=1.7,
                )
            )
        side_view = _longitudinal_view(layout, (0.0, 0.0, 1.0))
        if side_view is not None:
            tip = side_view.point3d((0.0, 0.0, thickness or 0.0))
            compartments, text, payload = _gtol("flatness", 0.1, style)
            out.append(
                feature_frame(tip, compartments, text, style, view=side_view.name,
                              payload=payload, importance=2.7)
            )
    return out


def _plan_generic(part, layout, style, rng, datums) -> list[Annotation]:
    """Housing and valve body: size every feature of size, locate every pattern.

    Deliberately less opinionated than the three hand-written plans above. These two
    families exist to broaden the distribution rather than to be the sharpest test, and
    SPEC.md section 13 lists them first in the cut order -- an elaborate scheme here would
    be effort spent on the part of the benchmark most likely to be dropped.
    """
    out: list[Annotation] = []
    for feature in part.features:
        if not feature.is_feature_of_size or "diameter" not in feature.nominal:
            continue
        if feature.nominal.get("count", 1) > 1:
            out.extend(_pattern_annotations(layout, style, feature, datums))
            continue
        ann = _size_annotation(
            part, layout, style, rng, feature,
            importance=2.4 if feature.meta.get("datum") else 1.2,
        )
        if ann is not None:
            out.append(ann)

    primary = next((d for d in datums if d.feature_type == "planar_face"), None)
    bore = next(
        (f for f in part.features if f.meta.get("datum") and f.axis is not None), None
    )
    if primary is not None and bore is not None:
        view = _longitudinal_view(layout, bore.axis[1]) or layout.placements[0]
        tip = _surface_anchor(view, bore)
        compartments, text, payload = _gtol(
            "perpendicularity", 0.05, style, diametral=True,
            refs=((primary.label, None),),
        )
        out.append(
            feature_frame(tip, compartments, text, style, view=view.name,
                          payload=payload, importance=3.0)
        )
    if primary is not None:
        view = layout.placements[0]
        feature = next(
            (f for f in part.features if f.meta.get("datum") == primary.label), None
        )
        if feature is not None:
            edge = _edge_on_view(layout, _feature_normal(part, feature) or (0, 0, 1)) or view
            tip = edge.point3d(feature.anchor)
            compartments, text, payload = _gtol("flatness", 0.08, style)
            out.append(
                feature_frame(tip, compartments, text, style, view=edge.name,
                              payload=payload, importance=2.6)
            )
    return out


_PLANS = {
    "flange": _plan_flange,
    "shaft": _plan_shaft,
    "plate_bracket": _plan_plate,
}


def plan_drawing(
    part: BuiltPart,
    layout: SheetLayout,
    style: HouseStyle,
    rng: np.random.Generator,
) -> tuple[list[DatumRecord], list[Annotation]]:
    """Everything the sheet will say about this part, before placement.

    Returns the datums and the callouts separately because the schema does, and because
    datums must be decided first -- a positional tolerance cannot reference a datum the
    drawing has not established.
    """
    datums = _plan_datums(part, layout, style)
    plan = _PLANS.get(part.family, _plan_generic)
    annotations = plan(part, layout, style, rng, datums)
    _mark_critical(annotations, rng)
    return datums, annotations


#: Roughly what fraction of a drawing's callouts a quality engineer would ring as critical
#: to quality (SPEC.md section 6). Not a hard count: on a sheet of eight callouts the number
#: that matters is one or two, and forcing exactly 15% would make it a property of how
#: crowded the sheet is rather than of what the part has to do.
CRITICAL_FRACTION = 0.15


def _criticality_score(payload: dict[str, Any]) -> float:
    """How likely this callout is to be the one that scraps a part if it is wrong.

    Criticality is sampled with weights rather than uniformly, because on a real drawing it
    is not random: the characteristics a quality engineer rings are the ones that decide
    whether the part assembles. A positional tolerance on a bolt pattern and a fitted bore
    are the usual answers; a chamfer angle and a general-toleranced overall length are not.
    Tightness matters too -- the same feature toleranced to 0.02 and to 0.5 are different
    risks, and the tolerance is the designer telling us which.
    """
    if payload.get("kind") == "geometric_tolerance":
        symbol = payload.get("gtol_symbol")
        base = 3.0 if symbol == "position" else 1.5
        value = payload.get("gtol_value") or 1.0
        # A tighter zone is a stronger statement of intent.
        return base * (1.0 + min(2.0, 0.1 / max(value, 1e-3)))

    if payload.get("kind") != "dimension":
        return 0.2

    # Features of size with a stated tolerance: a fit class is the strongest signal a
    # drawing has that a dimension is functional.
    if payload.get("fit_class"):
        return 3.0
    upper, lower = payload.get("upper_tol"), payload.get("lower_tol")
    if upper is None or lower is None:
        return 0.3
    span = abs(upper - lower)
    nominal = abs(payload.get("nominal") or 1.0)
    relative = span / max(nominal, 1e-6)
    return 2.0 if relative < 0.002 else 1.0


def _mark_critical(annotations: list[Annotation], rng: np.random.Generator) -> None:
    """Tag a weighted sample of callouts ``is_critical`` (SPEC.md section 6).

    This drives the tier-4 cost metric in ``evalkit``: a missed critical characteristic is
    charged several times a missed ordinary one. Without it every drawing would have a CTQ
    recall of nothing over nothing and the tier would be unmeasurable.
    """
    eligible = [a for a in annotations if a.payload.get("kind") in
                {"dimension", "geometric_tolerance"}]
    if not eligible:
        return

    count = max(1, round(CRITICAL_FRACTION * len(annotations)))
    weights = np.array([_criticality_score(a.payload) for a in eligible], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(
        len(eligible), size=min(count, len(eligible)), replace=False, p=weights
    )
    for index in np.atleast_1d(chosen):
        eligible[int(index)].payload["is_critical"] = True


# --- sheet decorations ---------------------------------------------------------------------


def _pattern_centres(feature: SemanticFeature) -> tuple[tuple[float, float, float], ...]:
    """Where each member of a hole pattern sits, in model coordinates.

    Reconstructed from the pattern's recorded parameters rather than from the face index,
    because the index gives one face per hole with no ordering and no way to know which
    belongs to which. The phase of a polar pattern is taken from the recorded anchor, which
    is one of the holes -- assuming a phase of zero would rotate every centre mark off its
    hole on any family that starts its array elsewhere.
    """
    count = int(feature.nominal.get("count", 1))
    if count <= 1:
        # The axis, not the recorded anchor. A single cylinder's anchor sits on its own
        # surface, so marking there draws the cross out at the rim instead of at the centre
        # -- and since the arm length scales with the diameter, a 315 mm flange rim gets a
        # 33 mm cross hanging off the side of the view.
        return (_axis_foot(feature),)

    z = feature.anchor[2]
    if feature.meta.get("pattern") == "polar":
        radius = feature.nominal.get("bolt_circle", 0.0) / 2
        if radius <= 0:
            return (feature.anchor,)
        phase = math.atan2(feature.anchor[1], feature.anchor[0])
        return tuple(
            (
                radius * math.cos(phase + 2 * math.pi * i / count),
                radius * math.sin(phase + 2 * math.pi * i / count),
                z,
            )
            for i in range(count)
        )

    if feature.meta.get("pattern") == "rectangular":
        hx = feature.nominal.get("pitch_x", 0.0) / 2
        hy = feature.nominal.get("pitch_y", 0.0) / 2
        return tuple(
            (sx * hx, sy * hy, z) for sx in (-1, 1) for sy in (-1, 1)
        )
    return (feature.anchor,)


def sheet_decorations(
    part: BuiltPart, layout: SheetLayout, style: HouseStyle
) -> tuple[Primitive, ...]:
    """Centre marks, bolt-circle phantom lines, and cutting-plane indications.

    None of these carry a ground-truth entry, which is exactly why they belong here rather
    than among the annotations: they are the drawing conventions a reader uses to interpret
    the callouts, and a benchmark image without them is easier than a real drawing in a way
    that would flatter every model measured on it. The bolt-circle phantom is the clearest
    case -- without it, the basic ``⌀200`` locating the pattern points at empty paper.
    """
    prims: list[Primitive] = []

    for feature in part.features:
        if "diameter" not in feature.nominal:
            continue
        axis = feature.axis[1] if feature.axis else (0.0, 0.0, 1.0)
        circular = _circular_view(layout, axis)
        if circular is None:
            continue

        centres = _pattern_centres(feature)
        arm = max(2.0, feature.nominal["diameter"] / 2 * circular.scale + 1.5)
        for centre in centres:
            cx, cy = circular.point3d(centre)
            prims.append(
                Primitive("line", ((cx - arm, cy), (cx + arm, cy)),
                          width=style.line_centre, layer="centre")
            )
            prims.append(
                Primitive("line", ((cx, cy - arm), (cx, cy + arm)),
                          width=style.line_centre, layer="centre")
            )

        bolt_circle = feature.nominal.get("bolt_circle")
        if bolt_circle:
            origin = circular.point3d(
                (0.0, 0.0, feature.anchor[2])
            )
            prims.append(
                Primitive(
                    "circle",
                    (origin, (bolt_circle / 2 * circular.scale, 0.0)),
                    width=style.line_centre,
                    layer="centre",
                )
            )

    prims.extend(_cutting_plane(layout, style))
    return tuple(prims)


def _cutting_plane(layout: SheetLayout, style: HouseStyle) -> list[Primitive]:
    """The cutting-plane line, its arrows and its letters, drawn in the parent view.

    A section view is meaningless without it: it says where the cut was taken and which way
    the viewer is looking. The arrows point along the *section view's* view direction, which
    is the part that is easy to get backwards -- an arrow pointing the other way describes
    the opposite half of the part, and the section drawn beside it would then contradict it.
    """
    prims: list[Primitive] = []
    parent = layout.placements[0]
    px0, py0, px1, py1 = parent.sheet_bounds

    for placement in layout.placements:
        spec = placement.spec
        if spec.section_normal is None or spec.section_letter is None:
            continue

        # The cutting plane passes through the model origin; find where that plane crosses
        # the parent view by projecting a point and a direction lying in it.
        normal2 = parent.view.transform.direction_2d(spec.section_normal)
        n = math.hypot(*normal2)
        if n < 1e-9:
            # The plane is edge-on to the parent view as well, so there is no line to draw.
            continue
        normal2 = (normal2[0] / n, normal2[1] / n)
        along = (-normal2[1], normal2[0])
        through = parent.point3d((0.0, 0.0, 0.0))

        # The line has to span the view plus a small overshoot, and no more. Sizing it from
        # the view's diagonal -- the obvious shortcut -- overshoots by a factor of root two
        # on a square view, which on a large sheet runs the line off the frame and takes one
        # of its two arrows and letters with it.
        over = 8.0
        half = (
            abs(along[0]) * (px1 - px0) + abs(along[1]) * (py1 - py0)
        ) / 2 + over
        a = (through[0] - along[0] * half, through[1] - along[1] * half)
        b = (through[0] + along[0] * half, through[1] + along[1] * half)
        prims.append(
            Primitive("line", (a, b), width=style.line_border * 0.8, layer="centre")
        )

        view_dir2 = parent.view.transform.direction_2d(placement.view.transform.direction)
        vn = math.hypot(*view_dir2)
        arrow = (view_dir2[0] / vn, view_dir2[1] / vn) if vn > 1e-9 else normal2
        size = style.arrow_length * 2.0
        for end in (a, b):
            tip = (end[0] + arrow[0] * size, end[1] + arrow[1] * size)
            prims.append(
                Primitive("line", (end, tip), width=style.line_dimension, layer="annotation")
            )
            back = math.atan2(end[1] - tip[1], end[0] - tip[0])
            spread = math.radians(12.0)
            prims.append(
                Primitive(
                    "triangle",
                    (
                        tip,
                        (tip[0] + math.cos(back - spread) * style.arrow_length,
                         tip[1] + math.sin(back - spread) * style.arrow_length),
                        (tip[0] + math.cos(back + spread) * style.arrow_length,
                         tip[1] + math.sin(back + spread) * style.arrow_length),
                    ),
                    filled=True,
                    width=style.line_dimension,
                    layer="annotation",
                )
            )
            prims.append(
                Primitive(
                    "text",
                    ((end[0] - arrow[0] * size * 0.9 - normal2[0] * size * 0.4,
                      end[1] - arrow[1] * size * 0.9 - normal2[1] * size * 0.4),),
                    text=spec.section_letter,
                    height=style.text_height * 1.15,
                    anchor="middle",
                    layer="annotation",
                )
            )
    return prims
