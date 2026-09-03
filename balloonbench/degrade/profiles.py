"""The six named degradation profiles, and the entry point that runs one.

SPEC.md section 8 names them: ``clean``, ``office_scan``, ``scan_heavy``, ``photocopy_gen3``,
``phone_photo``, ``blueprint_legacy``. Each is a story about how a drawing reached the person
reading it, and the transforms are chosen and ordered to tell that story rather than to
sample a menu of effects. A sheet is creased before it is copied; a phone photograph has a
lens and a room light but no toner streak; a diazo print is tinted before it is scanned, not
after.

Profiles are the benchmark's independent variable. Because each one is applied from an
explicit seed and recorded in ``provenance.degradation_profile``, a result can be reported
per profile -- "this model holds up on office_scan and collapses on photocopy_gen3" -- which
is the finding, rather than a single averaged number that hides where the failure is.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from balloonbench.degrade import clutter, geometry, photometric
from balloonbench.degrade.base import Sample, Transform, apply_transforms
from balloonbench.schema import Drawing

__all__ = ["PROFILES", "degrade", "profile_names"]


def _t(fn, name: str, probability: float = 1.0, **kwargs) -> Transform:
    """Bind a transform's keyword arguments while keeping it seeded and named."""
    if kwargs:
        def bound(sample, rng, _fn=fn, _kw=kwargs):
            return _fn(sample, rng, **_kw)
    else:
        bound = fn
    return Transform(name=name, fn=bound, probability=probability)


#: Ordered transform lists. Order is physical: paper is marked and damaged, then creased and
#: photographed or scanned, then the sensor and the file format have their turn.
PROFILES: dict[str, tuple[Transform, ...]] = {
    # The control condition. Present so that every metric has a baseline measured through
    # exactly the same code path as the degraded conditions, rather than against the
    # generator's own output -- which would confound "degradation hurt" with "the pipeline
    # did something".
    "clean": (),
    "office_scan": (
        _t(clutter.stamp, "stamp", 0.35),
        _t(clutter.handwritten_note, "handwritten_note", 0.3),
        _t(clutter.punch_holes, "punch_holes", 0.4),
        _t(geometry.rotate, "rotate", 0.9, max_degrees=1.5),
        _t(photometric.uneven_illumination, "uneven_illumination", 0.5, strength=0.12),
        _t(photometric.grain, "grain", 0.6, strength=0.035),
        _t(photometric.gaussian_noise, "gaussian_noise", 0.7, sigma=3.0),
        _t(photometric.jpeg, "jpeg", 1.0, low=70, high=92),
    ),
    "scan_heavy": (
        _t(clutter.revision_cloud, "revision_cloud", 0.5),
        _t(clutter.stamp, "stamp", 0.6),
        _t(clutter.handwritten_note, "handwritten_note", 0.6),
        _t(clutter.red_pen_correction, "red_pen_correction", 0.4),
        _t(clutter.punch_holes, "punch_holes", 0.6),
        _t(clutter.torn_corner, "torn_corner", 0.25),
        _t(geometry.fold, "fold", 0.5),
        _t(geometry.rotate, "rotate", 1.0, max_degrees=3.0),
        _t(geometry.scale_nonuniform, "scale_nonuniform", 0.5),
        _t(clutter.photocopier_edge, "photocopier_edge", 0.6),
        _t(photometric.bleed_through, "bleed_through", 0.5),
        _t(photometric.uneven_illumination, "uneven_illumination", 0.8),
        _t(photometric.grain, "grain", 0.8),
        _t(photometric.gaussian_noise, "gaussian_noise", 0.8, sigma=7.0),
        _t(photometric.salt_and_pepper, "salt_and_pepper", 0.5),
        _t(photometric.jpeg, "jpeg", 1.0, low=45, high=75),
    ),
    # Three generations of copying. The defining features are not noise but *loss*: toner
    # streaks, thin lines dropping out, and contrast washing away.
    "photocopy_gen3": (
        _t(clutter.previous_ballooning, "previous_ballooning", 0.55),
        _t(clutter.stamp, "stamp", 0.5),
        _t(clutter.punch_holes, "punch_holes", 0.5),
        _t(geometry.rotate, "rotate", 1.0, max_degrees=2.5),
        _t(clutter.photocopier_edge, "photocopier_edge", 0.85),
        _t(photometric.toner_streaks, "toner_streaks", 0.8),
        _t(photometric.low_contrast, "low_contrast", 1.0, low=0.5, high=0.72),
        _t(photometric.dropout, "dropout", 0.7),
        _t(photometric.grain, "grain", 0.7, strength=0.08),
        _t(photometric.gaussian_noise, "gaussian_noise", 0.8, sigma=8.0),
        _t(photometric.salt_and_pepper, "salt_and_pepper", 0.6, amount=0.004),
        _t(photometric.jpeg, "jpeg", 1.0, low=40, high=65),
    ),
    # A photograph of a sheet on a desk. Lens distortion and keystone, a room light, and no
    # copier artifacts at all -- the failure modes are geometric rather than tonal.
    "phone_photo": (
        _t(clutter.handwritten_note, "handwritten_note", 0.35),
        _t(clutter.previous_ballooning, "previous_ballooning", 0.3),
        _t(geometry.keystone, "keystone", 1.0, strength=0.03),
        _t(geometry.barrel, "barrel", 0.8, max_k=0.05),
        _t(geometry.rotate, "rotate", 0.8, max_degrees=2.5),
        _t(photometric.uneven_illumination, "uneven_illumination", 1.0, strength=0.3),
        _t(photometric.gaussian_noise, "gaussian_noise", 0.9, sigma=5.0),
        _t(photometric.jpeg, "jpeg", 1.0, low=55, high=85),
    ),
    # An old diazo print. Tinted first, because the print was blue before anyone scanned it.
    "blueprint_legacy": (
        _t(clutter.handwritten_note, "handwritten_note", 0.4),
        _t(clutter.stamp, "stamp", 0.35),
        _t(geometry.fold, "fold", 0.7),
        _t(photometric.blueprint_tint, "blueprint_tint", 1.0),
        _t(photometric.sepia, "sepia", 0.3, strength=0.35),
        _t(geometry.rotate, "rotate", 0.9, max_degrees=2.0),
        _t(photometric.uneven_illumination, "uneven_illumination", 0.8, strength=0.25),
        _t(photometric.grain, "grain", 0.9, strength=0.07),
        _t(photometric.bleed_through, "bleed_through", 0.4),
        _t(photometric.gaussian_noise, "gaussian_noise", 0.7, sigma=5.0),
        _t(photometric.jpeg, "jpeg", 1.0, low=50, high=78),
    ),
}


def profile_names() -> tuple[str, ...]:
    return tuple(PROFILES)


def degrade(
    image_path: str | Path,
    drawing: Drawing,
    profile: str,
    seed: int,
) -> Sample:
    """Apply a named profile to a rendered drawing and its ground truth.

    ``seed`` alone fixes the result: which optional steps run, and every parameter each one
    draws. The returned sample's ``drawing`` records the profile in ``provenance``, so a
    degraded image carries the name of the condition it belongs to and results can be sliced
    by it without a separate manifest.

    >>> sorted(profile_names())
    ['blueprint_legacy', 'clean', 'office_scan', 'phone_photo', 'photocopy_gen3', 'scan_heavy']
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown degradation profile {profile!r}; known: {sorted(PROFILES)}")

    sample = Sample.load(image_path, drawing)
    rng = np.random.default_rng(seed)
    sample = apply_transforms(sample, PROFILES[profile], rng)
    sample.drawing.provenance.degradation_profile = profile
    return sample
