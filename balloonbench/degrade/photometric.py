"""Photometric degradation: what happens to the pixels, not to where they are.

Every transform here leaves geometry alone, so boxes pass through untouched. That is the
reason the module is separate from :mod:`balloonbench.degrade.geometry` rather than the two
being one grab-bag of effects -- it makes "does this step need to move the ground truth?" a
question answered by which file the function lives in, instead of one a reviewer has to
re-derive for each new effect.

The effects are ordered here roughly as they occur physically: the paper ages and picks up
grain, light falls on it unevenly, the sensor adds noise, and the file is finally squashed
into a JPEG. A profile that applies them in that order produces something that looks like a
scan; one that JPEGs first and then adds grain produces something that looks like a filter.
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from balloonbench.degrade.base import Sample

__all__ = [
    "bleed_through",
    "blueprint_tint",
    "gaussian_noise",
    "grain",
    "jpeg",
    "low_contrast",
    "salt_and_pepper",
    "sepia",
    "toner_streaks",
    "uneven_illumination",
]


def _array(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32)


def _image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def jpeg(sample: Sample, rng: np.random.Generator, *, low: int = 40, high: int = 85) -> Sample:
    """Re-encode as JPEG, leaving its ringing and blocking behind.

    Applied by actually round-tripping through the encoder rather than by simulating block
    artifacts. The artifacts that matter are the ones that appear around high-contrast thin
    lines -- exactly what a drawing is made of -- and no simulation of them is as faithful
    as the encoder itself.
    """
    quality = int(rng.integers(low, high + 1))
    buffer = io.BytesIO()
    sample.image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    return sample.with_image(Image.open(buffer).convert("RGB"), f"jpeg_q{quality}")


def gaussian_noise(sample: Sample, rng: np.random.Generator, *, sigma: float = 6.0) -> Sample:
    array = _array(sample.image)
    noise = rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    return sample.with_image(_image(array + noise), "gaussian_noise")


def salt_and_pepper(
    sample: Sample, rng: np.random.Generator, *, amount: float = 0.002
) -> Sample:
    """Isolated black and white pixels, as from dust on the platen and sensor dropouts.

    Salt and pepper is applied per pixel across all three channels together, not per
    channel: a speck of dust is grey, and independently corrupting the channels would give
    coloured confetti that no scanner produces.
    """
    array = _array(sample.image)
    height, width = array.shape[:2]
    mask = rng.random((height, width))
    array[mask < amount / 2] = 0.0
    array[mask > 1.0 - amount / 2] = 255.0
    return sample.with_image(_image(array), "salt_and_pepper")


def grain(sample: Sample, rng: np.random.Generator, *, strength: float = 0.06) -> Sample:
    """Paper texture: low-frequency mottling rather than per-pixel noise.

    Generated at a coarse resolution and scaled up, because paper grain is a property of the
    fibre, several pixels across at scanning resolution. Per-pixel noise would be sensor
    noise, which :func:`gaussian_noise` already provides and which looks quite different.
    """
    width, height = sample.image.size
    small = rng.normal(0.0, 1.0, size=(max(2, height // 24), max(2, width // 24)))
    texture = np.asarray(
        Image.fromarray(
            ((small - small.min()) / max(float(np.ptp(small)), 1e-6) * 255).astype(np.uint8)
        ).resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.float32,
    )
    texture = (texture - texture.mean()) / max(texture.std(), 1e-6)
    array = _array(sample.image) * (1.0 + strength * texture[..., None])
    return sample.with_image(_image(array), "grain")


def uneven_illumination(
    sample: Sample, rng: np.random.Generator, *, strength: float = 0.22
) -> Sample:
    """A smooth brightness gradient, as from a lamp or a phone shadow across the page."""
    width, height = sample.image.size
    ys, xs = np.mgrid[0:height, 0:width]
    cx = float(rng.uniform(0.15, 0.85)) * width
    cy = float(rng.uniform(0.15, 0.85)) * height
    radius = math.hypot(width, height)
    distance = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / radius
    gain = 1.0 + strength * (0.5 - distance)
    return sample.with_image(
        _image(_array(sample.image) * gain[..., None]), "uneven_illumination"
    )



def low_contrast(
    sample: Sample, rng: np.random.Generator, *, low: float = 0.55, high: float = 0.85
) -> Sample:
    """Wash the image out, as a tired photocopier does.

    Contrast is reduced toward mid-grey and then the whole thing is lifted, because a faded
    copy is not merely low contrast -- its blacks go grey while its paper stays near white,
    which is what makes the thin lines the first thing to disappear.
    """
    factor = float(rng.uniform(low, high))
    reduced = ImageEnhance.Contrast(sample.image).enhance(factor)
    lifted = ImageEnhance.Brightness(reduced).enhance(float(rng.uniform(1.0, 1.08)))
    return sample.with_image(lifted, "low_contrast")


def blueprint_tint(sample: Sample, rng: np.random.Generator) -> Sample:
    """The cyan cast of a diazo print, or of a blue-line copy.

    Implemented as a duotone mapping from luminance rather than as a channel scale. A true
    blueprint has no neutral greys at all: paper is pale cyan and line work is deep blue,
    and scaling the channels of a black-and-white image leaves the greys neutral.
    """
    grey = np.asarray(ImageOps.grayscale(sample.image), dtype=np.float32) / 255.0
    paper = np.array([214.0, 235.0, 245.0])
    ink = np.array([18.0, 42.0, 96.0])
    jitter = float(rng.uniform(-8.0, 8.0))
    array = ink + (paper - ink) * grey[..., None] + jitter
    return sample.with_image(_image(array), "blueprint_tint")


def sepia(sample: Sample, rng: np.random.Generator, *, strength: float = 0.75) -> Sample:
    """Age the paper toward yellow-brown."""
    array = _array(sample.image)
    grey = array @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    tone = np.stack(
        [grey * 1.07, grey * 0.98, grey * 0.82], axis=-1
    ) + np.array([12.0, 4.0, -6.0], dtype=np.float32)
    amount = strength * float(rng.uniform(0.7, 1.0))
    return sample.with_image(_image(array * (1 - amount) + tone * amount), "sepia")


def bleed_through(
    sample: Sample, rng: np.random.Generator, *, strength: float = 0.16
) -> Sample:
    """Faint ink from the reverse side showing through thin paper.

    The sheet's own content is mirrored, blurred and multiplied back in at low strength.
    Using the drawing itself as its own reverse side is a cheat, but a defensible one: what
    the effect has to look like is *plausible technical linework* seen through paper, and
    nothing available is more plausible than a drawing.
    """
    reverse = sample.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).filter(
        ImageFilter.GaussianBlur(radius=float(rng.uniform(1.5, 3.0)))
    )
    array = _array(sample.image)
    ghost = _array(reverse)
    amount = strength * float(rng.uniform(0.6, 1.0))
    # Multiplicative, because ink absorbs light: bleed-through darkens paper, it never
    # brightens ink. Adding it would lighten the black lines it crossed.
    blended = array * (1.0 - amount + amount * ghost / 255.0)
    return sample.with_image(_image(blended), "bleed_through")


def toner_streaks(sample: Sample, rng: np.random.Generator, *, count: int = 3) -> Sample:
    """Vertical bands of toner starvation and excess, the signature of a worn drum.

    Streaks run the full height of the page and are constant along it, because the drum
    turns as the page feeds: a defect at one point on its circumference marks a stripe down
    the sheet. A streak with an end partway down would be a smudge, not a toner fault.
    """
    width, height = sample.image.size
    gain = np.ones(width, dtype=np.float32)
    for _ in range(int(rng.integers(1, count + 1))):
        centre = float(rng.uniform(0, width))
        half = float(rng.uniform(0.004, 0.02)) * width
        depth = float(rng.uniform(-0.28, 0.16))
        xs = np.arange(width, dtype=np.float32)
        gain *= 1.0 + depth * np.exp(-(((xs - centre) / half) ** 2))
    array = _array(sample.image) * gain[None, :, None]
    return sample.with_image(_image(array), "toner_streaks")


def dropout(sample: Sample, rng: np.random.Generator, *, threshold: int = 96) -> Sample:
    """Thin lines breaking up, as a photocopy loses the faintest strokes.

    Applied only to pixels that are already dark, and only to some of them, so that what
    disappears is the edge of a stroke rather than a random scatter of holes. This is the
    degradation that most directly attacks a drawing: dimension lines are one pixel wide at
    300 DPI and are the first thing a third-generation copy loses.
    """
    array = _array(sample.image)
    grey = array.mean(axis=2)
    ink = grey < threshold
    holes = rng.random(grey.shape) < 0.06
    array[ink & holes] = 255.0
    return sample.with_image(_image(array), "dropout")
