"""Geometric degradation, with the ground truth carried through every warp.

CLAUDE.md states the rule this module exists to honour: *every geometric transform in
`degrade/` must apply the same homography to the ground-truth boxes as to the image.* The
design here makes that structural rather than a habit. A warp is an object that knows two
maps -- :meth:`Warp.forward`, source pixel to destination pixel, and :meth:`Warp.inverse`,
its opposite -- and :meth:`Warp.apply` uses both: the inverse to resample the image and the
forward to move the boxes. There is no code path that resamples pixels without also mapping
boxes, because they are the same method.

The inverse is the one that resamples, which is worth stating because it reads backwards.
To fill destination pixel *d* you must know which source pixel landed there, and that is
``inverse(d)``. Using the forward map to push source pixels into the destination leaves
unfilled holes wherever the mapping expands, which is why every image-warping library works
this way and why the boxes -- which genuinely are being pushed forward -- use the other one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from balloonbench.degrade.base import Sample

__all__ = [
    "Barrel",
    "Fold",
    "Homography",
    "Warp",
    "barrel",
    "fold",
    "keystone",
    "rotate",
    "scale_nonuniform",
]


class Warp:
    """A point map that can move an image and its ground truth consistently."""

    name = "warp"

    def forward(self, pts: np.ndarray) -> np.ndarray:
        """Source pixel coordinates to destination. Used for boxes."""
        raise NotImplementedError

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        """Destination pixel coordinates to source. Used for resampling."""
        raise NotImplementedError

    def apply(self, sample: Sample, fill: int = 255) -> Sample:
        image = self._resample(sample.image, fill)
        out = sample.with_image(image, self.name)
        return out.map_boxes(self.forward, name=self.name, size=image.size)

    def _resample(self, image: Image.Image, fill: int) -> Image.Image:
        from scipy.ndimage import map_coordinates

        width, height = image.size
        ys, xs = np.mgrid[0:height, 0:width]
        dest = np.column_stack([xs.ravel().astype(float), ys.ravel().astype(float)])
        src = self.inverse(dest)
        # map_coordinates indexes (row, col), so the source columns swap order here.
        coords = np.vstack([src[:, 1], src[:, 0]])

        array = np.asarray(image, dtype=np.float32)
        channels = [
            map_coordinates(
                array[..., c], coords, order=1, mode="constant", cval=float(fill)
            ).reshape(height, width)
            for c in range(array.shape[2])
        ]
        out = np.clip(np.stack(channels, axis=-1), 0, 255).astype(np.uint8)
        return Image.fromarray(out, mode="RGB")


@dataclass
class Homography(Warp):
    """A projective transform, given as the 3x3 matrix that maps source to destination."""

    matrix: np.ndarray
    name: str = "homography"

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=float).reshape(3, 3)
        self._inverse = np.linalg.inv(self.matrix)

    def forward(self, pts: np.ndarray) -> np.ndarray:
        return _project(self.matrix, pts)

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        return _project(self._inverse, pts)

    def _resample(self, image: Image.Image, fill: int) -> Image.Image:
        """Pillow's own projective resampler, which is far faster than a generic remap.

        Pillow's ``PERSPECTIVE`` coefficients are the *inverse* map, for the same
        destination-to-source reason described in the module docstring, and they are the
        first eight entries of that matrix normalised so its last entry is one.
        """
        m = self._inverse / self._inverse[2, 2]
        return image.transform(
            image.size,
            Image.Transform.PERSPECTIVE,
            data=tuple(m.ravel()[:8]),
            resample=Image.Resampling.BICUBIC,
            fillcolor=(fill, fill, fill),
        )


def _project(matrix: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    homogeneous = np.column_stack([pts, np.ones(len(pts))])
    out = homogeneous @ matrix.T
    w = np.where(np.abs(out[:, 2]) < 1e-12, 1e-12, out[:, 2])
    return np.column_stack([out[:, 0] / w, out[:, 1] / w])


def _about_centre(matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Conjugate a transform so it acts about the image centre rather than the origin."""
    cx, cy = size[0] / 2.0, size[1] / 2.0
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=float)
    back = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], dtype=float)
    return back @ matrix @ to_origin


# --- named geometric transforms ------------------------------------------------------------


def rotate(sample: Sample, rng: np.random.Generator, *, max_degrees: float = 3.0) -> Sample:
    """A small rotation, as from a sheet laid slightly askew on a scanner bed.

    Kept small deliberately. SPEC.md section 8 says plus or minus three degrees, and that is
    not timidity: a scan rotated far enough to matter gets straightened by the operator, so
    large rotations are not what the real distribution contains.
    """
    angle = math.radians(float(rng.uniform(-max_degrees, max_degrees)))
    c, s = math.cos(angle), math.sin(angle)
    matrix = _about_centre(
        np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), sample.size
    )
    return Homography(matrix, name="rotate").apply(sample)


def keystone(
    sample: Sample, rng: np.random.Generator, *, strength: float = 0.02
) -> Sample:
    """Perspective from a page lifting off the platen, or from a phone held off-axis.

    Built by nudging the four corners and solving for the homography that takes the original
    corners there, rather than by writing a matrix directly. Corner positions are the thing
    with a physical meaning -- how far the page lifted -- and a matrix with a plausible-
    looking perspective term can easily fold the image onto itself.
    """
    width, height = sample.size
    src = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=float
    )
    jitter = np.array(
        [
            [rng.uniform(-1, 1) * strength * width, rng.uniform(-1, 1) * strength * height]
            for _ in range(4)
        ]
    )
    dst = src + jitter
    return Homography(_solve_homography(src, dst), name="keystone").apply(sample)


def scale_nonuniform(
    sample: Sample, rng: np.random.Generator, *, max_ratio: float = 0.03
) -> Sample:
    """Slightly different scale in x and y, as from a scanner's feed rate drifting."""
    sx = 1.0 + float(rng.uniform(-max_ratio, max_ratio))
    sy = 1.0 + float(rng.uniform(-max_ratio, max_ratio))
    matrix = _about_centre(
        np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]), sample.size
    )
    return Homography(matrix, name="scale_nonuniform").apply(sample)


def _solve_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 3x3 matrix taking four source points to four destination points."""
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.asarray(rows, dtype=float)
    b = dst.reshape(-1)
    solution = np.linalg.solve(a, b)
    return np.append(solution, 1.0).reshape(3, 3)


@dataclass
class Barrel(Warp):
    """Radial lens distortion, as from a phone camera.

    ``k`` is the usual first radial coefficient, applied to a radius normalised by half the
    image diagonal so that its magnitude means the same thing at any resolution. Positive
    ``k`` is pincushion, negative is barrel.
    """

    k: float
    size: tuple[int, int]
    name: str = "barrel"

    @property
    def _centre(self) -> np.ndarray:
        return np.array([self.size[0] / 2.0, self.size[1] / 2.0])

    @property
    def _norm(self) -> float:
        return math.hypot(*self._centre)

    def forward(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        rel = (pts - self._centre) / self._norm
        r2 = (rel**2).sum(axis=1, keepdims=True)
        return self._centre + rel * (1.0 + self.k * r2) * self._norm

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        """Undo the radial map by fixed-point iteration.

        The forward map is a cubic in the radius, so its inverse has no convenient closed
        form. For the small coefficients a real lens produces the iteration converges in a
        handful of steps, and five is well past the point where the result stops moving at
        pixel precision.
        """
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        target = (pts - self._centre) / self._norm
        guess = target.copy()
        for _ in range(5):
            r2 = (guess**2).sum(axis=1, keepdims=True)
            guess = target / (1.0 + self.k * r2)
        return self._centre + guess * self._norm


def barrel(sample: Sample, rng: np.random.Generator, *, max_k: float = 0.06) -> Sample:
    k = float(rng.uniform(-max_k, max_k))
    return Barrel(k=k, size=sample.size).apply(sample)


@dataclass
class Fold(Warp):
    """A page crease: a small local displacement across a line, plus a brightness ridge.

    A fold is not a global transform, and modelling it as one would be wrong in a way that
    matters for the boxes -- the displacement is confined to a band a few millimetres wide,
    so a callout on the far side of the sheet must not move at all. The displacement decays
    with a Gaussian falloff away from the crease, which both looks right and keeps the map
    invertible.
    """

    position: float
    vertical: bool
    amplitude: float
    width: float
    size: tuple[int, int]
    name: str = "fold"

    def _offset(self, pts: np.ndarray, sign: float) -> np.ndarray:
        axis = 0 if self.vertical else 1
        distance = pts[:, axis] - self.position
        falloff = np.exp(-((distance / self.width) ** 2))
        shift = sign * self.amplitude * falloff
        out = pts.copy()
        out[:, 1 - axis] = out[:, 1 - axis] + shift
        return out

    def forward(self, pts: np.ndarray) -> np.ndarray:
        return self._offset(np.atleast_2d(np.asarray(pts, dtype=float)), 1.0)

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        # The displacement is perpendicular to the axis the falloff is measured along, so
        # the falloff is unchanged by the shift and negating it inverts the map exactly.
        return self._offset(np.atleast_2d(np.asarray(pts, dtype=float)), -1.0)

    def apply(self, sample: Sample, fill: int = 255) -> Sample:
        out = super().apply(sample, fill)
        return out.with_image(self._ridge(out.image), "fold_ridge")

    def _ridge(self, image: Image.Image) -> Image.Image:
        """The bright line a crease leaves where the paper catches the light."""
        width, height = image.size
        axis_values = (
            np.arange(width, dtype=np.float32)
            if self.vertical
            else np.arange(height, dtype=np.float32)
        )
        profile = np.exp(-(((axis_values - self.position) / (self.width * 0.45)) ** 2))
        gain = 1.0 + 0.22 * profile
        array = np.asarray(image, dtype=np.float32)
        # The ridge runs along the crease, so the gain profile is broadcast across the
        # other axis: columns for a vertical fold, rows for a horizontal one.
        array = array * (gain[None, :, None] if self.vertical else gain[:, None, None])
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def fold(sample: Sample, rng: np.random.Generator) -> Sample:
    width, height = sample.size
    vertical = bool(rng.random() < 0.5)
    span = width if vertical else height
    return Fold(
        position=float(rng.uniform(0.3, 0.7) * span),
        vertical=vertical,
        amplitude=float(rng.uniform(1.5, 5.0)),
        width=float(rng.uniform(0.02, 0.05) * span),
        size=sample.size,
    ).apply(sample)
