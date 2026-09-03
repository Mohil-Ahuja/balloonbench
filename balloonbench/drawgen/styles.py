"""House styles: the axes along which two correct drawings of the same part differ.

SPEC.md section 7.3 asks for three to five distinct styles, sampled per drawing and recorded
in ``provenance.house_style``. The point is not decoration. Every axis here is a real
convention split that changes what an extractor has to parse, and recording which style a
drawing used is what lets a result be sliced afterwards -- "this model loses twenty points
on limit-form tolerances" is a finding; "this model scores 0.71" is not.

The axes, and why each is worth varying:

``tolerance_bias``
    ``44±0.05``, ``44 +0.05/-0.00``, ``44.05/43.95`` and ``⌀44 H7`` all state a tolerance,
    and the last two state it without ever writing the word. The limit form is the trap: a
    reader that takes the first number as nominal is wrong by half the tolerance band, and
    it looks like a rounding error rather than a parse failure.

``decimals`` / ``trailing_zeros``
    ISO practice tends to write ``44``; ASME-influenced practice writes ``44.00``. The
    trailing zeros also carry meaning about implied precision under a general tolerance
    note, so dropping them is not cosmetic.

``datum_style``
    The filled triangle is current. The older ``-A-`` between dashes is still on any drawing
    that has been revised rather than redrawn, which is a large share of what a real shop
    actually receives.

``zone_prefix``
    A positional tolerance on a cylindrical feature has a cylindrical zone, written ``⌀0.25``.
    Drawings that omit the ⌀ exist and mean a different (rectangular) zone; a benchmark that
    only ever shows one form cannot tell whether a model reads the symbol or assumes it.

``text_position`` and ``arrowhead``
    Purely visual, and exactly the kind of variation that separates a detector which has
    learned dimension *structure* from one that has memorised one renderer's output.

``units``
    Imperial sheets change the decimal convention, the tolerance magnitudes and the symbols.
    Kept to a minority of sheets, matching the population the benchmark targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["STYLES", "HouseStyle", "get_style", "sample_style", "style_names"]


@dataclass(frozen=True)
class HouseStyle:
    """One drawing office's conventions.

    Every field that affects what appears on the sheet is here rather than in the renderer,
    so a drawing is reproducible from ``(seed, style name)`` alone.
    """

    name: str

    # --- text and numbers ---
    #: Character height in millimetres for dimension text. ISO 3098 sizes are 2.5, 3.5, 5.
    text_height: float = 3.5
    #: Decimal places on a nominal dimension.
    decimals: int = 2
    #: Decimal places on a tolerance deviation. Conventionally at least as many as the
    #: nominal carries, because a tolerance finer than its own dimension is unreadable.
    tol_decimals: int = 2
    trailing_zeros: bool = True
    #: Decimal places inside a feature control frame.
    gtol_decimals: int = 2

    # --- tolerance presentation ---
    #: Relative weights over the schema's tol styles, sampled per dimension. Weights rather
    #: than a single choice because a real drawing mixes forms -- a fit on the bore, a
    #: bilateral on a length, a general note covering the rest.
    tolerance_bias: dict[str, float] = field(
        default_factory=lambda: {
            "bilateral": 0.45,
            "unilateral": 0.15,
            "limit": 0.15,
            "fit": 0.25,
        }
    )
    #: Share of dimensions left to the general tolerance note instead of being toleranced
    #: individually. Real drawings tolerance far fewer dimensions than a generator wants to.
    general_tolerance_share: float = 0.55

    # --- symbols ---
    #: ``filled_triangle`` (current) or ``dashed`` (the older ``-A-`` form).
    datum_style: str = "filled_triangle"
    #: Whether a positional tolerance on a cylindrical feature writes its ⌀ zone prefix.
    zone_prefix: bool = True

    # --- geometry of the annotation itself ---
    #: ``above`` puts dimension text over an unbroken line; ``broken`` interrupts the line.
    text_position: str = "above"
    #: ``filled``, ``open`` or ``tick``.
    arrowhead: str = "filled"
    arrow_length: float = 3.0
    #: Gap between the feature and the start of its extension line, in millimetres.
    extension_gap: float = 1.0
    #: Overshoot of an extension line past the dimension line.
    extension_overshoot: float = 2.0
    #: First dimension line offset from the part outline, and the step between chained
    #: tiers. Consistent tiers are what makes a real drawing look ordered.
    first_offset: float = 10.0
    tier_step: float = 8.0

    # --- line weights, in millimetres of pen width ---
    line_visible: float = 0.5
    line_hidden: float = 0.25
    line_centre: float = 0.25
    line_dimension: float = 0.25
    line_border: float = 0.7
    #: Hatch spacing for section views.
    hatch_spacing: float = 3.0
    hatch_angle: float = 45.0

    # --- sheet ---
    units: str = "mm"
    #: Font file, relative to ``assets/fonts``. Kept per style so a style change can carry a
    #: font change, but every entry must be a vendored font -- never a system one.
    font: str = "osifont/osifont-lgpl3fe.ttf"

    def __post_init__(self) -> None:
        if self.datum_style not in ("filled_triangle", "dashed"):
            raise ValueError(f"{self.name}: bad datum_style {self.datum_style!r}")
        if self.text_position not in ("above", "broken"):
            raise ValueError(f"{self.name}: bad text_position {self.text_position!r}")
        if self.arrowhead not in ("filled", "open", "tick"):
            raise ValueError(f"{self.name}: bad arrowhead {self.arrowhead!r}")
        if self.units not in ("mm", "inch"):
            raise ValueError(f"{self.name}: bad units {self.units!r}")
        if self.tol_decimals < self.decimals:
            raise ValueError(
                f"{self.name}: tolerance decimals ({self.tol_decimals}) finer than the "
                f"nominal's ({self.decimals}) is unreadable on a sheet"
            )
        total = sum(self.tolerance_bias.values())
        if total <= 0:
            raise ValueError(f"{self.name}: tolerance_bias sums to {total}")

    def length(self, mm: float) -> float:
        """A model-space length in the units this sheet is drawn in.

        The solids are modelled in millimetres and always will be -- ISO 286 fits, preferred
        number series and thread tables are all metric, and re-deriving them in inches would
        be a second source of truth. An imperial sheet is therefore a *presentation* of a
        metric part, and this is the single point where that presentation happens. Every
        number that reaches the sheet or the ground truth passes through here, so a drawing
        that says ``inch`` cannot end up carrying millimetre values -- which would be a
        ground-truth error stated in the units field, the worst place to hide one.
        """
        return mm / 25.4 if self.units == "inch" else mm

    def pick_tolerance_style(
        self, rng: np.random.Generator, allowed: tuple[str, ...]
    ) -> str:
        """Sample a tolerance presentation restricted to what this dimension supports.

        ``allowed`` matters: a fit class is meaningless on a plate thickness, and a
        bilateral form is wrong for a hole that was drawn from an ISO 286 fit. Filtering
        first and renormalising second keeps the style's bias intact within whatever the
        dimension can actually carry.
        """
        weights = np.array([self.tolerance_bias.get(s, 0.0) for s in allowed], dtype=float)
        if weights.sum() <= 0:
            return allowed[0]
        return str(rng.choice(allowed, p=weights / weights.sum()))


#: The sampled styles. Five, spanning the axes above rather than five near-copies: an
#: unremarkable ISO sheet, an ASME-influenced one, an older drawing carrying legacy
#: conventions, a tight-tolerance precision sheet, and an imperial one.
STYLES: tuple[HouseStyle, ...] = (
    HouseStyle(
        name="iso_metric_clean",
        text_height=3.5,
        decimals=1,
        tol_decimals=2,
        trailing_zeros=False,
        gtol_decimals=2,
        datum_style="filled_triangle",
        zone_prefix=True,
        text_position="above",
        arrowhead="filled",
        tolerance_bias={"bilateral": 0.5, "unilateral": 0.1, "limit": 0.1, "fit": 0.3},
        general_tolerance_share=0.6,
    ),
    HouseStyle(
        name="asme_decimal",
        text_height=3.5,
        decimals=2,
        tol_decimals=3,
        trailing_zeros=True,
        gtol_decimals=3,
        datum_style="filled_triangle",
        zone_prefix=True,
        text_position="broken",
        arrowhead="filled",
        tolerance_bias={"bilateral": 0.3, "unilateral": 0.35, "limit": 0.2, "fit": 0.15},
        general_tolerance_share=0.4,
        first_offset=12.0,
        tier_step=9.0,
    ),
    HouseStyle(
        name="legacy_shop",
        text_height=4.0,
        decimals=1,
        tol_decimals=2,
        trailing_zeros=False,
        gtol_decimals=2,
        # The legacy sheet is the one that omits the zone prefix and uses the dashed datum.
        # Both are period-correct, and both are things a model can only get right by
        # reading rather than assuming.
        datum_style="dashed",
        zone_prefix=False,
        text_position="above",
        arrowhead="open",
        tolerance_bias={"bilateral": 0.35, "unilateral": 0.1, "limit": 0.45, "fit": 0.1},
        general_tolerance_share=0.7,
        line_visible=0.6,
        hatch_spacing=2.5,
    ),
    HouseStyle(
        name="precision_metric",
        text_height=2.5,
        decimals=2,
        tol_decimals=3,
        trailing_zeros=True,
        gtol_decimals=3,
        datum_style="filled_triangle",
        zone_prefix=True,
        text_position="above",
        arrowhead="tick",
        tolerance_bias={"bilateral": 0.25, "unilateral": 0.15, "limit": 0.3, "fit": 0.3},
        general_tolerance_share=0.25,
        first_offset=9.0,
        tier_step=7.0,
        line_visible=0.35,
        line_dimension=0.18,
        hatch_spacing=2.0,
    ),
    HouseStyle(
        name="imperial_shop",
        text_height=3.5,
        # Imperial drawings carry more decimals because an inch is a coarse unit: a
        # three-place decimal inch is 0.025 mm, so three places is the working default and
        # four is a precision callout.
        decimals=3,
        tol_decimals=3,
        trailing_zeros=True,
        gtol_decimals=3,
        datum_style="filled_triangle",
        zone_prefix=True,
        text_position="broken",
        arrowhead="filled",
        tolerance_bias={"bilateral": 0.4, "unilateral": 0.3, "limit": 0.3, "fit": 0.0},
        general_tolerance_share=0.5,
        units="inch",
    ),
)

_BY_NAME: dict[str, HouseStyle] = {s.name: s for s in STYLES}

#: Sampling weights. The imperial sheet is deliberately a minority: the benchmark targets
#: metric drawings, and an imperial sheet is a variation to be robust to rather than the
#: population being measured.
_STYLE_WEIGHTS: dict[str, float] = {
    "iso_metric_clean": 0.30,
    "asme_decimal": 0.22,
    "legacy_shop": 0.22,
    "precision_metric": 0.18,
    "imperial_shop": 0.08,
}


def style_names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def get_style(name: str) -> HouseStyle:
    if name not in _BY_NAME:
        raise KeyError(f"unknown house style {name!r}; known: {sorted(_BY_NAME)}")
    return _BY_NAME[name]


def sample_style(rng: np.random.Generator) -> HouseStyle:
    """Draw a house style.

    >>> import numpy as np
    >>> sample_style(np.random.default_rng(0)).name in style_names()
    True
    """
    names = list(_BY_NAME)
    weights = np.array([_STYLE_WEIGHTS[n] for n in names], dtype=float)
    return _BY_NAME[str(rng.choice(names, p=weights / weights.sum()))]
