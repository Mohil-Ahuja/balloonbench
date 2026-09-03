"""Baseline 3: read the PDF's own text, parse it, and ask a model only for the rest.

SPEC.md section 11 states the prediction this baseline exists to test: *it will beat pure
VLMs by a wide margin on native PDFs and be useless on scans.* Quantifying that split is the
useful result, because it says routing on input type matters more than model choice -- and
it is a result that only appears if the baseline is honest about the second half.

So this module refuses to fake it. A raster PDF has no text to extract, and rather than
falling back to running the whole sheet through a model and quietly reporting the number as
"vector hybrid", it returns nothing and records why. The zero belongs in the table.

The pipeline is four steps:

1. **Extract.** ``pdfplumber`` gives words with positions in PDF points.
2. **Cluster.** Words are grouped into callouts geometrically -- along a line by gap, then
   across lines by vertical proximity, because a bilateral tolerance is drawn as a stack.
3. **Parse.** Each cluster's text goes through the shared grammar. What parses becomes a
   characteristic with a box; what does not is collected.
4. **Ask.** Only the leftovers are cropped and sent to a model, if one was given.

Step 4 being last is the whole point. Every callout the grammar resolves is one that cost
nothing, took no tokens, and cannot be hallucinated.

**Coordinates.** PDF points are converted to pixels of the rendered image with a single
scale factor, because ``drawgen`` rasterises the page at a uniform DPI. The Y axis needs no
flip: ``pdfplumber`` already reports ``top`` from the top of the page, in the same sense as
image pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from balloonbench.baselines.base import (
    BaselineResult,
    PromptConfig,
    call_model,
    extract_json,
    to_prediction,
)
from balloonbench.baselines.cache import ResponseCache
from balloonbench.baselines.grammar import ParseError, parse_callout
from balloonbench.baselines.prompts import schema_description
from balloonbench.baselines.providers import Provider, ProviderError
from balloonbench.evalkit.prediction import Prediction
from balloonbench.schema import Characteristic

__all__ = ["NAME", "Cluster", "cluster_words", "extract_words", "predict"]

NAME = "vector_hybrid"

#: A callout's words sit closer together than two callouts do. Both thresholds are in
#: multiples of the text height rather than in points, so they hold at any drawing scale.
GAP_RATIO = 0.9
LINE_GAP_RATIO = 0.8

#: Below this many extracted words a PDF is a scan with, at most, a stray text layer. The
#: baseline reports that rather than pretending to have read it.
VECTOR_MIN_WORDS = 8

#: Padding around an unresolved cluster before it is cropped for the model. A callout's
#: meaning often sits just outside its text -- the leader it hangs from, the frame around a
#: basic dimension -- so the crop is deliberately generous.
CROP_PAD = 24


FALLBACK = PromptConfig(
    system=(
        "You are reading a fragment of a mechanical engineering drawing. Answer only about "
        "what is visible in the fragment."
    ),
    template=(
        "This crop is {width} by {height} pixels, taken from ({offset_x}, {offset_y}) of a "
        "larger sheet. A text extractor read the following from it but could not interpret "
        "it:\n\n  {text}\n\nSay what this callout means, if it is one.\n\n{schema}"
    ),
    max_tokens=2048,
    temperature=0.0,
    variant="vector-fallback-v1",
)


@dataclass
class Cluster:
    """A group of words that look like one callout, with its box in PDF points."""

    words: list[dict[str, Any]] = field(default_factory=list)
    #: Text turned on its side, as a vertical dimension is written. Its glyphs are joined
    #: without spaces, since the PDF stores them one per word.
    rotated: bool = False

    @property
    def text(self) -> str:
        separator = "" if self.rotated else " "
        return separator.join(word["text"] for word in self.words)

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (
            min(w["x0"] for w in self.words),
            min(w["top"] for w in self.words),
            max(w["x1"] for w in self.words),
            max(w["bottom"] for w in self.words),
        )

    @property
    def height(self) -> float:
        return max(w["bottom"] - w["top"] for w in self.words)


def extract_words(pdf_path: str | Path, page: int = 0) -> tuple[list[dict[str, Any]], float]:
    """Words with positions, and the page width in points.

    ``pdfplumber`` and ``pypdfium2`` are both used in this repository and neither is
    ``pymupdf``: the core package is Apache-2.0 and must stay importable without the AGPL
    extra (see the licensing rules in CLAUDE.md).
    """
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as document:
        if page >= len(document.pages):
            return [], 0.0
        sheet = document.pages[page]
        words = sheet.extract_words(
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=["size", "upright"],
        )
        return [dict(word) for word in words], float(sheet.width)


def cluster_words(words: list[dict[str, Any]]) -> list[Cluster]:
    """Group words into callouts by position alone.

    Upright and rotated text are clustered separately, and the separation is not a nicety.
    A drawing turns its vertical dimensions on their side, and a PDF stores each rotated
    glyph as its own word because the glyphs are not horizontally adjacent -- so ``35.50``
    on a vertical dimension line arrives as five one-character words stacked bottom to top.
    Read with the horizontal rules it becomes five callouts of one character each; read with
    the upright ones it merges into whatever row it happens to cross. Both were observed
    before this split existed, and between them they cost a third of one family's recall.

    Within each orientation there are two passes: join along the reading direction by gap,
    then merge the lines that sit directly across it, because a unilateral tolerance is
    drawn as a stack and its two halves are one callout.
    """
    if not words:
        return []

    upright = [w for w in words if w.get("upright", True)]
    rotated = [w for w in words if not w.get("upright", True)]
    return _cluster_upright(upright) + _cluster_rotated(rotated)


def _cluster_upright(words: list[dict[str, Any]]) -> list[Cluster]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))

    lines: list[Cluster] = []
    current = Cluster(words=[ordered[0]])
    for word in ordered[1:]:
        previous = current.words[-1]
        height = max(previous["bottom"] - previous["top"], 1e-6)
        same_line = abs(word["top"] - previous["top"]) <= height * 0.6
        close = (word["x0"] - previous["x1"]) <= height * GAP_RATIO
        if same_line and close:
            current.words.append(word)
        else:
            lines.append(current)
            current = Cluster(words=[word])
    lines.append(current)
    return _merge_stacks(lines)


def _cluster_rotated(words: list[dict[str, Any]]) -> list[Cluster]:
    """Clusters of text turned on its side, read in the direction it was written.

    Grouped down a column and then reversed, because text rotated a quarter turn
    anticlockwise -- which is how a drawing writes a vertical dimension -- reads from the
    bottom of the sheet upwards. Joining the glyphs in the order the PDF lists them would
    spell every vertical dimension backwards, and ``05.53`` is a number, so nothing
    downstream would notice.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["x0"], 1), w["top"]))

    columns: list[list[dict[str, Any]]] = []
    current = [ordered[0]]
    for word in ordered[1:]:
        previous = current[-1]
        width = max(previous["x1"] - previous["x0"], 1e-6)
        same_column = abs(word["x0"] - previous["x0"]) <= width * 0.6
        close = (word["top"] - previous["bottom"]) <= width * GAP_RATIO
        if same_column and close:
            current.append(word)
        else:
            columns.append(current)
            current = [word]
    columns.append(current)

    return _merge_stacks(
        [Cluster(words=list(reversed(column)), rotated=True) for column in columns]
    )


def _merge_stacks(lines: list[Cluster]) -> list[Cluster]:
    merged: list[Cluster] = []
    for line in lines:
        for other in merged:
            if _stacked(other, line):
                other.words.extend(line.words)
                break
        else:
            merged.append(line)
    return merged


def _stacked(a: Cluster, b: Cluster) -> bool:
    """Whether ``b`` sits directly above or below ``a``, as the halves of one callout do.

    Two conditions, and the first is the one that matters. Requiring the horizontal overlap
    to be most of the *narrower* box stops a merge from chaining along a row: without it,
    one merge widens the cluster, the widened cluster then overlaps its neighbour, and a
    whole line of separate callouts collapses into a single string spanning the sheet. The
    second condition insists they really are stacked rather than side by side, since two
    callouts on one line trivially share their vertical extent.
    """
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    overlap = min(ax1, bx1) - max(ax0, bx0)
    narrower = min(ax1 - ax0, bx1 - bx0)
    if overlap <= 0 or overlap < 0.5 * max(narrower, 1e-6):
        return False
    if abs((ax0 + ax1) / 2 - (bx0 + bx1) / 2) > 0.6 * max(ax1 - ax0, bx1 - bx0):
        return False
    height = max(a.height, b.height)
    gap = by0 - ay1 if by0 >= ay1 else ay0 - by1
    if gap < -0.3 * height:
        return False
    return gap <= height * LINE_GAP_RATIO


def sheet_regions(pdf_path: str | Path, page: int = 0):
    """``(frame, excluded)`` in PDF points, read from the sheet's own drawn rectangles.

    A drawing states its own structure geometrically, and this reads it rather than being
    told. The border frame is the largest rectangle on the page; anything outside it is
    margin furniture -- the zone letters and numbers along the edges, which parse
    beautifully as dimensions and are not callouts. A metadata block is a rectangle inside
    the frame anchored to its edge: that is the title block, whose part number, scale and
    default surface finish are sheet metadata rather than characteristics of the part.

    No ground truth is consulted. If the page has no frame, nothing is excluded and every
    cluster is considered, which is the right behaviour for a sheet drawn without one.
    """
    import pdfplumber

    with pdfplumber.open(str(pdf_path)) as document:
        if page >= len(document.pages):
            return None, []
        sheet = document.pages[page]
        page_area = float(sheet.width) * float(sheet.height)
        rects = [
            (float(r["x0"]), float(r["top"]), float(r["x1"]), float(r["bottom"]))
            for r in sheet.rects
        ]

    def area(box) -> float:
        return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])

    frames = [r for r in rects if area(r) >= 0.6 * page_area]
    if not frames:
        return None, []
    frame = max(frames, key=area)
    frame_area = area(frame)
    frame_height = frame[3] - frame[1]

    excluded = [
        r
        for r in rects
        if r is not frame
        and area(r) >= 0.02 * frame_area
        and r[0] >= frame[0] - 1 and r[3] <= frame[3] + 1
        # Anchored to an edge of the frame: a title or revision block, not a view.
        and (abs(r[3] - frame[3]) < 0.02 * frame_height
             or abs(r[1] - frame[1]) < 0.02 * frame_height)
    ]
    return frame, excluded


def _inside(box, region) -> bool:
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]


def predict(
    image_path: str | Path,
    *,
    provider: Provider | None = None,
    model: str = "",
    drawing_id: str,
    cache: ResponseCache,
    config: PromptConfig | None = None,
    pdf_path: str | Path | None = None,
    fallback: bool = True,
    sheet: str = "unknown",
    projection: str = "unknown",
    units: str = "mm",
) -> BaselineResult:
    """Read one drawing from its vector PDF, asking a model only for what will not parse."""
    image_path = Path(image_path)
    pdf = Path(pdf_path) if pdf_path is not None else _pdf_beside(image_path)

    result = BaselineResult(prediction=to_prediction(None, drawing_id=drawing_id))
    if pdf is None or not pdf.exists():
        result.errors.append("no PDF beside the image; this baseline needs vector input")
        return result

    words, page_width = extract_words(pdf)
    if len(words) < VECTOR_MIN_WORDS:
        # The honest zero. SPEC.md section 11 predicts it for scans, and reporting it as a
        # zero rather than silently switching to a pure-VLM path is what makes the
        # prediction testable.
        result.errors.append(
            f"only {len(words)} words of vector text; this looks like a scan"
        )
        result.prediction = Prediction(
            drawing_id=drawing_id,
            units=units,
            meta={"baseline": NAME, "vector": False, "words": len(words)},
        )
        return result

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = width / page_width if page_width else 1.0

        frame, excluded = sheet_regions(pdf)
        characteristics: list[Characteristic] = []
        unresolved: list[Cluster] = []
        by_grammar = 0
        by_model = 0
        skipped = 0
        for cluster in cluster_words(words):
            if frame is not None and (
                not _inside(cluster.box, frame)
                or any(_inside(cluster.box, region) for region in excluded)
            ):
                skipped += 1
                continue
            box = _to_pixels(cluster.box, scale, width, height)
            if box is None:
                continue
            try:
                parsed = parse_callout(cluster.text)
            except ParseError:
                unresolved.append(cluster)
                continue
            item = _to_characteristic(parsed.as_payload(), box, len(characteristics) + 1)
            if item is None:
                unresolved.append(cluster)
            else:
                characteristics.append(item)
                by_grammar += 1

        asked = 0
        if fallback and provider is not None and model:
            for cluster in unresolved:
                box = _to_pixels(cluster.box, scale, width, height)
                if box is None:
                    continue
                asked += 1
                item = _ask_about(
                    cluster, box, image, image_path,
                    result=result, provider=provider, model=model,
                    config=config or FALLBACK, cache=cache,
                    drawing_id=drawing_id, units=units,
                    index=len(characteristics) + 1,
                )
                if item is not None:
                    characteristics.append(item)
                    by_model += 1

    for new_id, item in enumerate(characteristics, start=1):
        item.id = new_id

    result.prediction = Prediction(
        drawing_id=drawing_id,
        characteristics=characteristics,
        units=units,
        meta={
            "baseline": NAME,
            "model": model,
            "vector": True,
            "words": len(words),
            "clusters": by_grammar + len(unresolved),
            # The headline this baseline exists to report: how much of the sheet the parser
            # handled alone, and how much had to be paid for.
            "parsed_by_grammar": by_grammar,
            "sent_to_model": asked,
            "answered_by_model": by_model,
            "outside_the_frame": skipped,
        },
    )
    return result


def _pdf_beside(image_path: Path) -> Path | None:
    stem = image_path.stem
    while True:
        candidate = image_path.with_name(f"{stem}.pdf")
        if candidate.exists():
            return candidate
        if "_" not in stem:
            return None
        stem = stem.rsplit("_", 1)[0]


def _to_pixels(box, scale: float, width: int, height: int):
    x0, y0, x1, y1 = (value * scale for value in box)
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [x0, y0, x1, y1]


def _to_characteristic(payload: dict[str, Any], box, index: int) -> Characteristic | None:
    """Build a characteristic, or ``None`` when what parsed cannot be one.

    A grammar parse can be perfectly correct and still not be a legal characteristic: the
    grammar reads ``⏥ 0.05 A`` because it is written on real drawings, and the schema
    rejects it because a form tolerance has nothing to reference. Refusing here means the
    baseline does not emit output it knows to be invalid.
    """
    # ``count`` is a property of a pattern rather than a schema field; the grammar reports
    # it and the schema has nowhere to put it.
    fields = {k: v for k, v in payload.items() if k != "count"}
    try:
        return Characteristic.model_validate(
            {"id": index, "view": "unknown", "bbox": box, **fields}
        )
    except Exception:  # noqa: BLE001 - any validation failure means "not a characteristic"
        return None


def _ask_about(
    cluster: Cluster,
    box,
    image: Image.Image,
    image_path: Path,
    *,
    result: BaselineResult,
    provider: Provider,
    model: str,
    config: PromptConfig,
    cache: ResponseCache,
    drawing_id: str,
    units: str,
    index: int,
) -> Characteristic | None:
    """Crop one unresolved cluster and ask the model what it says."""
    x0, y0, x1, y1 = box
    crop_box = (
        int(max(0, x0 - CROP_PAD)),
        int(max(0, y0 - CROP_PAD)),
        int(min(image.width, x1 + CROP_PAD)),
        int(min(image.height, y1 + CROP_PAD)),
    )
    directory = Path(cache.root) / "fragments" / image_path.stem
    directory.mkdir(parents=True, exist_ok=True)
    crop_path = directory / ("_".join(str(v) for v in crop_box) + ".png")
    if not crop_path.exists():
        image.crop(crop_box).save(crop_path)

    rendered = config.render(
        width=crop_box[2] - crop_box[0],
        height=crop_box[3] - crop_box[1],
        offset_x=crop_box[0],
        offset_y=crop_box[1],
        text=cluster.text,
        schema=schema_description(),
    )
    try:
        text, hit = call_model(
            provider,
            model=model,
            config=config,
            rendered=rendered,
            image_path=crop_path,
            cache=cache,
            extra=(("fragment", crop_path.stem),),
        )
    except ProviderError as exc:
        result.errors.append(str(exc))
        result.cache_misses += 1
        return None

    result.raw_responses.append(text)
    result.cache_hits += int(hit)
    result.cache_misses += int(not hit)

    answered = to_prediction(extract_json(text), drawing_id=drawing_id, units=units)
    if not answered.characteristics:
        return None
    item = answered.characteristics[0]
    # The model saw a crop and answered in crop coordinates, if it gave coordinates at all.
    # The extractor already knows exactly where the text was, so its box is used instead --
    # it is the more reliable of the two by a wide margin.
    item.bbox = list(box)
    item.leader_target_bbox = None
    item.id = index
    return item
