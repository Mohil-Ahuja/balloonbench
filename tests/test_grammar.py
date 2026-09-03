"""Milestone 6 acceptance tests: the callout grammar and the vector-hybrid baseline.

PLAN.md's M6 gate is *grammar passes a 300-string corpus*, and the corpus lives beside this
file at ``tests/data/callout_corpus.jsonl``. Every line is a string a drawing might carry
and what it means; the gate is that all of them parse to what they say, and that the
deliberate nonsense among them is refused rather than parsed into something plausible.

Refusal matters as much as acceptance here. A parser that reads ``SECTION A-A`` as a
dimension of 0 or ``⌖`` as a positional tolerance with no value has not failed loudly, it
has put a fabricated characteristic into the results -- so roughly a sixth of the corpus is
strings that must *not* parse, and they are asserted as firmly as the ones that must.

The vector-hybrid tests then check the pipeline the grammar feeds: that the text really is
read out of the PDF, that a scan is reported as a scan instead of being quietly rerouted
through a model, and that a run with no provider makes no calls at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from balloonbench.baselines.cache import ResponseCache
from balloonbench.baselines.grammar import (
    ParseError,
    normalise,
    parse_callout,
    parse_many,
)
from balloonbench.baselines.providers import ScriptedProvider
from balloonbench.baselines.vector_hybrid import (
    VECTOR_MIN_WORDS,
    Cluster,
    cluster_words,
    extract_words,
    predict,
    sheet_regions,
)
from balloonbench.drawgen.generate import generate_drawing
from balloonbench.evalkit.metrics import evaluate_drawing
from balloonbench.partgen.registry import load_families
from balloonbench.schema import Characteristic

load_families()

CORPUS_PATH = Path(__file__).parent / "data" / "callout_corpus.jsonl"
TEST_DPI = 150.0


def _corpus() -> list[dict]:
    cases = []
    for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if "_comment" in case:
            continue
        cases.append(case)
    return cases


CORPUS = _corpus()


def _label(case: dict) -> str:
    return case["text"] or "<empty>"


# --- the gate ----------------------------------------------------------------------------


def test_the_corpus_is_the_size_the_spec_asks_for():
    """SPEC.md section 11 asks for about 300 strings, and a shrinking corpus is how a gate
    stops being a gate."""
    assert len(CORPUS) >= 300
    negatives = [c for c in CORPUS if c.get("error")]
    assert len(negatives) >= 25, "too few strings that must be refused"
    kinds = {c.get("kind") for c in CORPUS if not c.get("error")}
    assert kinds == {"dimension", "geometric_tolerance", "thread", "surface_finish"}


@pytest.mark.parametrize("case", CORPUS, ids=_label)
def test_the_corpus_parses_to_what_it_says(case):
    text = case["text"]

    if case.get("error"):
        with pytest.raises(ParseError):
            parse_callout(text)
        return

    parsed = parse_callout(text)
    assert parsed.kind == case["kind"], f"{text!r} read as {parsed.kind}"
    assert parsed.raw_text == text, "the original text must survive the parse"

    for key, expected in case.items():
        if key in {"text", "kind", "error"} or key.startswith("_"):
            continue
        if key == "n_refs":
            got = len(parsed.fields.get("datum_refs", []))
        else:
            got = parsed.fields.get(key)
        if isinstance(expected, float) and isinstance(got, int | float):
            assert got == pytest.approx(expected), f"{text!r}: {key}"
        else:
            assert got == expected, f"{text!r}: {key} is {got!r}"


# --- the parts of the grammar worth stating on their own ---------------------------------


def test_normalisation_folds_spellings_without_folding_case():
    """``M12`` is a thread and ``m`` is not, so case survives; ``Ø`` and ``DIA`` do not."""
    assert normalise("Ø44") == "⌀44"
    assert normalise("DIA 44") == "⌀ 44"
    assert normalise("%%c44") == "⌀44"
    assert normalise("4x ⌀12") == "4X ⌀12"
    assert normalise("M12X1.75") == "M12X1.75"
    assert normalise("  44 \n ±0.05 ") == "44 ±0.05"


def test_a_limit_dimension_is_stored_as_deviations():
    """Schema rule R4: both bounds are signed deviations from nominal, whatever the sheet
    prints. The renderer decides display; storage is not its business."""
    parsed = parse_callout("44.05/43.95")
    assert parsed.fields["nominal"] == pytest.approx(44.0)
    assert parsed.fields["upper_tol"] == pytest.approx(0.05)
    assert parsed.fields["lower_tol"] == pytest.approx(-0.05)


def test_a_reversed_deviation_pair_is_put_back_in_order():
    parsed = parse_callout("44 -0.05/+0.05")
    assert parsed.fields["upper_tol"] == pytest.approx(0.05)
    assert parsed.fields["lower_tol"] == pytest.approx(-0.05)


def test_the_datum_sequence_keeps_its_order_and_its_modifiers():
    parsed = parse_callout("⌖ ⌀0.05 Ⓜ A B Ⓜ C")
    assert [(r["label"], r["modifier"]) for r in parsed.fields["datum_refs"]] == [
        ("A", None),
        ("B", "MMC"),
        ("C", None),
    ]


def test_rfs_is_the_absence_of_a_modifier_not_a_modifier():
    """Writing (S) says only what silence already says, so it must not become a value that
    a later comparison would call different from silence."""
    assert parse_callout("⌖ ⌀0.05 (S) A").fields["material_modifier"] is None
    assert parse_callout("⌖ ⌀0.05 A").fields["material_modifier"] is None


def test_the_parser_refuses_rather_than_guesses():
    """The design decision this file exists to protect: a wrong parse enters the results as
    a confident answer, while a refusal is visibly a refusal and can be routed to a model."""
    for text in ("SECTION A-A", "⌖", "44 ±", "REV C"):
        with pytest.raises(ParseError):
            parse_callout(text)


def test_a_parse_error_says_where_it_gave_up():
    with pytest.raises(ParseError) as caught:
        parse_callout("44 +0.05/")
    assert caught.value.position > 0
    assert caught.value.text


def test_parse_many_keeps_the_failures():
    parsed, failed = parse_many(["⌀44", "SEE NOTE 3", "R5"])
    assert [p.fields["nominal"] for p in parsed] == [44.0, 5.0]
    assert len(failed) == 1


def test_a_kind_hint_narrows_the_grammar_without_overriding_it():
    assert parse_callout("M12", kind_hint="thread").kind == "thread"
    with pytest.raises(ParseError):
        parse_callout("SEE DETAIL A", kind_hint="dimension")


def test_what_the_grammar_reads_is_not_always_a_legal_characteristic():
    """Real drawings carry a flatness with a datum reference; it is written and it is wrong.
    The grammar reads it, because reading is its job, and the schema is what rejects it."""
    parsed = parse_callout("⏥ 0.05 A")
    assert parsed.fields["datum_refs"]
    with pytest.raises(Exception, match="form tolerance"):
        Characteristic.model_validate(
            {"id": 1, "view": "front", "bbox": [0, 0, 10, 10], **parsed.as_payload()}
        )


# --- the vector-hybrid baseline ----------------------------------------------------------


@pytest.fixture(scope="module")
def drawing(tmp_path_factory):
    out = tmp_path_factory.mktemp("vector")
    return generate_drawing("plate_bracket", 5, out, dpi=TEST_DPI)


def test_words_come_out_of_the_pdf_with_positions(drawing):
    words, page_width = extract_words(drawing.paths["pdf"])
    assert len(words) > VECTOR_MIN_WORDS
    assert page_width > 0
    assert {"text", "x0", "x1", "top", "bottom"} <= set(words[0])


def test_the_frame_and_the_title_block_are_found_from_the_sheet_itself(drawing):
    """Read off the drawn rectangles, never from the ground truth. Zone marks live outside
    the frame and title-block text is sheet metadata; both parse beautifully as dimensions
    and neither is a characteristic."""
    frame, excluded = sheet_regions(drawing.paths["pdf"])
    assert frame is not None
    assert excluded, "no title block found"
    for region in excluded:
        assert region[0] >= frame[0] - 1 and region[2] <= frame[2] + 1


def test_rotated_text_is_read_in_the_direction_it_was_written():
    """A vertical dimension arrives one glyph per word, bottom to top. Joined in the order
    the PDF lists them it spells the number backwards -- and ``05.53`` is a number, so
    nothing downstream would notice."""
    column = [
        {"text": "0", "x0": 100, "x1": 112, "top": 200, "bottom": 212, "upright": False},
        {"text": "5", "x0": 100, "x1": 112, "top": 212, "bottom": 224, "upright": False},
        {"text": ".", "x0": 100, "x1": 112, "top": 224, "bottom": 230, "upright": False},
        {"text": "5", "x0": 100, "x1": 112, "top": 230, "bottom": 242, "upright": False},
        {"text": "3", "x0": 100, "x1": 112, "top": 242, "bottom": 254, "upright": False},
    ]
    clusters = cluster_words(column)
    assert len(clusters) == 1
    assert clusters[0].text == "35.50"
    assert parse_callout(clusters[0].text).fields["nominal"] == pytest.approx(35.5)


def test_upright_and_rotated_text_do_not_merge_with_each_other():
    words = [
        {"text": "71.00", "x0": 100, "x1": 160, "top": 200, "bottom": 212, "upright": True},
        {"text": "4", "x0": 130, "x1": 142, "top": 214, "bottom": 226, "upright": False},
    ]
    assert len(cluster_words(words)) == 2


def test_a_stacked_tolerance_is_one_callout_not_two():
    words = [
        {"text": "44", "x0": 100, "x1": 130, "top": 200, "bottom": 212, "upright": True},
        {"text": "±0.05", "x0": 100, "x1": 140, "top": 214, "bottom": 226, "upright": True},
    ]
    clusters = cluster_words(words)
    assert len(clusters) == 1
    assert parse_callout(clusters[0].text).fields["upper_tol"] == pytest.approx(0.05)


def test_a_merge_cannot_chain_along_a_row():
    """The failure this rule exists for: one merge widens a cluster, the widened cluster
    then overlaps its neighbour, and a whole row collapses into one string spanning the
    sheet. Observed before the centre-proximity rule existed."""
    words = [
        {"text": "56.80", "x0": 100, "x1": 160, "top": 200, "bottom": 212, "upright": True},
        {"text": "4X", "x0": 60, "x1": 90, "top": 214, "bottom": 226, "upright": True},
        {"text": "E", "x0": 900, "x1": 912, "top": 206, "bottom": 218, "upright": True},
    ]
    clusters = cluster_words(words)
    assert all(c.box[2] - c.box[0] < 400 for c in clusters)


def test_the_baseline_reads_a_drawing_without_calling_a_model(drawing):
    """The point of the baseline: what the grammar resolves costs nothing, takes no tokens,
    and cannot be hallucinated."""
    provider = ScriptedProvider(responses=["{}"])
    result = predict(
        drawing.paths["png"],
        provider=provider,
        model="unused",
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=drawing.paths["png"].parent / "cache"),
        fallback=False,
        units=drawing.drawing.units,
    )
    assert provider.calls == []
    assert result.prediction.meta["vector"] is True
    assert result.prediction.meta["parsed_by_grammar"] >= 5

    score = evaluate_drawing(drawing.drawing, result.prediction)
    assert score.precision is not None and score.precision >= 0.7
    assert score.recall is not None and score.recall >= 0.6


def test_the_baseline_reports_a_scan_as_a_scan(drawing, tmp_path):
    """SPEC.md section 11 predicts this baseline is useless on scans. Rerouting the sheet
    through a model and reporting the number as 'vector hybrid' would hide the finding the
    baseline exists to produce, so the zero is returned as a zero."""
    import pypdfium2 as pdfium
    from PIL import Image

    raster = tmp_path / "scan.pdf"
    document = pdfium.PdfDocument(str(drawing.paths["pdf"]))
    try:
        page = document[0].render(scale=0.5).to_pil()
    finally:
        document.close()
    Image.new("RGB", page.size, "white").paste(page)
    page.save(raster, "PDF")

    result = predict(
        drawing.paths["png"],
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
        pdf_path=raster,
    )
    assert result.prediction.characteristics == []
    assert result.prediction.meta["vector"] is False
    assert "scan" in result.errors[0]


def test_a_missing_pdf_is_an_error_not_a_silent_empty_answer(drawing, tmp_path):
    lonely = tmp_path / "no_pdf_here.png"
    lonely.write_bytes(drawing.paths["png"].read_bytes())
    result = predict(
        lonely,
        drawing_id=drawing.drawing_id,
        cache=ResponseCache(root=tmp_path / "cache"),
    )
    assert result.errors and "needs vector input" in result.errors[0]


def test_the_cluster_box_is_the_union_of_its_words():
    cluster = Cluster(
        words=[
            {"text": "a", "x0": 10, "x1": 20, "top": 5, "bottom": 15},
            {"text": "b", "x0": 25, "x1": 40, "top": 6, "bottom": 16},
        ]
    )
    assert cluster.box == (10, 5, 40, 16)
    assert cluster.text == "a b"
