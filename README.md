# BalloonBench

**A synthetic-first benchmark, evaluation harness, and geometry-grounded verifier for GD&T
extraction from 2D engineering drawings.**

Before a machined part can be quoted, inspected or approved, a quality engineer reads every
dimension and geometric tolerance off a 2D drawing, numbers them ("ballooning"), and
transcribes them into an inspection plan. Vision-language models can now attempt that
extraction. The blocker is not extraction — it is **knowing whether the extraction is
correct**, and there is no public labelled dataset of engineering drawings with GD&T ground
truth against which to find out.

BalloonBench closes that gap in three parts:

- a **generator** that synthesises realistic drawings whose ground truth is known *by
  construction*, because the parts and their annotations are produced programmatically;
- an **evaluation harness** whose metrics map to what a quality engineer actually audits,
  with baselines for frontier VLMs, a vector-PDF parser, and a fine-tuned detector;
- a **verifier** that cross-checks every extracted characteristic against the source solid
  model and labels it `verified` / `contradicted` / `unverifiable`, turning a probabilistic
  extraction into an auditable artifact.

The full specification and the build plan are kept as internal design documents and are not
distributed with the repository; module docstrings that cite them by section carry the
reasoning inline, so no explanation lives only in a file you cannot read.

---

## Status

Under construction, milestone by milestone.

| Milestone | State |
|---|---|
| M0 — environment, frozen schema, tests, CI | **done** |
| M1 — `partgen` (5 part families) | **done** |
| M2 — `drawgen` (projection, layout, annotation, render) | **done** |
| M3 — `degrade` (6 realism profiles) | **done** |
| M4 — `evalkit` (matching + 4 metric tiers) | **done** |
| M5 — VLM baselines (zero-shot + structured) | **done** (harness; the paid run is yours to launch) |
| M6 — callout grammar + vector-hybrid baseline | **done** |
| M7 — `verifier` | next |
| M8 — 50 hand-labelled real drawings | not started |
| M9–M11 — detector baseline, demo, write-up | not started |

---

## Quickstart

```bash
conda create -y -n balloonbench python=3.12
conda activate balloonbench
pip install -e ".[dev,vision]"

python scripts/check_env.py   # must print "ok" before anything else
pytest
```

The OCCT binding is **OCP** (shipped with CadQuery), not `pythonocc-core`. The two are
separate bindings that each bundle their own copy of the OCCT shared libraries and must not
be loaded into one interpreter, so `pythonocc-core` must never be added as a dependency and
OCCT is always imported as `from OCP.<Package> import ...`.

### Generating drawings

```bash
balloonbench families                     # what can be built
balloonbench parts flange -n 5            # solids only, as STEP
balloonbench draw flange -n 5 --dpi 300   # full drawings
balloonbench draw all -n 40 -o data/bench # a benchmark slice
```

Each drawing writes six artifacts: a vector `PDF`, a `PNG` at the requested DPI, a `DXF`
(for the vector-hybrid baseline), the `STEP` solid, the ground-truth `JSON`, and an
`overlay.png` with the ground-truth boxes drawn on the render for human QA.

A run is fully described by `(family, seed, count)`. House style, projection convention and
sheet size are sampled from the seed; pin one with `--style`, `--projection` or `--sheet` to
build a slice that differs from the default population in exactly one recorded way.

```bash
balloonbench validate data/drawings/flange/flange-00003.json
```

### Degrading them

```bash
balloonbench profiles                                        # the six conditions
balloonbench degrade data/drawings/flange/*.png -p scan_heavy
balloonbench degrade data/drawings/flange/*.png -p all -o data/bench
```

Six named profiles stand for six ways a drawing reaches the person reading it: `clean`,
`office_scan`, `scan_heavy`, `photocopy_gen3`, `phone_photo` and `blueprint_legacy`. Each is
an ordered list of seeded, individually optional transforms — rotation, keystone, lens
distortion, creases, grain, toner streaks, dropout, JPEG, plus stamps, punch holes,
handwritten notes and previous ballooning drawn over the sheet. Order is physical: paper is
marked and creased before it is copied, and the sensor and the file format come last.

The ground truth is degraded with the image, never shared with the clean version. Every
geometric warp maps the bounding boxes through the same transform as the pixels, and a
callout warped off the sheet is dropped rather than clamped — so a degraded sample's JSON
describes the degraded image and nothing else. `clean` runs the identical code path with no
transforms, so it is a baseline for the other five rather than for the generator's output.

A profile plus a seed reproduces a sample exactly, and the profile name is recorded in
`provenance.degradation_profile` so results can be reported per condition instead of
averaged into a single number that hides where a model fails.

### Evaluating predictions

```bash
balloonbench evaluate preds/*.json --gt data/drawings --out results/my-run --name my-run
balloonbench evaluate preds/*.json --gt data/drawings --no-bbox --out results/my-run-nb
```

A prediction is a JSON file naming the `drawing_id` it answers, with a list of
characteristics in the ground-truth schema. Predictions are parsed leniently: anything that
cannot be a valid characteristic — a flatness carrying a datum reference, a modifier on a
surface profile — is kept with the reason, counted as a false positive, and reported as a
schema violation rather than discarded, so a model cannot improve its precision by emitting
output that will not load.

Matching is a global assignment over a cost matrix combining box IoU with a semantic
distance, solved with the Hungarian algorithm; pairs above a threshold are rejected and
become a miss plus a false alarm. `--no-bbox` matches on content alone for models that
cannot produce coordinates. It is a materially easier task, so it is reported as its own run
and never averaged with a localised one.

Four metric tiers are reported: **detection** (precision/recall/F1, broken out by kind and
by GD&T symbol), **correctness given a match** (nominal, tolerance exact and within one last
significant digit, symbol, modifier, ordered datum references), **structure** (normalised
datum graph edit distance and full-drawing exact match), and **cost** (CTQ recall and a
weighted error total). The cost weights, with a written justification for each, live in
`balloonbench/evalkit/error_costs.json`. Every run writes `report.json` with per-drawing
rows, `report.md`, and plots of accuracy against degradation profile, symbol and drawing
complexity. `docs/metrics.md` explains what each number means, and what it deliberately
does not.

### Running a baseline

```bash
balloonbench baselines                                   # what can be run
balloonbench predict vlm_zeroshot --gt data/drawings     --model claude-opus-5 --dry-run                       # what it would cost
balloonbench predict vlm_zeroshot --gt data/drawings --model claude-opus-5
balloonbench predict vlm_structured --gt data/drawings --model claude-opus-5 -k 3
```

Two baselines so far. `vlm_zeroshot` is the control: one image, one prompt describing the
output schema, one answer, no retries. `vlm_structured` is the same model asked properly —
a worked reading order, an overlapping grid of crops merged back into sheet coordinates, and
a self-consistency vote over *k* samples that keeps only what most of them agree on. Holding
the model fixed and varying only the asking is what makes the gap between them a measurement
of prompting rather than of capability.

This is the one command in the tool that spends money, so it is built to spend as little as
possible: every response is cached on disk under a key covering the model, the exact prompt,
the image bytes, the temperature and the sample index, and the runner resumes by skipping
drawings that already have a prediction. `--dry-run` reports the number of calls a run would
make and contacts nobody.

`--model` is required and recorded verbatim in `manifest.json` alongside the prompt variant,
temperature, sample count and cache statistics. A default tracking a vendor's moving alias
would make two runs of the same command incomparable while both claimed the same model.

API keys are read from the environment (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GOOGLE_API_KEY`); the SDKs live behind the optional `baselines` extra and none is imported
until it is used.

### Reading a native PDF without a model

`vector_hybrid` extracts the PDF's own text with `pdfplumber`, clusters the words into
callouts geometrically, and parses them with a recursive-descent grammar
(`baselines/grammar.py`) shared with the detector baseline. Only the strings the grammar
cannot resolve are cropped and sent to a model.

```bash
balloonbench predict vector_hybrid --gt data/drawings --model claude-opus-5
```

On the synthetic set at 150 DPI, with **no model calls at all**, it scores:

| Family | Precision | Recall | F1 |
|---|---|---|---|
| flange | 0.97 | 0.78 | 0.87 |
| housing | 1.00 | 0.69 | 0.82 |
| plate_bracket | 0.98 | 0.90 | 0.94 |
| shaft | 1.00 | 0.88 | 0.94 |
| valve_body | 1.00 | 0.77 | 0.87 |
| **overall** | **0.99** | **0.82** | **0.90** |

Given a match, it reads the symbol, the material modifier and the datum sequence correctly
every time, and the nominal 94% of the time. Full-drawing exact match is zero, because it
has no way to know which view a callout annotates and records `view` as `"unknown"` rather
than guessing.

On a scan it returns nothing and says so. That zero is the point: SPEC's prediction is that
routing on input type matters more than model choice, and rerouting a raster sheet through a
model while still calling the result "vector hybrid" would hide exactly the finding the
baseline exists to produce.

The grammar accepts what real drawings write, not only what this repository renders —
AutoCAD's `%%c`, shop abbreviations like `TP` and `TIR`, comma decimals, inch fractions,
`(M)` alongside `Ⓜ` — and refuses anything ambiguous rather than guessing, because a wrong
parse enters the results as a confident answer while a refusal can be routed to a model. It
is gated on a 323-string corpus at `tests/data/callout_corpus.jsonl`, a quarter of which are
strings that must **not** parse.

### House styles

Five, sampled per drawing and recorded in `provenance.house_style` so results can be sliced
by them afterwards. They differ on tolerance presentation (`44±0.05` / `44 +0.05/-0.00` /
`44.05/43.95` / `⌀44 H7`), decimal convention, datum symbol form (filled triangle vs the
older `-A-`), whether a positional tolerance writes its `⌀` zone prefix, arrowhead style,
text in or above the dimension line, and metric vs imperial sheets.

---

## Scope

### In scope (v1)

Rotational parts, flanges, plates and brackets, simple prismatic housings, and simplified
valve bodies. Linear, diameter, radius, angular and chamfer dimensions; bilateral,
unilateral, limit and ISO-fit tolerances; thread callouts; surface finish symbols. Geometric
tolerances covering form (flatness, straightness, circularity, cylindricity), orientation
(perpendicularity, parallelism, angularity), location (position, concentricity, symmetry),
runout (circular, total) and profile of a surface. Feature control frames with up to three
datum references, MMC/LMC/RFS material condition modifiers, datum feature symbols, title and
revision blocks. Orthographic, section, detail and isometric-inset views, in both first- and
third-angle projection.

### Explicitly out of scope (v1)

A narrow benchmark that has been validated beats a broad one that has not. The following are
**not** covered, and this list is binding:

- composite feature control frames
- profile of a line, tangent plane, free state, statistical tolerance
- datum targets
- weld symbols
- assembly drawings, BOM tables, exploded views
- sheet metal flat patterns and bend tables
- non-English drawings
- hand-sketched drawings

---

## Licensing

The repository is **Apache-2.0**. Two dependency licences are handled deliberately:

- **PyMuPDF** is AGPL-3.0 and is isolated behind the optional extra
  `balloonbench[agpl]`. Nothing under `balloonbench/` imports it; the core rasterisation and
  vector-extraction path uses `pypdfium2` (BSD-3) and `pdfplumber` (MIT).
- **Ultralytics YOLO** is AGPL-3.0 and is not used. The detector baseline is HuggingFace
  `DetrForObjectDetection` (Apache-2.0).

Drawings are rendered with **osifont**, vendored in `assets/fonts/osifont/` under
LGPL-3.0 **with the GPL font exception**. The exception is the point: it means the PDFs and
PNGs BalloonBench generates do not inherit the font's licence just by embedding it, so the
dataset can be distributed on its own terms. The font binaries are unmodified upstream
builds; see `assets/fonts/osifont/LICENSE.md` for provenance and both licence texts.

ASME Y14.5 and ISO 1101 are copyrighted standards. BalloonBench implements the *conventions*
they describe — conventions are not copyrightable — but reproduces no text, table or figure
from either standard. Every rule implemented here is described in our own words.

The real-drawing split will carry a `manifest.yaml` recording source URL, licence, retrieval
date and any redaction for every sheet, so that anyone can verify our right to distribute it.
