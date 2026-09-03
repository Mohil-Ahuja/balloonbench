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
| M4 — `evalkit` (matching + 4 metric tiers) | next |
| M5–M6 — VLM and vector-hybrid baselines | not started |
| M7 — `verifier` | not started |
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
