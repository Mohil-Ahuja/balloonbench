# The verifier

Everything else in this repository measures a model against a label. This module measures a
drawing against a part.

Given a STEP file and a list of extracted characteristics, it decides for each one whether
the geometry agrees. That is a different question from "did the model read the sheet
correctly", and it can be asked without any ground truth at all — which is what makes it
deployable on a customer's own drawings, where no ground truth exists.

## The asymmetry everything follows from

> A verifier that flags correct extractions is worse than no verifier.

A false contradiction sends a quality engineer to re-read a characteristic that was right.
After a few of those, nobody opens the report again, and a tool nobody opens has a recall of
zero regardless of what its table says. So every design decision in this module trades
recall away for precision, deliberately:

- **Three verdicts, not two.** Anything uncertain is `unverifiable`. `contradicted` is
  reserved for cases where the geometry positively disagrees.
- **Ambiguity is reported, not resolved.** A symmetric part genuinely has the same distance
  in several places, and the drawing does not say which one a dimension means. Saying so is
  right; picking one and calling it verified is not.
- **Near misses are never contradictions.** Something outside the stated tolerance but close
  to it is where a real tolerance interpretation, a fillet, or a hair of modelling
  difference lives.
- **Drawing defects are a separate stream.** A positional tolerance referencing one planar
  datum is a defect in the drawing, whoever read it. Reporting it as an extraction error
  would blame the model for the draughtsman's mistake.

## The B-rep index

`brep_index.py` is useful on its own. It loads any STEP file and answers what the solid
actually has:

| Query | What it answers |
|---|---|
| `diameters`, `radii` | every cylinder and cone, since a cast wall with draft is a cone |
| `plane_pairs()` | parallel faces and their separations — a wall thickness, a boss height |
| `axis_distances()` | hole to hole |
| `axis_to_plane_distances()` | hole to edge, the commonest dimension on a plate |
| `hole_patterns()` | coaxial clusters, with `uniform` distinguishing a bolt circle from a rectangular pattern |
| `envelope` | overall extents |
| `tap_drill_for(d)` | the hole a tapped callout is modelled as |

Three properties matter:

**It measures, it does not read labels.** The index never consults `partgen`'s semantic
features. Geometry that arrives pre-labelled with what it was meant to be would make every
check circular, and the same code has to work on an imported STEP that nobody here built.

**It tolerates bad input.** Degenerate faces, slivers and unclassifiable surfaces are
skipped and counted, not raised. `stats()` reports the count, which is the robustness
statistic for a stress test on public 3D datasets.

**It returns distances, not yes or no.** A check decides for itself whether a near miss is a
match, an ambiguity, or a contradiction. Folding that decision into the index would put the
conservatism in the wrong file.

## The five checks

**`size_exists`** — does a cylinder, plane pair, axis distance, bolt circle or envelope
extent of that size exist? Four-way: one match verifies, several distinct matches are
ambiguous, a near miss is unverifiable, nothing at all contradicts. A contradiction carries
the nearest real values, and a suggested correction when one clearly stands out.

**`datum_dof`** — a rigid body has six degrees of freedom. Each datum in the frame removes
some of what is left, so a secondary cannot re-remove what the primary already took. A
locating tolerance needs the frame to fix a position; an orienting one needs only a
direction; runout needs an axis to spin about. One deliberate leniency: a frame including an
axis-bearing datum may leave rotation about that axis free, because a circular bolt pattern
positioned from its own bore is how flanges are drawn, and flagging it would fire on almost
every flange ever made.

**`mmc_consistency`** — a material modifier promises a bonus that grows with the feature's
departure from maximum material, so it needs a size tolerance to depart within. The check
computes the bonus and the virtual condition boundary, and flags the boundary when it leaves
no material between neighbouring holes. Where the size is governed by the title block's
general note it reports `unverifiable` rather than a defect: the note *is* a size tolerance,
and calling that a defect flagged five correctly drawn flanges out of eight.

**`tolerance_stack`** — a set of dimensions summing to another is a chain. If both the chain
and the overall carry tolerances, the same distance is constrained twice and the two will
disagree on every real part. Bounded hard — chains of two or three, exact sums only, same
view — because the alternative is finding coincidences.

**`unit_sanity`** — the largest dimension and the largest envelope extent should be the same
order of magnitude. A ratio near 25.4 or 1/25.4 is mm/inch confusion, which is silent,
expensive, and invisible to every other check because it is *every* number at once.

## What it catches, measured

Five families × 8 seeds, 303 characteristics, injected errors compared against the clean
report so only *new* findings count:

| | |
|---|---|
| **False positives on correct ground truth** | **1.3%** (4 of 303) |
| `perturb_nominal` — a misread digit | 82% |
| `underconstrain_drf` — frame stripped to one datum | 100% |
| `undeclared_datum` — reference to a letter never established | 100% |
| `unit_confusion` — whole sheet in the wrong units | 100% |
| `swap_datum`, `shuffle_datum_refs`, `flip_modifier`, `drop_characteristic` | undetectable by geometry |

The last row is the honest one, and the harness proves it rather than asserting it. An
injection is labelled undetectable only when the verifier's own sufficiency rule says both
the original and the damaged frame constrain the part adequately — meaning the damaged
drawing is still internally consistent and still satisfiable by the solid. Nothing working
from geometry alone can distinguish them, and pushing it to try is precisely how the
false-positive rate climbs. Those errors are `evalkit`'s tier-3 datum graph distance's to
catch, and it has ground truth to compare against.

## Four things the measurement itself taught us

Each is now a named test, because each was a silent wrong answer before it was found:

1. **A drafted wall is a cone.** Matching only cylinders contradicted the valve body's ⌀100
   body wall on every seed.
2. **A tapped hole is modelled at its drill diameter.** An M14 callout finds a ⌀12 hole. The
   first version of the exemption accepted any cylinder between 0.75 and 0.98 of the stated
   size, which let a nominal misread as 125 be waved through because the part had a ⌀100
   face. It now consults a pitch table, and only for standard thread sizes.
3. **Every rectangle's corners are equidistant from its centre**, so distance alone calls a
   four-hole rectangular pattern a bolt circle. Only equal angular spacing makes it one.
4. **A recall harness must diff against the clean report.** An earlier version counted any
   finding at all and scored the verifier as catching errors it had not noticed, using
   complaints it had been making about the undamaged drawing all along.
