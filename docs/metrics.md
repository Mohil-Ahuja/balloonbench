# How a prediction is scored

This document explains the choices behind `balloonbench/evalkit` — what each number means,
and what it deliberately does not mean. The code carries the reasoning inline; this is the
version for someone deciding whether to trust a leaderboard entry.

## Matching comes first, and it is a choice

A prediction has no balloon numbers that correspond to the ground truth's, so every metric
rests on a correspondence the harness invents. Getting that correspondence wrong moves every
downstream number, which makes it the first thing to be sceptical about.

The cost of pairing a ground-truth callout with a prediction is

```
cost = w_geo * (1 - IoU(bbox)) + w_sem * semantic_distance
```

with the assignment solved globally by the Hungarian algorithm, not greedily. Greedy
matching lets a confident prediction claim a callout that a second prediction fits better,
and the displacement cascades: on a plate with four identical corner holes it is the
difference between finding all four and mislabelling all four.

Two pairs can never match, whatever their boxes look like:

* **A kind mismatch.** A predicted flatness is not a dimension. Allowing a high-IoU pair
  through here would score as a correct detection with wrong content, borrowing from tier 2
  to flatter tier 1.
* **Disjoint boxes**, in bbox mode. A strong semantic score must not drag a prediction onto
  a callout on the other side of the sheet.

Both cost exactly 1.0, and `max_cost` is required to lie strictly below 1.0 so neither can
ever be admitted by a loosened threshold.

`max_cost` itself defaults to 0.6, which does admit a pair whose box overlaps well but whose
content is wholly wrong. That is intentional: *found it and read it wrong* and *did not find
it* are different failures, and the point of having tiers is to report them separately
rather than to collapse them into one number.

### `--no-bbox` is a different task, not a lenient setting

Many VLMs cannot produce reliable coordinates. Semantic-only matching gives them a fair
number, but it is an easier problem — there is no localisation to get wrong — so the two are
reported as separate runs and never averaged. Published extraction results routinely conflate
the two; keeping them apart is part of what this benchmark is for.

## The four tiers

| Tier | Question | Metrics |
|---|---|---|
| 1 | Did it find the callout? | precision, recall, F1, broken out by kind and by GD&T symbol |
| 2 | Having found it, did it read it right? | `nominal_exact`, `tolerance_exact`, `tolerance_within_1lsd`, `symbol_accuracy`, `modifier_accuracy`, `datum_ref_exact` |
| 3 | Is the drawing wired together correctly? | datum graph edit distance, full-drawing exact match |
| 4 | What would the errors have cost? | CTQ recall, weighted error cost |

Three rules apply throughout.

**Every rate carries its counts.** `100.0% (3/3)` and `100.0% (300/300)` are the same
fraction and very different evidence.

**Aggregation sums counts, never averages rates.** Averaging per-drawing rates would give a
two-callout sheet the same weight as a twenty-callout one.

**A tier-2 rate is computed only over the pairs where its question applies.** Scoring
`modifier_accuracy` over dimensions, where the answer is trivially "no modifier, correct",
would inflate every model's score by the same meaningless amount.

### Why `tolerance_within_1lsd` reads the text

The last significant digit is a property of how the drawing is *written*, not of the number:
`⌀44.00 ±0.05` claims two decimal places and `⌀44 ±0.5` claims one, so the same absolute
error is one unit in the last place on the first and ten on the second. The metric takes the
digit count from the transcribed text for that reason.

### The datum graph distance is a restricted edit distance

General graph edit distance is NP-hard. The version here is not an approximation of it — it
is the exact answer to a smaller question, made smaller by a fact about the domain: **a
datum's label is its identity.** Datum A in the prediction is datum A in the ground truth or
it is nothing at all. With the node correspondence fixed, the edit distance is the symmetric
difference of the node and edge sets, which is exact, cheap, and readable as "how many wires
are wrong".

Edges carry their position in the reference sequence, because `|A|B|` and `|B|A|` constrain
different degrees of freedom. A set-based comparison would score a reversed frame as correct,
which is precisely the silent failure this tier exists to catch: every value right, every box
right, and the part measured from the wrong faces.

Matched characteristics are keyed by their ground-truth id so the two graphs are comparable;
an unmatched characteristic on either side keys uniquely and contributes its edges as pure
insertions or deletions.

## The cost weights, and why they are what they are

The weights live in `balloonbench/evalkit/error_costs.json`, each with its justification
beside it. They are a proxy, chosen to be defensible rather than precise — every shop would
quote them differently, and the file exists so that a reader who disagrees can change one
number and re-run rather than argue with a hard-coded constant.

| Error | Cost | The argument |
|---|---|---|
| Missed characteristic | 3.0 | It never reaches the inspection plan, so nothing downstream catches it. Strictly worse than a wrong value, which a first-article measurement contradicts. |
| Spurious characteristic | 1.0 | Costs inspection time and an argument with the supplier, but it is visible: someone looks for it and does not find it. |
| Wrong nominal | 1.0 | The reference error — one wrong value on a non-critical feature. |
| Wrong tolerance | 1.5 | Less likely to be noticed than a wrong nominal. A part measured against a loosened tolerance passes inspection and fails in assembly. |
| Wrong symbol | 2.0 | A different geometric characteristic is a different measurement with different fixturing, not merely a different label. |
| Wrong modifier | 2.0 | Dropping MMC silently removes bonus tolerance and rejects conforming parts; adding it accepts non-conforming ones. This is where models fail most, which is why it is not cheap. |
| Wrong datum references | 2.5 | The reference frame decides where the zone is. Every value can be right and the part still measured from the wrong faces. |
| Schema violation | 1.0 | A surcharge on top of the spurious charge, for output that would not load into a PPAP form at all. |

A characteristic tagged `is_critical` multiplies its charges by **4.0**. Getting a critical
characteristic wrong does not cost paperwork, it costs a part — scrap, a containment sort, or
a returned shipment. Four is a deliberately conservative stand-in for a ratio that is much
larger in practice.

Errors are charged per field, not per callout: a frame with the wrong symbol *and* the wrong
datum references is two distinct problems for the inspector, and charging only the larger
would make a model that fails twice look no worse than one that fails once.

## What the harness does not claim

* **It does not decide whether a drawing is well-dimensioned.** That is the verifier's job
  (`balloonbench/verifier`), and it is a different question.
* **It does not reward well-formed output.** Predictions that cannot be parsed as
  characteristics are counted as false positives, not dropped. A model cannot raise its
  precision by emitting garbage.
* **It does not average across degradation profiles by default.** A single number over six
  conditions hides where a model actually breaks, which is the finding the benchmark is
  built to produce.
