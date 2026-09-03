"""The prompts, kept in one file because they are an experimental variable.

The gap between ``vlm_zeroshot`` and ``vlm_structured`` is supposed to measure how much of a
model's failure is prompting and how much is capability (SPEC.md section 11). That
comparison is only meaningful if the two prompts differ in the ways being tested and in no
others, which is far easier to check when they sit next to each other than when each is
buried in the baseline that sends it.

Every prompt describes the output schema in our own words. ASME Y14.5 and ISO 1101 are
copyrighted, so no rule text, table or figure from either is reproduced here or anywhere
else in the repository -- the conventions are implemented and described, never quoted.

The enum lists are built from ``balloonbench.schema`` at import rather than typed out. A
prompt that offered a symbol the schema does not accept would produce predictions that
cannot be parsed, and the resulting "schema violation" would be the benchmark's fault
rather than the model's.
"""

from __future__ import annotations

from typing import get_args

from balloonbench.baselines.base import PromptConfig
from balloonbench.schema import DimType, GtolSymbol, Kind, TolStyle

__all__ = ["STRUCTURED", "TILE", "ZEROSHOT", "schema_description"]


def _enum(annotation) -> str:
    return " | ".join(get_args(annotation))


def schema_description() -> str:
    """The output contract, written for a model rather than for a validator."""
    return f"""Return a single JSON object with one key, "characteristics", whose value is a
list. Each element describes one callout on the drawing and uses these fields:

  id                  integer, 1-based, in reading order
  kind                {_enum(Kind)}
  view                the view the callout annotates, e.g. "front", "top", "right",
                      "section_A-A"; use "unknown" if you cannot tell
  bbox                [x0, y0, x1, y1] in pixels of the image given, origin top-left,
                      tightly around the callout text or the feature control frame

For kind = "dimension" also give:
  dim_type            {_enum(DimType)}
  nominal             the number, without its symbol
  upper_tol           signed deviation from nominal, e.g. 0.025
  lower_tol           signed deviation from nominal, e.g. -0.025
  tol_style           {_enum(TolStyle)}
  fit_class           e.g. "H7", "g6"; null if the callout states no fit
  is_basic            true only if the value is boxed as theoretically exact
  is_reference        true only if the value is written in parentheses

For kind = "geometric_tolerance" also give:
  gtol_symbol         {_enum(GtolSymbol)}
  gtol_value          the tolerance zone value in the first compartment
  gtol_zone           "diametral" if the value carries a diameter symbol, else "linear"
  material_modifier   "MMC", "LMC", or null when neither is shown
  datum_refs          ordered list of {{"label": "A", "modifier": null}}, primary first

For every element also give:
  raw_text            what the callout says, transcribed as literally as you can

Rules the output must obey, which follow from what the symbols mean:
  - A tolerance that locates or orients a feature (position, perpendicularity,
    parallelism, angularity, concentricity, symmetry, either runout) must reference at
    least one datum.
  - A form tolerance (flatness, straightness, circularity, cylindricity) must reference
    none, because it describes a surface against itself.
  - A material modifier applies only to a feature that has a size, so never on a form
    tolerance, on profile of a surface, or on a runout.
  - upper_tol and lower_tol are always deviations from nominal, even where the drawing
    prints the two limits directly.
  - A basic (boxed) dimension is theoretically exact: both deviations are zero.

Report only what is drawn. Do not infer a tolerance that is not shown, and do not repeat a
callout that appears once. Output the JSON object and nothing else."""


ZEROSHOT = PromptConfig(
    system=(
        "You are reading a mechanical engineering drawing the way a quality engineer does "
        "when they prepare an inspection plan: every dimension and every geometric "
        "tolerance on the sheet, transcribed exactly as drawn."
    ),
    template=(
        "This is a {sheet} engineering drawing in {projection} projection, "
        "{width} by {height} pixels.\n\n"
        "List every dimension, geometric tolerance, thread callout, surface finish symbol "
        "and general note on it.\n\n{schema}"
    ),
    max_tokens=8192,
    temperature=0.0,
    variant="zeroshot-v1",
)


STRUCTURED = PromptConfig(
    system=(
        "You are reading a mechanical engineering drawing the way a quality engineer does "
        "when they prepare an inspection plan. Work through the sheet methodically and "
        "return only the JSON object you are asked for."
    ),
    template=(
        "This is a {sheet} engineering drawing in {projection} projection, "
        "{width} by {height} pixels.\n\n"
        "Work in this order, then answer:\n"
        "1. Identify each view on the sheet and what it shows.\n"
        "2. Find the datum feature symbols and note their letters.\n"
        "3. Read every feature control frame compartment by compartment: symbol, zone "
        "value and whether it carries a diameter symbol, material modifier, then the "
        "datum references in order.\n"
        "4. Read every dimension, including its tolerance form.\n"
        "5. Read the remaining notes, thread callouts and surface finish symbols.\n\n"
        "{schema}"
    ),
    max_tokens=8192,
    # Non-zero because this baseline votes over several samples. At zero the three draws
    # would be identical and the self-consistency pass would measure nothing.
    temperature=0.4,
    variant="structured-v1",
)


TILE = PromptConfig(
    system=STRUCTURED.system,
    template=(
        "This is a crop from a larger engineering drawing. The crop is {width} by {height} "
        "pixels and its top-left corner is at ({offset_x}, {offset_y}) in the full sheet.\n\n"
        "List every callout that is fully visible in this crop. Ignore any that is cut off "
        "at an edge: it will be read from another crop, and a partial reading would be "
        "counted twice.\n\n"
        "Give bbox coordinates relative to this crop; they will be translated.\n\n{schema}"
    ),
    max_tokens=4096,
    temperature=0.4,
    variant="tile-v1",
)
