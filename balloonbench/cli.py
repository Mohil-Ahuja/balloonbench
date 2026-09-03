"""BalloonBench command line. Subcommands are added as their milestones land."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console

from balloonbench.schema import SCHEMA_VERSION, Drawing

app = typer.Typer(
    add_completion=False,
    help="BalloonBench: GD&T extraction benchmark, harness and verifier.",
)
console = Console()


@app.command()
def version() -> None:
    """Print the package and schema versions."""
    from balloonbench import __version__

    console.print(f"balloonbench {__version__} (schema {SCHEMA_VERSION})")


# typer builds its parameter metadata from default values, so these must be constructed
# once at module level rather than in the signature (ruff B008).
_PATHS_ARG = typer.Argument(..., help="Ground-truth JSON files to validate.")
_FAMILY_ARG = typer.Argument(..., help="Part family, or 'all'.")
_COUNT_OPT = typer.Option(10, "--count", "-n", help="How many parts to build.")
_SEED_OPT = typer.Option(0, "--seed", "-s", help="First seed.")
_OUT_OPT = typer.Option(Path("data/parts"), "--out", "-o", help="STEP output directory.")
_DRAW_OUT_OPT = typer.Option(
    Path("data/drawings"), "--out", "-o", help="Drawing output directory."
)
_DPI_OPT = typer.Option(300.0, "--dpi", help="Raster resolution for the PNG.")
_STYLE_OPT = typer.Option(
    None, "--style", help="House style name; sampled from the seed when omitted."
)
_PROJECTION_OPT = typer.Option(
    None, "--projection", help="first_angle or third_angle; sampled when omitted."
)
_SHEET_OPT = typer.Option(None, "--sheet", help="A4/A3/A2/A1/A0; sampled when omitted.")
_IMAGES_ARG = typer.Argument(..., help="Rendered PNGs to degrade; each needs a sibling .json.")
_PROFILE_OPT = typer.Option(
    ..., "--profile", "-p", help="Degradation profile name, or 'all'."
)
_DEG_OUT_OPT = typer.Option(
    Path("data/degraded"), "--out", "-o", help="Degraded output directory."
)
_DEG_SEED_OPT = typer.Option(0, "--seed", "-s", help="Degradation seed.")


@app.command()
def validate(paths: list[Path] = _PATHS_ARG) -> None:
    """Validate ground-truth JSON against the frozen schema."""
    failures = 0
    for path in paths:
        try:
            drawing = Drawing.from_json(path)
        except Exception as exc:  # noqa: BLE001 - report the reason, whatever it is
            failures += 1
            console.print(f"[red]FAIL[/red] {path}")
            console.print(f"       {exc}")
        else:
            console.print(
                f"[green] ok [/green] {path} "
                f"({len(drawing.characteristics)} characteristics, "
                f"{len(drawing.datums)} datums)"
            )
    if failures:
        raise typer.Exit(code=1)


@app.command("families")
def list_families() -> None:
    """List the registered part families."""
    from balloonbench.partgen.registry import load_families

    for name in load_families():
        console.print(name)


@app.command("parts")
def generate_parts(
    family: str = _FAMILY_ARG,
    count: int = _COUNT_OPT,
    start_seed: int = _SEED_OPT,
    out: Path = _OUT_OPT,
) -> None:
    """Build parts and export them as STEP.

    Seeds are consecutive from ``--seed``, so a run is fully described by
    ``(family, seed, count)`` and can be reproduced anywhere.
    """
    from balloonbench.partgen.registry import build_part, load_families
    from balloonbench.partgen.types import UnbuildableParams

    known = load_families()
    targets = list(known) if family == "all" else [family]
    for name in targets:
        if name not in known:
            console.print(f"[red]unknown family {name!r}; known: {known}")
            raise typer.Exit(code=1)

    built = rejected = 0
    for name in targets:
        for seed in range(start_seed, start_seed + count):
            try:
                part = build_part(name, seed, step_dir=out / name)
            except UnbuildableParams as exc:
                rejected += 1
                console.print(f"[yellow]skip[/yellow] {name} seed {seed}: {exc}")
                continue
            built += 1
            console.print(
                f"[green] ok [/green] {part.step_path}  "
                f"{len(part.faces)} faces, {len(part.features)} features"
            )

    console.print(f"\n{built} built, {rejected} rejected -> {out}")


@app.command("draw")
def generate_drawings(
    family: str = _FAMILY_ARG,
    count: int = _COUNT_OPT,
    start_seed: int = _SEED_OPT,
    out: Path = _DRAW_OUT_OPT,
    dpi: float = _DPI_OPT,
    style: str | None = _STYLE_OPT,
    projection: str | None = _PROJECTION_OPT,
    sheet: str | None = _SHEET_OPT,
) -> None:
    """Generate drawings: PDF, PNG, DXF, STEP, ground truth and a QA overlay.

    A run is fully described by ``(family, seed, count)``. The style, projection convention
    and sheet size are sampled from the seed unless given explicitly, so pinning one of
    them is how a slice of the benchmark is built -- ``--style legacy_shop`` gives a set
    that differs from the default population in exactly one recorded way.
    """
    from balloonbench.drawgen.generate import generate_drawing
    from balloonbench.drawgen.styles import get_style, style_names
    from balloonbench.partgen.registry import load_families

    known = load_families()
    targets = list(known) if family == "all" else [family]
    for name in targets:
        if name not in known:
            console.print(f"[red]unknown family {name!r}; known: {known}")
            raise typer.Exit(code=1)
    if style is not None and style not in style_names():
        console.print(f"[red]unknown style {style!r}; known: {list(style_names())}")
        raise typer.Exit(code=1)

    chosen = get_style(style) if style else None
    made = failed = 0
    for name in targets:
        for seed in range(start_seed, start_seed + count):
            try:
                bundle = generate_drawing(
                    name, seed, out / name, dpi=dpi, style=chosen,
                    projection=projection, sheet=sheet,
                )
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failed += 1
                console.print(f"[red]FAIL[/red] {name} seed {seed}: {exc}")
                continue
            made += 1
            console.print(
                f"[green] ok [/green] {bundle.drawing_id}  "
                f"{bundle.characteristics} characteristics, "
                f"{len(bundle.drawing.datums)} datums, "
                f"{bundle.style.name}, {bundle.drawing.sheet.size} "
                f"{bundle.drawing.projection}"
            )

    console.print(f"\n{made} drawn, {failed} failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


@app.command("profiles")
def list_profiles() -> None:
    """List the degradation profiles."""
    from balloonbench.degrade import profile_names

    for name in profile_names():
        console.print(name)


_GT_OPT = typer.Option(
    ..., "--gt", "-g", help="Ground-truth directory, searched recursively by drawing id."
)
_BASELINE_ARG = typer.Argument(..., help="Baseline name, e.g. vlm_zeroshot.")
_MODEL_OPT = typer.Option(
    ..., "--model", "-m", help="Exact model version string; recorded in the manifest."
)
_PROVIDER_OPT = typer.Option(
    None, "--provider", help="anthropic/openai/gemini; inferred from the model when omitted."
)
_PREDICT_OUT_OPT = typer.Option(
    Path("data/predictions"), "--out", "-o", help="Where predictions are written."
)
_CACHE_OPT = typer.Option(
    Path("data/cache/baselines"), "--cache", help="Response cache directory."
)
_LIMIT_OPT = typer.Option(None, "--limit", "-n", help="Stop after this many drawings.")
_SAMPLES_OPT = typer.Option(3, "--samples", "-k", help="Self-consistency samples.")
_NO_TILE_OPT = typer.Option(False, "--no-tile", help="Skip the tiled crops.")
_DRY_RUN_OPT = typer.Option(
    False, "--dry-run", help="Report what the run would cost and make no calls."
)


@app.command("baselines")
def list_baselines() -> None:
    """List the available extraction baselines."""
    from balloonbench.baselines.run import BASELINES

    for name in sorted(BASELINES):
        console.print(name)


@app.command("predict")
def run_baseline_command(
    baseline: str = _BASELINE_ARG,
    gt: Path = _GT_OPT,
    model: str = _MODEL_OPT,
    provider: str | None = _PROVIDER_OPT,
    out: Path = _PREDICT_OUT_OPT,
    cache_dir: Path = _CACHE_OPT,
    limit: int | None = _LIMIT_OPT,
    samples: int = _SAMPLES_OPT,
    no_tile: bool = _NO_TILE_OPT,
    dry_run: bool = _DRY_RUN_OPT,
) -> None:
    """Run an extraction baseline over a set of drawings.

    This is the one command in the tool that spends money. It resumes by default -- a
    drawing whose prediction already exists is skipped -- and every response is cached on
    disk, so a repeated run costs nothing. ``--dry-run`` reports the number of calls a run
    would make and contacts nobody, which is worth doing before pointing it at 500 sheets.

    ``--model`` is required and is recorded verbatim in the manifest. SPEC.md section 11
    asks for pinned versions, and a default tracking a vendor's moving alias would make two
    runs of the same command incomparable while both claimed the same model.
    """
    from balloonbench.baselines.cache import ResponseCache
    from balloonbench.baselines.providers import get_provider, provider_for_model
    from balloonbench.baselines.run import BASELINES, image_for, run_baseline

    if baseline not in BASELINES:
        console.print(f"[red]unknown baseline {baseline!r}; known: {sorted(BASELINES)}")
        raise typer.Exit(code=1)

    drawings = sorted(Path(gt).rglob("*.json"))
    drawings = [p for p in drawings if p.name != "manifest.json"]
    if limit is not None:
        drawings = drawings[:limit]
    if not drawings:
        console.print(f"[red]no ground-truth drawings found under {gt}")
        raise typer.Exit(code=1)

    provider_name = provider or provider_for_model(model)

    if dry_run:
        per_drawing = 1 if baseline == "vlm_zeroshot" else samples * (1 if no_tile else 5)
        missing = [p for p in drawings if image_for(p) is None]
        pending = [p for p in drawings if not (out / f"{p.stem}.json").exists()]
        console.print(
            f"{baseline} with {model} via {provider_name}\n"
            f"  {len(drawings)} drawings, {len(pending)} without a prediction yet\n"
            f"  up to {len(pending) * per_drawing} calls before the cache is consulted\n"
            f"  {len(missing)} drawings have no rendered image and would be skipped"
        )
        return

    manifest = run_baseline(
        baseline,
        drawings,
        out,
        provider=get_provider(provider_name),
        provider_name=provider_name,
        model=model,
        cache=ResponseCache(root=cache_dir),
        samples=samples,
        **({} if baseline == "vlm_zeroshot" else {"tile": not no_tile}),
        on_result=lambda drawing_id, result: console.print(
            f"[green] ok [/green] {drawing_id}  "
            f"{len(result.prediction.characteristics)} characteristics, "
            f"{result.calls} calls ({result.cache_hits} cached)"
            + (f" [red]{len(result.errors)} errors[/red]" if result.errors else "")
        ),
    )
    console.print(
        f"\n{manifest.n_drawings} drawings ({manifest.n_failed} failed), "
        f"{manifest.cache_hits} cache hits, {manifest.cache_misses} calls -> {out}"
    )
    if manifest.errors:
        console.print(f"[yellow]{len(manifest.errors)} errors; see manifest.json")


_VERIFY_ARG = typer.Argument(..., help="Ground-truth or prediction JSON files to verify.")
_STEP_OPT = typer.Option(
    None, "--step", help="STEP file or directory; found beside each JSON when omitted."
)
_VERIFY_OUT_OPT = typer.Option(
    None, "--out", "-o", help="Directory for the verification reports."
)


@app.command("verify")
def verify_command(
    paths: list[Path] = _VERIFY_ARG,
    step: Path | None = _STEP_OPT,
    out: Path | None = _VERIFY_OUT_OPT,
) -> None:
    """Check characteristics against the solid they describe.

    Each characteristic comes back verified, contradicted or unverifiable, alongside any
    defects found in the drawing itself. The unverifiable bucket is not a failure: it names
    the characteristics that need a person, which is what makes the output usable where a
    bare confidence score is not.

    The STEP file is found beside the JSON unless ``--step`` says otherwise, so the usual
    invocation is just the ground-truth file that ``drawgen`` already wrote.
    """
    from balloonbench.verifier import BrepIndex, verify_drawing

    failures = 0
    for path in paths:
        try:
            drawing = Drawing.from_json(path)
        except Exception as exc:  # noqa: BLE001 - report the reason, whatever it is
            failures += 1
            console.print(f"[red]FAIL[/red] {path}: {exc}")
            continue

        solid = _step_for(path, step, drawing)
        if solid is None:
            failures += 1
            console.print(f"[red]FAIL[/red] {path}: no STEP file found for this drawing")
            continue

        report = verify_drawing(drawing, BrepIndex.from_step(solid))
        summary = report.summary
        console.print(
            f"[green] ok [/green] {drawing.drawing_id}  "
            f"{summary['verified']} verified, "
            f"[red]{summary['contradicted']} contradicted[/red], "
            f"{summary['unverifiable']} unverifiable, "
            f"{len(report.drawing_defects)} drawing defects"
        )
        for entry in report.per_characteristic:
            if entry.verdict == "contradicted":
                console.print(f"    [red]#{entry.id}[/red] {entry.detail}")
        for defect in report.drawing_defects:
            console.print(f"    [yellow]{defect.type}[/yellow] {defect.detail}")

        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            report.write(out / f"{drawing.drawing_id}.verification.json")

    if failures:
        raise typer.Exit(code=1)


def _step_for(json_path: Path, step: Path | None, drawing: Drawing) -> Path | None:
    """The solid belonging to a drawing.

    Tries what the caller said, then the part reference the drawing carries, then a STEP
    file beside the JSON. The part reference is tried before the neighbour because a
    drawing knows which solid it describes and a directory listing only guesses.
    """
    if step is not None and step.is_file():
        return step
    candidates: list[Path] = []
    if drawing.part_ref:
        candidates += [Path(drawing.part_ref), json_path.parent / Path(drawing.part_ref).name]
    if step is not None and step.is_dir():
        candidates += sorted(step.glob(f"{json_path.stem}*.st*p"))
    candidates += sorted(json_path.parent.glob(f"{json_path.stem}*.st*p"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


_PRED_ARG = typer.Argument(..., help="Prediction JSON files, one per drawing.")
_REPORT_OUT_OPT = typer.Option(
    Path("results/latest"), "--out", "-o", help="Where the report is written."
)
_RUN_NAME_OPT = typer.Option("run", "--name", help="Run name, recorded in the report.")
_NO_BBOX_OPT = typer.Option(
    False, "--no-bbox", help="Match on semantic content alone; a different, easier task."
)
_MAX_COST_OPT = typer.Option(
    None, "--max-cost", help="Reject matches above this cost. Must lie in (0, 1)."
)


@app.command("evaluate")
def evaluate_predictions(
    predictions: list[Path] = _PRED_ARG,
    gt: Path = _GT_OPT,
    out: Path = _REPORT_OUT_OPT,
    name: str = _RUN_NAME_OPT,
    no_bbox: bool = _NO_BBOX_OPT,
    max_cost: float | None = _MAX_COST_OPT,
) -> None:
    """Score predictions against ground truth and write a report.

    Each prediction is paired with the ground truth whose ``drawing_id`` it names, so the
    two sets need not be in the same order or even the same shape -- a run that covers half
    the benchmark scores on that half and says so.

    ``--no-bbox`` is reported as a separate run, never merged with a localised one: without
    coordinates the task is materially easier, and averaging the two would quietly overstate
    what a model can do.
    """
    from balloonbench.evalkit.matching import MatchConfig, no_bbox_config
    from balloonbench.evalkit.metrics import aggregate, evaluate_drawing
    from balloonbench.evalkit.prediction import load_prediction
    from balloonbench.evalkit.report import markdown_table, write_report

    config = no_bbox_config() if no_bbox else MatchConfig()
    if max_cost is not None:
        config = replace(config, max_cost=max_cost)

    truth = {}
    for path in sorted(Path(gt).rglob("*.json")):
        try:
            drawing = Drawing.from_json(path)
        except Exception:  # noqa: BLE001 - not every JSON under the tree is a drawing
            continue
        truth[drawing.drawing_id] = drawing
    if not truth:
        console.print(f"[red]no ground-truth drawings found under {gt}")
        raise typer.Exit(code=1)

    scores = []
    missing = 0
    for path in predictions:
        prediction = load_prediction(path)
        drawing = truth.get(prediction.drawing_id)
        if drawing is None:
            missing += 1
            console.print(
                f"[yellow]skip[/yellow] {path}: no ground truth for "
                f"{prediction.drawing_id!r}"
            )
            continue
        scores.append(evaluate_drawing(drawing, prediction, config))

    if not scores:
        console.print("[red]nothing to score")
        raise typer.Exit(code=1)

    written = write_report(scores, out, run_name=name, meta={"no_bbox": no_bbox})
    console.print(markdown_table(aggregate(scores), title=name))
    console.print(f"\n{len(scores)} drawings scored ({missing} skipped) -> {written['json']}")


def _ground_truth_for(image: Path) -> Path | None:
    """The ground-truth JSON belonging to a rendered PNG.

    ``drawgen`` writes the raster with its resolution in the name (``flange-00003_150.png``)
    while the ground truth keeps the bare drawing id, so a plain suffix swap misses. Trailing
    underscore-separated qualifiers are dropped one at a time until a JSON turns up.
    """
    stem = image.stem
    while True:
        candidate = image.with_name(f"{stem}.json")
        if candidate.exists():
            return candidate
        if "_" not in stem:
            return None
        stem = stem.rsplit("_", 1)[0]


@app.command("degrade")
def degrade_images(
    images: list[Path] = _IMAGES_ARG,
    profile: str = _PROFILE_OPT,
    out: Path = _DEG_OUT_OPT,
    seed: int = _DEG_SEED_OPT,
) -> None:
    """Apply a degradation profile to rendered drawings and their ground truth.

    The ground truth is written alongside the degraded image, not shared with the clean one,
    because degradation moves boxes and drops callouts: a single JSON serving both would
    describe neither. ``--profile all`` writes one subdirectory per condition, which is the
    shape the harness expects when a result is reported per profile.
    """
    from balloonbench.degrade import degrade, profile_names

    known = profile_names()
    targets = list(known) if profile == "all" else [profile]
    for name in targets:
        if name not in known:
            console.print(f"[red]unknown profile {name!r}; known: {list(known)}")
            raise typer.Exit(code=1)

    made = failed = 0
    for name in targets:
        directory = out / name
        directory.mkdir(parents=True, exist_ok=True)
        for image in images:
            # A QA overlay has the boxes drawn on it. Degrading one would produce an image
            # whose painted boxes and whose ground truth disagree, which is exactly the
            # confusion the overlay exists to prevent.
            if image.stem.endswith("_overlay"):
                continue
            json_path = _ground_truth_for(image)
            if json_path is None:
                failed += 1
                console.print(f"[red]FAIL[/red] {image}: no ground truth alongside it")
                continue
            try:
                sample = degrade(image, Drawing.from_json(json_path), name, seed)
            except Exception as exc:  # noqa: BLE001 - report and keep going
                failed += 1
                console.print(f"[red]FAIL[/red] {image} [{name}]: {exc}")
                continue
            stem = image.stem
            sample.image.save(directory / f"{stem}.png")
            sample.drawing.to_json(directory / f"{stem}.json")
            made += 1
            console.print(
                f"[green] ok [/green] {directory / stem}.png  "
                f"{len(sample.drawing.characteristics)} characteristics, "
                f"{len(sample.applied)} transforms"
            )

    console.print(f"\n{made} degraded, {failed} failed -> {out}")
    if failed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
