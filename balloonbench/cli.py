"""BalloonBench command line. Subcommands are added as their milestones land."""

from __future__ import annotations

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


if __name__ == "__main__":
    app()
