"""Turning scores into the three artifacts a run has to leave behind.

SPEC.md section 10.3 asks for per-run JSON, a markdown table, and plots, committed to
``results/`` so the leaderboard is reproducible from the repo. The JSON is the record --
every per-drawing score, not just the summary, because a leaderboard entry that cannot be
re-aggregated or sliced afterwards is a claim rather than a result. The markdown is what a
human reads. The plots answer the four questions the benchmark was built to ask: how does
accuracy fall with degradation, which geometric symbols are hard, does a busy drawing hurt,
and how big is the synthetic-to-real gap.

Matplotlib is imported inside the plotting function and forced onto the ``Agg`` backend.
Importing it at module level would make every ``evalkit`` import pay for it, and leaving the
backend to the environment would let a machine with a display open windows during a batch
run -- or fail outright on one without.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from balloonbench.evalkit.metrics import DrawingScore, RunSummary, aggregate

__all__ = ["markdown_table", "write_plots", "write_report"]


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.1f}%"


def markdown_table(summary: RunSummary, *, title: str = "Run") -> str:
    """The human-readable summary: one headline table plus the two breakouts.

    Rates print with their counts. ``100.0% (3/3)`` and ``100.0% (300/300)`` are the same
    fraction and very different evidence, and a table that hides the denominator invites
    the reader to treat them alike.
    """

    def rate_row(name: str, rate) -> str:
        return f"| {name} | {_pct(rate.value)} | {rate.hits}/{rate.total} |"

    lines = [
        f"## {title}",
        "",
        f"{summary.n_drawings} drawings, "
        f"{summary.true_positives + summary.false_negatives} ground-truth characteristics.",
        "",
        "### Tier 1 — detection",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {_pct(summary.precision)} |",
        f"| Recall | {_pct(summary.recall)} |",
        f"| F1 | {_pct(summary.f1)} |",
        f"| False positives | {summary.false_positives} "
        f"(of which {summary.malformed} would not load) |",
        f"| False negatives | {summary.false_negatives} |",
        "",
        "| Recall by kind | Value | Count |",
        "|---|---|---|",
    ]
    lines += [rate_row(k, r) for k, r in sorted(summary.by_kind.items())]

    if summary.by_symbol:
        lines += ["", "| Recall by GD&T symbol | Value | Count |", "|---|---|---|"]
        lines += [rate_row(k, r) for k, r in sorted(summary.by_symbol.items())]

    lines += [
        "",
        "### Tier 2 — correctness given a match",
        "",
        "| Metric | Value | Count |",
        "|---|---|---|",
    ]
    lines += [rate_row(k, r) for k, r in sorted(summary.tier2.items())]

    lines += [
        "",
        "### Tier 3 — structure",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Mean datum graph edit distance | {summary.mean_graph_distance:.3f} |",
        f"| Full-drawing exact match | {_pct(summary.exact_drawing_rate)} "
        f"({summary.exact_drawings}/{summary.n_drawings}) |",
        "",
        "### Tier 4 — cost",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| CTQ recall | {_pct(summary.ctq.value)} "
        f"({summary.ctq.hits}/{summary.ctq.total}) |",
        f"| Weighted error cost | {summary.cost:.1f} |",
    ]
    if summary.cost_breakdown:
        lines += ["", "| Error type | Cost |", "|---|---|"]
        lines += [
            f"| {key} | {amount:.1f} |"
            for key, amount in sorted(
                summary.cost_breakdown.items(), key=lambda kv: -kv[1]
            )
        ]

    if summary.by_profile:
        lines += [
            "",
            "### By degradation profile",
            "",
            "| Profile | Drawings | Precision | Recall | F1 | Exact | Cost |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, block in summary.by_profile.items():
            detection = block["detection"]
            lines.append(
                f"| {name} | {block['n_drawings']} | "
                f"{_pct(detection['precision'])} | {_pct(detection['recall'])} | "
                f"{_pct(detection['f1'])} | "
                f"{_pct(block['structural']['exact_drawing_rate'])} | "
                f"{block['cost']['total']:.1f} |"
            )

    return "\n".join(lines) + "\n"


def write_plots(scores: list[DrawingScore], out_dir: Path) -> list[Path]:
    """The four SPEC.md section 10.3 plots, as far as the run supports them.

    A plot is skipped rather than drawn empty when the run has nothing to say -- a
    single-profile run has no accuracy-versus-profile curve, and drawing one point on an
    axis labelled as a trend would be a misleading picture, not a sparse one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def save(figure, name: str) -> None:
        path = out_dir / name
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    # 1. Accuracy against degradation profile: the benchmark's independent variable.
    profiles = sorted({s.profile or "unknown" for s in scores})
    if len(profiles) > 1:
        f1s = []
        for profile in profiles:
            subset = [s for s in scores if (s.profile or "unknown") == profile]
            f1s.append((aggregate(subset).f1 or 0.0) * 100)
        figure, axes = plt.subplots(figsize=(7, 4))
        axes.bar(profiles, f1s, color="#4C72B0")
        axes.set_ylabel("F1 (%)")
        axes.set_ylim(0, 100)
        axes.set_title("Detection F1 by degradation profile")
        axes.tick_params(axis="x", rotation=30)
        save(figure, "f1_by_profile.png")

    # 2. Recall by geometric symbol: which characteristics are hard to read.
    summary = aggregate(scores)
    if summary.by_symbol:
        names = sorted(summary.by_symbol)
        values = [(summary.by_symbol[n].value or 0.0) * 100 for n in names]
        figure, axes = plt.subplots(figsize=(7, 0.4 * len(names) + 2))
        axes.barh(names, values, color="#55A868")
        axes.set_xlabel("Recall (%)")
        axes.set_xlim(0, 100)
        axes.set_title("Recall by GD&T symbol")
        save(figure, "recall_by_symbol.png")

    # 3. Accuracy against how crowded the sheet is.
    if len(scores) > 2:
        xs = [s.n_gt for s in scores]
        ys = [(s.f1 or 0.0) * 100 for s in scores]
        figure, axes = plt.subplots(figsize=(6, 4))
        axes.scatter(xs, ys, alpha=0.6, color="#C44E52")
        axes.set_xlabel("Characteristics on the drawing")
        axes.set_ylabel("F1 (%)")
        axes.set_ylim(0, 100)
        axes.set_title("Accuracy vs. drawing complexity")
        save(figure, "f1_vs_complexity.png")

    # 4. The synthetic-to-real gap, once there is a real set to compare against.
    families = {s.family for s in scores if s.family}
    if {"synthetic", "real"} <= families:
        values = []
        for source in ("synthetic", "real"):
            subset = [s for s in scores if s.family == source]
            values.append((aggregate(subset).f1 or 0.0) * 100)
        figure, axes = plt.subplots(figsize=(4, 4))
        axes.bar(["synthetic", "real"], values, color=["#4C72B0", "#8172B2"])
        axes.set_ylabel("F1 (%)")
        axes.set_ylim(0, 100)
        axes.set_title("Synthetic vs. real")
        save(figure, "synthetic_vs_real.png")

    return written


def write_report(
    scores: list[DrawingScore],
    out_dir: Path,
    *,
    run_name: str,
    meta: dict[str, Any] | None = None,
    plots: bool = True,
) -> dict[str, Path]:
    """Write ``report.json``, ``report.md`` and the plots for one run.

    The JSON keeps every per-drawing score alongside the summary. Anyone re-slicing the
    result later -- by profile, by family, by how crowded the sheet was -- needs the rows,
    and a summary-only file forces a rerun to answer a question the data already contains.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(scores)

    payload = {
        "run": run_name,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "meta": meta or {},
        "summary": summary.as_dict(),
        "drawings": [score.as_dict() for score in scores],
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    md_path = out_dir / "report.md"
    md_path.write_text(markdown_table(summary, title=run_name), encoding="utf-8")

    written = {"json": json_path, "markdown": md_path}
    if plots:
        for path in write_plots(scores, out_dir):
            written[path.stem] = path
    return written
