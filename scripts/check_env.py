"""Environment gate for Milestone 0. Nothing else gets built until this prints ok.

SPEC.md section 3 writes this check against ``pythonocc-core`` (the ``OCC.Core.*``
namespace). We use OCP instead -- see PLAN.md section 0.1 -- so the same OCCT entry
points are imported from ``OCP.*``. The set of capabilities checked is identical: HLR
projection for drawgen, the surface adaptor for the verifier's B-rep index, STEP
read/write for the part round-trip, and the 2D/raster toolchain.

Run:  python scripts/check_env.py
"""

from __future__ import annotations

import importlib.metadata as md
import sys
from dataclasses import dataclass


@dataclass
class Check:
    label: str
    fn: object
    required: bool = True


def _ver(distribution: str) -> str:
    """Version from package metadata. Distributions move or drop ``__version__``
    between majors (pypdfium2 5.x dropped ``V_PYPDFIUM2``, jsonschema deprecated
    ``__version__``), so never read the attribute off the module."""
    try:
        return md.version(distribution)
    except md.PackageNotFoundError:
        return "unknown"


def _occt_hlr() -> str:
    from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt  # noqa: F401
    from OCP.HLRAlgo import HLRAlgo_Projector  # noqa: F401
    from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape  # noqa: F401

    return "HLRBRep_Algo, HLRBRep_HLRToShape, HLRAlgo_Projector"


def _occt_modelling() -> str:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    box = BRepPrimAPI_MakeBox(10.0, 20.0, 30.0).Shape()
    if not BRepCheck_Analyzer(box).IsValid():
        raise RuntimeError("BRepCheck_Analyzer rejected a unit box")
    return "BRepPrimAPI + BRepCheck_Analyzer (box is valid)"


def _occt_topology() -> str:
    from OCP.BRepAdaptor import BRepAdaptor_Surface  # noqa: F401
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.GeomAbs import GeomAbs_SurfaceType
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    cyl = BRepPrimAPI_MakeCylinder(5.0, 20.0).Shape()
    kinds = []
    exp = TopExp_Explorer(cyl, TopAbs_FACE)
    while exp.More():
        kinds.append(BRepAdaptor_Surface(TopoDS.Face_s(exp.Current())).GetType())
        exp.Next()
    if GeomAbs_SurfaceType.GeomAbs_Cylinder not in kinds:
        raise RuntimeError("no cylindrical face found on a cylinder")
    return f"BRepAdaptor_Surface classified {len(kinds)} faces on a cylinder"


def _occt_step() -> str:
    from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer  # noqa: F401

    return "STEPControl_Reader, STEPControl_Writer"


def _cadquery() -> str:
    import cadquery as cq

    n = len(cq.Workplane("XY").box(10, 10, 10).faces().vals())
    if n != 6:
        raise RuntimeError(f"expected 6 faces on a box, got {n}")
    return f"cadquery {cq.__version__} (box has 6 faces)"


def _ezdxf() -> str:
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.modelspace().add_line((0, 0), (10, 0))
    return f"ezdxf {ezdxf.__version__}"


def _numeric() -> str:
    import numpy as np
    import scipy
    from scipy.optimize import linear_sum_assignment  # noqa: F401

    return f"numpy {np.__version__}, scipy {scipy.__version__}"


def _schema() -> str:
    from jsonschema import Draft202012Validator

    from balloonbench.schema import SCHEMA_VERSION, load_json_schema

    Draft202012Validator.check_schema(load_json_schema())
    return (
        f"pydantic {_ver('pydantic')}, jsonschema {_ver('jsonschema')}, "
        f"balloonbench schema {SCHEMA_VERSION} (json schema is valid)"
    )


def _render_roundtrip() -> str:
    """The drawgen output path end to end: vector PDF out, raster back in.

    Checking the imports alone would miss the failure that actually matters -- a
    rasteriser that cannot open the PDF the renderer produces.
    """
    import io

    import pypdfium2 as pdfium
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(200, 100))
    c.rect(20, 20, 160, 60)
    c.showPage()
    c.save()

    doc = pdfium.PdfDocument(buf.getvalue())
    image = doc[0].render(scale=2).to_pil()
    if image.size != (400, 200):
        raise RuntimeError(f"expected a 400x200 raster, got {image.size}")
    return (
        f"reportlab {_ver('reportlab')} -> pypdfium2 {_ver('pypdfium2')} "
        f"-> pillow {_ver('pillow')} ({image.size[0]}x{image.size[1]})"
    )


def _svg() -> str:
    import svgwrite  # noqa: F401

    return f"svgwrite {_ver('svgwrite')}"


def _pdf_text() -> str:
    import pdfplumber  # noqa: F401

    return f"pdfplumber {_ver('pdfplumber')}"


def _clean_shutdown() -> str:
    """A native extension can import fine and still crash the interpreter on teardown.

    That failure is invisible to a normal check -- the module works, the output is
    correct, and only the exit code is wrong -- but it turns every CI step and every
    shell chain into a false failure. nlopt 2.11.0 alongside casadi 3.8.0 (both pulled in
    by cadquery) did exactly this, exiting 139 after a successful import. So spawn a
    child that imports the full stack and assert it exits 0.
    """
    import subprocess

    code = "import cadquery, ezdxf, pdfplumber, pypdfium2, OCP.HLRBRep"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True)
    if proc.returncode != 0:
        detail = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(
            f"child interpreter exited {proc.returncode} after importing the stack"
            + (f": {detail[-1]}" if detail else " (no traceback: a native crash)")
        )
    return f"child interpreter imports the full stack and exits {proc.returncode}"


#: Every codepoint the drawing renderer needs from the vendored font. A missing glyph does
#: not raise -- it renders as a tofu box -- so absence has to be checked, not assumed.
REQUIRED_GLYPHS: dict[str, int] = {
    # geometric characteristic symbols
    "straightness": 0x23E4, "flatness": 0x23E5, "circularity": 0x25CB,
    "cylindricity": 0x232D, "profile_surface": 0x2312, "perpendicularity": 0x27C2,
    "parallelism": 0x2225, "angularity": 0x2220, "position": 0x2316,
    "concentricity": 0x25CE, "symmetry": 0x232F, "circular_runout": 0x2197,
    "total_runout": 0x2330,
    # material condition and other frame modifiers
    "MMC": 0x24C2, "LMC": 0x24C1, "free_state": 0x24BB, "projected": 0x24C5,
    # dimensioning symbols
    "diameter": 0x2300, "counterbore": 0x2334, "countersink": 0x2335,
    "depth": 0x21A7, "square": 0x25A1, "arc": 0x2312,
    "plus_minus": 0x00B1, "degree": 0x00B0, "micro": 0x00B5,
}


def _drawing_font() -> str:
    """The vendored drawing font is present and carries every glyph the renderer needs.

    SPEC.md section 3 forbids relying on a system font: output has to be reproducible
    across machines, and most system fonts have no GD&T glyphs at all. Checking the
    codepoints here means a truncated download or a font swap fails the gate, rather than
    silently rendering boxes into a drawing whose ground truth then claims a symbol that
    is not on the page.
    """
    import pathlib  # noqa: PLC0415

    from fontTools.ttLib import TTFont  # noqa: PLC0415

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "assets" / "fonts" / "osifont" / "osifont-lgpl3fe.ttf"
    )
    if not path.exists():
        raise RuntimeError(f"vendored drawing font missing at {path}")

    font = TTFont(path)
    covered: set[int] = set()
    for table in font["cmap"].tables:
        covered |= set(table.cmap)

    missing = {n: hex(c) for n, c in REQUIRED_GLYPHS.items() if c not in covered}
    if missing:
        raise RuntimeError(f"drawing font is missing {len(missing)} glyphs: {missing}")

    licence = path.parent / "LICENSE.md"
    if not licence.exists():
        raise RuntimeError(f"vendored font has no licence notice beside it at {licence}")

    return (
        f"osifont ({len(covered)} glyphs) covers all {len(REQUIRED_GLYPHS)} required "
        f"symbols, licence vendored"
    )


def _handwriting_font() -> str:
    """The vendored handwriting font backs the clutter transforms in ``degrade``.

    The same reproducibility argument as the drawing font, with one addition: this one is
    drawn *over* a sheet whose ground truth was computed before it existed. If it fell back
    to a system font the marks would land differently on every machine, so a box that
    happens to survive here could be obliterated there -- a difference no test would
    attribute to the font.
    """
    import pathlib  # noqa: PLC0415

    from fontTools.ttLib import TTFont  # noqa: PLC0415

    path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "assets" / "fonts" / "caveat" / "Caveat.ttf"
    )
    if not path.exists():
        raise RuntimeError(f"vendored handwriting font missing at {path}")

    covered: set[int] = set()
    for table in TTFont(path)["cmap"].tables:
        covered |= set(table.cmap)

    # Handwritten notes, stamps and red-pen corrections use ASCII plus the few marks the
    # texts in ``degrade.clutter`` actually contain.
    needed = set(range(0x20, 0x7F)) | {0x00D8, 0x00B1, 0x00B0}
    missing = sorted(hex(c) for c in needed - covered)
    if missing:
        raise RuntimeError(f"handwriting font is missing {len(missing)} glyphs: {missing}")

    licence = path.parent / "OFL.txt"
    if not licence.exists():
        raise RuntimeError(f"vendored font has no licence notice beside it at {licence}")

    return f"Caveat ({len(covered)} glyphs) covers ASCII and the marks clutter needs"


def _no_agpl_in_core() -> str:
    """PLAN.md section 0.2: the core package stays Apache-2.0 clean, so nothing under
    balloonbench/ may import the AGPL-licensed pymupdf."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent / "balloonbench"
    pattern = re.compile(r"^\s*(?:import|from)\s+(?:fitz|pymupdf)\b", re.MULTILINE)
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    if offenders:
        raise RuntimeError(f"AGPL import in core package: {offenders}")
    return "no pymupdf import under balloonbench/"


def _cv() -> str:
    import cv2  # noqa: F401

    return f"opencv {_ver('opencv-python')}"


def _shapely() -> str:
    """Required, not optional: section-view hatching clips hatch lines against the cut
    region with shapely, and a section drawn without hatching is a wrong drawing."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    square = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    clipped = LineString([(-5, 5), (15, 5)]).intersection(unary_union([square]))
    if abs(clipped.length - 10.0) > 1e-9:
        raise RuntimeError(f"hatch clipping gave length {clipped.length}, expected 10")
    return f"shapely {_ver('shapely')} (hatch clipping works)"


CHECKS: list[Check] = [
    Check("OCCT modelling", _occt_modelling),
    Check("OCCT topology + surface adaptor", _occt_topology),
    Check("OCCT hidden line removal", _occt_hlr),
    Check("OCCT STEP io", _occt_step),
    Check("cadquery", _cadquery),
    Check("ezdxf", _ezdxf),
    Check("numpy / scipy", _numeric),
    Check("schema stack", _schema),
    Check("vector pdf -> raster round trip", _render_roundtrip),
    Check("svg", _svg),
    Check("pdf text extraction", _pdf_text),
    Check("clean interpreter shutdown", _clean_shutdown),
    Check("drawing font", _drawing_font),
    Check("handwriting font", _handwriting_font),
    Check("licence hygiene", _no_agpl_in_core),
    Check("opencv", _cv, required=False),
    Check("shapely", _shapely),
]


def main() -> int:
    print(f"python {sys.version.split()[0]}  ({sys.executable})\n")
    failures = 0
    for check in CHECKS:
        try:
            detail = check.fn()
        except Exception as exc:  # noqa: BLE001 - we want the reason, whatever it is
            mark = "FAIL" if check.required else "warn"
            failures += int(check.required)
            print(f"  [{mark}] {check.label}: {type(exc).__name__}: {exc}")
        else:
            print(f"  [ ok ] {check.label}: {detail}")

    print()
    if failures:
        print(f"{failures} required check(s) failed")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
