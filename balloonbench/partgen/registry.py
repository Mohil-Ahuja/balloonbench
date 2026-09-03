"""Family registry and the build driver.

One place that knows how to turn a seed into a validated :class:`BuiltPart`, so every
family gets the same treatment: reject-and-resample on unbuildable parameters, an OCCT
validity check before anything is exported, a deterministic face index, and a STEP
round-trip that is verified rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.IFSelect import IFSelect_ReturnStatus
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Reader, STEPControl_StepModelType, STEPControl_Writer
from OCP.TopoDS import TopoDS_Shape

from balloonbench.partgen.types import BuiltPart, FaceIndex, PartFamily, UnbuildableParams

__all__ = [
    "MAX_RESAMPLE_ATTEMPTS",
    "build_part",
    "export_step",
    "families",
    "get_family",
    "read_step",
    "register",
]

#: How many parameter draws to try before giving up on a seed. SPEC.md section 6.3 wants
#: reject-and-resample, but a family whose sampler rejects nearly everything is a bug in
#: the sampler, not bad luck, so the driver refuses to loop forever hiding it.
MAX_RESAMPLE_ATTEMPTS = 50

_REGISTRY: dict[str, PartFamily] = {}


def register(family: PartFamily) -> PartFamily:
    """Add a family to the registry. Usable as a decorator on the class instance."""
    if family.name in _REGISTRY:
        raise ValueError(f"a family named {family.name!r} is already registered")
    _REGISTRY[family.name] = family
    return family


def families() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_family(name: str) -> PartFamily:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no part family named {name!r}; registered families are {families()}"
        ) from None


# --- STEP io ------------------------------------------------------------------------


def export_step(shape: TopoDS_Shape, path: Path) -> Path:
    """Write ``shape`` to a STEP file, in millimetres.

    The unit is set explicitly. STEP files carry their own unit declaration, and a
    consumer that reads a file written with OCCT's ambient default has no way to know
    what it got -- which is precisely the mm/inch confusion the verifier's
    ``unit_sanity`` check exists to catch. Being sloppy here would mean the benchmark
    itself contains the error class it claims to detect.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    Interface_Static.SetCVal_s("write.step.unit", "MM")
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP export of {path} failed with status {status}")
    return path


def read_step(path: Path) -> TopoDS_Shape:
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(f"STEP read of {path} failed with status {status}")
    reader.TransferRoots()
    return reader.OneShape()


# --- the driver ---------------------------------------------------------------------


def build_part(
    family_name: str,
    seed: int,
    *,
    step_dir: Path | None = None,
    max_attempts: int = MAX_RESAMPLE_ATTEMPTS,
) -> BuiltPart:
    """Turn ``(family, seed)`` into a validated part.

    Determinism is the whole point: the same pair must give the same solid, the same face
    ids and the same feature description, on any machine, because the ground truth of
    every drawing generated from it is recorded as that seed.
    """
    family = get_family(family_name)
    rng = np.random.default_rng(seed)

    rejected: list[str] = []
    for _attempt in range(max_attempts):
        # Sampling is inside the try as well as building. A family that samples inside its
        # constraints still hits combinations with no legal answer -- a 63 mm flange with
        # a 25 mm bore has nowhere to put a bolt circle -- and it reports that by raising
        # from sample_params. Catching only build() would let one rejected draw kill the
        # whole seed instead of resampling it.
        try:
            params = family.sample_params(rng)
            shape = family.build(params)
        except UnbuildableParams as exc:
            rejected.append(str(exc))
            continue

        # Validate before export, per SPEC.md section 15: an invalid solid written to
        # STEP propagates a broken part into every module that reads it back.
        if not BRepCheck_Analyzer(shape).IsValid():
            rejected.append("BRepCheck_Analyzer rejected the solid")
            continue

        faces = FaceIndex(shape)
        features = family.describe(params, faces)
        part = BuiltPart(
            family=family_name,
            params=params,
            shape=shape,
            faces=faces,
            features=tuple(features),
            seed=seed,
        )
        part.validate()

        if step_dir is not None:
            part.step_path = export_step(
                shape, Path(step_dir) / f"{family_name}_{seed:08d}.step"
            )
        return part

    raise UnbuildableParams(
        f"{family_name}: {max_attempts} attempts from seed {seed} produced no buildable "
        f"part. First few reasons: {rejected[:5]}"
    )


def build_batch(
    family_name: str,
    seeds: list[int] | range,
    *,
    step_dir: Path | None = None,
) -> list[BuiltPart]:
    return [build_part(family_name, s, step_dir=step_dir) for s in seeds]


def load_families() -> tuple[str, ...]:
    """Import the family modules so their registrations run.

    Kept explicit rather than done at package import: importing a family pulls in OCCT
    modelling machinery, and ``balloonbench.schema`` must stay importable without it so
    that schema validation works in environments with no CAD stack (the labelling app in
    SPEC.md section 9, for one).
    """
    import importlib  # noqa: PLC0415
    import pkgutil  # noqa: PLC0415

    from balloonbench.partgen import families as families_pkg  # noqa: PLC0415

    # Discovered rather than listed. A hard-coded list has to be edited in lockstep with
    # the package, and gets it wrong in both directions: it breaks the moment a module is
    # absent, and it silently omits one that is present. An import error inside a module
    # that does exist still propagates.
    for module in pkgutil.iter_modules(families_pkg.__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{families_pkg.__name__}.{module.name}")

    return families()


def _describe_rejects(family_name: str, seed: int, attempts: int = 20) -> dict[str, Any]:
    """Diagnostic helper: how often does a family's sampler reject its own draws?

    A sampler with a high reject rate still works, but it is a signal that the parameter
    ranges disagree with the manufacturability constraints, which is worth knowing before
    it shows up as a slow generation run.
    """
    family = get_family(family_name)
    rng = np.random.default_rng(seed)
    reasons: list[str] = []
    ok = 0
    for _ in range(attempts):
        try:
            family.build(family.sample_params(rng))
        except UnbuildableParams as exc:
            reasons.append(str(exc))
        else:
            ok += 1
    return {"family": family_name, "built": ok, "rejected": len(reasons), "reasons": reasons}
