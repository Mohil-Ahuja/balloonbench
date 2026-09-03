"""Geometry-grounded verification: does the drawing agree with the solid? (SPEC.md 12)"""

from balloonbench.verifier.base import CheckContext, Defect, Verdict, Verification
from balloonbench.verifier.brep_index import BrepIndex, Candidate, HolePattern
from balloonbench.verifier.inject import INJECTORS, Injection, inject, injector_names
from balloonbench.verifier.report import CHECKS, VerificationReport, verify_drawing

__all__ = [
    "CHECKS",
    "INJECTORS",
    "BrepIndex",
    "Candidate",
    "CheckContext",
    "Defect",
    "HolePattern",
    "Injection",
    "Verdict",
    "Verification",
    "VerificationReport",
    "inject",
    "injector_names",
    "verify_drawing",
]
