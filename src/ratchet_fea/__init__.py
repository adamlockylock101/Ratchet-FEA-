"""STP-RB-001: fracture mechanics investigation of a PA66 ratchet belt strap.

A static linear elastic fracture mechanics study: a crack is seeded at a
perforation, the ratchet tension is applied, and K at the crack tip is compared
against the PA66 K_IC bracket over a sweep of crack lengths.

Inputs
    :mod:`ratchet_fea.geometry`    strap dimensions and crack definition
                                   (PLACEHOLDER values)
    :mod:`ratchet_fea.materials`   PA66 and steel property brackets, K_IC

Handbook screen (no FE stack needed)
    :mod:`ratchet_fea.analytical`  Kt, Newman crack-from-hole K, LEFM validity

Finite element
    :mod:`ratchet_fea.mesh`        perforated strip with a seeded crack
    :mod:`ratchet_fea.mechanics`   static plane-stress elastic solve
    :mod:`ratchet_fea.fracture`    J-integral and K extraction, crack closure
    :mod:`ratchet_fea.postprocess` K(a) curve, K_IC comparison, plots

Supporting
    :mod:`ratchet_fea.provenance`  where every number came from, and what it
                                   would take to replace it with a measurement

UNITS everywhere: mm, N, MPa; stress intensity in MPa*sqrt(mm). The K_IC
bracket is stored in MPa*sqrt(m), as datasheets quote it -- convert with
``materials.fracture_toughness_mpa_root_mm``.

Every dimension is currently a visual estimate and every material property a
literature range. Run either entry point under ``scripts/`` to print the list
of assumptions still to verify against a physical part.
"""

from . import analytical, geometry, materials, provenance

__all__ = [
    "analytical",
    "geometry",
    "materials",
    "provenance",
    "mesh",
    "mechanics",
    "fracture",
    "postprocess",
    "__version__",
]

__version__ = "0.3.0"


def __getattr__(name):
    """Import the FE modules lazily.

    They pull in scikit-fem, gmsh and matplotlib. The handbook screen needs
    none of those, so someone who only wants the closed-form answer should not
    have to install them.
    """
    if name in ("mesh", "mechanics", "fracture", "postprocess"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
