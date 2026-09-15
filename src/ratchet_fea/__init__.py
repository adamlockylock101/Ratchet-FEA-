"""STP-RB-001: fracture mechanics investigation of a PA66 ratchet belt strap.

Tier 1 -- closed-form screening
    :mod:`ratchet_fea.geometry`    strap dimensions (PLACEHOLDER values)
    :mod:`ratchet_fea.materials`   PA66 and steel property brackets
    :mod:`ratchet_fea.analytical`  stress concentration and swelling estimates

Tier 2 -- coupled FE
    :mod:`ratchet_fea.mesh`        perforated strip mesh via pygmsh/gmsh
    :mod:`ratchet_fea.diffusion`   transient Fickian moisture field
    :mod:`ratchet_fea.mechanics`   plane-stress solve with swelling eigenstrain
    :mod:`ratchet_fea.postprocess` extraction, plots, Tier 1 cross-check

Supporting
    :mod:`ratchet_fea.provenance`  where every number came from, and what it
                                   would take to replace it with a measurement

UNITS everywhere: mm, N, MPa, s, mm^2/s, moisture as a dimensionless mass
fraction.

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
    "diffusion",
    "mechanics",
    "postprocess",
    "__version__",
]

__version__ = "0.2.0"


def __getattr__(name):
    """Import the Tier 2 modules lazily.

    They pull in scikit-fem, gmsh and matplotlib. Tier 1 needs none of those,
    so someone who only wants the closed-form screen should not have to install
    them.
    """
    if name in ("mesh", "diffusion", "mechanics", "postprocess"):
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
