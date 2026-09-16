"""Shared fixtures.

FE fixtures are session-scoped and deliberately coarse: the point of a test is
to catch a wrong formula or a mis-tagged boundary, not to produce a
publication-quality K. Convergence is checked explicitly in
``test_fracture.py`` rather than paid for in every test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ratchet_fea.geometry import (
    PLACEHOLDER_GEOMETRY,
    CrackGeometry,
    CrackOrientation,
    StrapGeometry,
)


def _gmsh_unavailable() -> str:
    """Why gmsh cannot be used here, or an empty string if it can.

    Import is not enough: the gmsh wheel links against X11/GL shared libraries
    and raises OSError at import time when they are missing, which is the usual
    situation in a headless container.
    """
    try:
        import gmsh  # noqa: F401
        import pygmsh  # noqa: F401
    except ImportError:
        return "pygmsh/gmsh not installed"
    except OSError as exc:
        return f"gmsh shared library will not load: {exc}"
    return ""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_gmsh: test needs a working gmsh, so the handbook suite still "
        "runs where the FE stack is absent",
    )


def pytest_collection_modifyitems(config, items):
    reason = _gmsh_unavailable()
    if not reason:
        return
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "requires_gmsh" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def geometry() -> StrapGeometry:
    """The placeholder strap, unmodified."""
    return PLACEHOLDER_GEOMETRY


@pytest.fixture(scope="session")
def small_geometry() -> StrapGeometry:
    """A short three-hole window, for tests that need a real mesh but not detail.

    Three is the minimum that has an interior hole, which is where a crack gets
    seeded. The end margin is deliberately shorter than a full strap width to
    keep the mesh small.
    """
    return replace(PLACEHOLDER_GEOMETRY, n_holes=3, end_margin=10.0)


@pytest.fixture(scope="session")
def isolated_geometry() -> StrapGeometry:
    """A single-hole strip -- the configuration the handbook solutions describe.

    Used to check the FE against the handbook without the row-shielding effect
    confounding the comparison.
    """
    return replace(PLACEHOLDER_GEOMETRY, n_holes=1)


@pytest.fixture(scope="session")
def coarse_controls():
    from ratchet_fea.mesh import MeshControls

    return MeshControls(
        elements_around_hole=16, elements_along_crack=10, min_ligament_divisions=3
    )


@pytest.fixture(scope="session")
def transverse_crack() -> CrackGeometry:
    return CrackGeometry(length=2.0, orientation=CrackOrientation.TRANSVERSE)


@pytest.fixture(scope="session")
def longitudinal_crack() -> CrackGeometry:
    return CrackGeometry(length=2.0, orientation=CrackOrientation.LONGITUDINAL)


@pytest.fixture(scope="session")
def small_strip(small_geometry, coarse_controls):
    """An uncracked mesh."""
    from ratchet_fea.mesh import build_strip_mesh

    return build_strip_mesh(small_geometry, coarse_controls)


@pytest.fixture(scope="session")
def cracked_strip(small_geometry, coarse_controls, transverse_crack):
    from ratchet_fea.mesh import build_strip_mesh

    return build_strip_mesh(small_geometry, coarse_controls, transverse_crack)
