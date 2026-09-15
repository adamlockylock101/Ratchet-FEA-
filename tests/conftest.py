"""Shared fixtures.

Tier 2 fixtures are session-scoped and deliberately coarse: the point of a
test is to catch a wrong formula or a mis-tagged boundary, not to produce a
publication-quality stress. Convergence is checked explicitly in
``test_mechanics.py`` rather than paid for in every test.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, StrapGeometry

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
        "requires_gmsh: test needs a working gmsh, so the Tier 1 suite still "
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

    Three is the minimum that has an interior hole, so the interior/end
    distinction the post-processing relies on is actually exercised. The end
    margin is deliberately shorter than a full strap width to keep the mesh
    small; that makes the END holes truncation-affected, which is fine because
    conclusions are drawn from the interior one.
    """
    return replace(PLACEHOLDER_GEOMETRY, n_holes=3, end_margin=8.0)


@pytest.fixture(scope="session")
def two_hole_geometry() -> StrapGeometry:
    """A window with no interior hole at all, for the fallback path."""
    return replace(PLACEHOLDER_GEOMETRY, n_holes=2, end_margin=8.0)


@pytest.fixture(scope="session")
def plain_geometry() -> StrapGeometry:
    """A single-hole, unreinforced strip -- the configuration Tier 1 describes."""
    return replace(
        PLACEHOLDER_GEOMETRY,
        n_holes=1,
        steel_band_width=0.0,
        steel_band_thickness=0.0,
    )


@pytest.fixture(scope="session")
def coarse_controls():
    from ratchet_fea.mesh import MeshControls

    return MeshControls(elements_around_hole=16, min_ligament_divisions=3)


@pytest.fixture(scope="session")
def small_strip(small_geometry, coarse_controls):
    from ratchet_fea.mesh import build_strip_mesh

    return build_strip_mesh(small_geometry, coarse_controls)
