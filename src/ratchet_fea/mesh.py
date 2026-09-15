"""Tier 2: 2D mesh of the perforated strap, built from :mod:`geometry`.

Generates a plane mesh of the strap window -- a rectangle of
``modelled_length x strap_width`` with a row of ``n_holes`` circular
perforations -- using gmsh's OCC kernel through :mod:`pygmsh`, then converts it
to a :class:`skfem.MeshTri` with named boundaries and subdomains.

NOTHING HERE IS HARDCODED.  Every dimension comes from
:class:`~ratchet_fea.geometry.StrapGeometry` and every mesh size derives from
the hole radius and the ligament widths.  Replace the placeholder dimensions
with measured ones and the mesh regenerates at an equivalent density.

Named boundaries
----------------
``CUT_START`` / ``CUT_END``
    The x = 0 and x = L faces.  These are *model truncation planes* cut
    through a longer strap, not physical surfaces.  Mechanically they carry
    the remote tension; for diffusion they are zero-flux (symmetry), because
    material continues beyond them.
``SIDE_LOWER`` / ``SIDE_UPPER``
    The y = 0 and y = W long edges.  Real free surfaces, exposed to moisture.
``hole_0`` ... ``hole_{n-1}``
    Each perforation wall, tagged individually so post-processing can report
    a peak stress per hole and show how the end holes differ from the
    interior ones.

Named subdomains
----------------
``BAND``
    Footprint of the internal steel band -- a PA66/steel/PA66 laminate
    through the thickness.
``POLYMER``
    Everything outside the band footprint -- PA66 for the full thickness.
    Absent when the band spans the full strap width.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .geometry import StrapGeometry

__all__ = [
    "CUT_START",
    "CUT_END",
    "SIDE_LOWER",
    "SIDE_UPPER",
    "BAND",
    "POLYMER",
    "MeshControls",
    "StripMesh",
    "build_strip_mesh",
]

CUT_START = "cut_start"
CUT_END = "cut_end"
SIDE_LOWER = "side_lower"
SIDE_UPPER = "side_upper"
BAND = "band"
POLYMER = "polymer"

#: Tolerance for classifying gmsh entities by position, as a fraction of the
#: hole radius.  Geometric, so it scales with the part instead of assuming mm.
_REL_TOL = 1e-6


@dataclass(frozen=True)
class MeshControls:
    """Mesh density, expressed relative to the geometry rather than in mm.

    Stated this way the mesh stays equivalent when the placeholder dimensions
    are replaced by measured ones -- a 3 mm hole gets the same number of
    elements round it as a 4 mm hole did.
    """

    #: Elements around each hole's circumference.  Sets the resolution of the
    #: stress concentration, which is the quantity the whole model exists to
    #: produce.
    elements_around_hole: int = 48
    #: Bulk element size as a multiple of the hole-edge element size.
    bulk_size_factor: float = 4.0
    #: Minimum elements across the narrowest ligament, so the ligament is
    #: always resolved even if the pitch is tightened.
    min_ligament_divisions: int = 6
    #: Element size growth per unit distance away from a hole edge.
    grading: float = 0.35
    #: gmsh optimisation passes (Netgen). 0 disables.
    optimise: bool = True
    #: gmsh message verbosity: 0 silent, 5 debug.
    verbosity: int = 0

    def __post_init__(self) -> None:
        if self.elements_around_hole < 8:
            raise ValueError("need at least 8 elements around a hole to resolve Kt")
        if self.bulk_size_factor < 1.0:
            raise ValueError("bulk_size_factor must be >= 1")
        if self.min_ligament_divisions < 2:
            raise ValueError("min_ligament_divisions must be >= 2")
        if self.grading <= 0.0:
            raise ValueError("grading must be positive")

    def refined(self, factor: float = 2.0) -> "MeshControls":
        """Return controls with ``factor`` times the resolution.

        For mesh convergence studies: a Kt that moves by more than a couple of
        percent between ``controls`` and ``controls.refined()`` is not
        converged and should not be quoted.
        """
        if factor <= 0:
            raise ValueError("refinement factor must be positive")
        return replace(
            self,
            elements_around_hole=int(round(self.elements_around_hole * factor)),
            min_ligament_divisions=int(round(self.min_ligament_divisions * factor)),
        )

    def hole_edge_size(self, geometry: StrapGeometry) -> float:
        """Target element size on a hole wall, mm."""
        circumference = 2.0 * np.pi * geometry.hole_radius
        return circumference / self.elements_around_hole

    def bulk_size(self, geometry: StrapGeometry) -> float:
        """Target element size far from the holes, mm."""
        narrowest = min(geometry.side_ligament, geometry.inter_hole_ligament)
        by_ligament = narrowest / self.min_ligament_divisions
        return min(self.bulk_size_factor * self.hole_edge_size(geometry), by_ligament)


@dataclass
class StripMesh:
    """A generated mesh plus the geometry it came from."""

    mesh: object  # skfem.MeshTri -- typed loosely to keep skfem an optional import
    geometry: StrapGeometry
    controls: MeshControls

    @property
    def hole_boundaries(self) -> list[str]:
        """Boundary names for the perforations, in x order."""
        return [h for h in self.geometry.hole_labels if h in self.mesh.boundaries]

    @property
    def exposed_boundaries(self) -> list[str]:
        """Boundaries in contact with the environment.

        The long side edges and every hole wall.  The truncation planes are
        excluded: material continues past them, so no moisture enters there.
        """
        names = [
            n for n in (SIDE_LOWER, SIDE_UPPER) if n in self.mesh.boundaries
        ]
        return names + self.hole_boundaries

    @property
    def traction_boundary(self) -> str:
        """Boundary carrying the remote tension."""
        return CUT_END

    @property
    def restrained_boundary(self) -> str:
        """Boundary holding the longitudinal displacement."""
        return CUT_START

    @property
    def has_band_subdomain(self) -> bool:
        return bool(self.mesh.subdomains) and BAND in self.mesh.subdomains

    @property
    def n_elements(self) -> int:
        return int(self.mesh.t.shape[1])

    @property
    def n_vertices(self) -> int:
        return int(self.mesh.p.shape[1])

    def element_centroids(self) -> np.ndarray:
        """``(2, n_elements)`` element centroid coordinates."""
        return self.mesh.p[:, self.mesh.t].mean(axis=1)

    def band_element_mask(self) -> np.ndarray:
        """Boolean mask over elements: True where the steel band is present.

        Classified geometrically from :class:`StrapGeometry`, NOT from the
        gmsh subdomain tags, for two reasons:

        1.  The band footprint is exactly a rectangle in ``y``, so the
            geometric test is unambiguous. The mesh is fragmented along the
            band edge (see :func:`build_strip_mesh`), so no element straddles
            it and centroid classification is exact. :func:`_validate` checks
            that conformity holds.
        2.  ``skfem.io.meshio`` returns subdomain indices offset by the
            preceding line-cell block, so ``mesh.subdomains[BAND]`` does not
            index elements directly. Boundary (facet) tags are unaffected and
            are used as-is. Going through the geometry sidesteps the interop
            entirely rather than depending on the offset staying constant.
        """
        n = self.n_elements
        g = self.geometry
        if not g.has_steel_band:
            return np.zeros(n, dtype=bool)
        if g.band_full_width:
            return np.ones(n, dtype=bool)
        centroids = self.element_centroids()
        mid = g.strap_width / 2 + g.steel_band_offset
        half = g.steel_band_width / 2
        return np.abs(centroids[1] - mid) <= half

    def band_vertex_mask(self) -> np.ndarray:
        """Boolean mask over mesh vertices: True where the steel band is present.

        Vertices exactly on the band edge count as inside. That is the
        conservative choice for diffusion: inside the band the polymer skin is
        thinner, so it wets faster.
        """
        g = self.geometry
        n = self.n_vertices
        if not g.has_steel_band:
            return np.zeros(n, dtype=bool)
        if g.band_full_width:
            return np.ones(n, dtype=bool)
        mid = g.strap_width / 2 + g.steel_band_offset
        half = g.steel_band_width / 2
        tol = max(_REL_TOL * g.hole_radius, 1e-9)
        return np.abs(self.mesh.p[1] - mid) <= half + tol

    def summary(self) -> str:
        g, c = self.geometry, self.controls
        return "\n".join(
            [
                "TIER 2 MESH",
                "-----------",
                f"  domain            {g.modelled_length:g} x {g.strap_width:g} mm",
                f"  elements          {self.n_elements}",
                f"  vertices          {self.n_vertices}",
                f"  hole-edge size    {c.hole_edge_size(g):.4f} mm "
                f"({c.elements_around_hole} around each hole)",
                f"  bulk size         {c.bulk_size(g):.4f} mm",
                f"  boundaries        {', '.join(sorted(self.mesh.boundaries))}",
                f"  subdomains        "
                f"{', '.join(sorted(self.mesh.subdomains)) if self.mesh.subdomains else '(none)'}",
            ]
        )


def build_strip_mesh(
    geometry: StrapGeometry,
    controls: MeshControls | None = None,
):
    """Mesh the perforated strap window.

    Returns a :class:`StripMesh` wrapping a ``skfem.MeshTri`` whose boundaries
    and subdomains are named as documented at module level.

    Raises :class:`ImportError` with an actionable message if gmsh is missing
    or cannot load -- the gmsh wheel needs a handful of X11/GL shared libraries
    even when run headless.
    """
    controls = controls or MeshControls()
    try:
        import gmsh  # noqa: F401
        import pygmsh
        from skfem.io.meshio import from_meshio
    except OSError as exc:  # gmsh's shared library failed to load
        raise ImportError(
            f"gmsh is installed but its shared library will not load ({exc}). "
            "The gmsh wheel links against X11/GL libraries even in headless "
            "use; on Debian/Ubuntu install: libglu1-mesa libxft2 libxinerama1 "
            "libxcursor1 libxrender1 libxfixes3 libfontconfig1."
        ) from exc

    g = geometry
    length = g.modelled_length
    width = g.strap_width
    radius = g.hole_radius
    centres = g.hole_centres
    tol = max(_REL_TOL * radius, 1e-9)

    h_hole = controls.hole_edge_size(g)
    h_bulk = controls.bulk_size(g)

    band_present = g.has_steel_band and not g.band_full_width
    band_mid = width / 2 + g.steel_band_offset
    band_lo = band_mid - g.steel_band_width / 2
    band_hi = band_mid + g.steel_band_width / 2

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("General.Verbosity", controls.verbosity)

        plate = geom.add_rectangle([0.0, 0.0, 0.0], length, width)

        if band_present:
            # Fragmenting against the band footprint gives a conforming
            # interface between the reinforced and unreinforced strips, so the
            # laminate stiffness jump lands on element boundaries.
            band_rect = geom.add_rectangle(
                [0.0, band_lo, 0.0], length, band_hi - band_lo
            )
            solids: Iterable = geom.boolean_fragments(plate, band_rect)
        else:
            solids = [plate]

        disks = [
            geom.add_disk([float(cx), float(cy), 0.0], radius) for cx, cy in centres
        ]
        geom.boolean_difference(solids, disks)
        geom.synchronize()

        _tag_surfaces(gmsh, band_present, band_lo, band_hi, tol)
        _tag_curves(gmsh, g, centres, length, width, radius, tol)

        geom.set_mesh_size_callback(
            _size_callback(centres, radius, h_hole, h_bulk, controls.grading)
        )
        if controls.optimise:
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        meshio_mesh = geom.generate_mesh(dim=2, order=1)

    skfem_mesh = from_meshio(meshio_mesh)
    strip = StripMesh(mesh=skfem_mesh, geometry=g, controls=controls)
    _validate(strip)
    return strip


def _size_callback(centres, radius, h_hole, h_bulk, grading):
    """Element size: fine on the hole walls, growing linearly away from them."""
    cx = np.asarray(centres, dtype=float)[:, 0]
    cy = np.asarray(centres, dtype=float)[:, 1]

    def callback(dim, tag, x, y, z, lc):
        distance = np.min(np.hypot(x - cx, y - cy)) - radius
        return float(min(h_bulk, h_hole + grading * max(distance, 0.0)))

    return callback


def _tag_surfaces(gmsh, band_present: bool, band_lo: float, band_hi: float, tol: float):
    """Assign BAND / POLYMER physical groups by surface centre of mass."""
    band_tags: list[int] = []
    polymer_tags: list[int] = []
    for dim, tag in gmsh.model.getEntities(2):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        inside_band = band_present and (band_lo - tol) <= com[1] <= (band_hi + tol)
        (band_tags if inside_band else polymer_tags).append(tag)

    if not band_present:
        # A single region: name it BAND if the band spans the strap, else
        # POLYMER. Callers use band_element_mask(), which handles both.
        gmsh.model.addPhysicalGroup(2, polymer_tags, name=POLYMER)
        return

    if band_tags:
        gmsh.model.addPhysicalGroup(2, band_tags, name=BAND)
    if polymer_tags:
        gmsh.model.addPhysicalGroup(2, polymer_tags, name=POLYMER)


def _tag_curves(gmsh, geometry, centres, length, width, radius, tol):
    """Assign boundary physical groups by curve position.

    Hole walls are identified by their centre of mass coinciding with a hole
    centre; the four outer edges by having zero extent in x or y at the right
    coordinate.  Classifying by position rather than by entity tag keeps this
    robust against gmsh renumbering entities after boolean operations.
    """
    groups: dict[str, list[int]] = {
        CUT_START: [],
        CUT_END: [],
        SIDE_LOWER: [],
        SIDE_UPPER: [],
    }
    for label in geometry.hole_labels:
        groups[label] = []

    for dim, tag in gmsh.model.getEntities(1):
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.occ.getBoundingBox(dim, tag)

        hole_index = _matching_hole(com, centres, tol)
        if hole_index is not None:
            groups[geometry.hole_labels[hole_index]].append(tag)
            continue

        if abs(xmin) < tol and abs(xmax) < tol:
            groups[CUT_START].append(tag)
        elif abs(xmin - length) < tol and abs(xmax - length) < tol:
            groups[CUT_END].append(tag)
        elif abs(ymin) < tol and abs(ymax) < tol:
            groups[SIDE_LOWER].append(tag)
        elif abs(ymin - width) < tol and abs(ymax - width) < tol:
            groups[SIDE_UPPER].append(tag)
        # Anything left over is the internal band interface, which needs no tag.

    for name, tags in groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, name=name)


def _matching_hole(com, centres, tol) -> int | None:
    """Index of the hole whose centre coincides with this curve's centroid.

    A full circle's centre of mass is its centre; no outer edge of the
    rectangle can share that position, so this is unambiguous.
    """
    for i, (cx, cy) in enumerate(centres):
        if abs(com[0] - cx) < tol and abs(com[1] - cy) < tol:
            return i
    return None


def _validate(strip: StripMesh) -> None:
    """Fail loudly if the mesh did not come out as the geometry demands.

    A silently mis-tagged boundary would produce a plausible-looking but wrong
    stress field, which is the worst possible outcome for an investigation.
    """
    mesh = strip.mesh
    g = strip.geometry
    boundaries = mesh.boundaries or {}

    missing = [
        n
        for n in (CUT_START, CUT_END, SIDE_LOWER, SIDE_UPPER)
        if n not in boundaries
    ]
    if missing:
        raise RuntimeError(
            f"mesh is missing expected boundaries {missing}; "
            "gmsh entity classification failed"
        )

    found_holes = strip.hole_boundaries
    if len(found_holes) != g.n_holes:
        raise RuntimeError(
            f"tagged {len(found_holes)} hole boundaries but geometry has "
            f"{g.n_holes}; check hole placement against the strap envelope"
        )

    # Meshed area should equal rectangle minus holes, to a few percent. A large
    # shortfall means a boolean operation silently removed the wrong region.
    expected = g.modelled_length * g.strap_width - g.n_holes * np.pi * g.hole_radius**2
    actual = float(np.sum(_element_areas(mesh)))
    if not np.isclose(actual, expected, rtol=0.02):
        raise RuntimeError(
            f"meshed area {actual:.3f} mm^2 differs from expected {expected:.3f} "
            "mm^2 by more than 2%; the boolean geometry is wrong"
        )

    _validate_band_conformity(strip)


def _validate_band_conformity(strip: StripMesh) -> None:
    """No element may straddle the steel band's edge.

    The band edge is a jump in section stiffness. An element spanning it would
    smear that jump over its own width, and would also break the centroid-based
    classification in :meth:`StripMesh.band_element_mask`. The geometry is
    fragmented along the band edge precisely so this cannot happen, so a
    failure here means the boolean fragment did not take.
    """
    g = strip.geometry
    if not g.has_steel_band or g.band_full_width:
        return

    mid = g.strap_width / 2 + g.steel_band_offset
    half = g.steel_band_width / 2
    tol = max(_REL_TOL * g.hole_radius, 1e-9)

    y = strip.mesh.p[1][strip.mesh.t]
    for edge in (mid - half, mid + half):
        above = (y > edge + tol).any(axis=0)
        below = (y < edge - tol).any(axis=0)
        straddling = int(np.sum(above & below))
        if straddling:
            raise RuntimeError(
                f"{straddling} element(s) straddle the steel band edge at "
                f"y = {edge:g} mm; the mesh does not conform to the band "
                "footprint, so the section stiffness jump would be smeared"
            )


def _element_areas(mesh) -> np.ndarray:
    p, t = mesh.p, mesh.t
    x = p[0, t]
    y = p[1, t]
    return 0.5 * np.abs(
        (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    )
