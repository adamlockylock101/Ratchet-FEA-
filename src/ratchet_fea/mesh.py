"""2D mesh of the perforated strap, optionally with a crack seeded at a hole.

Generates a plane mesh of the strap window -- a rectangle of
``modelled_length x strap_width`` with a row of ``n_holes`` perforations --
using gmsh's OCC kernel through :mod:`pygmsh`, then converts it to a
:class:`skfem.MeshTri` with named boundaries.

NOTHING HERE IS HARDCODED.  Every dimension comes from
:class:`~ratchet_fea.geometry.StrapGeometry` and every mesh size derives from
the hole radius, the ligament widths and the crack length.  Replace the
placeholder dimensions with measured ones and the mesh regenerates at an
equivalent density.

Seeding a crack
---------------
A crack is a surface, not a thin slot, so it cannot be cut out of the geometry.
It is made in three steps:

1.  the crack line is imprinted onto the plate with an OCC fragment, so mesh
    edges lie exactly along it;
2.  the mesh is graded down toward the crack tip, where the stress field is
    singular;
3.  the nodes along the crack line are **duplicated**, and the elements on one
    side are remapped onto the duplicates -- everywhere except at the tip,
    which stays shared. That is what makes it a crack tip rather than a cut.

The crack faces then become ordinary traction-free boundaries with no special
treatment, which is exactly what a crack is.

Named boundaries
----------------
``CUT_START`` / ``CUT_END``
    The x = 0 and x = L faces.  Model truncation planes cut through a longer
    strap, not physical surfaces: they carry the remote tension.
``SIDE_LOWER`` / ``SIDE_UPPER``
    The y = 0 and y = W long edges -- real free surfaces.
``hole_0`` ... ``hole_{n-1}``
    Each perforation wall, tagged individually.
``CRACK_UPPER`` / ``CRACK_LOWER``
    The two crack faces, present only on a cracked mesh.  Separated so that
    :mod:`ratchet_fea.fracture` can measure the opening between them, which is
    how crack closure is detected rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import CrackGeometry, StrapGeometry

__all__ = [
    "CUT_START",
    "CUT_END",
    "SIDE_LOWER",
    "SIDE_UPPER",
    "CRACK_UPPER",
    "CRACK_LOWER",
    "MeshControls",
    "StripMesh",
    "build_strip_mesh",
]

CUT_START = "cut_start"
CUT_END = "cut_end"
SIDE_LOWER = "side_lower"
SIDE_UPPER = "side_upper"
CRACK_UPPER = "crack_upper"
CRACK_LOWER = "crack_lower"

#: Tolerance for classifying entities by position, as a fraction of the hole
#: radius. Geometric, so it scales with the part instead of assuming mm.
_REL_TOL = 1e-6


@dataclass(frozen=True)
class MeshControls:
    """Mesh density, expressed relative to the geometry rather than in mm.

    Stated this way the mesh stays equivalent when the placeholder dimensions
    are replaced by measured ones -- a 3 mm hole gets the same number of
    elements round it as a 4 mm hole did, and a 0.5 mm crack is resolved as
    finely as a 5 mm one.
    """

    #: Elements around each hole's circumference.
    elements_around_hole: int = 48
    #: Elements along the crack, from mouth to tip. Sets how finely the
    #: near-tip field is resolved, which is what the J-integral needs.
    elements_along_crack: int = 40
    #: Tip element size as a fraction of the crack length. The J-integral is
    #: a domain form and does not need the singularity resolved pointwise, but
    #: the contour has to sit in a well-resolved field.
    tip_size_fraction: float = 0.02
    #: Bulk element size as a multiple of the hole-edge element size.
    bulk_size_factor: float = 4.0
    #: Minimum elements across the narrowest ligament.
    min_ligament_divisions: int = 6
    #: Element size growth per unit distance away from a refined feature.
    grading: float = 0.35
    optimise: bool = True
    verbosity: int = 0

    def __post_init__(self) -> None:
        if self.elements_around_hole < 8:
            raise ValueError("need at least 8 elements around a hole to resolve Kt")
        if self.elements_along_crack < 4:
            raise ValueError("need at least 4 elements along a crack")
        if not 0.0 < self.tip_size_fraction < 1.0:
            raise ValueError("tip_size_fraction must lie in (0, 1)")
        if self.bulk_size_factor < 1.0:
            raise ValueError("bulk_size_factor must be >= 1")
        if self.min_ligament_divisions < 2:
            raise ValueError("min_ligament_divisions must be >= 2")
        if self.grading <= 0.0:
            raise ValueError("grading must be positive")

    def refined(self, factor: float = 2.0) -> "MeshControls":
        """Return controls with ``factor`` times the resolution.

        For convergence studies: a K that moves by more than a couple of
        percent between ``controls`` and ``controls.refined()`` is not
        converged and should not be quoted.
        """
        if factor <= 0:
            raise ValueError("refinement factor must be positive")
        return replace(
            self,
            elements_around_hole=int(round(self.elements_around_hole * factor)),
            elements_along_crack=int(round(self.elements_along_crack * factor)),
            min_ligament_divisions=int(round(self.min_ligament_divisions * factor)),
            tip_size_fraction=self.tip_size_fraction / factor,
        )

    def hole_edge_size(self, geometry: StrapGeometry) -> float:
        """Target element size on a hole wall, mm."""
        return 2.0 * np.pi * geometry.hole_radius / self.elements_around_hole

    def crack_size(self, crack: CrackGeometry) -> float:
        """Target element size along the crack, mm."""
        return crack.length / self.elements_along_crack

    def tip_size(self, crack: CrackGeometry) -> float:
        """Target element size at the crack tip, mm."""
        return crack.length * self.tip_size_fraction

    def bulk_size(self, geometry: StrapGeometry) -> float:
        """Target element size far from any refined feature, mm."""
        narrowest = min(geometry.side_ligament, geometry.inter_hole_ligament)
        return min(
            self.bulk_size_factor * self.hole_edge_size(geometry),
            narrowest / self.min_ligament_divisions,
        )


@dataclass
class StripMesh:
    """A generated mesh plus the geometry and crack it came from."""

    mesh: object  # skfem.MeshTri
    geometry: StrapGeometry
    controls: MeshControls
    crack: CrackGeometry | None = None

    @property
    def is_cracked(self) -> bool:
        return self.crack is not None

    @property
    def hole_boundaries(self) -> list:
        return [h for h in self.geometry.hole_labels if h in self.mesh.boundaries]

    @property
    def traction_boundary(self) -> str:
        return CUT_END

    @property
    def restrained_boundary(self) -> str:
        return CUT_START

    @property
    def n_elements(self) -> int:
        return int(self.mesh.t.shape[1])

    @property
    def n_vertices(self) -> int:
        return int(self.mesh.p.shape[1])

    def element_centroids(self) -> np.ndarray:
        return self.mesh.p[:, self.mesh.t].mean(axis=1)

    def crack_tip(self) -> np.ndarray:
        if self.crack is None:
            raise ValueError("this mesh has no crack")
        return self.geometry.crack_tip(self.crack)

    def crack_mouth(self) -> np.ndarray:
        if self.crack is None:
            raise ValueError("this mesh has no crack")
        return self.geometry.crack_mouth(self.crack)

    def summary(self) -> str:
        g, c = self.geometry, self.controls
        lines = [
            "MESH",
            "----",
            f"  domain            {g.modelled_length:g} x {g.strap_width:g} mm",
            f"  elements          {self.n_elements}",
            f"  vertices          {self.n_vertices}",
            f"  hole-edge size    {c.hole_edge_size(g):.4f} mm "
            f"({c.elements_around_hole} around each hole)",
            f"  bulk size         {c.bulk_size(g):.4f} mm",
        ]
        if self.crack is not None:
            lines += [
                f"  crack             {self.crack.length:g} mm, "
                f"{self.crack.orientation.value}, from "
                f"{g.hole_labels[self.crack.hole_index if self.crack.hole_index is not None else g.default_crack_hole()]}",
                f"  crack tip at      ({self.crack_tip()[0]:.3f}, {self.crack_tip()[1]:.3f}) mm",
                f"  tip element size  {c.tip_size(self.crack):.5f} mm",
                f"  ligament ahead    {g.remaining_ligament(self.crack):.3f} mm",
            ]
        lines.append(f"  boundaries        {', '.join(sorted(self.mesh.boundaries))}")
        return "\n".join(lines)


def build_strip_mesh(
    geometry: StrapGeometry,
    controls: MeshControls | None = None,
    crack: CrackGeometry | None = None,
):
    """Mesh the perforated strap window, optionally with a crack.

    Raises :class:`ImportError` with an actionable message if gmsh is missing
    or cannot load -- the gmsh wheel needs a handful of X11/GL shared libraries
    even when run headless.
    """
    controls = controls or MeshControls()
    try:
        import gmsh  # noqa: F401
        import pygmsh
        from skfem.io.meshio import from_meshio
    except OSError as exc:
        raise ImportError(
            f"gmsh is installed but its shared library will not load ({exc}). "
            "The gmsh wheel links against X11/GL libraries even in headless "
            "use; on Debian/Ubuntu install: libglu1-mesa libxft2 libxinerama1 "
            "libxcursor1 libxrender1 libxfixes3 libfontconfig1."
        ) from exc

    g = geometry
    if crack is not None and not g.crack_is_admissible(crack):
        raise ValueError(
            f"a {crack.length:g} mm {crack.orientation.value} crack does not fit: "
            f"the most this geometry can hold is "
            f"{g.max_crack_length(crack):g} mm before it breaks through"
        )

    mouth = g.crack_mouth(crack) if crack is not None else None
    tip = g.crack_tip(crack) if crack is not None else None

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("General.Verbosity", controls.verbosity)

        plate = geom.add_rectangle([0.0, 0.0, 0.0], g.modelled_length, g.strap_width)
        disks = [
            geom.add_disk([float(cx), float(cy), 0.0], g.hole_radius)
            for cx, cy in g.hole_centres
        ]
        geom.boolean_difference(plate, disks)
        geom.synchronize()

        if crack is not None:
            _imprint_crack(gmsh, mouth, tip)

        geom.set_mesh_size_callback(
            _size_callback(g, controls, tip, crack)
        )
        if controls.optimise:
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

        meshio_mesh = geom.generate_mesh(dim=2, order=1)

    mesh = from_meshio(meshio_mesh)
    if crack is not None:
        mesh = _split_crack_nodes(mesh, mouth, tip, g)
    mesh = _tag_boundaries(mesh, g, controls, mouth, tip)

    strip = StripMesh(mesh=mesh, geometry=g, controls=controls, crack=crack)
    _validate(strip)
    return strip


def _imprint_crack(gmsh, mouth, tip) -> None:
    """Fragment the plate with the crack line so mesh edges lie along it.

    ``fragment`` rather than ``embed`` because the crack mouth sits ON the hole
    boundary, and embedding a curve whose endpoint touches the surface boundary
    is unreliable; fragmenting imprints it cleanly either way.
    """
    surfaces = gmsh.model.getEntities(2)
    p0 = gmsh.model.occ.addPoint(float(mouth[0]), float(mouth[1]), 0.0)
    p1 = gmsh.model.occ.addPoint(float(tip[0]), float(tip[1]), 0.0)
    line = gmsh.model.occ.addLine(p0, p1)
    gmsh.model.occ.fragment(surfaces, [(1, line)])
    gmsh.model.occ.synchronize()


def _size_callback(geometry, controls, tip, crack):
    """Element size: fine on the hole walls and finest at the crack tip."""
    centres = np.asarray(geometry.hole_centres, dtype=float)
    cx, cy = centres[:, 0], centres[:, 1]
    radius = geometry.hole_radius
    h_hole = controls.hole_edge_size(geometry)
    h_bulk = controls.bulk_size(geometry)
    h_tip = controls.tip_size(crack) if crack is not None else None
    h_crack = controls.crack_size(crack) if crack is not None else None
    grading = controls.grading

    def callback(dim, tag, x, y, z, lc):
        from_hole = np.min(np.hypot(x - cx, y - cy)) - radius
        size = min(h_bulk, h_hole + grading * max(from_hole, 0.0))
        if tip is not None:
            from_tip = float(np.hypot(x - tip[0], y - tip[1]))
            # Grade up from the tip, but never coarser than the crack-line size
            # until well away from the crack.
            size = min(size, h_tip + grading * from_tip, h_crack + grading * from_tip)
        return float(size)

    return callback


def _split_crack_nodes(mesh, mouth, tip, geometry):
    """Duplicate the nodes along the crack line to open it into a real crack.

    The tip node is deliberately left shared: that single connected node is
    what distinguishes a crack from a cut all the way through.
    """
    from skfem import MeshTri

    p, t = mesh.p.copy(), mesh.t.copy()
    segment = tip - mouth
    length = float(np.linalg.norm(segment))
    unit = segment / length
    normal = np.array([-unit[1], unit[0]])
    tol = _tolerance(geometry)

    relative = p - mouth[:, None]
    along = unit @ relative
    across = normal @ relative
    # Nodes on the crack line, from the mouth up to but excluding the tip.
    on_crack = (np.abs(across) < tol) & (along > -tol) & (along < length - tol)
    indices = np.where(on_crack)[0]
    if len(indices) == 0:
        raise RuntimeError(
            "no mesh nodes landed on the crack line; the imprint did not take"
        )

    duplicates = np.arange(p.shape[1], p.shape[1] + len(indices))
    p_split = np.hstack([p, p[:, indices]])

    centroids = p[:, t].mean(axis=1)
    below = (normal @ (centroids - mouth[:, None])) < 0.0

    t_split = t.copy()
    for original, duplicate in zip(indices, duplicates):
        for row in range(t.shape[0]):
            t_split[row, (t[row] == original) & below] = duplicate

    return MeshTri(p_split, t_split)


def _tolerance(geometry) -> float:
    return max(_REL_TOL * geometry.hole_radius, 1e-9)


def _hole_facet_tolerance(geometry, controls) -> float:
    """How far a facet midpoint may sit inside the true hole radius.

    A circle meshed with ``n`` straight segments has facet midpoints pulled in
    by the chord sagitta, ``r (1 - cos(pi/n))``. A fixed fraction of the radius
    would either miss every facet on a coarse mesh or sweep up neighbouring
    material on a fine one, so the tolerance is derived from the resolution
    actually requested -- with headroom, since gmsh does not place exactly
    ``n`` segments.
    """
    sagitta = geometry.hole_radius * (
        1.0 - np.cos(np.pi / controls.elements_around_hole)
    )
    return max(3.0 * sagitta, 1e-9)


def _tag_boundaries(mesh, geometry, controls, mouth, tip):
    """Re-tag boundaries geometrically.

    Splitting the crack nodes rebuilds the facet numbering, so any tags carried
    over from gmsh would silently point at the wrong facets. Re-deriving them
    from position is both simpler and safe against that.
    """
    g = geometry
    tol = _tolerance(g)
    length = g.modelled_length
    width = g.strap_width

    selectors = {
        CUT_START: lambda x: x[0] < tol,
        CUT_END: lambda x: x[0] > length - tol,
        SIDE_LOWER: lambda x: x[1] < tol,
        SIDE_UPPER: lambda x: x[1] > width - tol,
    }
    hole_tol = _hole_facet_tolerance(g, controls)
    for index, (cx, cy) in enumerate(g.hole_centres):
        selectors[g.hole_labels[index]] = (
            lambda x, cx=cx, cy=cy: np.abs(
                np.hypot(x[0] - cx, x[1] - cy) - g.hole_radius
            )
            < hole_tol
        )

    boundaries = {}
    for name, selector in selectors.items():
        facets = mesh.facets_satisfying(selector, boundaries_only=True)
        if len(facets):
            boundaries[name] = facets

    if mouth is not None:
        upper, lower = _crack_face_facets(mesh, mouth, tip, tol)
        if len(upper):
            boundaries[CRACK_UPPER] = upper
        if len(lower):
            boundaries[CRACK_LOWER] = lower

    # Explicit index arrays rather than predicates: the two crack faces are
    # geometrically coincident, so no midpoint test could separate them.
    return mesh.with_boundaries(boundaries)


def _crack_face_facets(mesh, mouth, tip, tol):
    """The two crack faces, separated by which side of the crack they sit on.

    They are geometrically coincident, so position alone cannot tell them
    apart; the adjacent element's centroid can.
    """
    segment = tip - mouth
    length = float(np.linalg.norm(segment))
    unit = segment / length
    normal = np.array([-unit[1], unit[0]])

    midpoints = mesh.p[:, mesh.facets].mean(axis=1)
    relative = midpoints - mouth[:, None]
    along = unit @ relative
    across = normal @ relative

    boundary = mesh.boundary_facets()
    on_crack = np.where((np.abs(across) < tol) & (along > tol) & (along < length))[0]
    on_crack = np.intersect1d(on_crack, boundary)

    elements = mesh.f2t[0, on_crack]
    centroids = mesh.p[:, mesh.t[:, elements]].mean(axis=1)
    side = normal @ (centroids - mouth[:, None])
    return on_crack[side > 0], on_crack[side <= 0]


def _validate(strip: StripMesh) -> None:
    """Fail loudly if the mesh did not come out as the geometry demands."""
    mesh = strip.mesh
    g = strip.geometry
    boundaries = mesh.boundaries or {}

    missing = [
        n for n in (CUT_START, CUT_END, SIDE_LOWER, SIDE_UPPER) if n not in boundaries
    ]
    if missing:
        raise RuntimeError(
            f"mesh is missing expected boundaries {missing}; "
            "geometric boundary classification failed"
        )

    if len(strip.hole_boundaries) != g.n_holes:
        raise RuntimeError(
            f"tagged {len(strip.hole_boundaries)} hole boundaries but geometry "
            f"has {g.n_holes}; check hole placement against the strap envelope"
        )

    expected = g.modelled_length * g.strap_width - g.n_holes * np.pi * g.hole_radius**2
    actual = float(np.sum(_element_areas(mesh)))
    if not np.isclose(actual, expected, rtol=0.02):
        raise RuntimeError(
            f"meshed area {actual:.3f} mm^2 differs from expected {expected:.3f} "
            "mm^2 by more than 2%; the boolean geometry is wrong"
        )

    if strip.crack is not None:
        for name in (CRACK_UPPER, CRACK_LOWER):
            if name not in boundaries:
                raise RuntimeError(
                    f"cracked mesh has no {name} boundary; the node split did "
                    "not open the crack"
                )
        upper, lower = boundaries[CRACK_UPPER], boundaries[CRACK_LOWER]
        if len(upper) != len(lower):
            raise RuntimeError(
                f"crack faces have {len(upper)} and {len(lower)} facets; they "
                "must match, or the split was asymmetric"
            )
        _validate_tip_is_connected(strip)


def _validate_tip_is_connected(strip: StripMesh) -> None:
    """The tip node must be shared, or the 'crack' is a cut right through.

    A cut-through would give a completely different and silently wrong answer,
    so it is worth an explicit check rather than trusting the split logic.
    """
    mesh = strip.mesh
    tip = strip.crack_tip()
    tol = _tolerance(strip.geometry)
    at_tip = np.where(np.hypot(mesh.p[0] - tip[0], mesh.p[1] - tip[1]) < tol)[0]
    if len(at_tip) != 1:
        raise RuntimeError(
            f"{len(at_tip)} nodes sit at the crack tip; exactly one is required "
            "or the crack is cut clean through"
        )


def _element_areas(mesh) -> np.ndarray:
    p, t = mesh.p, mesh.t
    x, y = p[0, t], p[1, t]
    return 0.5 * np.abs(
        (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    )
