"""Stress intensity factors from the FE solution.

Two independent extractions, because a single one cannot be checked:

:func:`j_integral`
    The domain (equivalent domain integral) form of Rice's J.  Robust, uses
    the energy field rather than the singular stress directly, and comes with
    a free self-check: J must not depend on which annulus around the tip it is
    evaluated over.  :func:`j_integral_domain_independence` measures that
    spread and it is reported on every run.

:func:`k_from_opening`
    Displacement extrapolation from the crack-flank opening.  Completely
    different information -- displacements rather than energy -- so agreement
    between the two is meaningful.  It also yields the crack opening itself,
    which is how crack CLOSURE is detected rather than assumed.

Both are then checked against the handbook solutions in
:mod:`ratchet_fea.analytical`.

Why closure matters here
------------------------
A crack running along the strap, from one perforation toward the next, lies in
the hole's compressive hoop lobe.  Under axial tension its faces are pressed
together.  A linear elastic model has no contact and will happily let them
interpenetrate, reporting a confident and meaningless K.  :func:`k_from_opening`
measures the sign of the opening and refuses to report a mode I K when the
faces overlap, which is the physically correct answer: a closed crack has no
mode I driving force.

UNITS: mm / N / MPa, so J is in N/mm and K in MPa*sqrt(mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CrackGeometry, CrackOrientation, StrapGeometry
from .mechanics import MechanicsModel, MechanicsResult
from .mechanics import solve as solve_mechanics
from .mesh import CRACK_LOWER, CRACK_UPPER, MeshControls, StripMesh, build_strip_mesh

__all__ = [
    "CrackTipResult",
    "j_integral",
    "j_integral_domain_independence",
    "k_from_opening",
    "crack_opening",
    "evaluate_crack",
    "sweep_crack_length",
]


@dataclass
class CrackTipResult:
    """Everything extracted at one crack tip, at one crack length."""

    crack_length: float
    orientation: CrackOrientation
    #: J-integral, N/mm.
    j: float
    #: ``sqrt(E J)`` -- the effective K the energy release rate implies.
    k_from_j: float
    #: K from crack-flank displacement extrapolation.
    k_from_cod: float
    #: Spread of J across evaluation annuli, as a fraction of the mean. Small
    #: means the domain integral is converged.
    domain_independence: float
    #: True when the crack faces overlap, i.e. the crack is held shut.
    is_closed: bool
    #: Smallest (most negative) opening found on the flanks, mm.
    min_opening: float
    remaining_ligament: float
    model: MechanicsModel

    @property
    def k(self) -> float:
        """The K to quote: zero for a closed crack, else the J-based value.

        J is preferred over the displacement extrapolation because it uses the
        whole near-tip field rather than a handful of flank points; the COD
        value is kept as the cross-check.
        """
        return 0.0 if self.is_closed else self.k_from_j

    @property
    def partially_closed(self) -> bool:
        """Some of the flank overlaps, even though the tip region opens.

        Happens as a longitudinal crack reaches across the ligament and its tip
        enters the next hole's tensile lobe while the mouth is still shut. The
        linear model has no contact, so the overlapping part is free to
        interpenetrate; that makes the body more compliant than it really is,
        so the reported K is an UPPER BOUND.
        """
        return (not self.is_closed) and self.min_opening < 0.0

    @property
    def extraction_agreement(self) -> float:
        """``k_from_cod / k_from_j``. Should sit near 1."""
        if self.k_from_j == 0.0:
            return float("nan")
        return self.k_from_cod / self.k_from_j


# ---------------------------------------------------------------------------
# J-integral, domain form
# ---------------------------------------------------------------------------


def _tip_radius_limit(strip: StripMesh) -> float:
    """Largest annulus radius that stays clear of every other free surface.

    The domain form assumes the only surfaces inside the annulus are the crack
    flanks, which are traction free AND parallel to the crack. A hole wall or a
    strap edge inside the annulus breaks that and adds a surface term nobody
    has computed, so the annulus must not reach one.
    """
    g = strip.geometry
    crack = strip.crack
    tip = strip.crack_tip()

    limits = [crack.length]  # back along the crack to the hole wall
    for cx, cy in g.hole_centres:
        distance = float(np.hypot(tip[0] - cx, tip[1] - cy)) - g.hole_radius
        limits.append(abs(distance))
    limits.append(abs(tip[1]))
    limits.append(abs(g.strap_width - tip[1]))
    limits.append(abs(tip[0]))
    limits.append(abs(g.modelled_length - tip[0]))
    return max(min(limits), 0.0)


def _q_field(strip: StripMesh, inner: float, outer: float) -> np.ndarray:
    """Nodal ``q``: 1 inside ``inner`` of the tip, 0 outside ``outer``.

    A linear ramp between the two. Only the gradient enters J, so any smooth
    enough shape works; this is the standard choice.
    """
    if not 0.0 < inner < outer:
        raise ValueError("require 0 < inner < outer radius")
    tip = strip.crack_tip()
    r = np.hypot(strip.mesh.p[0] - tip[0], strip.mesh.p[1] - tip[1])
    q = (outer - r) / (outer - inner)
    return np.clip(q, 0.0, 1.0)


def j_integral(
    result: MechanicsResult,
    inner: float | None = None,
    outer: float | None = None,
) -> float:
    """Rice's J via the equivalent domain integral, N/mm.

        J = int_A [ sigma_ij du_j/dx_k d_k - W d_i ] dq/dx_i dA

    where ``d`` is the unit vector along the crack and ``W`` the strain energy
    density. Written with ``d`` explicitly rather than in crack-local axes, so
    both crack orientations go through the same code with no rotation step to
    get wrong.
    """
    from skfem import Functional

    strip = result.strip
    if strip.crack is None:
        raise ValueError("this mesh has no crack, so J is not defined")

    limit = _tip_radius_limit(strip)
    if outer is None:
        outer = 0.6 * limit
    if inner is None:
        inner = 0.3 * limit
    if outer > limit * 1.001:
        raise ValueError(
            f"outer radius {outer:g} mm reaches another free surface "
            f"({limit:g} mm away); the domain form is not valid there"
        )

    basis = result.basis
    scalar_basis = basis.with_element(_p1())
    q = scalar_basis.interpolate(_q_field(strip, inner, outer))
    disp = basis.interpolate(result.displacement)
    d0, d1 = strip.crack.direction

    @Functional
    def integrand(w):
        du = w["disp"].grad  # du[j, k] = du_j / dx_k
        dq = w["q"].grad
        sxx, syy, sxy = w["sxx"], w["syy"], w["sxy"]

        # Strain energy density, from the stresses already recovered.
        energy = 0.5 * (
            sxx * du[0, 0] + syy * du[1, 1] + sxy * (du[0, 1] + du[1, 0])
        )

        # Directional derivative of displacement along the crack.
        dud0 = du[0, 0] * d0 + du[0, 1] * d1
        dud1 = du[1, 0] * d0 + du[1, 1] * d1

        term0 = sxx * dud0 + sxy * dud1 - energy * d0
        term1 = sxy * dud0 + syy * dud1 - energy * d1
        return term0 * dq[0] + term1 * dq[1]

    return float(
        integrand.assemble(
            basis, disp=disp, q=q, sxx=result.sxx, syy=result.syy, sxy=result.sxy
        )
    )


def _p1():
    from skfem import ElementTriP1

    return ElementTriP1()


def j_integral_domain_independence(
    result: MechanicsResult, n_rings: int = 5
) -> tuple:
    """``(mean J, relative spread)`` over several annuli.

    J is path independent for an elastic body with traction-free crack faces,
    so the spread is a direct measure of discretisation error. A few percent is
    healthy; tens of percent means the near-tip mesh is too coarse and the
    number should not be quoted.
    """
    limit = _tip_radius_limit(result.strip)
    values = []
    for fraction in np.linspace(0.35, 0.85, n_rings):
        outer = fraction * limit
        values.append(j_integral(result, inner=0.45 * outer, outer=outer))
    values = np.asarray(values)
    mean = float(values.mean())
    if mean == 0.0:
        return 0.0, 0.0
    return mean, float(np.ptp(values) / abs(mean))


# ---------------------------------------------------------------------------
# Crack opening and displacement extrapolation
# ---------------------------------------------------------------------------


def crack_opening(result: MechanicsResult) -> tuple:
    """``(distance behind tip, opening)`` sampled along the crack flanks, mm.

    Positive opening means the faces have separated. Negative means the linear
    model has let them pass through each other, which is how a closed crack
    shows up in an analysis with no contact.
    """
    from skfem import ElementTriP2, ElementVector, FacetBasis

    strip = result.strip
    mesh = strip.mesh
    if strip.crack is None:
        raise ValueError("this mesh has no crack")

    element = ElementVector(ElementTriP2())
    mouth = strip.crack_mouth()
    tip = strip.crack_tip()
    segment = tip - mouth
    length = float(np.linalg.norm(segment))
    unit = segment / length
    normal = np.array(strip.crack.normal)

    samples = {}
    for name, sign in ((CRACK_UPPER, 1.0), (CRACK_LOWER, -1.0)):
        fb = FacetBasis(mesh, element, facets=mesh.boundaries[name])
        coords = np.asarray(fb.global_coordinates())
        u = np.asarray(fb.interpolate(result.displacement))
        along = unit[0] * (coords[0] - mouth[0]) + unit[1] * (coords[1] - mouth[1])
        normal_u = normal[0] * u[0] + normal[1] * u[1]
        order = np.argsort(along.ravel())
        samples[name] = (along.ravel()[order], normal_u.ravel()[order], sign)

    upper_along, upper_u, _ = samples[CRACK_UPPER]
    lower_along, lower_u, _ = samples[CRACK_LOWER]
    if len(upper_along) != len(lower_along):
        raise RuntimeError("crack flanks have different sample counts")
    if not np.allclose(upper_along, lower_along, atol=1e-6 * length):
        raise RuntimeError("crack flank samples do not line up")

    opening = upper_u - lower_u
    behind_tip = length - upper_along
    keep = behind_tip > 0
    return behind_tip[keep], opening[keep]


def k_from_opening(
    result: MechanicsResult,
    fit_range: tuple = (0.05, 0.4),
) -> tuple:
    """``(K_I, is_closed, min_opening)`` from crack-flank displacements.

    For plane stress the mode I near-tip opening is

        COD(r) = (8 K_I / E) sqrt(r / (2 pi))

    -- independent of Poisson's ratio -- so ``K_I(r) = (E/8) COD sqrt(2 pi / r)``
    is constant near the tip and can be extrapolated to ``r = 0``.  The fit
    uses flank points between ``fit_range`` fractions of the crack length:
    close enough that the singular term dominates, far enough that the element
    at the tip is not being asked to carry the answer on its own.

    Returns ``K_I = 0`` and ``is_closed = True`` if the faces overlap.
    """
    r, opening = crack_opening(result)
    if len(r) == 0:
        raise RuntimeError("no crack flank samples found")

    length = result.strip.crack.length
    min_opening = float(np.min(opening))
    # Judge closure on the tip half of the crack: the mouth can lift slightly
    # even on a crack that is shut where it matters.
    near_tip = r <= 0.5 * length
    if np.any(near_tip) and float(np.mean(opening[near_tip])) <= 0.0:
        return 0.0, True, min_opening

    lo, hi = fit_range
    window = (r >= lo * length) & (r <= hi * length) & (opening > 0)
    if window.sum() < 3:
        window = opening > 0
    if window.sum() < 2:
        return 0.0, True, min_opening

    e = result.model.youngs_modulus
    k_local = (e / 8.0) * opening[window] * np.sqrt(2.0 * np.pi / r[window])
    # Extrapolate the local estimates linearly to r = 0.
    slope, intercept = np.polyfit(r[window], k_local, 1)
    return float(intercept), False, min_opening


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def evaluate_crack(
    geometry: StrapGeometry,
    crack: CrackGeometry,
    model: MechanicsModel | None = None,
    controls: MeshControls | None = None,
) -> CrackTipResult:
    """Mesh, solve and extract K for one crack length."""
    model = model or MechanicsModel.from_materials(geometry)
    strip = build_strip_mesh(geometry, controls, crack)
    result = solve_mechanics(strip, model)

    j, spread = j_integral_domain_independence(result)
    k_cod, is_closed, min_opening = k_from_opening(result)

    # J is an energy and cannot be negative for an opening crack; a small
    # negative value on a closed crack is the interpenetration showing up.
    k_j = float(np.sqrt(model.youngs_modulus * j)) if j > 0 else 0.0

    return CrackTipResult(
        crack_length=crack.length,
        orientation=crack.orientation,
        j=j,
        k_from_j=k_j,
        k_from_cod=k_cod,
        domain_independence=spread,
        is_closed=is_closed,
        min_opening=min_opening,
        remaining_ligament=geometry.remaining_ligament(crack),
        model=model,
    )


def sweep_crack_length(
    geometry: StrapGeometry,
    crack_lengths,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    model: MechanicsModel | None = None,
    controls: MeshControls | None = None,
    progress: bool = False,
) -> list:
    """Evaluate K at each crack length. Returns a list of :class:`CrackTipResult`.

    Every length gets its own mesh, so the near-tip refinement follows the
    crack rather than being fixed once at the start.
    """
    model = model or MechanicsModel.from_materials(geometry)
    results = []
    for index, length in enumerate(np.atleast_1d(crack_lengths)):
        crack = CrackGeometry(length=float(length), orientation=orientation)
        if not geometry.crack_is_admissible(crack):
            raise ValueError(
                f"crack length {length:g} mm does not fit; the most this "
                f"geometry can hold is {geometry.max_crack_length(crack):g} mm"
            )
        if progress:
            print(
                f"    a = {length:6.3f} mm  ({index + 1}/{len(np.atleast_1d(crack_lengths))})",
                flush=True,
            )
        results.append(evaluate_crack(geometry, crack, model, controls))
    return results
