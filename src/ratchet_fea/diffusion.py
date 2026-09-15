"""Tier 2: transient Fickian moisture diffusion over the strap mesh.

Solves

    dc/dt = D grad^2 c

with the diffusivity bracket from :mod:`materials` and the exposed surfaces
identified by :mod:`mesh`, and returns a thickness-averaged concentration field
sampled at a sequence of times, ready to drive the swelling eigenstrain in
:mod:`mechanics`.

Why a plane model needs a through-thickness correction
------------------------------------------------------
A plane-stress FE model lives in the strap's mid-plane, so a 2D diffusion solve
on that mesh only sees moisture entering through the long side edges and the
hole walls.  But the strap's two big faces are exposed too, and they are much
closer to the interior: for the placeholder dimensions the through-thickness
path is ~1.1-1.5 mm against ~3 mm in-plane, and diffusion time goes as the
square of distance.  Solving the in-plane problem alone would predict the
strap wetting through roughly seven times slower than it really does.

This is handled exactly rather than fudged.  For a prism -- any 2D cross
section extruded through a uniform thickness -- with a uniform initial
concentration and the same fixed concentration on every surface, the Fickian
solution separates:

    (c_s - c) / (c_s - c_0)  =  u_2D(x, y, t) * u_1D(z, t)

(Crank, *The Mathematics of Diffusion*, 2nd ed., section 2.5.4: solutions for a
rectangular parallelepiped as products of plane-sheet solutions.)  So the code
solves the in-plane problem by FE, evaluates the through-thickness plane-sheet
solution analytically, and multiplies.  Averaging over the thickness -- which
is what a plane-stress model needs -- leaves the in-plane FE field multiplied
by the thickness-averaged 1D factor.

VERIFY: the product solution assumes the same boundary concentration on every
exposed surface and a uniform initial state.  It also assumes the steel band is
impermeable and perfectly bonded (see
:meth:`~ratchet_fea.geometry.StrapGeometry.diffusion_half_thickness`).  A
debonded band would wick moisture along its interface far faster than this
model allows, which would be a qualitatively different and worse case.

VERIFY: Fick's law with a constant D is an approximation for water in
polyamide.  Real PA66 sorption is often slightly non-Fickian, and D rises with
both temperature and concentration.  The diffusivity bracket spans a decade,
which dwarfs that error, but it means the *timing* of any result here is good
to an order of magnitude and no better.

UNITS: mm / s / mm^2 s^-1, concentration as a dimensionless mass fraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.special import erfc

from .geometry import StrapGeometry
from .materials import PA66_MOISTURE, MoistureTransport
from .mesh import StripMesh

__all__ = [
    "MoistureState",
    "DiffusionModel",
    "DiffusionResult",
    "slab_uptake_fraction",
    "slab_remaining_fraction",
    "log_time_grid",
    "solve_in_plane",
    "solve",
]

#: Below this value of the Fourier number D t / h^2 the short-time (sqrt-t)
#: series is used; above it the long-time exponential series. Both converge
#: comfortably here, so the switch point is not sensitive.
_SHORT_TIME_TAU = 0.25


class MoistureState(str, Enum):
    """Named equilibrium moisture states, resolved against the bracket."""

    DRY = "dry"
    RH50 = "50% RH"
    IMMERSED = "immersed"

    def content(self, moisture: MoistureTransport, corner: str = "nominal") -> float:
        """Equilibrium moisture mass fraction for this state."""
        if self is MoistureState.DRY:
            return 0.0
        if self is MoistureState.RH50:
            return moisture.saturation_50rh.at(corner)
        return moisture.saturation_immersed.at(corner)


# ---------------------------------------------------------------------------
# Through-thickness plane-sheet solution (analytic)
# ---------------------------------------------------------------------------


def slab_uptake_fraction(
    time: np.ndarray | float,
    diffusivity: float,
    half_thickness: float,
    n_terms: int = 20,
) -> np.ndarray:
    """Fractional moisture uptake ``M_t / M_inf`` of a plane sheet.

    ``half_thickness`` is the half-thickness of a sheet exposed on both faces,
    or equivalently the full depth of a layer exposed on one face and sealed on
    the other.

    Uses Crank's short-time series for ``D t / h^2 < 0.25`` and the long-time
    exponential series above it, so the result is accurate to better than 1e-6
    across the whole range with only 20 terms.  A single series would either
    converge slowly at small times or need hundreds of terms.
    """
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    if half_thickness <= 0.0:
        raise ValueError("half_thickness must be positive")

    t = np.atleast_1d(np.asarray(time, dtype=float))
    if np.any(t < 0.0):
        raise ValueError("time must be non-negative")

    tau = diffusivity * t / half_thickness**2
    out = np.zeros_like(tau)

    short = tau < _SHORT_TIME_TAU
    long = ~short

    # --- long-time series (Crank eq. 4.18) ---------------------------------
    if np.any(long):
        tl = tau[long]
        total = np.zeros_like(tl)
        for n in range(n_terms):
            m = 2 * n + 1
            total += (8.0 / (m * m * math.pi**2)) * np.exp(
                -(m**2) * math.pi**2 * tl / 4.0
            )
        out[long] = 1.0 - total

    # --- short-time series (Crank eq. 4.20) --------------------------------
    # M_t/M_inf = 2 sqrt(tau) [ 1/sqrt(pi) + 2 sum_n (-1)^n ierfc(n/sqrt(tau)) ]
    # Note the 1/sqrt(pi) sits INSIDE the bracket and applies to the leading
    # term only, not to the ierfc sum. Pulling it out front is an easy slip and
    # scales the correction terms by sqrt(pi); the cross-check in
    # tests/test_diffusion.py against the long-time series catches it.
    if np.any(short):
        ts = tau[short]
        result = np.zeros_like(ts)
        positive = ts > 0.0
        tp = ts[positive]
        root = np.sqrt(tp)
        series = np.zeros_like(tp)
        for n in range(1, n_terms + 1):
            series += ((-1.0) ** n) * _ierfc(n / root)
        result[positive] = 2.0 * root * (1.0 / math.sqrt(math.pi) + 2.0 * series)
        out[short] = result

    return np.clip(out, 0.0, 1.0)


def slab_remaining_fraction(
    time: np.ndarray | float,
    diffusivity: float,
    half_thickness: float,
    n_terms: int = 20,
) -> np.ndarray:
    """``1 - M_t/M_inf``: the thickness-averaged *unabsorbed* fraction.

    This is the ``u_1D`` factor of the product solution described at module
    level, already averaged over the thickness -- which is exactly the form a
    plane-stress model needs.
    """
    return 1.0 - slab_uptake_fraction(time, diffusivity, half_thickness, n_terms)


def _ierfc(z: np.ndarray) -> np.ndarray:
    """Integral of the complementary error function, ``ierfc(z)``."""
    z = np.asarray(z, dtype=float)
    return np.exp(-(z**2)) / math.sqrt(math.pi) - z * erfc(z)


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionModel:
    """Everything the diffusion solve needs, derived from geometry + materials.

    Build it with :meth:`between` rather than by hand, so that every value
    stays traceable to :mod:`geometry` and :mod:`materials`.
    """

    diffusivity: float
    c_initial: float
    c_surface: float
    #: Effective through-thickness half-thickness inside the band footprint.
    half_thickness_band: float
    #: Effective through-thickness half-thickness outside it.
    half_thickness_plain: float
    #: Include the through-thickness product factor. Turning this off isolates
    #: the pure in-plane field, which is useful for verification but is NOT a
    #: physically complete model of the strap.
    include_through_thickness: bool = True
    n_series_terms: int = 20
    corner: str = "nominal"
    label: str = ""

    def __post_init__(self) -> None:
        if self.diffusivity <= 0.0:
            raise ValueError("diffusivity must be positive")
        if self.c_initial < 0.0 or self.c_surface < 0.0:
            raise ValueError("moisture contents cannot be negative")
        if self.c_initial == self.c_surface:
            raise ValueError(
                "c_initial equals c_surface -- there is no moisture change to "
                "solve for; pick different start and end states"
            )
        for name in ("half_thickness_band", "half_thickness_plain"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def between(
        cls,
        geometry: StrapGeometry,
        start: MoistureState = MoistureState.DRY,
        end: MoistureState = MoistureState.IMMERSED,
        moisture: MoistureTransport | None = None,
        corner: str = "nominal",
        include_through_thickness: bool = True,
    ) -> "DiffusionModel":
        """Model a transition between two named equilibrium states.

        ``start=DRY, end=IMMERSED`` is absorption; reversing them is
        desorption.  Desorption matters because constrained *swelling* puts the
        polymer in compression, which does not crack it, whereas constrained
        *shrinkage* on drying puts the same magnitude into tension.
        """
        moisture = moisture or PA66_MOISTURE
        return cls(
            diffusivity=moisture.diffusivity.at(corner),
            c_initial=start.content(moisture, corner),
            c_surface=end.content(moisture, corner),
            half_thickness_band=geometry.diffusion_half_thickness(in_band_region=True),
            half_thickness_plain=geometry.diffusion_half_thickness(
                in_band_region=False
            ),
            include_through_thickness=include_through_thickness,
            corner=corner,
            label=f"{start.value} -> {end.value}",
        )

    @property
    def is_absorption(self) -> bool:
        return self.c_surface > self.c_initial

    @property
    def moisture_change(self) -> float:
        """Signed change in moisture content from start to equilibrium."""
        return self.c_surface - self.c_initial

    @property
    def governing_half_thickness(self) -> float:
        """Whichever through-thickness path wets fastest."""
        return min(self.half_thickness_band, self.half_thickness_plain)

    def characteristic_time(self) -> float:
        """Time to reach half the equilibrium uptake through the thickness, s."""
        return 0.19685 * self.governing_half_thickness**2 / self.diffusivity

    def summary(self) -> str:
        return "\n".join(
            [
                "DIFFUSION MODEL",
                "---------------",
                f"  transition        {self.label or 'custom'} "
                f"({'absorption' if self.is_absorption else 'desorption'})",
                f"  c_initial         {self.c_initial:.4f} mass fraction",
                f"  c_surface         {self.c_surface:.4f} mass fraction",
                f"  diffusivity       {self.diffusivity:.3e} mm^2/s "
                f"[{self.corner} bracket corner]",
                f"  half-thickness    {self.half_thickness_band:g} mm (band) / "
                f"{self.half_thickness_plain:g} mm (plain)",
                f"  through-thickness {'included' if self.include_through_thickness else 'EXCLUDED'}",
                f"  half-uptake time  {self.characteristic_time() / 86400.0:.1f} d",
            ]
        )


@dataclass
class DiffusionResult:
    """Concentration history over the mesh."""

    times: np.ndarray
    #: ``(n_times, n_vertices)`` thickness-averaged moisture mass fraction.
    concentration: np.ndarray
    #: ``(n_times, n_vertices)`` in-plane unabsorbed fraction ``u_2D``.
    in_plane_remaining: np.ndarray
    #: ``(n_times, n_vertices)`` through-thickness factor ``u_1D``.
    thickness_remaining: np.ndarray
    #: ``(n_times,)`` area-weighted global uptake, ``M_t / M_inf``.
    uptake_fraction: np.ndarray
    model: DiffusionModel

    def at_time(self, index: int) -> np.ndarray:
        return self.concentration[index]

    @property
    def n_times(self) -> int:
        return len(self.times)

    def time_to_uptake(self, fraction: float) -> float:
        """Interpolated time at which global uptake reaches ``fraction``."""
        if not 0.0 < fraction < 1.0:
            raise ValueError("fraction must lie strictly between 0 and 1")
        u = self.uptake_fraction
        if u[-1] < fraction:
            return float("nan")
        return float(np.interp(fraction, u, self.times))


# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------


def log_time_grid(t_end: float, n_steps: int = 40, t_start: float | None = None):
    """Logarithmically spaced times from ``t_start`` to ``t_end``, plus t = 0.

    Diffusion transients are self-similar in ``sqrt(t)``, so uniform steps waste
    almost all their effort on the flat tail while missing the early gradient --
    which is where the differential-swelling stress peaks.  Log spacing puts the
    resolution where the physics is.
    """
    if t_end <= 0:
        raise ValueError("t_end must be positive")
    if n_steps < 2:
        raise ValueError("need at least 2 steps")
    if t_start is None:
        t_start = t_end / 1.0e4
    if not 0 < t_start < t_end:
        raise ValueError("require 0 < t_start < t_end")
    return np.concatenate([[0.0], np.geomspace(t_start, t_end, n_steps)])


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve_in_plane(
    mesh,
    exposed_boundaries,
    diffusivity: float,
    times: np.ndarray,
) -> np.ndarray:
    """Solve the normalised in-plane diffusion problem.

    Returns ``(n_times, n_vertices)`` of ``u_2D``, the *unabsorbed* fraction:
    1 in still-pristine material, 0 where the surface concentration has been
    reached.  Working in this normalised variable makes the boundary condition
    homogeneous, so absorption and desorption are the same solve.

    Boundaries not listed in ``exposed_boundaries`` get the natural zero-flux
    condition.  For the strap that means the model truncation planes, which is
    right: material continues past them.
    """
    from skfem import Basis, ElementTriP1, condense, solve as skfem_solve
    from skfem.models.poisson import laplace, mass

    times = np.asarray(times, dtype=float)
    if times[0] != 0.0:
        raise ValueError("time grid must start at t = 0")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time grid must be strictly increasing after t = 0")

    basis = Basis(mesh, ElementTriP1())
    stiffness = laplace.assemble(basis)
    mass_matrix = _lumped(mass.assemble(basis))
    dirichlet = basis.get_dofs(tuple(exposed_boundaries))

    n_dofs = basis.N
    u = np.ones(n_dofs)

    history = np.empty((len(times), n_dofs))
    # The true initial condition is u = 1 everywhere, including on the exposed
    # surfaces: at t = 0 no moisture has entered yet. The Dirichlet condition
    # applies from the first step onward. Recording the boundary as already wet
    # at t = 0 would report a few percent of spurious uptake before any time
    # had passed.
    history[0] = u

    # Backward Euler. Unconditionally stable, so the log-spaced steps below are
    # safe, and its numerical damping suppresses the spurious oscillation an
    # incompatible initial condition (u = 1 in the interior, 0 on the boundary)
    # would otherwise excite in a Crank-Nicolson scheme.
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        system = mass_matrix + dt * diffusivity * stiffness
        rhs = mass_matrix @ u
        u = skfem_solve(*condense(system, rhs, D=dirichlet))
        # Clip only the round-off level excursions a direct solve can leave;
        # mass lumping above already rules out the physical-scale overshoot.
        u = np.clip(u, 0.0, 1.0)
        history[i] = u

    return history


def solve(
    strip: StripMesh,
    model: DiffusionModel,
    times: np.ndarray,
) -> DiffusionResult:
    """Full transient solve: in-plane FE field times the analytic thickness factor.

    Returns thickness-averaged concentration, which is what a plane-stress
    mechanics model consumes.
    """
    times = np.asarray(times, dtype=float)

    u_plane = solve_in_plane(
        strip.mesh, strip.exposed_boundaries, model.diffusivity, times
    )

    n_vertices = u_plane.shape[1]
    if model.include_through_thickness:
        in_band = strip.band_vertex_mask()
        f_band = slab_remaining_fraction(
            times, model.diffusivity, model.half_thickness_band, model.n_series_terms
        )
        f_plain = slab_remaining_fraction(
            times, model.diffusivity, model.half_thickness_plain, model.n_series_terms
        )
        u_thick = np.where(in_band[None, :], f_band[:, None], f_plain[:, None])
    else:
        u_thick = np.ones((len(times), n_vertices))

    remaining = u_plane * u_thick
    concentration = model.c_surface + (model.c_initial - model.c_surface) * remaining

    weights = _vertex_areas(strip.mesh)
    total_area = weights.sum()
    uptake = np.clip(1.0 - (remaining @ weights) / total_area, 0.0, 1.0)

    return DiffusionResult(
        times=times,
        concentration=concentration,
        in_plane_remaining=u_plane,
        thickness_remaining=u_thick,
        uptake_fraction=uptake,
        model=model,
    )


def _lumped(mass_matrix):
    """Row-sum (lumped) mass matrix.

    The initial condition is discontinuous -- pristine material everywhere, the
    surface concentration imposed from the first step -- and a consistent mass
    matrix responds to that with a spurious oscillation in the nodes next to
    the boundary: the concentration overshoots, then relaxes back, so the field
    is briefly non-monotone in time. Row-sum lumping restores the discrete
    maximum principle for backward Euler, which keeps the solution bounded and
    monotone. It costs formal accuracy that is far below the decade-wide
    uncertainty in the diffusivity itself, and the verification test against
    the analytic plane-sheet series still passes to within 2%.
    """
    from scipy.sparse import diags

    return diags(np.asarray(mass_matrix.sum(axis=1)).ravel()).tocsr()


def _vertex_areas(mesh) -> np.ndarray:
    """Lumped nodal areas, for area-weighted averages over the mesh."""
    p, t = mesh.p, mesh.t
    x, y = p[0, t], p[1, t]
    areas = 0.5 * np.abs(
        (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0])
    )
    weights = np.zeros(p.shape[1])
    # mesh.t is (n_vertices_per_element, n_elements), so the per-vertex
    # contributions must be tiled down the first axis to line up with
    # t.ravel(). np.repeat would scatter each element's area to three
    # unrelated vertices and silently skew every area-weighted average.
    np.add.at(weights, t.ravel(), np.tile(areas / t.shape[0], (t.shape[0], 1)).ravel())
    return weights
