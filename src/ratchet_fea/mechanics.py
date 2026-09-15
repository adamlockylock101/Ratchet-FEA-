"""Tier 2: plane-stress elastic solve with a moisture swelling eigenstrain.

Solves the membrane problem for the perforated strap under

*   remote tension applied at the model's downstream cut face, and
*   a swelling eigenstrain ``eps_sw = beta * (c - c_ref) * I`` driven by the
    moisture field from :mod:`diffusion`,

and reports the PA66 stress -- specifically at the hole walls, where the strap
is cracking.

The through-thickness laminate
------------------------------
The strap is not homogeneous through its thickness: a steel band runs inside
it.  Rather than model that in 3D, the section is condensed to a membrane using
the A-matrix of classical laminate theory.  For in-plane loading of a laminate
with no bending, every layer shares the same in-plane strain, so the section
stiffness is the thickness-weighted sum of the layers' plane-stress stiffnesses

    A = sum_k  t_k Q_k

and the eigenstrain produces a section force

    N_sw = sum_k  t_k Q_k eps_sw_k

with ``eps_sw = 0`` in the steel, because steel neither absorbs water nor
swells.  That single fact is the whole mechanism: the polymer wants to grow,
the steel will not let it, and the mismatch has to go somewhere as stress.

Ply-level stress recovery is what matters
-----------------------------------------
The smeared membrane stress ``N/t`` is not a quantity that can be compared with
a PA66 strength -- it averages polymer and steel together.  After solving for
the membrane strain field, :func:`recover_polymer_stress` evaluates the stress
in the PA66 layer alone:

    sigma_PA66 = Q_PA66 : (eps - eps_sw_PA66)

In the fully-constrained limit (``eps -> 0``) this reduces exactly to Tier 1's
``E beta dc / (1 - nu)``, which is the cross-check :mod:`postprocess` runs.

SIGN CONVENTION AND WHAT IT MEANS
---------------------------------
Tension is positive.  Constrained *swelling* during absorption produces
COMPRESSIVE polymer stress.  Compression does not open a crack, so a large
negative number here is not by itself an explanation of the failure.  What does
produce tension is a moisture *gradient*: material that is still dry, held by
neighbouring material that has already swollen, is pulled into tension.  That
is why the results are reported as both maximum and minimum principal stress
over time, and why the entry point runs a desorption case as well.

LIMITATION -- THROUGH-THICKNESS GRADIENTS ARE AVERAGED AWAY.  This is a
membrane model fed a thickness-averaged concentration, so it cannot see the
skin-in-tension state that a drying surface develops over a still-wet core.
That is the classic moisture-driven surface cracking mechanism in polyamides
and it needs a through-thickness (Tier 3) model.  Do not read a low tensile
stress here as evidence that moisture is harmless.

UNITS: mm / N / MPa.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import StrapGeometry
from .materials import (
    PA66_MOISTURE,
    STEEL_BAND,
    MoistureTransport,
    PolymerGrade,
    pa66_properties,
)
from .mesh import CUT_END, CUT_START, StripMesh

__all__ = [
    "MechanicsModel",
    "MechanicsResult",
    "solve",
    "recover_polymer_stress",
    "principal_stresses",
    "von_mises",
    "hole_edge_stresses",
    "stress_concentration_factors",
]


@dataclass(frozen=True)
class MechanicsModel:
    """Section, material and load definition for the membrane solve.

    Build with :meth:`from_materials`; every field traces back to
    :mod:`geometry` or :mod:`materials`.
    """

    # --- section (through-thickness build-up) -------------------------------
    strap_thickness: float
    band_thickness: float
    strap_width: float

    # --- steel band ---------------------------------------------------------
    band_modulus: float
    band_poisson: float

    # --- polymer ------------------------------------------------------------
    swelling_coefficient: float
    #: Moisture content at which the polymer is stress-free.
    #:
    #: VERIFY: assumed to be the dry-as-moulded state, i.e. the strap was
    #: assembled around the band dry and every subsequent gram of water is a
    #: mismatch. If the band is instead bonded in after conditioning, or if the
    #: moulding left its own residual stress, this reference moves and the
    #: whole stress history shifts with it.
    stress_free_moisture: float = 0.0
    #: When True the polymer modulus is evaluated pointwise from the local
    #: concentration, so the water that drives the swelling also softens the
    #: polymer resisting it. Turning it off overstates swelling stress by 2-3x.
    moisture_dependent_modulus: bool = True
    #: Used only when ``moisture_dependent_modulus`` is False.
    fixed_modulus: float = 1350.0
    fixed_poisson: float = 0.41

    # --- load ---------------------------------------------------------------
    remote_tension: float = 0.0

    corner: str = "nominal"

    def __post_init__(self) -> None:
        if self.strap_thickness <= 0.0:
            raise ValueError("strap_thickness must be positive")
        if not 0.0 <= self.band_thickness < self.strap_thickness:
            raise ValueError("band_thickness must be in [0, strap_thickness)")
        if self.strap_width <= 0.0:
            raise ValueError("strap_width must be positive")

    @classmethod
    def from_materials(
        cls,
        geometry: StrapGeometry,
        moisture: MoistureTransport | None = None,
        steel: PolymerGrade | None = None,
        corner: str = "nominal",
        include_band: bool = True,
        include_tension: bool = True,
        moisture_dependent_modulus: bool = True,
        stress_free_moisture: float = 0.0,
    ) -> "MechanicsModel":
        moisture = moisture or PA66_MOISTURE
        steel = steel or STEEL_BAND
        use_band = include_band and geometry.has_steel_band
        from .materials import pa66_at_moisture

        e_fixed, nu_fixed, _, _ = pa66_at_moisture(stress_free_moisture, corner)
        return cls(
            strap_thickness=geometry.strap_thickness,
            band_thickness=geometry.steel_band_thickness if use_band else 0.0,
            strap_width=geometry.strap_width,
            band_modulus=steel.youngs_modulus.at(corner),
            band_poisson=steel.poisson_ratio.at(corner),
            swelling_coefficient=moisture.swelling_coefficient.at(corner),
            stress_free_moisture=stress_free_moisture,
            moisture_dependent_modulus=moisture_dependent_modulus,
            fixed_modulus=e_fixed,
            fixed_poisson=nu_fixed,
            remote_tension=geometry.service_tension if include_tension else 0.0,
            corner=corner,
        )

    # ------------------------------------------------------------- variants
    def without_band(self) -> "MechanicsModel":
        """Plain PA66 strip -- used for the Tier 1 Kt cross-check."""
        return replace(self, band_thickness=0.0)

    def without_tension(self) -> "MechanicsModel":
        """Swelling only, no mechanical load."""
        return replace(self, remote_tension=0.0)

    def with_tension(self, force: float) -> "MechanicsModel":
        return replace(self, remote_tension=force)

    # -------------------------------------------------------------- section
    @property
    def polymer_thickness_in_band(self) -> float:
        return self.strap_thickness - self.band_thickness

    @property
    def stiffness_ratio(self) -> float:
        """Band membrane stiffness divided by polymer membrane stiffness.

        Well above 1 means the polymer is effectively fully constrained and
        Tier 1's rigid-constraint formula should be a good approximation.
        """
        polymer = self.fixed_modulus * self.polymer_thickness_in_band
        if polymer <= 0.0:
            return float("inf")
        return (self.band_modulus * self.band_thickness) / polymer

    @property
    def remote_line_load(self) -> float:
        """Tension per unit width applied at the cut face, N/mm."""
        return self.remote_tension / self.strap_width

    def summary(self) -> str:
        return "\n".join(
            [
                "MECHANICS MODEL",
                "---------------",
                f"  section            PA66 {self.polymer_thickness_in_band:g} mm"
                + (
                    f" + steel {self.band_thickness:g} mm"
                    if self.band_thickness > 0
                    else " (no band)"
                ),
                f"  band/polymer stiffness ratio  {self.stiffness_ratio:.1f}"
                + (" (polymer effectively fully constrained)" if self.stiffness_ratio > 10 else ""),
                f"  swelling coefficient {self.swelling_coefficient:g} per unit mass fraction",
                f"  stress-free moisture {self.stress_free_moisture:g}",
                "  modulus              "
                + (
                    "moisture dependent"
                    if self.moisture_dependent_modulus
                    else f"fixed at {self.fixed_modulus:g} MPa"
                ),
                f"  remote tension       {self.remote_tension:g} N "
                f"({self.remote_line_load:g} N/mm of width)",
                f"  bracket corner       {self.corner}",
            ]
        )


@dataclass
class MechanicsResult:
    """Solved displacement plus recovered PA66 stress."""

    displacement: np.ndarray
    #: PA66 stress components at quadrature points, each ``(n_elements, n_qp)``.
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    #: Nodal projections of the same, for plotting.
    nodal_max_principal: np.ndarray
    nodal_von_mises: np.ndarray
    #: Per-hole stress summaries, keyed by boundary name.
    hole_edges: dict
    model: MechanicsModel

    @property
    def max_principal(self) -> np.ndarray:
        s1, _ = principal_stresses(self.sxx, self.syy, self.sxy)
        return s1

    @property
    def min_principal(self) -> np.ndarray:
        _, s2 = principal_stresses(self.sxx, self.syy, self.sxy)
        return s2

    @property
    def von_mises_field(self) -> np.ndarray:
        return von_mises(self.sxx, self.syy, self.sxy)

    @property
    def peak_tensile(self) -> float:
        return float(np.max(self.max_principal))

    @property
    def peak_compressive(self) -> float:
        return float(np.min(self.min_principal))

    @property
    def peak_von_mises(self) -> float:
        return float(np.max(self.von_mises_field))


# ---------------------------------------------------------------------------
# Constitutive helpers
# ---------------------------------------------------------------------------


def plane_stress_moduli(youngs_modulus, poisson_ratio):
    """Plane-stress stiffness entries ``(q11, q12, q66)``.

    ``sigma_xx = q11 eps_xx + q12 eps_yy``, ``tau_xy = 2 q66 eps_xy``.
    """
    e = np.asarray(youngs_modulus, dtype=float)
    nu = np.asarray(poisson_ratio, dtype=float)
    if np.any(nu >= 0.5) or np.any(nu <= -1.0):
        raise ValueError("Poisson's ratio must lie in (-1, 0.5) for plane stress")
    denom = 1.0 - nu * nu
    return e / denom, nu * e / denom, e / (2.0 * (1.0 + nu))


def principal_stresses(sxx, syy, sxy):
    """In-plane principal stresses ``(sigma_1, sigma_2)``, sigma_1 >= sigma_2.

    The third principal stress is zero by the plane-stress assumption; callers
    comparing against a yield criterion should remember that.
    """
    mean = 0.5 * (sxx + syy)
    radius = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
    return mean + radius, mean - radius


def von_mises(sxx, syy, sxy):
    """Plane-stress von Mises equivalent stress."""
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def _polymer_properties_at(model: MechanicsModel, concentration):
    """PA66 modulus and Poisson ratio at each evaluation point."""
    if model.moisture_dependent_modulus:
        e, nu, _, _ = pa66_properties(concentration, corner=model.corner)
        return e, nu
    shape = np.shape(concentration)
    return (
        np.full(shape, model.fixed_modulus),
        np.full(shape, model.fixed_poisson),
    )


def _section_fields(model: MechanicsModel, concentration, in_band):
    """Membrane stiffness and swelling force at each evaluation point.

    Returns ``(a11, a12, a66, n_sw, q11_p, q12_p, q66_p, eps_sw)`` where the
    ``*_p`` entries are the PA66 layer's own plane-stress moduli, kept for ply
    stress recovery.
    """
    e_p, nu_p = _polymer_properties_at(model, concentration)
    q11_p, q12_p, q66_p = plane_stress_moduli(e_p, nu_p)

    in_band = np.broadcast_to(np.asarray(in_band, dtype=bool), np.shape(concentration))
    t_polymer = np.where(in_band, model.polymer_thickness_in_band, model.strap_thickness)
    t_band = np.where(in_band, model.band_thickness, 0.0)

    q11_s, q12_s, q66_s = plane_stress_moduli(model.band_modulus, model.band_poisson)

    a11 = t_polymer * q11_p + t_band * q11_s
    a12 = t_polymer * q12_p + t_band * q12_s
    a66 = t_polymer * q66_p + t_band * q66_s

    # Free swelling strain of the polymer layer. Steel contributes nothing.
    eps_sw = model.swelling_coefficient * (
        np.asarray(concentration, dtype=float) - model.stress_free_moisture
    )
    n_sw = t_polymer * (q11_p + q12_p) * eps_sw

    return a11, a12, a66, n_sw, q11_p, q12_p, q66_p, eps_sw


def recover_polymer_stress(exx, eyy, exy, q11, q12, q66, eps_sw):
    """PA66 ply stress from the membrane strain field.

    ``sigma = Q : (eps - eps_sw)``.  With ``eps = 0`` (rigid constraint) this
    gives ``sigma_xx = -(q11 + q12) eps_sw = -E beta dc / (1 - nu)``, i.e.
    exactly Tier 1's biaxial constrained-swelling stress, with the negative
    sign showing it is compressive during absorption.
    """
    mxx = exx - eps_sw
    myy = eyy - eps_sw
    return (
        q11 * mxx + q12 * myy,
        q12 * mxx + q11 * myy,
        2.0 * q66 * exy,
    )


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


def solve(
    strip: StripMesh,
    model: MechanicsModel,
    concentration: np.ndarray | None = None,
) -> MechanicsResult:
    """Plane-stress membrane solve at one instant.

    ``concentration`` is a nodal (P1) moisture field from :mod:`diffusion`.
    Pass ``None`` for the dry, stress-free-moisture state, which isolates the
    mechanical load.

    Boundary conditions
    -------------------
    The upstream cut face carries ``u_x = 0`` and the downstream one a uniform
    traction.  Both are model truncation planes rather than physical ends, so
    this represents a window in a longer strap loaded in tension; the
    ``end_margin`` in the geometry keeps them a St Venant distance from the
    first and last hole.  A single node on the upstream face is additionally
    held in ``y`` to remove the remaining rigid-body translation.
    """
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriP1,
        ElementTriP2,
        ElementVector,
        FacetBasis,
        LinearForm,
        condense,
        solve as skfem_solve,
    )
    from skfem.helpers import sym_grad

    mesh = strip.mesh
    # P2 displacement gives a linearly varying strain inside each element, which
    # is what makes a hole-edge stress converge on a mesh of this density.
    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element)

    scalar_basis = basis.with_element(ElementTriP1())
    if concentration is None:
        # The stress-free moisture state: isolates the mechanical load.
        c_nodal = np.full(mesh.p.shape[1], model.stress_free_moisture)
    else:
        c_nodal = np.asarray(concentration, dtype=float)
        if c_nodal.shape != (mesh.p.shape[1],):
            raise ValueError(
                f"concentration must be a nodal field of length {mesh.p.shape[1]}, "
                f"got shape {c_nodal.shape}"
            )
    c_qp = np.asarray(scalar_basis.interpolate(c_nodal))

    in_band_elements = strip.band_element_mask()
    in_band_qp = np.repeat(in_band_elements[:, None], c_qp.shape[1], axis=1)

    a11, a12, a66, n_sw, q11_p, q12_p, q66_p, eps_sw = _section_fields(
        model, c_qp, in_band_qp
    )

    @BilinearForm
    def membrane(u, v, w):
        eu = sym_grad(u)
        ev = sym_grad(v)
        return (
            w["a11"] * (eu[0, 0] * ev[0, 0] + eu[1, 1] * ev[1, 1])
            + w["a12"] * (eu[0, 0] * ev[1, 1] + eu[1, 1] * ev[0, 0])
            # engineering shear: gamma = 2 eps_xy, so gamma * dgamma = 4 eps dEps
            + 4.0 * w["a66"] * eu[0, 1] * ev[0, 1]
        )

    @LinearForm
    def swelling_force(v, w):
        ev = sym_grad(v)
        return w["n_sw"] * (ev[0, 0] + ev[1, 1])

    stiffness = membrane.assemble(basis, a11=a11, a12=a12, a66=a66)
    load = swelling_force.assemble(basis, n_sw=n_sw)

    if model.remote_tension != 0.0:
        facet_basis = FacetBasis(mesh, element, facets=mesh.boundaries[CUT_END])

        @LinearForm
        def traction(v, w):
            return w["tx"] * v[0]

        load = load + traction.assemble(
            facet_basis,
            tx=np.full(_quadrature_shape(facet_basis), model.remote_line_load),
        )

    fixed_x = basis.get_dofs(CUT_START).all("u^1")
    fixed_y = _single_y_constraint(basis, mesh)
    dirichlet = np.unique(np.concatenate([fixed_x, fixed_y]))

    displacement = skfem_solve(*condense(stiffness, load, D=dirichlet))

    grad = basis.interpolate(displacement).grad
    exx = grad[0, 0]
    eyy = grad[1, 1]
    exy = 0.5 * (grad[0, 1] + grad[1, 0])
    sxx, syy, sxy = recover_polymer_stress(
        exx, eyy, exy, q11_p, q12_p, q66_p, eps_sw
    )

    s1, _ = principal_stresses(sxx, syy, sxy)
    vm = von_mises(sxx, syy, sxy)
    nodal_s1 = _project_to_nodes(scalar_basis, s1)
    nodal_vm = _project_to_nodes(scalar_basis, vm)

    edges = hole_edge_stresses(strip, model, displacement, c_nodal)

    return MechanicsResult(
        displacement=displacement,
        sxx=sxx,
        syy=syy,
        sxy=sxy,
        nodal_max_principal=nodal_s1,
        nodal_von_mises=nodal_vm,
        hole_edges=edges,
        model=model,
    )


def _quadrature_shape(basis) -> tuple[int, int]:
    """``(n_elements_or_facets, n_quadrature_points)`` for a basis.

    Derived from the global coordinates rather than from internal attributes,
    so it works for both element and facet bases across skfem versions.
    """
    return np.asarray(basis.global_coordinates())[0].shape


def _single_y_constraint(basis, mesh) -> np.ndarray:
    """One transverse DOF held, to remove rigid-body translation in y.

    ``u_x = 0`` on the whole upstream face already removes x translation and
    rotation, so a single y constraint is the minimum that leaves the system
    non-singular without over-constraining it -- anything more would react the
    transverse swelling and corrupt the stress field.
    """
    dofs = basis.get_dofs(CUT_START)
    y_dofs = dofs.all("u^2")
    if len(y_dofs) == 0:
        raise RuntimeError("no transverse DOFs found on the upstream cut face")
    # Pick the one nearest the strap centreline so that the (symmetric)
    # transverse swelling is not biased to one side.
    nodal = dofs.nodal.get("u^2")
    if nodal is not None and len(nodal) > 0:
        coords = basis.doflocs[:, nodal]
        centre = mesh.p[1].mean()
        return np.array([nodal[np.argmin(np.abs(coords[1] - centre))]])
    return y_dofs[:1]


def _project_to_nodes(scalar_basis, field) -> np.ndarray:
    """L2-project a quadrature-point field onto P1 nodes, for plotting.

    Plotting only -- the peak values reported by :func:`hole_edge_stresses` are
    taken at the hole walls directly, without this smoothing step.
    """
    from skfem import LinearForm
    from skfem.models.poisson import mass
    from scipy.sparse.linalg import spsolve

    @LinearForm
    def rhs(v, w):
        return w["f"] * v

    m = mass.assemble(scalar_basis)
    b = rhs.assemble(scalar_basis, f=field)
    return spsolve(m.tocsc(), b)


def hole_edge_stresses(
    strip: StripMesh,
    model: MechanicsModel,
    displacement: np.ndarray,
    concentration: np.ndarray,
) -> dict:
    """PA66 stress evaluated on each hole wall.

    Sampling on the facets themselves, rather than reading a smoothed nodal
    field, keeps the peak sharp: nodal averaging across a hole edge pulls the
    value toward the interior and systematically under-reports Kt.

    Returns ``{hole_name: {...}}`` with the peak maximum-principal, peak von
    Mises and most-compressive stress on each hole, plus the stress evaluated
    around the wall for plotting.
    """
    from skfem import ElementTriP1, ElementTriP2, ElementVector, FacetBasis

    mesh = strip.mesh
    element = ElementVector(ElementTriP2())
    out: dict = {}

    for name in strip.hole_boundaries:
        facets = mesh.boundaries[name]
        fb = FacetBasis(mesh, element, facets=facets)
        sb = fb.with_element(ElementTriP1())

        c_qp = np.asarray(sb.interpolate(concentration))
        elements = mesh.f2t[0, facets]
        in_band = np.repeat(
            strip.band_element_mask()[elements][:, None], c_qp.shape[1], axis=1
        )
        _, _, _, _, q11, q12, q66, eps_sw = _section_fields(model, c_qp, in_band)

        grad = fb.interpolate(displacement).grad
        exx = grad[0, 0]
        eyy = grad[1, 1]
        exy = 0.5 * (grad[0, 1] + grad[1, 0])
        sxx, syy, sxy = recover_polymer_stress(exx, eyy, exy, q11, q12, q66, eps_sw)

        s1, s2 = principal_stresses(sxx, syy, sxy)
        vm = von_mises(sxx, syy, sxy)

        # Angular position around the hole, for plotting the wall distribution.
        centre = _hole_centre(strip, name)
        x = fb.global_coordinates()
        theta = np.arctan2(np.asarray(x)[1] - centre[1], np.asarray(x)[0] - centre[0])

        out[name] = {
            "max_principal": float(np.max(s1)),
            "min_principal": float(np.min(s2)),
            "von_mises": float(np.max(vm)),
            "mean_concentration": float(np.mean(c_qp)),
            "theta": theta.ravel(),
            "s1_around": s1.ravel(),
            "vm_around": vm.ravel(),
        }
    return out


def _hole_centre(strip: StripMesh, name: str) -> np.ndarray:
    index = strip.geometry.hole_labels.index(name)
    return strip.geometry.hole_centres[index]


def stress_concentration_factors(
    strip: StripMesh,
    model: MechanicsModel | None = None,
    force: float | None = None,
) -> dict:
    """FE stress concentration factors, for cross-checking Tier 1.

    Runs the *mechanical-only* case on a *plain PA66* section -- no steel band,
    no moisture -- because that is the configuration Tier 1's Howland/Peterson
    Kt describes.  Comparing anything else to that formula would not be
    like-for-like.

    Returns ``{hole_name: Kt_net}`` plus ``"net_section_stress"`` and
    ``"interior_mean"`` (the mean over holes that have neighbours on both
    sides, which is the value representative of a long strap).
    """
    geometry = strip.geometry
    if model is None:
        model = MechanicsModel.from_materials(geometry, moisture_dependent_modulus=False)
    force = geometry.service_tension if force is None else force

    check_model = model.without_band().with_tension(force)
    result = solve(strip, check_model, concentration=None)

    net_stress = force / geometry.net_section_area
    factors = {
        name: result.hole_edges[name]["max_principal"] / net_stress
        for name in strip.hole_boundaries
    }
    interior = [
        factors[name]
        for i, name in enumerate(strip.hole_boundaries)
        if geometry.is_interior_hole(i)
    ]
    interior_mean = float(np.mean(interior)) if interior else float("nan")
    all_mean = float(np.mean(list(factors.values())))
    return {
        "per_hole": factors,
        "net_section_stress": net_stress,
        "interior_mean": interior_mean,
        "all_mean": all_mean,
        # A window of one or two holes has no interior hole at all. Callers
        # want a number regardless, so fall back to the mean over every hole
        # and record that the fallback was taken.
        "representative_mean": interior_mean if interior else all_mean,
        "has_interior_holes": bool(interior),
    }
