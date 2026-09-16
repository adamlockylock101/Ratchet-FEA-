"""Static plane-stress linear elastic solve for the perforated strap.

One load case: the ratchet tension.  No moisture, no time dependence, no
eigenstrain.  The job of this module is to produce a displacement and stress
field accurate enough near a crack tip for :mod:`ratchet_fea.fracture` to pull
a stress intensity factor out of it.

The steel band is deliberately NOT in the model
-----------------------------------------------
:class:`~ratchet_fea.geometry.StrapGeometry` carries the band's dimensions, but
the FE model is a homogeneous PA66 plate.  Two reasons, both deliberate:

1.  The band's existence is *inferred* from the failure photographs, not
    observed.  Building a load path around an unverified feature would make
    every result conditional on it.
2.  Ignoring it is conservative.  A bonded band would carry most of the
    tension and shield the polymer, lowering K.  A model that says "the crack
    does not run" without the band says it more strongly with one.

:func:`band_load_sharing` reports how much the band would take if it is real,
so the size of that conservatism is visible rather than hidden.

UNITS: mm / N / MPa.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .geometry import StrapGeometry
from .materials import (
    SERVICE_CONDITION,
    STEEL_BAND,
    MoistureCondition,
    PolymerGrade,
    grade_for,
)
from .mesh import CUT_END, CUT_START, StripMesh

__all__ = [
    "MechanicsModel",
    "MechanicsResult",
    "solve",
    "plane_stress_moduli",
    "principal_stresses",
    "von_mises",
    "band_load_sharing",
]


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
    """In-plane principal stresses ``(sigma_1, sigma_2)``, sigma_1 >= sigma_2."""
    mean = 0.5 * (sxx + syy)
    radius = np.sqrt((0.5 * (sxx - syy)) ** 2 + sxy**2)
    return mean + radius, mean - radius


def von_mises(sxx, syy, sxy):
    """Plane-stress von Mises equivalent stress."""
    return np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)


def band_load_sharing(
    geometry: StrapGeometry,
    grade: PolymerGrade | None = None,
    corner: str = "nominal",
) -> float:
    """Fraction of the ratchet tension a bonded steel band would carry.

    Iso-strain through the thickness, so the split is by ``E * t``.  Returns 0
    when the geometry has no band.

    The FE model ignores the band, so a value near 1 means the FE stresses --
    and therefore K -- are overstated by roughly ``1/(1 - this)``.  That is
    conservatism, not error, but it should be quoted alongside any margin.
    """
    if not geometry.has_steel_band:
        return 0.0
    grade = grade or grade_for(SERVICE_CONDITION)
    polymer = grade.youngs_modulus.at(corner) * geometry.polymer_thickness_in_band
    steel = STEEL_BAND.youngs_modulus.at(corner) * geometry.steel_band_thickness
    return steel / (steel + polymer)


@dataclass(frozen=True)
class MechanicsModel:
    """Material and load definition for the static solve."""

    youngs_modulus: float
    poisson_ratio: float
    #: Remote tensile stress on the gross section, MPa.
    remote_stress: float
    condition: MoistureCondition = SERVICE_CONDITION
    corner: str = "nominal"

    def __post_init__(self) -> None:
        if self.youngs_modulus <= 0.0:
            raise ValueError("Young's modulus must be positive")
        if not -1.0 < self.poisson_ratio < 0.5:
            raise ValueError("Poisson's ratio must lie in (-1, 0.5) for plane stress")

    @classmethod
    def from_materials(
        cls,
        geometry: StrapGeometry,
        force: float | None = None,
        condition: MoistureCondition = SERVICE_CONDITION,
        corner: str = "nominal",
    ) -> "MechanicsModel":
        grade = grade_for(condition)
        force = geometry.service_tension if force is None else force
        return cls(
            youngs_modulus=grade.youngs_modulus.at(corner),
            poisson_ratio=grade.poisson_ratio.at(corner),
            remote_stress=force / geometry.gross_section_area,
            condition=condition,
            corner=corner,
        )

    def with_remote_stress(self, stress: float) -> "MechanicsModel":
        return replace(self, remote_stress=stress)

    @property
    def moduli(self) -> tuple:
        return plane_stress_moduli(self.youngs_modulus, self.poisson_ratio)

    def summary(self) -> str:
        return "\n".join(
            [
                "MECHANICS MODEL",
                "---------------",
                f"  material          PA66, {self.condition.value} "
                f"[{self.corner} bracket corner]",
                f"  Young's modulus   {self.youngs_modulus:g} MPa",
                f"  Poisson's ratio   {self.poisson_ratio:g}",
                f"  remote stress     {self.remote_stress:.3f} MPa "
                "(gross section, steel band ignored)",
            ]
        )


@dataclass
class MechanicsResult:
    """Solved displacement and the stress recovered from it."""

    displacement: np.ndarray
    basis: object
    sxx: np.ndarray
    syy: np.ndarray
    sxy: np.ndarray
    model: MechanicsModel
    strip: StripMesh

    @property
    def von_mises_field(self) -> np.ndarray:
        return von_mises(self.sxx, self.syy, self.sxy)

    @property
    def max_principal(self) -> np.ndarray:
        return principal_stresses(self.sxx, self.syy, self.sxy)[0]

    @property
    def peak_tensile(self) -> float:
        return float(np.max(self.max_principal))

    def nodal(self, field) -> np.ndarray:
        """L2-project a quadrature-point field onto P1 nodes, for plotting."""
        return _project_to_nodes(self.basis.with_element(_p1()), field)


def _p1():
    from skfem import ElementTriP1

    return ElementTriP1()


def solve(strip: StripMesh, model: MechanicsModel) -> MechanicsResult:
    """Plane-stress solve under remote tension.

    Boundary conditions
    -------------------
    ``u_x = 0`` on the upstream cut face and a uniform traction on the
    downstream one.  Both are model truncation planes rather than physical
    ends, so this represents a window in a longer strap loaded in tension; the
    geometry's ``end_margin`` (one strap width by default) keeps them a
    St Venant distance from the perforations and the crack.  A single node on
    the upstream face is additionally held in ``y`` to remove the remaining
    rigid-body translation -- the minimum that leaves the system non-singular
    without reacting the transverse contraction.

    The crack faces need no treatment at all: after the node split in
    :mod:`ratchet_fea.mesh` they are ordinary free surfaces, which is exactly
    what a traction-free crack is.
    """
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriP2,
        ElementVector,
        FacetBasis,
        LinearForm,
        condense,
        solve as skfem_solve,
    )
    from skfem.helpers import sym_grad

    mesh = strip.mesh
    # P2 displacement gives linearly varying strain inside each element, which
    # is what lets a crack-tip field converge on a mesh of this density.
    element = ElementVector(ElementTriP2())
    basis = Basis(mesh, element)
    q11, q12, q66 = model.moduli

    @BilinearForm
    def elasticity(u, v, w):
        eu = sym_grad(u)
        ev = sym_grad(v)
        return (
            q11 * (eu[0, 0] * ev[0, 0] + eu[1, 1] * ev[1, 1])
            + q12 * (eu[0, 0] * ev[1, 1] + eu[1, 1] * ev[0, 0])
            # engineering shear: gamma = 2 eps_xy, so gamma * dgamma = 4 eps dEps
            + 4.0 * q66 * eu[0, 1] * ev[0, 1]
        )

    @LinearForm
    def traction(v, w):
        return w["tx"] * v[0]

    stiffness = elasticity.assemble(basis)
    facet_basis = FacetBasis(mesh, element, facets=mesh.boundaries[CUT_END])
    load = traction.assemble(
        facet_basis,
        tx=np.full(_quadrature_shape(facet_basis), model.remote_stress),
    )

    dirichlet = np.unique(
        np.concatenate(
            [basis.get_dofs(CUT_START).all("u^1"), _single_y_constraint(basis, mesh)]
        )
    )
    displacement = skfem_solve(*condense(stiffness, load, D=dirichlet))

    grad = basis.interpolate(displacement).grad
    exx, eyy = grad[0, 0], grad[1, 1]
    exy = 0.5 * (grad[0, 1] + grad[1, 0])

    return MechanicsResult(
        displacement=displacement,
        basis=basis,
        sxx=q11 * exx + q12 * eyy,
        syy=q12 * exx + q11 * eyy,
        sxy=2.0 * q66 * exy,
        model=model,
        strip=strip,
    )


def _quadrature_shape(basis) -> tuple:
    """``(n_elements_or_facets, n_quadrature_points)`` for a basis."""
    return np.asarray(basis.global_coordinates())[0].shape


def _single_y_constraint(basis, mesh) -> np.ndarray:
    """One transverse DOF held, to remove rigid-body translation in y."""
    dofs = basis.get_dofs(CUT_START)
    nodal = dofs.nodal.get("u^2")
    if nodal is None or len(nodal) == 0:
        y_dofs = dofs.all("u^2")
        if len(y_dofs) == 0:
            raise RuntimeError("no transverse DOFs on the upstream cut face")
        return y_dofs[:1]
    coords = basis.doflocs[:, nodal]
    centre = mesh.p[1].mean()
    return np.array([nodal[np.argmin(np.abs(coords[1] - centre))]])


def _project_to_nodes(scalar_basis, field) -> np.ndarray:
    """L2-project a quadrature-point field onto P1 nodes. Plotting only."""
    from scipy.sparse.linalg import spsolve
    from skfem import LinearForm
    from skfem.models.poisson import mass

    @LinearForm
    def rhs(v, w):
        return w["f"] * v

    return spsolve(mass.assemble(scalar_basis).tocsc(), rhs.assemble(scalar_basis, f=field))
