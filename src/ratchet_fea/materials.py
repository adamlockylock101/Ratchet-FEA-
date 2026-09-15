"""Material property brackets for STP-RB-001.

UNITS: MPa for stiffness and strength, mm^2/s for diffusivity, dimensionless
mass fraction for moisture content, and linear strain per unit mass fraction
for the coefficient of moisture expansion.  See README.md.

Why brackets and not numbers
----------------------------
Unfilled PA66 is the textbook case of a polymer whose properties are dominated
by moisture.  Between dry-as-moulded (DAM) and 50% RH conditioned, the modulus
falls by roughly a factor of two and the yield strength by nearly as much.  At
water saturation it falls further still.  Quoting a single modulus for "PA66"
is meaningless, so every property here is a :class:`Bracket` and every
conclusion is checked at both ends of it.

This matters doubly for the moisture-swelling mechanism: the same water that
drives the swelling eigenstrain also softens the polymer that resists it, and
the two effects partly cancel.  :func:`pa66_at_moisture` interpolates between
the conditioned states so the coupling is represented rather than ignored.

=============================================================================
!! NO PROPERTY HERE IS FROM A DATASHEET FOR THE ACTUAL GRADE IN THE PART !!
=============================================================================
These are published ranges for generic unfilled PA66.  If the strap is glass
filled, impact modified, or plasticised, several of these move by more than
their own bracket width.  Identify the grade (moulder's part record, FTIR, or
a DSC melt point plus ash test) and replace the brackets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .provenance import Bracket, Documented, Provenance, Source

__all__ = [
    "MoistureCondition",
    "ElasticProperties",
    "StrengthProperties",
    "PolymerGrade",
    "MoistureTransport",
    "PA66_DAM",
    "PA66_RH50",
    "PA66_SATURATED",
    "PA66_MOISTURE",
    "STEEL_BAND",
    "pa66_at_moisture",
    "pa66_properties",
    "PA66_CONDITIONS",
    "STEEL_SWELLING_COEFFICIENT",
    "all_material_objects",
]


#: Tolerance below zero that counts as interpolation round-off rather than an
#: error. Nine orders of magnitude below a realistic moisture content, so it
#: cannot mask a genuine sign problem.
_MOISTURE_ROUNDOFF = 1e-9


class MoistureCondition(str, Enum):
    """Conditioning states at which PA66 properties are commonly quoted."""

    DAM = "dry-as-moulded"
    RH50 = "conditioned-50%RH"
    SATURATED = "water-saturated"


@dataclass(frozen=True)
class ElasticProperties(Documented):
    """Isotropic linear-elastic constants."""

    youngs_modulus: Bracket
    poisson_ratio: Bracket


@dataclass(frozen=True)
class StrengthProperties(Documented):
    """Strength limits used to form margins.  Screening only.

    VERIFY: these are short-term, room-temperature, monotonic values.  The
    strap is failing under repeated use, so the governing limit is more likely
    a fatigue or static-fatigue (creep rupture) threshold well below yield.
    Tier 3 should replace these with cyclic data at the right moisture state.
    """

    yield_strength: Bracket
    tensile_strength: Bracket


@dataclass(frozen=True)
class PolymerGrade(Documented):
    """A polymer in one specific conditioning state."""

    name: str
    condition: MoistureCondition
    elastic: ElasticProperties
    strength: StrengthProperties
    #: Equilibrium moisture content at this condition, mass fraction.
    moisture_content: float = 0.0

    @property
    def doc_label(self) -> str:
        return f"{self.name} @ {self.condition.value}"

    @property
    def youngs_modulus(self) -> Bracket:
        return self.elastic.youngs_modulus

    @property
    def poisson_ratio(self) -> Bracket:
        return self.elastic.poisson_ratio

    @property
    def yield_strength(self) -> Bracket:
        return self.strength.yield_strength

    @property
    def tensile_strength(self) -> Bracket:
        return self.strength.tensile_strength



@dataclass(frozen=True)
class MoistureTransport(Documented):
    """Fickian transport and swelling properties for the polymer."""

    #: Fickian diffusivity of water in the polymer at service temperature.
    diffusivity: Bracket
    #: Equilibrium uptake in air at 50% RH, mass fraction.
    saturation_50rh: Bracket
    #: Equilibrium uptake fully immersed / at 100% RH, mass fraction.
    saturation_immersed: Bracket
    #: Coefficient of moisture expansion: linear strain per unit mass fraction.
    swelling_coefficient: Bracket


# ---------------------------------------------------------------------------
# PA66, unfilled -- literature brackets
# ---------------------------------------------------------------------------
# VERIFY: replace with the datasheet for the grade actually used in the strap.

_LIT = Provenance.LITERATURE
_GRADE_CHECK = "Identify the grade (moulder records / FTIR + DSC + ash) and use its datasheet."

PA66_DAM = PolymerGrade(
    name="PA66 (unfilled)",
    condition=MoistureCondition.DAM,
    moisture_content=0.0,
    elastic=ElasticProperties(
        youngs_modulus=Bracket(
            2800.0,
            3300.0,
            Source(_LIT, "MPa", "Tensile modulus, dry as moulded, 23 C.", _GRADE_CHECK),
        ),
        poisson_ratio=Bracket(
            0.35,
            0.41,
            Source(
                _LIT,
                "-",
                "Rarely quoted for the specific grade; the bracket covers the "
                "usual spread for unfilled polyamides.",
                "Rosette strain gauge or DIC on a tensile coupon.",
            ),
        ),
    ),
    strength=StrengthProperties(
        yield_strength=Bracket(
            75.0,
            90.0,
            Source(_LIT, "MPa", "Tensile yield, dry as moulded, 23 C.", _GRADE_CHECK),
        ),
        tensile_strength=Bracket(
            75.0,
            95.0,
            Source(_LIT, "MPa", "Tensile strength at break, dry as moulded.", _GRADE_CHECK),
        ),
    ),
)

PA66_RH50 = PolymerGrade(
    name="PA66 (unfilled)",
    condition=MoistureCondition.RH50,
    moisture_content=0.025,
    elastic=ElasticProperties(
        youngs_modulus=Bracket(
            1000.0,
            1700.0,
            Source(
                _LIT,
                "MPa",
                "Conditioned to equilibrium at 50% RH. Roughly half the DAM "
                "value; water plasticises the amorphous phase.",
                _GRADE_CHECK,
            ),
        ),
        poisson_ratio=Bracket(
            0.38,
            0.44,
            Source(_LIT, "-", "Rises with moisture as the polymer softens.", "DIC on a conditioned coupon."),
        ),
    ),
    strength=StrengthProperties(
        yield_strength=Bracket(
            40.0,
            55.0,
            Source(_LIT, "MPa", "Tensile yield, 50% RH conditioned, 23 C.", _GRADE_CHECK),
        ),
        tensile_strength=Bracket(
            50.0,
            70.0,
            Source(_LIT, "MPa", "Tensile strength, 50% RH conditioned.", _GRADE_CHECK),
        ),
    ),
)

PA66_SATURATED = PolymerGrade(
    name="PA66 (unfilled)",
    condition=MoistureCondition.SATURATED,
    moisture_content=0.085,
    elastic=ElasticProperties(
        youngs_modulus=Bracket(
            600.0,
            1100.0,
            Source(
                _LIT,
                "MPa",
                "Water saturated. Well above Tg for the wet amorphous phase, "
                "so the modulus is low and rate dependent.",
                _GRADE_CHECK,
            ),
        ),
        poisson_ratio=Bracket(
            0.40,
            0.45,
            Source(_LIT, "-", "Approaching incompressible as the polymer softens.", "DIC on a saturated coupon."),
        ),
    ),
    strength=StrengthProperties(
        yield_strength=Bracket(
            30.0,
            45.0,
            Source(_LIT, "MPa", "Tensile yield, water saturated, 23 C.", _GRADE_CHECK),
        ),
        tensile_strength=Bracket(
            40.0,
            60.0,
            Source(_LIT, "MPa", "Tensile strength, water saturated.", _GRADE_CHECK),
        ),
    ),
)

#: Ordered by moisture content -- used by :func:`pa66_at_moisture`.
PA66_CONDITIONS = (PA66_DAM, PA66_RH50, PA66_SATURATED)


PA66_MOISTURE = MoistureTransport(
    diffusivity=Bracket(
        1.0e-7,
        1.0e-6,
        Source(
            _LIT,
            "mm^2/s",
            "Water in unfilled PA66 near 23 C (1e-13 .. 1e-12 m^2/s). Spans a "
            "full decade and is strongly temperature dependent, so the *timing* "
            "of any result is only good to an order of magnitude. The bracket "
            "is log-scaled; its nominal is the geometric mean.",
            "Gravimetric sorption on a coupon of known thickness; fit sqrt(t).",
        ),
        scale="log",
    ),
    saturation_50rh=Bracket(
        0.020,
        0.030,
        Source(
            _LIT,
            "-",
            "Equilibrium uptake in 50% RH air, mass fraction of dry polymer.",
            "Weigh a coupon dry, then after conditioning to constant mass.",
        ),
    ),
    saturation_immersed=Bracket(
        0.075,
        0.090,
        Source(
            _LIT,
            "-",
            "Equilibrium uptake fully immersed. Relevant if the strap is worn "
            "against skin, washed, or used outdoors.",
            "Immerse a coupon to constant mass at service temperature.",
        ),
    ),
    swelling_coefficient=Bracket(
        0.20,
        0.30,
        Source(
            _LIT,
            "1/(mass fraction)",
            "Linear strain per unit moisture mass fraction. 0.25 means a 2.5% "
            "mass uptake gives 0.63% linear strain. This is THE parameter the "
            "swelling mechanism turns on -- it multiplies straight through to "
            "the constrained-swelling stress.",
            "Measure a coupon's length dry and conditioned; divide strain by "
            "mass uptake.",
        ),
    ),
)


# ---------------------------------------------------------------------------
# Steel reinforcing band
# ---------------------------------------------------------------------------
STEEL_BAND = PolymerGrade(  # reusing the container; it is just an elastic body
    name="steel band",
    condition=MoistureCondition.DAM,
    moisture_content=0.0,
    elastic=ElasticProperties(
        youngs_modulus=Bracket(
            200000.0,
            210000.0,
            Source(
                Provenance.LITERATURE,
                "MPa",
                "Any carbon or stainless strip. The exact value barely matters: "
                "the band is ~50x stiffer per unit width than the polymer, so "
                "the polymer is close to fully constrained either way.",
                "Confirm the band is steel and not a stiff polymer or glass tape.",
            ),
        ),
        poisson_ratio=Bracket(
            0.29,
            0.31,
            Source(Provenance.LITERATURE, "-", "Standard for steel.", "None needed."),
        ),
    ),
    strength=StrengthProperties(
        yield_strength=Bracket(
            250.0,
            600.0,
            Source(
                Provenance.LITERATURE,
                "MPa",
                "Not used to form margins; the band is nowhere near yield.",
                "None needed unless the band itself is found cracked.",
            ),
        ),
        tensile_strength=Bracket(
            350.0,
            800.0,
            Source(Provenance.LITERATURE, "MPa", "Not used.", "None needed."),
        ),
    ),
)

#: The steel band does not take up water and does not swell.
STEEL_SWELLING_COEFFICIENT = 0.0


def pa66_properties(moisture, corner: str = "nominal"):
    """Vectorised PA66 properties at an array of moisture contents.

    Returns ``(E, nu, sigma_y, sigma_u)`` as arrays shaped like ``moisture``,
    in MPa (dimensionless for Poisson's ratio).  The FE model calls this with
    the concentration field evaluated at every quadrature point, so it has to
    be vectorised; :func:`pa66_at_moisture` is the scalar wrapper.

    Values are clamped at both ends rather than extrapolated.  Below the
    dry-as-moulded state there is no less moisture to have; above saturation
    the polymer is not absorbing more water, it is doing something else
    (voiding, hydrolysis) that a linear-elastic model cannot represent anyway.
    """
    c = np.asarray(moisture, dtype=float)
    # A field that dries all the way to zero reaches exactly 0.0 at its nodes,
    # and interpolating that onto quadrature points can land a few ulp below
    # zero -- a P1 basis function evaluated on a facet is not guaranteed to be
    # non-negative to the last bit. Clamp that away, but still reject anything
    # negative enough to be a real error rather than round-off.
    if np.any(c < -_MOISTURE_ROUNDOFF):
        raise ValueError(
            f"moisture content cannot be negative (minimum {np.min(c):g})"
        )
    c = np.clip(c, 0.0, None)

    knots = sorted(PA66_CONDITIONS, key=lambda g: g.moisture_content)
    xs = np.array([g.moisture_content for g in knots])
    # np.interp clamps to the end values outside the knot range, which is the
    # behaviour we want.
    return tuple(
        np.interp(c, xs, np.array([getter(g).at(corner) for g in knots]))
        for getter in (
            lambda g: g.youngs_modulus,
            lambda g: g.poisson_ratio,
            lambda g: g.yield_strength,
            lambda g: g.tensile_strength,
        )
    )


def pa66_at_moisture(
    moisture: float, corner: str = "nominal"
) -> tuple[float, float, float, float]:
    """Interpolate PA66 properties to an arbitrary moisture content.

    Returns ``(youngs_modulus, poisson_ratio, yield_strength, tensile_strength)``
    in MPa (and dimensionless for Poisson's ratio) at moisture mass fraction
    ``moisture``, taking each property at the given bracket ``corner``.

    The mechanism under investigation is a competition: more water means more
    swelling eigenstrain, but also a softer polymer to resist it.  Holding the
    modulus at its dry value would overstate the swelling stress by 2-3x, so
    the FE model evaluates the modulus pointwise from the local concentration.

    VERIFY: the interpolation is piecewise linear in mass fraction between the
    three conditioned states above.  The real modulus-vs-moisture curve is
    sigmoidal, with the steepest drop as the wet Tg passes room temperature
    somewhere near 2-3% uptake.  Piecewise linear through the 50% RH point
    captures the drop crudely but will misplace it if the service temperature
    is not close to 23 C.
    """
    return tuple(float(v) for v in pa66_properties(moisture, corner))  # type: ignore[return-value]


def all_material_objects() -> list:
    """Every material object, for the verification report."""
    return [PA66_DAM, PA66_RH50, PA66_SATURATED, PA66_MOISTURE, STEEL_BAND]
