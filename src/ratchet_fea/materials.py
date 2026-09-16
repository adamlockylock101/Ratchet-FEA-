"""Material property brackets for STP-RB-001.

UNITS: MPa for stiffness and strength.  Fracture toughness is stored in
MPa*sqrt(m), because that is the unit every datasheet and paper quotes it in,
and a bracket nobody can read against a datasheet is a bracket nobody will
maintain.  :func:`fracture_toughness_mpa_root_mm` converts it into the
repository's mm/N/MPa system at the single point of use.  See README.md.

Why brackets and not numbers
----------------------------
Unfilled PA66 is the textbook case of a polymer whose properties are dominated
by conditioning.  Between dry-as-moulded (DAM) and 50% RH conditioned, the
modulus falls by roughly a factor of two and the yield strength by nearly as
much -- while fracture toughness moves the OTHER way, because the absorbed
water plasticises the amorphous phase and makes the polymer tougher.  Quoting a
single number for "PA66" is meaningless, so every property here is a
:class:`Bracket` and every conclusion is checked at both ends of it.

The three conditioning states are NOT a moisture model.  This repository does
not model moisture.  They are simply the three states the literature reports
properties at; the service state is 50% RH conditioned.

=============================================================================
!! NO PROPERTY HERE IS FROM A DATASHEET FOR THE ACTUAL GRADE IN THE PART !!
=============================================================================
These are published ranges for generic unfilled PA66.  If the strap is glass
filled, impact modified, or plasticised, several of these move by more than
their own bracket width.  Identify the grade (moulder's part record, FTIR, or
a DSC melt point plus ash test) and replace the brackets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .provenance import Bracket, Documented, Provenance, Source

__all__ = [
    "MoistureCondition",
    "ElasticProperties",
    "StrengthProperties",
    "PolymerGrade",
    "PA66_DAM",
    "PA66_RH50",
    "PA66_SATURATED",
    "PA66_CONDITIONS",
    "SERVICE_CONDITION",
    "STEEL_BAND",
    "MPA_ROOT_M_TO_MPA_ROOT_MM",
    "fracture_toughness_mpa_root_mm",
    "grade_for",
    "all_material_objects",
]

#: Fracture toughness unit conversion. K has dimensions stress * sqrt(length),
#: so changing the length unit from m to mm multiplies K by sqrt(1000) = 31.62.
#: Getting this wrong scales every fracture margin in the study by 31.6, which
#: is the easiest available route to a confidently wrong conclusion.
MPA_ROOT_M_TO_MPA_ROOT_MM = math.sqrt(1000.0)


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
    """Strength and toughness limits used to form margins.  Screening only.

    VERIFY: these are short-term, room-temperature, monotonic values.  The
    strap is failing under repeated use, so the governing limit may well be a
    fatigue threshold or a static-fatigue (creep rupture) one, both below the
    monotonic values here.  A crack that will not run under a single pull can
    still grow a little on every pull.
    """

    yield_strength: Bracket
    tensile_strength: Bracket
    #: Plane-strain fracture toughness, MPa*sqrt(m) -- the unit datasheets use.
    #: Convert with :func:`fracture_toughness_mpa_root_mm` before comparing
    #: against anything this repository computes.
    fracture_toughness: Bracket


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

    @property
    def fracture_toughness(self) -> Bracket:
        """Plane-strain fracture toughness bracket, in MPa*sqrt(m)."""
        return self.strength.fracture_toughness



# ---------------------------------------------------------------------------
# PA66, unfilled -- literature brackets
# ---------------------------------------------------------------------------
# VERIFY: replace with the datasheet for the grade actually used in the strap.

_LIT = Provenance.LITERATURE
_GRADE_CHECK = "Identify the grade (moulder records / FTIR + DSC + ash) and use its datasheet."
_KIC_CHECK = (
    "Datasheets rarely quote K_IC for polyamides. Measure it: ASTM D5045 "
    "SENB or compact tension on razor-notched specimens, conditioned to the "
    "service state, at the service temperature. Check the validity criteria "
    "in the standard -- unfilled PA66 often fails them and needs J or the "
    "essential work of fracture instead."
)

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
        fracture_toughness=Bracket(
            2.5,
            4.0,
            Source(
                _LIT,
                "MPa*sqrt(m)",
                "Dry as moulded. The brittlest state: with no absorbed water "
                "to plasticise the amorphous phase, PA66 is at its least "
                "tough. This is the conservative corner for a fracture check.",
                _KIC_CHECK,
            ),
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
        fracture_toughness=Bracket(
            3.0,
            5.5,
            Source(
                _LIT,
                "MPa*sqrt(m)",
                "50% RH conditioned -- the service state. Toughness rises with "
                "conditioning even as strength falls, because the absorbed "
                "water plasticises the amorphous phase.",
                _KIC_CHECK,
            ),
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
        fracture_toughness=Bracket(
            3.5,
            6.5,
            Source(
                _LIT,
                "MPa*sqrt(m)",
                "Water saturated. Toughest and most ductile state, and the one "
                "where a single K_IC number is least meaningful: the crack tip "
                "yields over a region comparable to the ligament, so LEFM is "
                "being used outside its validity. See "
                "analytical.lefm_validity().",
                _KIC_CHECK,
            ),
        ),
    ),
)

#: The three conditioning states, ordered driest first.
PA66_CONDITIONS = (PA66_DAM, PA66_RH50, PA66_SATURATED)

#: The state a belt strap actually lives in. Used as the default everywhere.
#:
#: VERIFY: assumes ordinary indoor service. A strap used outdoors, washed, or
#: worn against skin sits closer to saturated; one kept in a hot dry cab sits
#: closer to dry-as-moulded, which is also the BRITTLEST state.
SERVICE_CONDITION = MoistureCondition.RH50


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
        fracture_toughness=Bracket(
            50.0,
            200.0,
            Source(
                Provenance.LITERATURE,
                "MPa*sqrt(m)",
                "Not used. The band is nowhere near fracture, and this model "
                "does not represent cracks in it.",
                "None needed.",
            ),
        ),
    ),
)

def grade_for(condition: MoistureCondition = SERVICE_CONDITION) -> PolymerGrade:
    """The PA66 property set for a conditioning state."""
    for grade in PA66_CONDITIONS:
        if grade.condition is condition:
            return grade
    raise ValueError(f"no PA66 properties for condition {condition!r}")


def fracture_toughness_mpa_root_mm(
    grade: PolymerGrade, corner: str = "nominal"
) -> float:
    """Fracture toughness in MPa*sqrt(mm), ready to compare against a computed K.

    The bracket is stored in MPa*sqrt(m) so it can be read straight off a
    datasheet. Everything this repository computes is in MPa*sqrt(mm). This is
    the one place the two meet, so it is the one place the factor of 31.62 can
    be got wrong -- which is why it is a named function with a test on it
    rather than a multiplication scattered through the code.
    """
    return grade.fracture_toughness.at(corner) * MPA_ROOT_M_TO_MPA_ROOT_MM


def all_material_objects() -> list:
    """Every material object, for the verification report."""
    return [PA66_DAM, PA66_RH50, PA66_SATURATED, STEEL_BAND]
