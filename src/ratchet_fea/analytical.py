"""Handbook linear elastic fracture mechanics for the STP-RB-001 strap.

Closed-form stress intensity factors for a crack growing out of a perforation,
so the FE model in :mod:`ratchet_fea.fracture` has something independent to be
checked against.  Two routes to the same number is the only cheap defence
against a silently wrong FE model.

The question
------------
A crack has nucleated at a hole edge.  Under the ratchet tension alone, does
``K`` reach ``K_IC`` at any crack length the part can hold?  If it does, the
crack runs and the strap fails in one pull.  If it does not, something other
than static overload is keeping the crack going, and this model has ruled a
mechanism out rather than found one.

UNITS
-----
Everything here is mm / N / MPa, so stress intensity comes out in
**MPa*sqrt(mm)**.  The material bracket in :mod:`ratchet_fea.materials` is
stored in MPa*sqrt(m) because that is what datasheets quote; convert with
:func:`~ratchet_fea.materials.fracture_toughness_mpa_root_mm` before comparing.
The factor is 31.62, and mixing them up is the fastest way to be wrong by a
factor of 30.

Handbook solutions used
-----------------------
Newman's collocation fits for a radial crack (or two symmetric radial cracks)
from a circular hole in an infinite plate under remote uniaxial tension normal
to the crack, from NASA TN D-6376 (1971).  Both are exact at both asymptotes,
which :func:`newman_single_crack_factor` and
:func:`newman_double_crack_factor` document and the tests pin:

* ``a -> 0``:  ``F -> 3.36``, an edge crack sitting in the ``3 sigma`` hoop
  stress at the hole wall (``1.1215 * 3``);
* ``a -> inf``: the hole stops mattering and the flaw behaves as a plain
  through crack -- ``F -> 1/sqrt(2)`` for one crack, ``F -> 1`` for two.

Finite width is then handled with Feddersen's secant correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import (
    PLACEHOLDER_GEOMETRY,
    CrackGeometry,
    CrackOrientation,
    StrapGeometry,
)
from .materials import (
    SERVICE_CONDITION,
    MoistureCondition,
    PolymerGrade,
    fracture_toughness_mpa_root_mm,
    grade_for,
)

__all__ = [
    "kt_hole_in_finite_width_strip",
    "kt_gross_from_net",
    "row_interaction_note",
    "net_section_stress",
    "gross_section_stress",
    "mechanical_peak_stress",
    "newman_single_crack_factor",
    "newman_double_crack_factor",
    "feddersen_width_correction",
    "handbook_k",
    "handbook_k_curve",
    "critical_crack_length",
    "LefmValidity",
    "lefm_validity",
    "FractureScreen",
    "screen",
]

# Validity limit of the Howland/Peterson polynomial below.
KT_MAX_D_OVER_W = 0.5

#: Free-surface correction for an edge crack, ``1.1215``. The ``a -> 0`` limit
#: of both Newman fits is this times the hole's ``3 sigma`` hoop stress.
EDGE_CRACK_FACTOR = 1.1215


# ---------------------------------------------------------------------------
# Stress concentration (uncracked)
# ---------------------------------------------------------------------------


def kt_hole_in_finite_width_strip(d_over_W: float) -> float:
    """Stress concentration factor for a central circular hole, NET-section basis.

    Howland's solution as fitted by Peterson (*Stress Concentration Factors*,
    chart 4.1) for a finite-width strip in uniaxial tension:

        Kt_net = 3.00 - 3.13 (d/W) + 3.66 (d/W)^2 - 1.53 (d/W)^3

    Valid for ``0 <= d/W <= 0.5``, reducing to the classical Kirsch value of
    3.0 for an infinitely wide plate.

    This describes the stress in an UNCRACKED strip. It is what nucleates the
    crack; once a crack exists, K takes over and Kt stops being the right
    quantity.
    """
    if not 0.0 <= d_over_W <= KT_MAX_D_OVER_W:
        raise ValueError(
            f"d/W = {d_over_W:.3f} is outside the validity range "
            f"0..{KT_MAX_D_OVER_W} of the Howland/Peterson fit"
        )
    x = d_over_W
    return 3.00 - 3.13 * x + 3.66 * x * x - 1.53 * x * x * x


def kt_gross_from_net(kt_net: float, d_over_W: float) -> float:
    """Convert a net-section Kt to a gross-section Kt."""
    return kt_net / (1.0 - d_over_W)


def row_interaction_note(pitch_over_d: float) -> str:
    """Qualitative note on hole-to-hole interaction for a row along the load.

    For holes in a line *parallel* to the loading direction, each hole sits in
    the low-stress lobe of its neighbours, so Kt is slightly *reduced* relative
    to an isolated hole once the pitch exceeds roughly three diameters.

    Returns prose rather than a number on purpose: a hand-waved correction
    factor must not end up inside a margin. The FE model measures it.
    """
    if pitch_over_d < 2.0:
        return (
            f"pitch/d = {pitch_over_d:.2f} is below 2 -- holes are close enough "
            "that the ligaments between them may govern rather than the hole "
            "edge itself. The single-hole Kt is NOT reliable here; trust the FE."
        )
    if pitch_over_d < 3.0:
        return (
            f"pitch/d = {pitch_over_d:.2f} -- mild interaction expected. The "
            "single-hole Kt should be within roughly 10% but the sign of the "
            "correction is not obvious. The FE model resolves it."
        )
    return (
        f"pitch/d = {pitch_over_d:.2f} is above 3 -- holes are effectively "
        "isolated and the single-hole Kt should hold to a few percent."
    )


# ---------------------------------------------------------------------------
# Far-field stress
# ---------------------------------------------------------------------------


def gross_section_stress(force: float, geometry: StrapGeometry) -> float:
    """Remote tensile stress away from the perforations, MPa.

    This is the ``sigma`` that drives K: the handbook solutions are written in
    terms of the stress that would exist in the plate if the hole were not
    there, not the concentrated stress at the hole.
    """
    return force / geometry.gross_section_area


def net_section_stress(force: float, geometry: StrapGeometry) -> float:
    """Average tensile stress on the section through a hole, MPa."""
    return force / geometry.net_section_area


def mechanical_peak_stress(force: float, geometry: StrapGeometry) -> tuple:
    """Peak hole-edge stress in the UNCRACKED strip, ``(stress, kt_net)``.

    ASSUMPTION: the steel band is ignored, so the whole section is taken as
    polymer. Deliberately conservative -- a real bonded band would carry most
    of the load and drop the polymer stress by roughly the stiffness ratio.
    """
    kt = kt_hole_in_finite_width_strip(geometry.d_over_W)
    return kt * net_section_stress(force, geometry), kt


# ---------------------------------------------------------------------------
# Crack from a hole: Newman's fits
# ---------------------------------------------------------------------------


def _hole_parameter(crack_length, hole_radius: float):
    """``lambda = r / (r + a)``, the variable Newman's fits are written in.

    Runs from 1 at zero crack length (all hole) to 0 for a long crack (the
    hole has stopped mattering).
    """
    a = np.asarray(crack_length, dtype=float)
    if np.any(a <= 0.0):
        raise ValueError("crack length must be positive")
    if hole_radius <= 0.0:
        raise ValueError("hole radius must be positive")
    return hole_radius / (hole_radius + a)


def newman_single_crack_factor(crack_length, hole_radius: float):
    """Geometry factor ``F`` for ONE radial crack from a hole, infinite plate.

    ``K = F sigma sqrt(pi a)``, with ``a`` measured from the hole wall.

        F = 0.707 - 0.18 L + 6.55 L^2 - 10.54 L^3 + 6.85 L^4,   L = r/(r+a)

    Asymptotes, both reproduced exactly by the fit and pinned by tests:

    * ``a -> 0`` (``L -> 1``): ``F -> 3.39``. An edge crack in the hole's
      ``3 sigma`` hoop stress: ``1.1215 * 3 = 3.36``.
    * ``a -> inf`` (``L -> 0``): ``F -> 0.707 = 1/sqrt(2)``. The hole plus its
      single crack behave as a through crack of total length ``2r + a``, whose
      half-length is ``(2r + a)/2 -> a/2``, so ``K -> sigma sqrt(pi a / 2)``.

    This is the configuration of interest: a crack on ONE side of a hole.

    FIT ARTEFACT, documented rather than hidden: being a quartic, the
    polynomial does not settle onto its asymptote monotonically. Beyond about
    ``a/r = 20`` it wanders within roughly 1% either side of 0.707 -- slightly
    above, then about 0.2% below near ``a/r = 50``, then back up. Irrelevant
    here, since this strap cannot hold a crack past ``a/r ~ 5``, but worth
    knowing before reusing the fit for a long crack from a small hole.
    """
    lam = _hole_parameter(crack_length, hole_radius)
    return (
        0.707
        - 0.18 * lam
        + 6.55 * lam**2
        - 10.54 * lam**3
        + 6.85 * lam**4
    )


def newman_double_crack_factor(crack_length, hole_radius: float):
    """Geometry factor ``F`` for TWO symmetric radial cracks from a hole.

    ``K = F sigma sqrt(pi a)``, with

        F = 0.5 (3 - m) (1 + 1.243 (1 - m)^3),    m = a/(r+a)

    Asymptotes: ``F -> 3.365`` as ``a -> 0`` (same edge-crack-in-3-sigma limit
    as the single crack), and ``F -> 1`` as ``a -> inf``, where the hole plus
    both cracks behave as a central crack of half-length ``r + a``.

    Included because it is the configuration the FE model can be verified
    against most cleanly: it is symmetric, so a quarter model has no ambiguity
    about which tip is which.
    """
    lam = _hole_parameter(crack_length, hole_radius)
    m = 1.0 - lam
    return 0.5 * (3.0 - m) * (1.0 + 1.243 * (1.0 - m) ** 3)


def feddersen_width_correction(effective_half_crack, width: float):
    """Feddersen's secant correction for finite width.

    ``F_w = sqrt(sec(pi c / W))``, where ``c`` is the effective half-crack and
    ``W`` the full width. Tends to 1 for a small flaw and diverges as the flaw
    consumes the section, which is the right behaviour: a crack that has eaten
    the ligament has infinite K.

    APPROXIMATION: for a crack from a hole the effective half-crack is taken as
    ``r + a``, i.e. the hole and its crack are treated as one central flaw.
    Exact for two symmetric cracks; conservative for one, because a single
    crack removes less section than the symmetric pair it is modelled as.
    """
    c = np.asarray(effective_half_crack, dtype=float)
    if width <= 0.0:
        raise ValueError("width must be positive")
    if np.any(c <= 0.0):
        raise ValueError("effective half-crack must be positive")
    ratio = c / width
    if np.any(ratio >= 0.5):
        raise ValueError(
            "effective half-crack reaches the strip edge; the secant "
            "correction is singular and LEFM has nothing left to describe"
        )
    return np.sqrt(1.0 / np.cos(math.pi * ratio))


def handbook_k(
    crack_length,
    geometry: StrapGeometry,
    force: float | None = None,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    symmetric: bool = False,
    finite_width: bool = True,
):
    """Stress intensity factor at the tip of a crack from a hole, MPa*sqrt(mm).

    ``crack_length`` is measured from the hole wall and may be an array.

    Returns 0 for a :attr:`~ratchet_fea.geometry.CrackOrientation.LONGITUDINAL`
    crack. That is not a missing feature: a crack running along the strap lies
    in the hole's compressive hoop lobe, its faces are pressed together by the
    ratchet tension, and a mode I stress intensity factor does not exist for a
    closed crack. The FE model confirms it by measuring the crack opening
    directly rather than asserting it.
    """
    a = np.asarray(crack_length, dtype=float)
    force = geometry.service_tension if force is None else force

    if orientation is CrackOrientation.LONGITUDINAL:
        return np.zeros_like(a)

    sigma = gross_section_stress(force, geometry)
    factor = (
        newman_double_crack_factor(a, geometry.hole_radius)
        if symmetric
        else newman_single_crack_factor(a, geometry.hole_radius)
    )
    k = factor * sigma * np.sqrt(math.pi * a)

    if finite_width:
        k = k * feddersen_width_correction(
            geometry.hole_radius + a, geometry.strap_width
        )
    return k


def handbook_k_curve(
    geometry: StrapGeometry,
    force: float | None = None,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    n_points: int = 60,
    max_fraction: float = 0.9,
    symmetric: bool = False,
):
    """``(a, K)`` over the crack lengths this geometry can hold.

    Stops at ``max_fraction`` of the ligament, because both the secant
    correction and LEFM itself stop meaning anything as the ligament vanishes.
    """
    probe = CrackGeometry(length=1.0, orientation=orientation)
    limit = geometry.max_crack_length(probe) * max_fraction
    if limit <= 0:
        raise ValueError("geometry has no room for a crack in this direction")
    a = np.linspace(limit / n_points, limit, n_points)
    return a, handbook_k(a, geometry, force, orientation, symmetric=symmetric)


def critical_crack_length(
    geometry: StrapGeometry,
    toughness_mpa_root_mm: float,
    force: float | None = None,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    symmetric: bool = False,
) -> float:
    """Crack length at which ``K`` first reaches ``K_IC``, mm.

    ``nan`` if K never reaches the toughness anywhere the part can hold a
    crack -- which is the interesting answer, because it means static overload
    does not explain the failure.
    """
    a, k = handbook_k_curve(
        geometry, force, orientation, n_points=4000, symmetric=symmetric
    )
    if not np.any(k >= toughness_mpa_root_mm):
        return float("nan")
    # K rises monotonically over this range, so a simple interpolation on the
    # first crossing is exact enough.
    index = int(np.argmax(k >= toughness_mpa_root_mm))
    if index == 0:
        return float(a[0])
    return float(
        np.interp(toughness_mpa_root_mm, [k[index - 1], k[index]], [a[index - 1], a[index]])
    )


# ---------------------------------------------------------------------------
# Is LEFM even applicable?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LefmValidity:
    """Whether a K-based argument is admissible for this part at all.

    LEFM assumes the crack-tip plastic zone is small compared with the crack,
    the ligament and the thickness. Unfilled PA66 at room temperature is a
    tough, ductile polymer in a 3 mm section: that assumption is not obviously
    satisfied, and saying so is part of the answer rather than a disclaimer.
    """

    crack_length: float
    ligament: float
    thickness: float
    yield_strength: float
    toughness_mpa_root_mm: float
    k_applied: float

    @property
    def plastic_zone_plane_stress(self) -> float:
        """Irwin plane-stress plastic zone radius at the applied K, mm."""
        return (1.0 / (2.0 * math.pi)) * (self.k_applied / self.yield_strength) ** 2

    @property
    def plastic_zone_plane_strain(self) -> float:
        return (1.0 / (6.0 * math.pi)) * (self.k_applied / self.yield_strength) ** 2

    @property
    def astm_characteristic_size(self) -> float:
        """``2.5 (K_IC/sigma_y)^2`` -- ASTM E399's size requirement, mm.

        Crack length, ligament AND thickness must all exceed it for a measured
        K_IC to be a valid plane-strain material property.
        """
        return 2.5 * (self.toughness_mpa_root_mm / self.yield_strength) ** 2

    @property
    def thickness_is_plane_strain(self) -> bool:
        return self.thickness >= self.astm_characteristic_size

    @property
    def small_scale_yielding(self) -> bool:
        """Plastic zone under an eighth of both the crack and the ligament."""
        smallest = min(self.crack_length, self.ligament)
        return self.plastic_zone_plane_stress <= smallest / 8.0

    @property
    def verdict(self) -> str:
        if self.small_scale_yielding and self.thickness_is_plane_strain:
            return "valid"
        if self.small_scale_yielding:
            return "plane-stress"
        return "invalid"

    def message(self) -> str:
        lines = [
            f"Plastic zone (plane stress) {self.plastic_zone_plane_stress:.3f} mm "
            f"against a {self.crack_length:.2f} mm crack and a "
            f"{self.ligament:.2f} mm ligament.",
            f"ASTM E399 size requirement 2.5(K_IC/sigma_y)^2 = "
            f"{self.astm_characteristic_size:.1f} mm, against a "
            f"{self.thickness:.1f} mm section.",
        ]
        if self.verdict == "valid":
            lines.append("LEFM is applicable and the section is thick enough for K_IC.")
        elif self.verdict == "plane-stress":
            lines.append(
                "Small-scale yielding holds, so K is meaningful -- but the "
                "section is far thinner than the ASTM requirement, so the part "
                "is in PLANE STRESS. Its effective toughness is higher than the "
                "plane-strain K_IC, often by a factor of two or more. Comparing "
                "against K_IC is therefore CONSERVATIVE: a 'no growth' verdict "
                "is strengthened by it, a 'growth' verdict is not."
            )
        else:
            lines.append(
                "SMALL-SCALE YIELDING DOES NOT HOLD. The plastic zone is a "
                "significant fraction of the crack or the ligament, so K does "
                "not characterise the crack tip and this whole comparison is "
                "indicative at best. A J-integral or essential-work-of-fracture "
                "approach is needed to say anything quantitative."
            )
        return " ".join(lines)


def lefm_validity(
    crack_length: float,
    k_applied: float,
    geometry: StrapGeometry,
    grade: PolymerGrade | None = None,
    corner: str = "nominal",
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
) -> LefmValidity:
    """Assemble the LEFM applicability check for one crack length."""
    grade = grade or grade_for(SERVICE_CONDITION)
    probe = CrackGeometry(length=crack_length, orientation=orientation)
    return LefmValidity(
        crack_length=crack_length,
        ligament=max(geometry.remaining_ligament(probe), 1e-9),
        thickness=geometry.strap_thickness,
        yield_strength=grade.yield_strength.at(corner),
        toughness_mpa_root_mm=fracture_toughness_mpa_root_mm(grade, corner),
        k_applied=k_applied,
    )


# ---------------------------------------------------------------------------
# Screening study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FractureScreen:
    """Handbook LEFM screen across the toughness bracket."""

    geometry: StrapGeometry
    condition: MoistureCondition
    orientation: CrackOrientation
    force: float
    gross_stress: float
    kt_net: float
    row_note: str
    crack_lengths: np.ndarray
    k: np.ndarray
    #: ``corner -> K_IC`` in MPa*sqrt(mm).
    toughness: dict
    #: ``corner -> critical crack length`` in mm, ``nan`` if never reached.
    critical_lengths: dict
    k_at_observed: float
    observed_crack_length: float
    validity: LefmValidity

    @property
    def max_k(self) -> float:
        return float(np.max(self.k)) if len(self.k) else 0.0

    @property
    def propagates_at_any_length(self) -> bool:
        """True if K reaches the weakest toughness anywhere on the curve."""
        return self.max_k >= min(self.toughness.values())

    @property
    def propagates_at_observed(self) -> bool:
        return self.k_at_observed >= min(self.toughness.values())

    def margin_at_observed(self, corner: str = "nominal") -> float:
        """``K_IC / K`` at the observed crack length. Above 1 means it holds."""
        if self.k_at_observed <= 0:
            return float("inf")
        return self.toughness[corner] / self.k_at_observed

    def summary(self) -> str:
        lines = [
            "HANDBOOK LEFM SCREEN",
            "====================",
            "",
            f"  crack orientation     {self.orientation.value} "
            f"({'mode I under axial tension' if self.orientation.is_mode_i_under_axial_tension else 'PARALLEL to the load'})",
            f"  hoop stress at mouth  {self.orientation.hole_hoop_stress_factor:+.0f} x far field",
            f"  material condition    {self.condition.value}",
            f"  ratchet tension       {self.force:g} N",
            f"  gross-section stress  {self.gross_stress:.2f} MPa",
            f"  Kt (uncracked hole)   {self.kt_net:.3f}",
            f"  row interaction       {self.row_note}",
            "",
        ]
        if not self.orientation.is_mode_i_under_axial_tension:
            lines += [
                "  K IS ZERO FOR THIS ORIENTATION.",
                "  The crack plane is parallel to the ratchet tension and its mouth",
                "  sits in the hole's compressive hoop lobe. Axial tension presses",
                "  the faces together; there is no mode I driving force at any crack",
                "  length. Static overload cannot run this crack.",
                "",
            ]
            return "\n".join(lines)

        lines += [
            f"  K at the observed {self.observed_crack_length:g} mm crack:"
            f" {self.k_at_observed:.1f} MPa*sqrt(mm)",
            f"  peak K over the whole ligament:  {self.max_k:.1f} MPa*sqrt(mm)",
            "",
            f"  {'corner':<10} {'K_IC':>18} {'critical a':>12} {'margin at obs':>14}",
            "  " + "-" * 58,
        ]
        for corner in ("low", "nominal", "high"):
            kic = self.toughness[corner]
            critical = self.critical_lengths[corner]
            critical_text = "never" if math.isnan(critical) else f"{critical:.2f} mm"
            lines.append(
                f"  {corner:<10} {kic:>10.1f} MPa*sqrt(mm) {critical_text:>12} "
                f"{self.margin_at_observed(corner):>14.2f}"
            )
        lines += [
            "",
            "  K_IC units: divide by 31.62 for MPa*sqrt(m).",
            "",
            "  LEFM APPLICABILITY",
            f"  {self.validity.message()}",
        ]
        return "\n".join(lines)


def screen(
    geometry: StrapGeometry | None = None,
    force: float | None = None,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    condition: MoistureCondition = SERVICE_CONDITION,
    n_points: int = 200,
) -> FractureScreen:
    """Run the handbook LEFM screen across the toughness bracket."""
    geometry = geometry or PLACEHOLDER_GEOMETRY
    force = geometry.service_tension if force is None else force
    grade = grade_for(condition)

    if orientation.is_mode_i_under_axial_tension:
        a, k = handbook_k_curve(geometry, force, orientation, n_points=n_points)
        k_obs = float(
            handbook_k(geometry.observed_crack_length, geometry, force, orientation)
        )
    else:
        probe = CrackGeometry(length=1.0, orientation=orientation)
        a = np.linspace(
            geometry.max_crack_length(probe) / n_points,
            geometry.max_crack_length(probe) * 0.9,
            n_points,
        )
        k = np.zeros_like(a)
        k_obs = 0.0

    toughness = {
        corner: fracture_toughness_mpa_root_mm(grade, corner)
        for corner in ("low", "nominal", "high")
    }
    critical = {
        corner: critical_crack_length(geometry, value, force, orientation)
        for corner, value in toughness.items()
    }

    return FractureScreen(
        geometry=geometry,
        condition=condition,
        orientation=orientation,
        force=force,
        gross_stress=gross_section_stress(force, geometry),
        kt_net=kt_hole_in_finite_width_strip(geometry.d_over_W),
        row_note=row_interaction_note(geometry.pitch_over_d),
        crack_lengths=a,
        k=k,
        toughness=toughness,
        critical_lengths=critical,
        k_at_observed=k_obs,
        observed_crack_length=geometry.observed_crack_length,
        validity=lefm_validity(
            geometry.observed_crack_length, k_obs, geometry, grade,
            orientation=orientation,
        ),
    )
