"""Tier 1: closed-form screening of the STP-RB-001 strap.

This is the cheap check that decides whether an FE model is worth building.
It asks two questions:

1.  Does the *mechanical* service load, concentrated at a perforation, get
    anywhere near the strength of PA66?
2.  Does *constrained moisture swelling* against the internal steel band?

Everything is evaluated across the material brackets in :mod:`materials`, so
the answer is a range, not a number.  A conclusion that survives both bracket
corners is worth acting on; one that flips is a measurement request.

All formulae here are textbook and are cited in place.  Tier 2
(:mod:`mesh`, :mod:`diffusion`, :mod:`mechanics`) exists to check the two
assumptions Tier 1 cannot: that the holes do not interact, and that the
swelling field is uniform.

UNITS: mm / N / MPa / s, as everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import PLACEHOLDER_GEOMETRY, StrapGeometry
from .materials import (
    PA66_MOISTURE,
    MoistureTransport,
    pa66_at_moisture,
)

__all__ = [
    "kt_hole_in_finite_width_strip",
    "kt_gross_from_net",
    "row_interaction_note",
    "net_section_stress",
    "mechanical_peak_stress",
    "constrained_swelling_stress",
    "fickian_half_time",
    "Tier1Case",
    "Tier1Result",
    "screen",
]

# Validity limit of the Howland/Peterson polynomial below.
KT_MAX_D_OVER_W = 0.5


def kt_hole_in_finite_width_strip(d_over_W: float) -> float:
    """Stress concentration factor for a central circular hole, NET-section basis.

    Howland's solution as fitted by Peterson (*Stress Concentration Factors*,
    chart 4.1) for a finite-width strip in uniaxial tension:

        Kt_net = 3.00 - 3.13 (d/W) + 3.66 (d/W)^2 - 1.53 (d/W)^3

    Valid for ``0 <= d/W <= 0.5``.  It reduces to the classical Kirsch value of
    3.0 for an infinitely wide plate.

    ASSUMPTION (checked by Tier 2): a *single* hole in an otherwise plain
    strip.  The real strap has a row of them.  See :func:`row_interaction_note`.
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
    to an isolated hole once the pitch exceeds roughly three diameters.  Below
    about two diameters the ligaments start to control and the behaviour turns
    over.

    This is exactly the assumption Tier 2 is built to test, which is why the
    function returns prose rather than a number: do not put a hand-waved
    correction factor into a margin.
    """
    if pitch_over_d < 2.0:
        return (
            f"pitch/d = {pitch_over_d:.2f} is below 2 -- holes are close enough "
            "that the ligaments between them may govern rather than the hole "
            "edge itself. The single-hole Kt is NOT reliable here; trust Tier 2."
        )
    if pitch_over_d < 3.0:
        return (
            f"pitch/d = {pitch_over_d:.2f} -- mild interaction expected. The "
            "single-hole Kt should be within roughly 10% but the sign of the "
            "correction is not obvious. Tier 2 resolves it."
        )
    return (
        f"pitch/d = {pitch_over_d:.2f} is above 3 -- holes are effectively "
        "isolated for the mechanical load case and the single-hole Kt should "
        "hold to a few percent."
    )


def net_section_stress(force: float, geometry: StrapGeometry) -> float:
    """Average tensile stress on the section through a hole, MPa."""
    return force / geometry.net_section_area


def mechanical_peak_stress(force: float, geometry: StrapGeometry) -> tuple[float, float]:
    """Peak hole-edge stress under remote tension alone.

    Returns ``(peak_stress, kt_net)`` in MPa and dimensionless.

    ASSUMPTION: the steel band is ignored here, so the whole section is taken as
    polymer.  This is deliberately conservative for the mechanical case -- a
    real steel band would carry most of the load and drop the polymer stress by
    roughly the stiffness ratio.  Tier 2's laminate model includes the band.
    """
    kt = kt_hole_in_finite_width_strip(geometry.d_over_W)
    return kt * net_section_stress(force, geometry), kt


def constrained_swelling_stress(
    youngs_modulus: float,
    poisson_ratio: float,
    swelling_coefficient: float,
    moisture_change: float,
    constraint: str = "biaxial",
) -> float:
    """Stress produced when swelling is prevented, MPa (magnitude).

    A free strain ``eps = beta * dc`` that is fully prevented produces

        uniaxial   sigma = E eps                 (restrained in one direction)
        biaxial    sigma = E eps / (1 - nu)      (restrained in-plane, free through thickness)
        triaxial   sigma = E eps / (1 - 2 nu)    (restrained in all three directions)

    For this strap, ``biaxial`` is the right default: the steel band restrains
    the polymer in the strap plane but nothing restrains the free faces, so the
    part thickens instead.

    SIGN CONVENTION: the returned value is a magnitude.  During *absorption*
    the constrained stress is COMPRESSIVE -- which on its own does not crack a
    polymer.  It is during *desorption* (drying out between uses, or a wet skin
    drying over a still-wet core) that the same magnitude appears in TENSION.
    That asymmetry is why Tier 2 runs both directions.
    """
    free_strain = swelling_coefficient * moisture_change
    if constraint == "uniaxial":
        factor = 1.0
    elif constraint == "biaxial":
        factor = 1.0 / (1.0 - poisson_ratio)
    elif constraint == "triaxial":
        if poisson_ratio >= 0.5:
            raise ValueError("triaxial constraint is singular at nu = 0.5")
        factor = 1.0 / (1.0 - 2.0 * poisson_ratio)
    else:
        raise ValueError(f"unknown constraint mode {constraint!r}")
    return abs(youngs_modulus * free_strain * factor)


def fickian_half_time(half_thickness: float, diffusivity: float) -> float:
    """Time for a slab to reach half its equilibrium moisture uptake, s.

    From the standard plane-sheet solution (Crank, *The Mathematics of
    Diffusion*, eq. 4.18): ``M_t/M_inf = 0.5`` at ``D t / h^2 = 0.19685``,
    where ``h`` is the HALF-thickness for a sheet exposed on both faces.
    """
    if diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")
    return 0.19685 * half_thickness**2 / diffusivity


# ---------------------------------------------------------------------------
# Screening study
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier1Case:
    """One evaluation of the screening check at a chosen bracket corner."""

    label: str
    corner: str
    moisture_content: float
    youngs_modulus: float
    poisson_ratio: float
    yield_strength: float
    tensile_strength: float
    swelling_coefficient: float

    # Results
    kt_net: float = 0.0
    net_stress: float = 0.0
    mechanical_peak: float = 0.0
    swelling_stress: float = 0.0
    combined_peak: float = 0.0

    @property
    def mechanical_utilisation(self) -> float:
        """Peak mechanical stress as a fraction of yield."""
        return self.mechanical_peak / self.yield_strength

    @property
    def swelling_utilisation(self) -> float:
        return self.swelling_stress / self.yield_strength

    @property
    def combined_utilisation(self) -> float:
        return self.combined_peak / self.yield_strength


@dataclass(frozen=True)
class Tier1Result:
    """Outcome of the Tier 1 screen, across bracket corners."""

    geometry: StrapGeometry
    cases: tuple[Tier1Case, ...]
    kt_net: float
    kt_gross: float
    row_note: str
    half_time_seconds: float
    half_time_bracket: tuple[float, float]
    in_plane_half_time_seconds: float
    #: Wetting half-time at each evaluated bracket corner. The diffusivity
    #: bracket spans a decade, so a cross-check against a Tier 2 run must use
    #: the SAME corner the FE run used or it will disagree by that decade.
    half_time_by_corner: dict = field(default_factory=dict)

    def half_time_at(self, corner: str) -> float:
        """Wetting half-time at a given bracket corner, seconds."""
        if corner in self.half_time_by_corner:
            return self.half_time_by_corner[corner]
        return self.half_time_seconds

    def case(self, label: str, corner: str) -> Tier1Case:
        for c in self.cases:
            if c.label == label and c.corner == corner:
                return c
        raise KeyError(f"no case {label!r}/{corner!r}")

    @property
    def mechanical_alone_is_benign(self) -> bool:
        """True if mechanical load stays below yield at every bracket corner."""
        return all(c.mechanical_utilisation < 1.0 for c in self.cases)

    @property
    def swelling_is_significant(self) -> bool:
        """True if constrained swelling reaches yield at any bracket corner.

        This is the Tier 1 finding that justifies building Tier 2.
        """
        return any(c.swelling_utilisation >= 1.0 for c in self.cases)

    def summary(self) -> str:
        lines = [
            "TIER 1 SCREENING RESULT",
            "=======================",
            "",
            f"  Kt (net section, single hole)   {self.kt_net:.3f}",
            f"  Kt (gross section)              {self.kt_gross:.3f}",
            f"  row interaction                 {self.row_note}",
            "",
            "  Through-thickness wetting half-time (Mt/Minf = 0.5):",
            f"    nominal D  {_fmt_days(self.half_time_seconds)}",
            f"    bracket    {_fmt_days(self.half_time_bracket[0])} .. "
            f"{_fmt_days(self.half_time_bracket[1])}",
            f"  In-plane equivalent half-time    {_fmt_days(self.in_plane_half_time_seconds)}",
            "    (the smaller of the two governs; if they differ by more than ~5x,",
            "     the faster path sets the timing and a plane-stress in-plane model",
            "     alone will MISTIME the result -- see diffusion.py product solution)",
            "",
            "  Stress check (MPa), utilisation = peak / yield:",
            "",
        ]
        header = (
            f"  {'case':<26} {'E':>7} {'sig_y':>6} {'mech':>7} {'swell':>7} "
            f"{'comb':>7} {'util':>6}"
        )
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for c in self.cases:
            lines.append(
                f"  {c.label + ' [' + c.corner + ']':<26} "
                f"{c.youngs_modulus:7.0f} {c.yield_strength:6.1f} "
                f"{c.mechanical_peak:7.1f} {c.swelling_stress:7.1f} "
                f"{c.combined_peak:7.1f} {c.combined_utilisation:6.2f}"
            )
        lines += [
            "",
            "  CONCLUSIONS",
            f"    mechanical load alone stays below yield everywhere: "
            f"{self.mechanical_alone_is_benign}",
            f"    constrained swelling reaches yield somewhere:        "
            f"{self.swelling_is_significant}",
        ]
        return "\n".join(lines)


def _fmt_days(seconds: float) -> str:
    days = seconds / 86400.0
    if days < 1.0:
        return f"{seconds / 3600.0:.1f} h"
    if days < 365.0:
        return f"{days:.1f} d"
    return f"{days / 365.0:.2f} y"


def screen(
    geometry: StrapGeometry | None = None,
    moisture: MoistureTransport | None = None,
    corners: tuple[str, ...] = ("low", "nominal", "high"),
) -> Tier1Result:
    """Run the Tier 1 screen over the material brackets.

    Two exposure cases are evaluated:

    ``50% RH``      the strap equilibrated to ordinary indoor humidity;
    ``immersed``    the strap wet through, as it would be if washed, worn
                    against skin, or used outdoors.

    Both are evaluated at each requested bracket corner.  Moisture-softened
    modulus and strength are taken from :func:`materials.pa66_at_moisture`, so
    the softening that partly offsets the swelling is included.
    """
    geometry = geometry or PLACEHOLDER_GEOMETRY
    moisture = moisture or PA66_MOISTURE

    kt_net = kt_hole_in_finite_width_strip(geometry.d_over_W)
    kt_gross = kt_gross_from_net(kt_net, geometry.d_over_W)
    net_sigma = net_section_stress(geometry.service_tension, geometry)
    mech_peak = kt_net * net_sigma

    exposures = {
        "50% RH": moisture.saturation_50rh,
        "immersed": moisture.saturation_immersed,
    }

    cases: list[Tier1Case] = []
    for label, sat_bracket in exposures.items():
        for corner in corners:
            dc = sat_bracket.at(corner)
            beta = moisture.swelling_coefficient.at(corner)
            E, nu, sy, su = pa66_at_moisture(dc, corner=corner)
            swell = constrained_swelling_stress(E, nu, beta, dc, constraint="biaxial")
            # The swelling field is uniform in Tier 1, so it is not amplified by
            # the hole the way the remote load is -- a uniform equi-biaxial field
            # has a hoop stress at a hole edge equal to twice the far field for
            # the in-plane components, but the *difference* from the far field is
            # what a uniform eigenstrain produces, which for an unconstrained
            # hole edge is zero. Tier 2 resolves this properly; Tier 1 simply
            # superposes the magnitudes, which is the conservative reading.
            combined = mech_peak + swell
            cases.append(
                Tier1Case(
                    label=label,
                    corner=corner,
                    moisture_content=dc,
                    youngs_modulus=E,
                    poisson_ratio=nu,
                    yield_strength=sy,
                    tensile_strength=su,
                    swelling_coefficient=beta,
                    kt_net=kt_net,
                    net_stress=net_sigma,
                    mechanical_peak=mech_peak,
                    swelling_stress=swell,
                    combined_peak=combined,
                )
            )

    h_band = geometry.diffusion_half_thickness(in_band_region=True)
    d_nom = moisture.diffusivity.nominal
    half_time_by_corner = {
        corner: fickian_half_time(h_band, moisture.diffusivity.at(corner))
        for corner in ("low", "nominal", "high")
    }
    return Tier1Result(
        geometry=geometry,
        cases=tuple(cases),
        kt_net=kt_net,
        kt_gross=kt_gross,
        row_note=row_interaction_note(geometry.pitch_over_d),
        half_time_seconds=fickian_half_time(h_band, d_nom),
        half_time_bracket=(
            fickian_half_time(h_band, moisture.diffusivity.high),
            fickian_half_time(h_band, moisture.diffusivity.low),
        ),
        in_plane_half_time_seconds=fickian_half_time(
            geometry.in_plane_diffusion_length, d_nom
        ),
        half_time_by_corner=half_time_by_corner,
    )
