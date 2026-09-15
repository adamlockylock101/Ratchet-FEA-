"""Tier 2: extract, plot and sanity-check the coupled results.

Pulls the hole-edge stress history out of the transient solve, plots stress
against moisture uptake, and cross-checks the FE model against the Tier 1
closed-form estimates in :mod:`analytical`.  The cross-check is the point: two
independent routes to the same number is the only cheap defence against a
silently wrong FE model.

Three comparisons are made:

``Kt``
    FE hole-edge stress concentration against the Howland/Peterson formula.
    These should agree to within the row-interaction effect -- which is
    precisely what Tier 2 was built to quantify, so a moderate difference is a
    *result*, not an error.  A factor-of-two difference is a bug.

``constrained swelling stress``
    FE polymer stress in the reinforced bulk against ``E beta dc / (1 - nu)``.
    These should agree closely, because in the bulk, far from holes and free
    edges, the FE model is solving the same problem the formula describes.

``wetting time``
    FE time to half uptake against the plane-sheet analytic value.  The FE
    result should be slightly faster, because it adds in-plane ingress on top
    of the through-thickness path.

A LIMITATION THE RESULTS MAKE UNAVOIDABLE
-----------------------------------------
:mod:`mechanics` condenses the PA66/steel/PA66 section to a membrane using
classical laminate theory.  CLT enforces the *resultant* traction on a free
edge, not the traction on each layer, so at a hole wall it lets the polymer and
steel plies carry equal and opposite self-equilibrating stress.  Reality
relaxes that mismatch to zero through interlaminar shear over a boundary layer
roughly one laminate thickness wide.

For the placeholder dimensions that boundary layer is ~3 mm -- larger than the
hole radius and half the inter-hole ligament.  So the swelling component of the
hole-edge stress is an UPPER BOUND, and resolving it properly needs a
through-thickness (Tier 3) model.  :func:`free_edge_validity` reports this
automatically against whatever dimensions are supplied, so the caveat updates
itself when real measurements replace the estimates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytical import Tier1Result, constrained_swelling_stress
from .diffusion import DiffusionResult
from .geometry import StrapGeometry
from .materials import pa66_at_moisture, pa66_properties
from .mechanics import MechanicsModel, MechanicsResult
from .mesh import StripMesh

__all__ = [
    "HoleHistory",
    "StressHistory",
    "Tier1Comparison",
    "FreeEdgeVerdict",
    "TensileAssessment",
    "MoistureCycleAssessment",
    "peak_tensile_assessment",
    "moisture_cycle_assessment",
    "relaxation_bracket",
    "stress_history",
    "bulk_element_mask",
    "free_edge_validity",
    "compare_with_tier1",
    "plot_hole_stress_history",
    "plot_stress_vs_uptake",
    "plot_tier1_comparison",
    "plot_moisture_cycle",
    "plot_field",
    "write_summary",
]

SECONDS_PER_DAY = 86400.0


# ---------------------------------------------------------------------------
# History extraction
# ---------------------------------------------------------------------------


@dataclass
class HoleHistory:
    """Stress history at one perforation."""

    name: str
    index: int
    is_interior: bool
    max_principal: np.ndarray
    min_principal: np.ndarray
    von_mises: np.ndarray
    mean_concentration: np.ndarray

    @property
    def peak_tensile(self) -> float:
        return float(np.max(self.max_principal))

    @property
    def peak_compressive(self) -> float:
        return float(np.min(self.min_principal))

    @property
    def peak_von_mises(self) -> float:
        return float(np.max(self.von_mises))


@dataclass
class StressHistory:
    """Everything the transient produced, indexed by time."""

    times: np.ndarray
    uptake_fraction: np.ndarray
    #: Indices into the diffusion result's time grid that these entries came
    #: from, so a stress step can be paired with its concentration field.
    indices: np.ndarray
    holes: list
    #: Peak tensile / compressive / von Mises anywhere in the polymer.
    global_max_principal: np.ndarray
    global_min_principal: np.ndarray
    global_von_mises: np.ndarray
    #: Polymer stress in the reinforced bulk, away from holes and free edges,
    #: where classical laminate theory is valid. The Tier 1 cross-check point.
    bulk_sxx: np.ndarray
    bulk_concentration: np.ndarray
    #: The final mechanics solve, kept for field plots.
    final_result: MechanicsResult
    diffusion: DiffusionResult
    mechanics_model: MechanicsModel
    geometry: StrapGeometry

    @property
    def times_days(self) -> np.ndarray:
        return self.times / SECONDS_PER_DAY

    @property
    def interior_holes(self) -> list:
        return [h for h in self.holes if h.is_interior]

    def representative_holes(self) -> list:
        """Interior holes if there are any, otherwise all of them.

        The end holes of the modelled window carry a higher concentration
        because nothing shields them on one side. That is real for the two
        holes at the ends of the strap's perforation row and misleading for
        every other hole, so quote the interior ones.
        """
        return self.interior_holes or self.holes

    def peak_tensile_envelope(self) -> np.ndarray:
        """Highest hole-edge tensile stress across representative holes."""
        holes = self.representative_holes()
        return np.max(np.vstack([h.max_principal for h in holes]), axis=0)

    def peak_compressive_envelope(self) -> np.ndarray:
        holes = self.representative_holes()
        return np.min(np.vstack([h.min_principal for h in holes]), axis=0)

    def peak_von_mises_envelope(self) -> np.ndarray:
        holes = self.representative_holes()
        return np.max(np.vstack([h.von_mises for h in holes]), axis=0)

    def yield_utilisation_envelope(self) -> np.ndarray:
        """Hole-edge von Mises divided by the moisture-adjusted yield strength.

        Uses the yield strength at each instant's local moisture content, so
        the softening that accompanies the swelling is accounted for on both
        sides of the ratio.
        """
        vm = self.peak_von_mises_envelope()
        holes = self.representative_holes()
        c = np.mean(np.vstack([h.mean_concentration for h in holes]), axis=0)
        sy = np.array([pa66_at_moisture(ci)[2] for ci in c])
        return vm / sy


def stress_history(
    strip: StripMesh,
    diffusion_result: DiffusionResult,
    model: MechanicsModel,
    every: int = 1,
    progress: bool = False,
) -> StressHistory:
    """Solve the mechanics at each stored time and collect the hole-edge peaks.

    ``every`` subsamples the diffusion time grid; the first and last steps are
    always included so the dry and equilibrium states are never lost.
    """
    from . import mechanics as mech

    indices = list(range(0, diffusion_result.n_times, max(1, every)))
    if indices[-1] != diffusion_result.n_times - 1:
        indices.append(diffusion_result.n_times - 1)

    geometry = strip.geometry
    names = strip.hole_boundaries
    n = len(indices)

    per_hole = {
        name: {
            "max_principal": np.empty(n),
            "min_principal": np.empty(n),
            "von_mises": np.empty(n),
            "mean_concentration": np.empty(n),
        }
        for name in names
    }
    g_max = np.empty(n)
    g_min = np.empty(n)
    g_vm = np.empty(n)
    bulk_sxx = np.empty(n)
    bulk_c = np.empty(n)

    bulk = bulk_element_mask(strip)
    result = None

    for k, i in enumerate(indices):
        if progress:
            print(
                f"    mechanics {k + 1}/{n} "
                f"(t = {diffusion_result.times[i] / SECONDS_PER_DAY:.2f} d)",
                flush=True,
            )
        result = mech.solve(strip, model, diffusion_result.concentration[i])
        for name in names:
            edge = result.hole_edges[name]
            for key in ("max_principal", "min_principal", "von_mises", "mean_concentration"):
                per_hole[name][key][k] = edge[key]
        g_max[k] = result.peak_tensile
        g_min[k] = result.peak_compressive
        g_vm[k] = result.peak_von_mises
        bulk_sxx[k] = float(np.mean(result.sxx[bulk])) if bulk.any() else np.nan
        bulk_c[k] = (
            float(np.mean(diffusion_result.concentration[i][_bulk_vertices(strip, bulk)]))
            if bulk.any()
            else np.nan
        )

    holes = [
        HoleHistory(
            name=name,
            index=idx,
            is_interior=geometry.is_interior_hole(idx),
            max_principal=per_hole[name]["max_principal"],
            min_principal=per_hole[name]["min_principal"],
            von_mises=per_hole[name]["von_mises"],
            mean_concentration=per_hole[name]["mean_concentration"],
        )
        for idx, name in enumerate(names)
    ]

    return StressHistory(
        times=diffusion_result.times[indices],
        uptake_fraction=diffusion_result.uptake_fraction[indices],
        indices=np.asarray(indices, dtype=int),
        holes=holes,
        global_max_principal=g_max,
        global_min_principal=g_min,
        global_von_mises=g_vm,
        bulk_sxx=bulk_sxx,
        bulk_concentration=bulk_c,
        final_result=result,
        diffusion=diffusion_result,
        mechanics_model=model,
        geometry=geometry,
    )


def bulk_element_mask(strip: StripMesh, clearance: float | None = None) -> np.ndarray:
    """Elements where the membrane (CLT) model is trustworthy.

    Excludes everything within ``clearance`` of a hole wall or a free side
    edge.  The default clearance is one strap thickness, which is the usual
    estimate of the laminate free-edge boundary layer width.  Inside that
    band, CLT's assumption that every layer shares the membrane strain breaks
    down, so those elements are not a fair comparison against the Tier 1
    constrained-swelling formula.
    """
    g = strip.geometry
    clearance = g.strap_thickness if clearance is None else clearance

    centroids = strip.element_centroids()
    x, y = centroids[0], centroids[1]

    ok = strip.band_element_mask() if g.has_steel_band else np.ones(len(x), dtype=bool)
    ok &= y > clearance
    ok &= y < g.strap_width - clearance
    for cx, cy in g.hole_centres:
        ok &= np.hypot(x - cx, y - cy) > g.hole_radius + clearance
    # Stay clear of the truncation planes too.
    ok &= x > g.end_margin / 2.0
    ok &= x < g.modelled_length - g.end_margin / 2.0
    return ok


def _bulk_vertices(strip: StripMesh, element_mask: np.ndarray) -> np.ndarray:
    """Vertices belonging to the bulk elements."""
    return np.unique(strip.mesh.t[:, element_mask].ravel())


@dataclass
class FreeEdgeVerdict:
    """Whether CLT is usable at the hole walls for these dimensions."""

    boundary_layer: float
    hole_radius: float
    inter_hole_ligament: float
    side_ligament: float
    bulk_element_fraction: float

    @property
    def narrowest_ligament(self) -> float:
        return min(self.inter_hole_ligament, self.side_ligament)

    @property
    def boundary_layer_ratio(self) -> float:
        """Free-edge boundary layer as a fraction of the narrowest ligament."""
        if self.narrowest_ligament <= 0:
            return float("inf")
        return self.boundary_layer / self.narrowest_ligament

    @property
    def hole_edge_is_reliable(self) -> bool:
        """True if the free-edge boundary layer is small next to the holes."""
        return self.boundary_layer_ratio <= 0.25

    @property
    def is_marginal(self) -> bool:
        """The boundary layer is comparable to, but not larger than, the ligament."""
        return 0.25 < self.boundary_layer_ratio <= 0.5

    @property
    def has_valid_bulk(self) -> bool:
        return self.bulk_element_fraction > 0.02

    def message(self) -> str:
        scale = (
            f"Laminate free-edge boundary layer ~{self.boundary_layer:g} mm "
            f"(one strap thickness) against a narrowest ligament of "
            f"{self.narrowest_ligament:g} mm -- a ratio of "
            f"{self.boundary_layer_ratio:.2f}. "
        )
        if self.hole_edge_is_reliable:
            return scale + (
                "That is small enough that the membrane model's hole-edge "
                "swelling stresses are usable."
            )
        if self.is_marginal:
            return scale + (
                "MARGINAL: classical laminate theory cannot relax the "
                "polymer/steel ply mismatch to zero at a free edge, and the "
                "boundary layer covers a noticeable part of the ligament, so "
                "treat the SWELLING component of the hole-edge stress as an "
                "upper bound. The mechanical stress concentration and the "
                "reinforced bulk stress are unaffected."
            )
        return (
            f"CAUTION: the laminate free-edge boundary layer is ~"
            f"{self.boundary_layer:g} mm (one strap thickness), against a hole "
            f"radius of {self.hole_radius:g} mm and ligaments of "
            f"{self.inter_hole_ligament:g} / {self.side_ligament:g} mm. Classical "
            "laminate theory cannot relax the polymer/steel ply mismatch to zero "
            "at a free edge, so the SWELLING part of the hole-edge stress is an "
            "UPPER BOUND. The mechanical stress concentration and the reinforced "
            "bulk stress are unaffected. Resolving the hole edge properly needs a "
            "through-thickness (Tier 3) model."
        )


def free_edge_validity(strip: StripMesh) -> FreeEdgeVerdict:
    """Report whether the membrane model can be trusted at the hole walls."""
    g = strip.geometry
    bulk = bulk_element_mask(strip)
    return FreeEdgeVerdict(
        boundary_layer=g.strap_thickness,
        hole_radius=g.hole_radius,
        inter_hole_ligament=g.inter_hole_ligament,
        side_ligament=g.side_ligament,
        bulk_element_fraction=float(bulk.mean()),
    )


# ---------------------------------------------------------------------------
# Tier 1 cross-check
# ---------------------------------------------------------------------------


@dataclass
class Tier1Comparison:
    """Side-by-side of the Tier 2 FE result and the Tier 1 closed form."""

    kt_tier1: float
    kt_tier2_interior: float
    kt_tier2_end: float

    swelling_tier1: float
    swelling_tier2_bulk: float

    half_time_tier1: float
    half_time_tier2: float

    #: Agreement within this factor counts as "same ballpark".
    tolerance: float = 2.0

    @staticmethod
    def _ratio(a: float, b: float) -> float:
        if b == 0.0 or not np.isfinite(a) or not np.isfinite(b):
            return float("nan")
        return a / b

    @property
    def kt_ratio(self) -> float:
        return self._ratio(self.kt_tier2_interior, self.kt_tier1)

    @property
    def swelling_ratio(self) -> float:
        return self._ratio(self.swelling_tier2_bulk, self.swelling_tier1)

    @property
    def half_time_ratio(self) -> float:
        return self._ratio(self.half_time_tier2, self.half_time_tier1)

    def _within(self, ratio: float) -> bool:
        if not np.isfinite(ratio) or ratio <= 0:
            return False
        return (1.0 / self.tolerance) <= ratio <= self.tolerance

    @property
    def agrees(self) -> bool:
        """True if every comparison sits inside the tolerance factor."""
        return all(
            self._within(r)
            for r in (self.kt_ratio, self.swelling_ratio, self.half_time_ratio)
        )

    def summary(self) -> str:
        rows = [
            ("Kt at hole (net section)", self.kt_tier1, self.kt_tier2_interior, self.kt_ratio, "-"),
            ("constrained swelling stress", self.swelling_tier1, self.swelling_tier2_bulk, self.swelling_ratio, "MPa"),
            ("time to half uptake", self.half_time_tier1 / SECONDS_PER_DAY, self.half_time_tier2 / SECONDS_PER_DAY, self.half_time_ratio, "d"),
        ]
        lines = [
            "TIER 1 <-> TIER 2 CROSS-CHECK",
            "=============================",
            "",
            f"  {'quantity':<28} {'Tier 1':>10} {'Tier 2':>10} {'ratio':>8}  {'':<5} ok",
            "  " + "-" * 68,
        ]
        for name, t1, t2, ratio, unit in rows:
            flag = "yes" if self._within(ratio) else "NO"
            lines.append(
                f"  {name:<28} {t1:10.3f} {t2:10.3f} {ratio:8.3f}  {unit:<5} {flag}"
            )
        lines += [
            "",
            f"  End-hole Kt (no shielding neighbour): {self.kt_tier2_end:.3f}",
            "",
            f"  Overall: Tier 2 {'agrees with' if self.agrees else 'DISAGREES with'} "
            f"Tier 1 within a factor of {self.tolerance:g}.",
        ]
        if np.isfinite(self.kt_ratio) and self.kt_ratio < 0.95:
            lines.append(
                f"  Tier 2 Kt is {100 * (1 - self.kt_ratio):.0f}% below the isolated-hole "
                "value. That is the row-interaction effect Tier 1 could not "
                "capture: holes in a line along the load shield each other."
            )
        return "\n".join(lines)


def compare_with_tier1(
    history: StressHistory,
    tier1: Tier1Result,
    strip: StripMesh,
    kt_tier2: dict,
) -> Tier1Comparison:
    """Build the Tier 1 / Tier 2 comparison.

    ``kt_tier2`` is the dict from
    :func:`~ratchet_fea.mechanics.stress_concentration_factors`, which runs the
    mechanical-only, band-free case that Tier 1's formula actually describes.
    """
    model = history.mechanics_model
    end_kts = [
        v
        for i, (k, v) in enumerate(kt_tier2["per_hole"].items())
        if not history.geometry.is_interior_hole(i)
    ]

    # Equilibrium bulk state: the last time step, where the in-plane and
    # through-thickness fields have both come to rest.
    c_bulk = float(history.bulk_concentration[-1])
    e, nu, _, _ = pa66_at_moisture(c_bulk, corner=model.corner)
    swelling_t1 = constrained_swelling_stress(
        e, nu, model.swelling_coefficient, c_bulk - model.stress_free_moisture,
        constraint="biaxial",
    )

    half_time_t2 = history.diffusion.time_to_uptake(0.5)

    return Tier1Comparison(
        # "representative" falls back to the mean over all holes when the
        # modelled window is too short to contain an interior one.
        kt_tier2_interior=float(
            kt_tier2.get("representative_mean", kt_tier2["interior_mean"])
        ),
        kt_tier1=tier1.kt_net,
        kt_tier2_end=float(np.mean(end_kts)) if end_kts else float("nan"),
        swelling_tier1=swelling_t1,
        # Compare magnitudes: the FE value is compressive (negative) during
        # absorption, which the Tier 1 formula reports as a magnitude.
        swelling_tier2_bulk=abs(float(history.bulk_sxx[-1])),
        # Use the same bracket corner the FE run used: the diffusivity bracket
        # spans a decade, so mixing corners would make the comparison fail by
        # that decade regardless of whether the models agree.
        half_time_tier1=tier1.half_time_at(model.corner),
        half_time_tier2=half_time_t2,
    )


# ---------------------------------------------------------------------------
# Peak tensile stress against the yield bracket
# ---------------------------------------------------------------------------


@dataclass
class TensileAssessment:
    """Peak hole-edge TENSILE stress, and what it is up against.

    Tension is the quantity that opens a crack, so it gets its own assessment
    rather than being read off a von Mises number that cannot tell tension from
    compression. The yield strength is evaluated at the moisture content the
    material is actually at when the peak occurs -- which matters a great deal
    here, because a strap that has dried out is both more highly stressed AND
    considerably stronger than the same strap wet.
    """

    case: str
    peak_stress: float
    peak_time: float
    peak_uptake: float
    moisture_at_peak: float
    #: Yield strength at the peak's moisture content, at each bracket corner.
    yield_low: float
    yield_nominal: float
    yield_high: float
    #: Equilibrium (end of transient) tensile stress, for comparison with the
    #: transient peak: the difference is the moisture-gradient contribution.
    equilibrium_stress: float
    stress_free_moisture: float
    #: True when the maximum occurs at t = 0, before the moisture change has
    #: started. That means the case never develops tension at all -- the
    #: "peak" is just the starting state.
    peak_is_initial_state: bool = False

    @property
    def utilisation_low(self) -> float:
        """Against the WEAKEST end of the yield bracket -- the pessimistic read."""
        return self.peak_stress / self.yield_low

    @property
    def utilisation_nominal(self) -> float:
        return self.peak_stress / self.yield_nominal

    @property
    def utilisation_high(self) -> float:
        return self.peak_stress / self.yield_high

    @property
    def utilisation_range(self) -> tuple:
        """``(best, worst)`` utilisation, ascending.

        The weakest end of the yield bracket gives the HIGHEST utilisation, so
        the low/high labels invert between the two. Ascending order avoids
        printing a range that reads backwards.
        """
        return (self.utilisation_high, self.utilisation_low)

    @property
    def reaches_yield(self) -> bool:
        """True if the peak reaches yield anywhere in the bracket."""
        return self.peak_stress >= self.yield_low

    @property
    def is_tensile(self) -> bool:
        return self.peak_stress > 0.0

    @property
    def develops_tension(self) -> bool:
        """True if the moisture change itself drives the hole edge into tension.

        Distinct from :attr:`is_tensile`: a case can show a small positive
        maximum at t = 0 purely from the mechanical load and then go straight
        into compression, which is not a tensile case in any useful sense.
        """
        return self.is_tensile and not self.peak_is_initial_state

    @property
    def gradient_contribution(self) -> float | None:
        """Transient overshoot above the equilibrium tensile stress, MPa.

        A drying hole wall shrinks against a still-wet interior, so it can be
        pulled harder mid-transient than the final uniform state pulls it.

        ``None`` when the comparison is meaningless -- when the peak is the
        starting state, or when equilibrium is compressive, the difference
        between the two is not a gradient effect, it is just the case changing
        sign along the way.
        """
        if self.peak_is_initial_state or self.equilibrium_stress <= 0.0:
            return None
        return self.peak_stress - self.equilibrium_stress

    def summary(self) -> str:
        lines = [
            f"PEAK TENSILE STRESS AT THE HOLE EDGE -- {self.case}",
            "=" * 60,
            "",
            f"  stress-free moisture assumed   {self.stress_free_moisture:.4f}",
            f"  peak tensile stress            {self.peak_stress:8.2f} MPa",
            f"    at t =                       {self.peak_time / SECONDS_PER_DAY:8.2f} d "
            f"({self.peak_uptake * 100:.0f}% through the moisture change)",
            f"    local moisture there         {self.moisture_at_peak:8.4f} mass fraction",
            f"  at equilibrium                 {self.equilibrium_stress:8.2f} MPa",
        ]
        gradient = self.gradient_contribution
        if gradient is not None:
            lines.append(
                f"  transient gradient adds        {gradient:8.2f} MPa"
            )
        elif self.peak_is_initial_state:
            lines.append(
                "  (the maximum is the STARTING state, before the moisture change "
                "begins --\n   this case never develops tension of its own)"
            )
        lines += [
            "",
            "  Against the PA66 yield bracket at that moisture content:",
            "",
            f"    {'corner':<10} {'yield':>8} {'utilisation':>12}",
            "    " + "-" * 32,
            f"    {'low':<10} {self.yield_low:8.1f} {self.utilisation_low:12.2f}",
            f"    {'nominal':<10} {self.yield_nominal:8.1f} {self.utilisation_nominal:12.2f}",
            f"    {'high':<10} {self.yield_high:8.1f} {self.utilisation_high:12.2f}",
            "",
        ]
        if not self.develops_tension:
            lines.append(
                "  The moisture change does not drive the hole edge into tension in "
                "this case.\n  With the stress-free state assumed above it moves "
                "toward compression, or\n  merely unloads without crossing zero. "
                "See --relaxation."
            )
        elif self.reaches_yield:
            lines.append(
                "  REACHES YIELD at the weak end of the bracket. Pin down the "
                "grade's yield\n  strength before drawing a conclusion -- see "
                "the verification table."
            )
        else:
            lines.append(
                f"  Stays below yield across the bracket "
                f"(worst case {self.utilisation_low:.2f} of yield)."
            )
        return "\n".join(lines)


def peak_tensile_assessment(history: StressHistory, case: str = "") -> TensileAssessment:
    """Find the worst tensile moment at the hole edge and size it against yield."""
    tension = history.peak_tensile_envelope()
    peak = int(np.argmax(tension))

    holes = history.representative_holes()
    # Clamped: a field dried all the way out interpolates to a round-off-level
    # negative, which would print as "-0.0000".
    moisture = max(0.0, float(np.mean([h.mean_concentration[peak] for h in holes])))

    yields = tuple(
        float(pa66_properties(moisture, corner=corner)[2])
        for corner in ("low", "nominal", "high")
    )

    return TensileAssessment(
        case=case or history.diffusion.model.label,
        peak_is_initial_state=peak == 0,
        peak_stress=float(tension[peak]),
        peak_time=float(history.times[peak]),
        peak_uptake=float(history.uptake_fraction[peak]),
        moisture_at_peak=moisture,
        yield_low=yields[0],
        yield_nominal=yields[1],
        yield_high=yields[2],
        equilibrium_stress=float(tension[-1]),
        stress_free_moisture=history.mechanics_model.stress_free_moisture,
    )


# ---------------------------------------------------------------------------
# Moisture cycling
# ---------------------------------------------------------------------------


@dataclass
class MoistureCycleAssessment:
    """The stress cycle a wet/dry excursion imposes, with no external load change.

    Full fatigue analysis is out of scope (see the README's Tier 3 section),
    but the stress range a moisture cycle produces is a direct output of the
    two transient cases and is worth stating: if it is a large fraction of the
    tensile strength at a strongly negative R ratio, the part is being fatigue
    loaded by humidity alone, and no amount of looking at the mechanical duty
    cycle will reveal it.
    """

    peak_tension: float
    peak_compression: float
    #: Tensile strength at the moisture content where the tensile peak occurs.
    uts_low: float
    uts_nominal: float
    uts_high: float
    #: Time for the bulk to complete half a wet/dry excursion.
    bulk_half_time: float
    #: Whether the hole wall, being an exposed surface, tracks ambient humidity
    #: far faster than the bulk does.
    hole_wall_tracks_ambient: bool = True

    @property
    def stress_range(self) -> float:
        """Peak-to-peak swing over a full wet/dry cycle, MPa."""
        return self.peak_tension - self.peak_compression

    @property
    def stress_amplitude(self) -> float:
        return 0.5 * self.stress_range

    @property
    def mean_stress(self) -> float:
        return 0.5 * (self.peak_tension + self.peak_compression)

    @property
    def r_ratio(self) -> float:
        """``sigma_min / sigma_max``. -1 is fully reversed."""
        if self.peak_tension == 0.0:
            return float("nan")
        return self.peak_compression / self.peak_tension

    def amplitude_over_uts(self, corner: str = "nominal") -> float:
        uts = {"low": self.uts_low, "nominal": self.uts_nominal, "high": self.uts_high}[
            corner
        ]
        return self.stress_amplitude / uts

    @property
    def is_significant(self) -> bool:
        """Amplitude above ~20% of UTS at a negative R is worth chasing.

        Unfilled PA66's fully-reversed fatigue strength at 1e6 cycles sits
        roughly in the 20-30% of UTS band. An amplitude at or above the bottom
        of that band means the moisture cycle alone puts the part in the
        finite-life regime.
        """
        return self.amplitude_over_uts("low") >= 0.20 and self.r_ratio < 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                "MOISTURE CYCLING -- THE STRESS CYCLE WITH NO LOAD CYCLE",
                "=" * 60,
                "",
                "  Combining the two cases above, one full wet/dry excursion swings",
                "  the hole-edge stress between:",
                "",
                f"    peak tension (dried)       {self.peak_tension:8.2f} MPa",
                f"    peak compression (wet)     {self.peak_compression:8.2f} MPa",
                f"    stress range               {self.stress_range:8.2f} MPa",
                f"    amplitude                  {self.stress_amplitude:8.2f} MPa",
                f"    mean stress                {self.mean_stress:8.2f} MPa",
                f"    R = sigma_min / sigma_max  {self.r_ratio:8.2f}",
                "",
                "  Amplitude as a fraction of PA66 tensile strength:",
                f"    strongest UTS {self.amplitude_over_uts('high'):.2f}   "
                f"nominal {self.amplitude_over_uts('nominal'):.2f}   "
                f"weakest UTS {self.amplitude_over_uts('low'):.2f}",
                "",
                f"  Bulk half-cycle time: {self.bulk_half_time / SECONDS_PER_DAY:.1f} d.",
            ]
        )

    def commentary(self) -> list[str]:
        """Prose for the run script. Returns unwrapped paragraphs."""
        out = []
        if self.is_significant:
            out.append(
                "WORTH CHASING. The moisture cycle alone drives a stress amplitude "
                f"of {self.stress_amplitude:.0f} MPa at R = {self.r_ratio:.1f}, which is "
                f"{self.amplitude_over_uts('high'):.0%} to "
                f"{self.amplitude_over_uts('low'):.0%} of PA66's tensile strength. "
                "Unfilled PA66's fully-reversed fatigue strength at 1e6 cycles is "
                "typically 20-30% of UTS, so this sits in or above the finite-life "
                "band -- from humidity alone, with the mechanical duty cycle "
                "completely unchanged. A fatigue assessment that only counts "
                "ratchet tightening cycles would miss it entirely."
            )
        else:
            out.append(
                f"Probably not the driver on its own. The moisture cycle swings the "
                f"hole edge by {self.stress_range:.0f} MPa at R = {self.r_ratio:.1f}, "
                f"an amplitude of {self.amplitude_over_uts('high'):.0%} to "
                f"{self.amplitude_over_uts('low'):.0%} of UTS, which is at or below "
                "PA66's usual fully-reversed endurance band. Worth re-checking if the "
                "stress-free or property assumptions move."
            )

        out.append(
            "TWO TIME SCALES, AND THE FAST ONE IS AT THE CRACK SITE. The bulk takes "
            f"about {self.bulk_half_time / SECONDS_PER_DAY:.0f} days to go half way "
            "through a moisture change, so full bulk wet/dry cycles are seasonal and "
            "few. But a hole wall is an exposed surface at every depth, so it "
            "equilibrates with ambient humidity in hours. The hole edge is therefore "
            "simultaneously the most highly stressed location AND the one that cycles "
            "fastest -- it can see near-full-amplitude excursions daily while the bulk "
            "barely moves. Counting cycles off the bulk time constant would "
            "under-count by orders of magnitude."
        )

        out.append(
            "WHY THIS IS NOT A FATIGUE ANALYSIS. There is no cycle counting here, no "
            "S-N or Paris-law data for the actual grade at the right moisture state, "
            "no account of the frequency and temperature effects that dominate polymer "
            "fatigue, no hysteretic self-heating, and no crack-growth model. The stress "
            "range above is an elastic, single-excursion result -- it says a cyclic "
            "driver exists and roughly how big it is, not how long the part lasts. "
            "Sizing that is Tier 3."
        )
        return out


def moisture_cycle_assessment(
    absorption: StressHistory, desorption: StressHistory
) -> MoistureCycleAssessment:
    """Combine a wetting and a drying case into the cycle they jointly describe.

    The tensile half comes from the drying case and the compressive half from
    the wetting case, because each case only traverses its own direction.
    """
    tension = float(np.max(desorption.peak_tensile_envelope()))
    compression = float(np.min(absorption.peak_compressive_envelope()))

    peak = int(np.argmax(desorption.peak_tensile_envelope()))
    holes = desorption.representative_holes()
    moisture_at_peak = float(np.mean([h.mean_concentration[peak] for h in holes]))
    uts = tuple(
        float(pa66_properties(moisture_at_peak, corner=corner)[3])
        for corner in ("low", "nominal", "high")
    )

    return MoistureCycleAssessment(
        peak_tension=tension,
        peak_compression=compression,
        uts_low=uts[0],
        uts_nominal=uts[1],
        uts_high=uts[2],
        bulk_half_time=desorption.diffusion.model.characteristic_time(),
    )


def relaxation_bracket(
    strip: StripMesh,
    history: StressHistory,
    conditioned_moisture: float,
    fractions: tuple = (0.0, 0.5, 1.0),
) -> dict:
    """Peak hole-edge tension against the assumed stress-free state.

    Re-solves the mechanics at the instant of peak tension for each relaxation
    fraction, so the reader can see how much of the answer rests on an
    assumption nobody has measured. One extra solve per fraction, not a whole
    extra transient.
    """
    from dataclasses import replace as dc_replace

    from . import mechanics as mech

    peak = int(np.argmax(history.peak_tensile_envelope()))
    concentration = history.diffusion.concentration[history.indices[peak]]
    geometry = history.geometry

    out = {}
    for fraction in fractions:
        stress_free = mech.relaxed_stress_free_moisture(conditioned_moisture, fraction)
        model = dc_replace(history.mechanics_model, stress_free_moisture=stress_free)
        result = mech.solve(strip, model, concentration)
        edges = [
            result.hole_edges[name]["max_principal"]
            for i, name in enumerate(strip.hole_boundaries)
            if geometry.is_interior_hole(i)
        ] or [e["max_principal"] for e in result.hole_edges.values()]
        out[fraction] = {
            "stress_free_moisture": stress_free,
            "peak_tensile": float(np.max(edges)),
        }
    return out


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _figure(figsize=(9.0, 5.5)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.figure(figsize=figsize)


def plot_hole_stress_history(history: StressHistory, path: Path) -> Path:
    """Peak stress at every hole edge against time."""
    plt, fig = _figure()
    ax = fig.add_subplot(111)
    days = np.maximum(history.times_days, history.times_days[1] / 10.0)

    for hole in history.holes:
        style = "-" if hole.is_interior else "--"
        label = f"{hole.name}{'' if hole.is_interior else ' (end)'}"
        ax.plot(days, hole.max_principal, style, label=f"{label} max principal")
    ax.plot(
        days,
        history.peak_compressive_envelope(),
        "k:",
        lw=2,
        label="interior min principal (envelope)",
    )

    ax.set_xscale("log")
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("time (days, log scale)")
    ax.set_ylabel("hole-edge principal stress (MPa)")
    ax.set_title(
        f"Hole-edge stress vs time -- {history.diffusion.model.label}\n"
        "tension positive; solid = interior holes, dashed = end holes"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_stress_vs_uptake(history: StressHistory, path: Path) -> Path:
    """Hole-edge stress and yield utilisation against global moisture uptake."""
    plt, fig = _figure(figsize=(9.0, 6.5))
    ax1 = fig.add_subplot(211)
    uptake = history.uptake_fraction

    ax1.plot(uptake, history.peak_tensile_envelope(), "-o", ms=3, label="max principal (tension)")
    ax1.plot(uptake, history.peak_compressive_envelope(), "-s", ms=3, label="min principal (compression)")
    ax1.plot(uptake, history.peak_von_mises_envelope(), "-^", ms=3, label="von Mises")
    if np.isfinite(history.bulk_sxx).any():
        ax1.plot(uptake, history.bulk_sxx, "--", color="0.4", label="reinforced bulk sigma_xx")
    ax1.axhline(0.0, color="0.6", lw=0.8)
    ax1.set_ylabel("stress (MPa)")
    ax1.set_title(
        f"Hole-edge stress vs moisture uptake -- {history.diffusion.model.label}\n"
        "(interior holes; tension positive)"
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(212, sharex=ax1)
    util = history.yield_utilisation_envelope()
    ax2.plot(uptake, util, "-o", ms=3, color="firebrick")
    ax2.axhline(1.0, color="k", ls="--", lw=1.0)
    ax2.text(
        0.02, 1.02, "yield", transform=ax2.get_yaxis_transform(), fontsize=8, va="bottom"
    )
    ax2.set_xlabel("global moisture uptake  M(t) / M(inf)")
    ax2.set_ylabel("von Mises / yield")
    ax2.set_title(
        "Yield utilisation, using the yield strength at the local moisture content",
        fontsize=9,
    )
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_tier1_comparison(comparison: Tier1Comparison, path: Path) -> Path:
    """Bar chart of the three Tier 1 / Tier 2 cross-checks."""
    plt, fig = _figure(figsize=(8.0, 4.5))
    ax = fig.add_subplot(111)

    labels = ["Kt at hole", "constrained\nswelling stress", "time to\nhalf uptake"]
    t1 = [
        comparison.kt_tier1,
        comparison.swelling_tier1,
        comparison.half_time_tier1 / SECONDS_PER_DAY,
    ]
    t2 = [
        comparison.kt_tier2_interior,
        comparison.swelling_tier2_bulk,
        comparison.half_time_tier2 / SECONDS_PER_DAY,
    ]
    # Normalise each pair to the Tier 1 value so three different units share
    # one axis; the raw numbers are annotated on the bars.
    norm_t1 = [1.0] * 3
    norm_t2 = [
        (b / a) if a not in (0.0,) and np.isfinite(a) and np.isfinite(b) else np.nan
        for a, b in zip(t1, t2)
    ]

    x = np.arange(3)
    ax.bar(x - 0.18, norm_t1, width=0.34, label="Tier 1 (closed form)", color="steelblue")
    ax.bar(x + 0.18, norm_t2, width=0.34, label="Tier 2 (FE)", color="darkorange")
    for i, (a, b) in enumerate(zip(t1, t2)):
        ax.text(i - 0.18, 1.02, f"{a:.3g}", ha="center", fontsize=8)
        if np.isfinite(norm_t2[i]):
            ax.text(i + 0.18, norm_t2[i] + 0.02, f"{b:.3g}", ha="center", fontsize=8)

    ax.axhspan(1 / comparison.tolerance, comparison.tolerance, color="green", alpha=0.08)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("value / Tier 1 value")
    ax.set_title(
        "Tier 2 FE against Tier 1 closed form\n"
        f"shaded band = agreement within a factor of {comparison.tolerance:g}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_field(
    strip: StripMesh,
    nodal_values: np.ndarray,
    path: Path,
    title: str,
    label: str,
    cmap: str = "viridis",
) -> Path:
    """Contour map of a nodal field over the perforated strip."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    mesh = strip.mesh
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

    g = strip.geometry
    aspect = g.modelled_length / max(g.strap_width, 1e-9)
    fig = plt.figure(figsize=(min(14.0, 3.0 + 1.6 * aspect), 3.6))
    ax = fig.add_subplot(111)
    art = ax.tripcolor(tri, nodal_values, shading="gouraud", cmap=cmap)
    ax.set_aspect("equal")
    ax.set_xlabel("x along strap (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(art, ax=ax, label=label, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_moisture_cycle(
    absorption: StressHistory,
    desorption: StressHistory,
    cycle: MoistureCycleAssessment,
    path: Path,
) -> Path:
    """Both halves of a wet/dry excursion on one stress axis, against log time.

    Two things the figure has to show:

    * the vertical extent -- the strap traverses a large fraction of its
      tensile strength at the hole edge with no change in mechanical load;
    * the two time scales -- the hole wall is an exposed surface, so it
      slams to its new stress within hours, while the reinforced bulk takes
      weeks to follow. Plotting against uptake fraction hides this, because
      the hole wall is done inside the first few percent.
    """
    plt, fig = _figure(figsize=(10.0, 5.8))
    ax = fig.add_subplot(111)

    def _days(history):
        # Substitute a small positive value for t = 0 so it is visible on a
        # log axis rather than silently dropped.
        days = history.times_days.copy()
        if len(days) > 1 and days[0] <= 0:
            days[0] = days[1] / 5.0
        return days

    for history, label, colour in (
        (absorption, "absorption (wetting)", "tab:blue"),
        (desorption, "desorption (drying)", "tab:red"),
    ):
        days = _days(history)
        ax.plot(
            days,
            history.peak_tensile_envelope(),
            "-o",
            ms=3.5,
            color=colour,
            label=f"{label} -- hole edge",
        )
        ax.plot(
            days,
            history.bulk_sxx,
            "--",
            lw=1.4,
            color=colour,
            alpha=0.6,
            label=f"{label} -- reinforced bulk",
        )

    ax.axhline(0.0, color="0.5", lw=0.9)
    ax.axhspan(
        cycle.peak_compression, cycle.peak_tension, color="0.75", alpha=0.22, zorder=0
    )
    ax.axhline(
        cycle.uts_low,
        color="firebrick",
        ls=":",
        lw=1.1,
        label="+/- UTS (weak end of bracket)",
    )
    ax.axhline(-cycle.uts_low, color="firebrick", ls=":", lw=1.1)

    ax.set_xscale("log")
    ax.annotate(
        f"cycle range {cycle.stress_range:.0f} MPa   R = {cycle.r_ratio:.2f}\n"
        f"amplitude {cycle.stress_amplitude:.0f} MPa "
        f"= {cycle.amplitude_over_uts('nominal'):.0%} of UTS",
        xy=(0.5, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="0.6", alpha=0.92),
    )

    bulk_days = cycle.bulk_half_time / SECONDS_PER_DAY
    ax.axvline(bulk_days, color="0.35", lw=1.0, ls="-.")
    ax.annotate(
        f"bulk half-response {bulk_days:.0f} d",
        xy=(bulk_days, 0.97),
        xycoords=ax.get_xaxis_transform(),
        xytext=(5, -2),
        textcoords="offset points",
        fontsize=7.5,
        color="0.3",
        va="top",
    )

    ax.set_xlabel("time (days, log scale)")
    ax.set_ylabel("principal stress (MPa)")
    ax.set_title(
        "One wet/dry moisture cycle -- tension positive\n"
        "no mechanical load changes anywhere on this plot; "
        "the hole edge leads the bulk by weeks"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------------


def write_summary(
    history: StressHistory,
    comparison: Tier1Comparison,
    verdict: FreeEdgeVerdict,
    path: Path,
    tensile: "TensileAssessment | None" = None,
    cycle: "MoistureCycleAssessment | None" = None,
) -> Path:
    """Dump the headline numbers as JSON, for diffing between runs."""
    payload = {
        "case": history.diffusion.model.label,
        "bracket_corner": history.mechanics_model.corner,
        "geometry": {
            k: v
            for k, v in history.geometry.__dict__.items()
            if isinstance(v, (int, float, bool))
        },
        "diffusion": {
            "diffusivity_mm2_s": history.diffusion.model.diffusivity,
            "c_initial": history.diffusion.model.c_initial,
            "c_surface": history.diffusion.model.c_surface,
            "half_uptake_days": history.diffusion.time_to_uptake(0.5) / SECONDS_PER_DAY,
            "final_uptake_fraction": float(history.uptake_fraction[-1]),
        },
        "holes": {
            h.name: {
                "interior": h.is_interior,
                "peak_tensile_MPa": h.peak_tensile,
                "peak_compressive_MPa": h.peak_compressive,
                "peak_von_mises_MPa": h.peak_von_mises,
            }
            for h in history.holes
        },
        "envelope": {
            "peak_tensile_MPa": float(np.max(history.peak_tensile_envelope())),
            "peak_compressive_MPa": float(np.min(history.peak_compressive_envelope())),
            "peak_von_mises_MPa": float(np.max(history.peak_von_mises_envelope())),
            "peak_yield_utilisation": float(np.max(history.yield_utilisation_envelope())),
        },
        "reinforced_bulk": {
            "final_sxx_MPa": float(history.bulk_sxx[-1]),
            "final_concentration": float(history.bulk_concentration[-1]),
        },
        "tier1_cross_check": {
            "kt_tier1": comparison.kt_tier1,
            "kt_tier2_interior": comparison.kt_tier2_interior,
            "kt_tier2_end": comparison.kt_tier2_end,
            "kt_ratio": comparison.kt_ratio,
            "swelling_tier1_MPa": comparison.swelling_tier1,
            "swelling_tier2_bulk_MPa": comparison.swelling_tier2_bulk,
            "swelling_ratio": comparison.swelling_ratio,
            "half_time_tier1_days": comparison.half_time_tier1 / SECONDS_PER_DAY,
            "half_time_tier2_days": comparison.half_time_tier2 / SECONDS_PER_DAY,
            "half_time_ratio": comparison.half_time_ratio,
            "agrees": comparison.agrees,
        },
        "model_validity": {
            "free_edge_boundary_layer_mm": verdict.boundary_layer,
            "hole_edge_swelling_stress_reliable": verdict.hole_edge_is_reliable,
            "message": verdict.message(),
        },
        "series": {
            "time_days": history.times_days.tolist(),
            "uptake_fraction": history.uptake_fraction.tolist(),
            "hole_edge_max_principal_MPa": history.peak_tensile_envelope().tolist(),
            "hole_edge_min_principal_MPa": history.peak_compressive_envelope().tolist(),
            "hole_edge_von_mises_MPa": history.peak_von_mises_envelope().tolist(),
        },
    }
    if tensile is not None:
        payload["peak_tensile"] = {
            "stress_MPa": tensile.peak_stress,
            "time_days": tensile.peak_time / SECONDS_PER_DAY,
            "uptake_fraction_at_peak": tensile.peak_uptake,
            "moisture_at_peak": tensile.moisture_at_peak,
            "equilibrium_stress_MPa": tensile.equilibrium_stress,
            "gradient_contribution_MPa": tensile.gradient_contribution,
            "stress_free_moisture": tensile.stress_free_moisture,
            "yield_low_MPa": tensile.yield_low,
            "yield_nominal_MPa": tensile.yield_nominal,
            "yield_high_MPa": tensile.yield_high,
            "utilisation_low": tensile.utilisation_low,
            "utilisation_nominal": tensile.utilisation_nominal,
            "utilisation_high": tensile.utilisation_high,
            "reaches_yield": tensile.reaches_yield,
        }

    if cycle is not None:
        payload["moisture_cycle"] = {
            "peak_tension_MPa": cycle.peak_tension,
            "peak_compression_MPa": cycle.peak_compression,
            "stress_range_MPa": cycle.stress_range,
            "stress_amplitude_MPa": cycle.stress_amplitude,
            "mean_stress_MPa": cycle.mean_stress,
            "r_ratio": cycle.r_ratio,
            "amplitude_over_uts_low": cycle.amplitude_over_uts("low"),
            "amplitude_over_uts_nominal": cycle.amplitude_over_uts("nominal"),
            "amplitude_over_uts_high": cycle.amplitude_over_uts("high"),
            "bulk_half_cycle_days": cycle.bulk_half_time / SECONDS_PER_DAY,
            "is_significant": cycle.is_significant,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path
