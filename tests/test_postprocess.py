"""Result extraction, the Tier 1 cross-check, and the plots.

The cross-check is the safety net for the whole Tier 2 model, so its logic is
tested with constructed values as well as against a real solve.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ratchet_fea.analytical import screen
from ratchet_fea.diffusion import DiffusionModel, MoistureState, log_time_grid
from ratchet_fea.diffusion import solve as solve_diffusion
from ratchet_fea.mechanics import MechanicsModel, stress_concentration_factors
from ratchet_fea.postprocess import (
    SECONDS_PER_DAY,
    FreeEdgeVerdict,
    MoistureCycleAssessment,
    TensileAssessment,
    Tier1Comparison,
    bulk_element_mask,
    compare_with_tier1,
    free_edge_validity,
    plot_field,
    plot_hole_stress_history,
    plot_stress_vs_uptake,
    plot_tier1_comparison,
    moisture_cycle_assessment,
    peak_tensile_assessment,
    plot_moisture_cycle,
    relaxation_bracket,
    stress_history,
    write_summary,
)


class TestTier1Comparison:
    @staticmethod
    def _comparison(**overrides) -> Tier1Comparison:
        base = dict(
            kt_tier1=2.6,
            kt_tier2_interior=2.3,
            kt_tier2_end=2.5,
            swelling_tier1=31.0,
            swelling_tier2_bulk=30.0,
            half_time_tier1=8.0 * SECONDS_PER_DAY,
            half_time_tier2=8.5 * SECONDS_PER_DAY,
        )
        base.update(overrides)
        return Tier1Comparison(**base)

    def test_ratios(self):
        c = self._comparison()
        assert c.kt_ratio == pytest.approx(2.3 / 2.6)
        assert c.swelling_ratio == pytest.approx(30.0 / 31.0)
        assert c.half_time_ratio == pytest.approx(8.5 / 8.0)

    def test_close_values_agree(self):
        assert self._comparison().agrees

    def test_a_factor_of_ten_disagrees(self):
        assert not self._comparison(swelling_tier2_bulk=310.0).agrees

    def test_disagreement_is_symmetric(self):
        """Ten times too small must fail just as ten times too large does."""
        assert not self._comparison(swelling_tier2_bulk=3.1).agrees

    def test_tolerance_is_configurable(self):
        assert self._comparison(kt_tier2_interior=6.0, tolerance=1.5).agrees is False
        assert self._comparison(kt_tier2_interior=6.0, tolerance=3.0).agrees is True

    def test_a_nan_never_counts_as_agreement(self):
        assert not self._comparison(half_time_tier2=float("nan")).agrees

    def test_a_zero_reference_never_counts_as_agreement(self):
        assert not self._comparison(swelling_tier1=0.0).agrees

    def test_summary_reports_each_comparison(self):
        text = self._comparison().summary()
        for expected in ("Kt at hole", "constrained swelling", "half uptake", "ratio"):
            assert expected in text

    def test_summary_explains_a_low_kt_ratio(self):
        """A Tier 2 Kt below Tier 1 is the row effect, not an error."""
        assert "row-interaction" in self._comparison(kt_tier2_interior=2.2).summary()


class TestFreeEdgeVerdict:
    @staticmethod
    def _verdict(boundary_layer, ligament):
        return FreeEdgeVerdict(
            boundary_layer=boundary_layer,
            hole_radius=2.0,
            inter_hole_ligament=ligament,
            side_ligament=ligament * 2,
            bulk_element_fraction=0.3,
        )

    def test_a_thin_boundary_layer_is_reliable(self):
        v = self._verdict(0.5, 10.0)
        assert v.hole_edge_is_reliable
        assert not v.is_marginal
        assert "usable" in v.message()

    def test_a_comparable_boundary_layer_is_marginal(self):
        v = self._verdict(3.0, 8.0)
        assert not v.hole_edge_is_reliable
        assert v.is_marginal
        assert "MARGINAL" in v.message()

    def test_a_boundary_layer_wider_than_the_ligament_is_a_caution(self):
        v = self._verdict(6.0, 6.0)
        assert not v.hole_edge_is_reliable
        assert not v.is_marginal
        assert "UPPER BOUND" in v.message()

    def test_message_always_quotes_the_ratio(self):
        for bl in (0.5, 3.0, 6.0):
            assert "ratio" in self._verdict(bl, 8.0).message() or "UPPER BOUND" in self._verdict(bl, 8.0).message()

    def test_ratio_uses_the_narrowest_ligament(self):
        v = self._verdict(2.0, 8.0)
        assert v.narrowest_ligament == 8.0
        assert v.boundary_layer_ratio == pytest.approx(0.25)


@pytest.mark.requires_gmsh
class TestNoInteriorHoles:
    """A window too short to contain an interior hole must still report."""

    def test_kt_falls_back_to_the_mean_over_all_holes(
        self, two_hole_geometry, coarse_controls
    ):
        from ratchet_fea.mesh import build_strip_mesh

        strip = build_strip_mesh(two_hole_geometry, coarse_controls)
        kt = stress_concentration_factors(strip)
        assert not kt["has_interior_holes"]
        assert np.isnan(kt["interior_mean"])
        assert kt["representative_mean"] == pytest.approx(kt["all_mean"])

    def test_envelope_falls_back_to_all_holes(
        self, two_hole_geometry, coarse_controls
    ):
        from ratchet_fea.mesh import build_strip_mesh

        strip = build_strip_mesh(two_hole_geometry, coarse_controls)
        model = DiffusionModel.between(two_hole_geometry)
        times = log_time_grid(20 * model.characteristic_time(), n_steps=4)
        diffusion = solve_diffusion(strip, model, times)
        history = stress_history(
            strip, diffusion, MechanicsModel.from_materials(two_hole_geometry), every=2
        )
        assert history.interior_holes == []
        assert history.representative_holes() == history.holes
        assert np.all(np.isfinite(history.peak_tensile_envelope()))


@pytest.mark.requires_gmsh
class TestBulkMask:
    def test_excludes_material_next_to_a_hole(self, small_strip, small_geometry):
        mask = bulk_element_mask(small_strip)
        centroids = small_strip.element_centroids()
        clearance = small_geometry.strap_thickness
        for cx, cy in small_geometry.hole_centres:
            distance = np.hypot(centroids[0, mask] - cx, centroids[1, mask] - cy)
            assert np.all(distance > small_geometry.hole_radius + clearance - 1e-9)

    def test_excludes_material_next_to_a_free_side_edge(
        self, small_strip, small_geometry
    ):
        mask = bulk_element_mask(small_strip)
        y = small_strip.element_centroids()[1, mask]
        clearance = small_geometry.strap_thickness
        assert y.min() > clearance - 1e-9
        assert y.max() < small_geometry.strap_width - clearance + 1e-9

    def test_bulk_lies_inside_the_reinforced_region(self, small_strip):
        mask = bulk_element_mask(small_strip)
        assert np.all(small_strip.band_element_mask()[mask])

    def test_a_larger_clearance_keeps_less(self, small_strip):
        loose = bulk_element_mask(small_strip, clearance=0.5).sum()
        tight = bulk_element_mask(small_strip, clearance=4.0).sum()
        assert tight < loose

    def test_something_survives_for_this_geometry(self, small_strip):
        """If nothing does, there is no region where the CLT comparison holds."""
        assert bulk_element_mask(small_strip).any()

    def test_validity_report_matches_the_geometry(self, small_strip, small_geometry):
        v = free_edge_validity(small_strip)
        assert v.boundary_layer == pytest.approx(small_geometry.strap_thickness)
        assert v.inter_hole_ligament == pytest.approx(small_geometry.inter_hole_ligament)
        assert v.has_valid_bulk


@pytest.mark.requires_gmsh
class TestStressHistory:
    @pytest.fixture(scope="class")
    def history(self, small_strip, small_geometry):
        model = DiffusionModel.between(
            small_geometry, MoistureState.DRY, MoistureState.IMMERSED
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=8)
        diffusion = solve_diffusion(small_strip, model, times)
        mech = MechanicsModel.from_materials(small_geometry)
        return stress_history(small_strip, diffusion, mech, every=2)

    def test_shapes_are_consistent(self, history):
        n = len(history.times)
        assert history.uptake_fraction.shape == (n,)
        assert history.indices.shape == (n,)
        for hole in history.holes:
            assert hole.max_principal.shape == (n,)
            assert hole.von_mises.shape == (n,)

    def test_first_and_last_steps_are_always_kept(self, history):
        """Subsampling must not lose the dry state or the equilibrium state."""
        assert history.indices[0] == 0
        assert history.uptake_fraction[0] == pytest.approx(0.0, abs=1e-9)
        assert history.uptake_fraction[-1] == pytest.approx(1.0, abs=1e-3)

    def test_indices_point_at_the_matching_concentration_field(self, history):
        for k, i in enumerate(history.indices):
            assert history.diffusion.times[i] == pytest.approx(history.times[k])

    def test_every_hole_is_tracked_and_classified(self, history, small_geometry):
        assert [h.name for h in history.holes] == small_geometry.hole_labels
        for hole in history.holes:
            assert hole.is_interior == small_geometry.is_interior_hole(hole.index)

    def test_interior_holes_are_used_for_the_envelope(self, history):
        """End holes see the model truncation, so they are not representative."""
        assert len(history.interior_holes) == 1
        assert history.representative_holes() == history.interior_holes

    def test_envelopes_bound_the_individual_holes(self, history):
        tens = history.peak_tensile_envelope()
        comp = history.peak_compressive_envelope()
        for hole in history.representative_holes():
            assert np.all(hole.max_principal <= tens + 1e-9)
            assert np.all(hole.min_principal >= comp - 1e-9)

    def test_times_in_days_conversion(self, history):
        assert np.allclose(history.times_days, history.times / SECONDS_PER_DAY)

    def test_bulk_stress_is_compressive_and_grows_with_uptake(self, history):
        """Constrained swelling: more water, more compression."""
        assert history.bulk_sxx[-1] < 0
        assert history.bulk_sxx[-1] < history.bulk_sxx[0]

    def test_bulk_concentration_rises_to_saturation(self, history):
        assert history.bulk_concentration[-1] > history.bulk_concentration[0]
        assert history.bulk_concentration[-1] == pytest.approx(
            history.diffusion.model.c_surface, rel=0.05
        )

    def test_yield_utilisation_is_positive_and_finite(self, history):
        util = history.yield_utilisation_envelope()
        assert np.all(util >= 0)
        assert np.all(np.isfinite(util))

    def test_utilisation_uses_the_softened_yield_strength(self, history):
        """Dividing by the dry yield strength would understate the utilisation."""
        from ratchet_fea.materials import pa66_at_moisture

        vm = history.peak_von_mises_envelope()
        util = history.yield_utilisation_envelope()
        dry_yield = pa66_at_moisture(0.0)[2]
        assert util[-1] > vm[-1] / dry_yield

    def test_peak_helpers_agree_with_the_arrays(self, history):
        for hole in history.holes:
            assert hole.peak_tensile == pytest.approx(np.max(hole.max_principal))
            assert hole.peak_compressive == pytest.approx(np.min(hole.min_principal))
            assert hole.peak_von_mises == pytest.approx(np.max(hole.von_mises))


@pytest.mark.requires_gmsh
class TestAgainstTier1:
    @pytest.fixture(scope="class")
    def outcome(self, small_strip, small_geometry):
        model = DiffusionModel.between(
            small_geometry, MoistureState.DRY, MoistureState.IMMERSED
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=8)
        diffusion = solve_diffusion(small_strip, model, times)
        mech = MechanicsModel.from_materials(small_geometry)
        history = stress_history(small_strip, diffusion, mech, every=2)
        kt = stress_concentration_factors(small_strip)
        tier1 = screen(geometry=small_geometry, corners=("nominal",))
        return history, compare_with_tier1(history, tier1, small_strip, kt)

    def test_the_two_tiers_agree(self, outcome):
        """The headline sanity check the whole Tier 2 model rests on."""
        _, comparison = outcome
        assert comparison.agrees

    def test_swelling_stress_matches_closely(self, outcome):
        """In the reinforced bulk the FE solves the same problem as the formula."""
        _, comparison = outcome
        assert comparison.swelling_ratio == pytest.approx(1.0, rel=0.15)

    def test_fe_kt_is_below_the_isolated_hole_value(self, outcome):
        _, comparison = outcome
        assert comparison.kt_tier2_interior < comparison.kt_tier1

    def test_wetting_time_is_the_same_order(self, outcome):
        _, comparison = outcome
        assert comparison.half_time_ratio == pytest.approx(1.0, rel=0.4)

    def test_end_holes_carry_more_than_interior_ones(self, outcome):
        _, comparison = outcome
        assert comparison.kt_tier2_end >= comparison.kt_tier2_interior or np.isnan(
            comparison.kt_tier2_interior
        )


@pytest.mark.requires_gmsh
class TestOutputs:
    @pytest.fixture(scope="class")
    def bundle(self, small_strip, small_geometry):
        model = DiffusionModel.between(small_geometry)
        times = log_time_grid(20 * model.characteristic_time(), n_steps=6)
        diffusion = solve_diffusion(small_strip, model, times)
        mech = MechanicsModel.from_materials(small_geometry)
        history = stress_history(small_strip, diffusion, mech, every=2)
        kt = stress_concentration_factors(small_strip)
        tier1 = screen(geometry=small_geometry, corners=("nominal",))
        comparison = compare_with_tier1(history, tier1, small_strip, kt)
        return small_strip, history, comparison, free_edge_validity(small_strip)

    def test_summary_json_round_trips(self, bundle, tmp_path):
        _, history, comparison, verdict = bundle
        path = write_summary(history, comparison, verdict, tmp_path / "summary.json")
        data = json.loads(path.read_text())

        assert data["tier1_cross_check"]["agrees"] is True
        assert set(data["holes"]) == set(history.geometry.hole_labels)
        assert len(data["series"]["time_days"]) == len(history.times)
        assert data["model_validity"]["message"]

    def test_summary_records_the_geometry_that_produced_it(self, bundle, tmp_path):
        _, history, comparison, verdict = bundle
        path = write_summary(history, comparison, verdict, tmp_path / "s.json")
        geometry = json.loads(path.read_text())["geometry"]
        assert geometry["strap_width"] == history.geometry.strap_width
        assert geometry["hole_diameter"] == history.geometry.hole_diameter

    def test_summary_creates_missing_directories(self, bundle, tmp_path):
        _, history, comparison, verdict = bundle
        path = write_summary(
            history, comparison, verdict, tmp_path / "deep" / "er" / "s.json"
        )
        assert path.exists()

    @pytest.mark.parametrize(
        "plotter,name",
        [
            (plot_hole_stress_history, "history.png"),
            (plot_stress_vs_uptake, "uptake.png"),
        ],
    )
    def test_history_plots_are_written(self, bundle, tmp_path, plotter, name):
        _, history, _, _ = bundle
        path = plotter(history, tmp_path / name)
        assert path.exists() and path.stat().st_size > 1000

    def test_comparison_plot_is_written(self, bundle, tmp_path):
        _, _, comparison, _ = bundle
        path = plot_tier1_comparison(comparison, tmp_path / "cmp.png")
        assert path.exists() and path.stat().st_size > 1000

    def test_field_plot_is_written(self, bundle, tmp_path):
        strip, history, _, _ = bundle
        path = plot_field(
            strip,
            history.final_result.nodal_von_mises,
            tmp_path / "field.png",
            "title",
            "MPa",
        )
        assert path.exists() and path.stat().st_size > 1000



class TestTensileAssessmentLogic:
    """Unit-level, with constructed values."""

    @staticmethod
    def _assessment(**overrides) -> TensileAssessment:
        base = dict(
            case="desorption",
            peak_stress=32.0,
            peak_time=0.05 * SECONDS_PER_DAY,
            peak_uptake=0.08,
            moisture_at_peak=0.0,
            yield_low=75.0,
            yield_nominal=82.5,
            yield_high=90.0,
            equilibrium_stress=29.7,
            stress_free_moisture=0.025,
        )
        base.update(overrides)
        return TensileAssessment(**base)

    def test_utilisations(self):
        a = self._assessment()
        assert a.utilisation_low == pytest.approx(32.0 / 75.0)
        assert a.utilisation_nominal == pytest.approx(32.0 / 82.5)
        assert a.utilisation_high == pytest.approx(32.0 / 90.0)

    def test_the_weakest_yield_gives_the_highest_utilisation(self):
        """Easy to print backwards; utilisation_range fixes the order."""
        a = self._assessment()
        assert a.utilisation_low > a.utilisation_high
        best, worst = a.utilisation_range
        assert best < worst
        assert worst == pytest.approx(a.utilisation_low)

    def test_reaches_yield_uses_the_weakest_corner(self):
        assert not self._assessment(peak_stress=70.0).reaches_yield
        assert self._assessment(peak_stress=80.0).reaches_yield

    def test_gradient_contribution_is_the_transient_excess(self):
        a = self._assessment(peak_stress=32.0, equilibrium_stress=29.7)
        assert a.gradient_contribution == pytest.approx(2.3)

    def test_gradient_contribution_is_undefined_when_the_peak_is_the_start(self):
        """Otherwise a compressive case reports a huge fictitious 'gradient'."""
        a = self._assessment(
            peak_stress=1.4, equilibrium_stress=-30.0, peak_is_initial_state=True
        )
        assert a.gradient_contribution is None
        assert "STARTING state" in a.summary()

    def test_gradient_contribution_is_undefined_against_compressive_equilibrium(self):
        a = self._assessment(peak_stress=5.0, equilibrium_stress=-10.0)
        assert a.gradient_contribution is None

    def test_a_case_peaking_at_t_zero_does_not_develop_tension(self):
        """A small positive value from the mechanical load is not a tensile case."""
        a = self._assessment(peak_stress=1.4, peak_is_initial_state=True)
        assert a.is_tensile
        assert not a.develops_tension
        assert "does not drive the hole edge into tension" in a.summary()

    def test_a_case_peaking_mid_transient_does_develop_tension(self):
        a = self._assessment(peak_stress=32.0, peak_is_initial_state=False)
        assert a.develops_tension

    def test_a_compressive_case_is_reported_as_not_tensile(self):
        a = self._assessment(peak_stress=-12.0)
        assert not a.is_tensile
        assert not a.develops_tension
        assert "does not drive the hole edge into tension" in a.summary()

    def test_summary_flags_reaching_yield(self):
        assert "REACHES YIELD" in self._assessment(peak_stress=80.0).summary()

    def test_summary_reports_the_whole_bracket(self):
        text = self._assessment().summary()
        for expected in ("low", "nominal", "high", "stress-free moisture"):
            assert expected in text


class TestMoistureCycleLogic:
    @staticmethod
    def _cycle(**overrides) -> MoistureCycleAssessment:
        base = dict(
            peak_tension=31.6,
            peak_compression=-31.1,
            uts_low=75.0,
            uts_nominal=85.0,
            uts_high=95.0,
            bulk_half_time=8.7 * SECONDS_PER_DAY,
        )
        base.update(overrides)
        return MoistureCycleAssessment(**base)

    def test_range_amplitude_and_mean(self):
        c = self._cycle()
        assert c.stress_range == pytest.approx(62.7)
        assert c.stress_amplitude == pytest.approx(31.35)
        assert c.mean_stress == pytest.approx(0.25)

    def test_r_ratio_is_near_fully_reversed(self):
        assert self._cycle().r_ratio == pytest.approx(-31.1 / 31.6)

    def test_a_purely_tensile_cycle_has_r_zero(self):
        assert self._cycle(peak_compression=0.0).r_ratio == pytest.approx(0.0)

    def test_amplitude_over_uts_uses_the_named_corner(self):
        c = self._cycle()
        assert c.amplitude_over_uts("low") > c.amplitude_over_uts("high")
        assert c.amplitude_over_uts("low") == pytest.approx(31.35 / 75.0)

    def test_significance_needs_both_amplitude_and_a_negative_r(self):
        assert self._cycle().is_significant
        # Same amplitude but entirely in tension: not a reversed cycle.
        assert not self._cycle(
            peak_tension=62.7, peak_compression=0.0
        ).is_significant
        # Reversed but small.
        assert not self._cycle(peak_tension=5.0, peak_compression=-5.0).is_significant

    def test_commentary_always_covers_the_caveats(self):
        for cycle in (self._cycle(), self._cycle(peak_tension=5.0, peak_compression=-5.0)):
            text = " ".join(cycle.commentary())
            assert "WHY THIS IS NOT A FATIGUE ANALYSIS" in text
            assert "TWO TIME SCALES" in text
            assert "Tier 3" in text

    def test_commentary_changes_with_significance(self):
        assert "WORTH CHASING" in " ".join(self._cycle().commentary())
        assert "WORTH CHASING" not in " ".join(
            self._cycle(peak_tension=5.0, peak_compression=-5.0).commentary()
        )

    def test_summary_reports_the_cycle(self):
        text = self._cycle().summary()
        for expected in ("stress range", "amplitude", "R = sigma_min"):
            assert expected in text


@pytest.mark.requires_gmsh
class TestDesorptionCase:
    """End to end: conditioned strap, dried out, against the yield bracket."""

    @pytest.fixture(scope="class")
    def dried(self, small_strip, small_geometry):
        from ratchet_fea.mechanics import relaxed_stress_free_moisture

        model = DiffusionModel.between(
            small_geometry, MoistureState.RH50, MoistureState.DRY
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=8)
        diffusion = solve_diffusion(small_strip, model, times)
        mech = MechanicsModel.from_materials(
            small_geometry,
            stress_free_moisture=relaxed_stress_free_moisture(model.c_initial, 1.0),
        )
        return stress_history(small_strip, diffusion, mech, every=2)

    def test_the_hole_edge_goes_into_tension(self, dried):
        """The headline: shrinkage against the band reverses the sign."""
        assessment = peak_tensile_assessment(dried, case="desorption")
        assert assessment.develops_tension
        assert assessment.peak_stress > 10.0

    def test_the_bulk_ends_in_tension_too(self, dried):
        """Not just a hole-edge effect -- the whole reinforced section is pulled."""
        assert dried.bulk_sxx[-1] > 0

    def test_moisture_runs_downhill(self, dried):
        assert dried.diffusion.concentration[-1].mean() < dried.diffusion.concentration[0].mean()

    def test_the_transient_peak_is_at_or_above_equilibrium(self, dried):
        """A drying hole wall shrinks against a still-wet interior."""
        a = peak_tensile_assessment(dried)
        assert a.develops_tension
        assert a.gradient_contribution is not None
        assert a.gradient_contribution >= -1e-9
        assert a.peak_time <= dried.times[-1]

    def test_yield_bracket_is_taken_at_the_dried_state(self, dried):
        """A dried strap is stronger; using the wet yield would overstate risk."""
        from ratchet_fea.materials import pa66_at_moisture

        a = peak_tensile_assessment(dried)
        assert a.yield_low > pa66_at_moisture(0.025, "low")[2]
        assert a.yield_low < a.yield_nominal < a.yield_high

    def test_relaxation_bracket_spans_unloading_to_full_tension(
        self, small_strip, dried
    ):
        bracket = relaxation_bracket(
            small_strip, dried, conditioned_moisture=0.025, fractions=(0.0, 0.5, 1.0)
        )
        peaks = [bracket[f]["peak_tensile"] for f in (0.0, 0.5, 1.0)]
        assert peaks[0] < peaks[1] < peaks[2]
        assert bracket[0.0]["stress_free_moisture"] == 0.0
        assert bracket[1.0]["stress_free_moisture"] == pytest.approx(0.025)

    def test_without_relaxation_drying_barely_produces_tension(
        self, small_strip, dried
    ):
        """The purely elastic reading: drying only unloads the compression."""
        bracket = relaxation_bracket(
            small_strip, dried, conditioned_moisture=0.025, fractions=(0.0,)
        )
        assert bracket[0.0]["peak_tensile"] < 5.0


@pytest.mark.requires_gmsh
class TestCycleEndToEnd:
    @pytest.fixture(scope="class")
    def both(self, small_strip, small_geometry):
        from ratchet_fea.mechanics import relaxed_stress_free_moisture

        out = {}
        for name, start, end, relaxation in (
            ("absorption", MoistureState.DRY, MoistureState.IMMERSED, 0.0),
            ("desorption", MoistureState.RH50, MoistureState.DRY, 1.0),
        ):
            model = DiffusionModel.between(small_geometry, start, end)
            times = log_time_grid(20 * model.characteristic_time(), n_steps=6)
            diffusion = solve_diffusion(small_strip, model, times)
            mech = MechanicsModel.from_materials(
                small_geometry,
                stress_free_moisture=relaxed_stress_free_moisture(
                    model.c_initial, relaxation
                ),
            )
            out[name] = stress_history(small_strip, diffusion, mech, every=2)
        return out

    def test_the_two_cases_straddle_zero(self, both):
        """Wetting compresses, drying pulls -- that is what makes it a cycle."""
        assert np.min(both["absorption"].peak_compressive_envelope()) < 0
        assert np.max(both["desorption"].peak_tensile_envelope()) > 0

    def test_cycle_assessment_combines_them(self, both):
        cycle = moisture_cycle_assessment(both["absorption"], both["desorption"])
        assert cycle.peak_tension > 0 > cycle.peak_compression
        assert cycle.stress_range == pytest.approx(
            cycle.peak_tension - cycle.peak_compression
        )
        assert -2.0 < cycle.r_ratio < 0.0

    def test_the_cycle_is_a_meaningful_fraction_of_strength(self, both):
        cycle = moisture_cycle_assessment(both["absorption"], both["desorption"])
        assert 0.05 < cycle.amplitude_over_uts("nominal") < 1.5

    def test_uts_is_taken_where_the_tensile_peak_is(self, both):
        """Tension peaks dry, and dry PA66 is the strong state."""
        from ratchet_fea.materials import pa66_at_moisture

        cycle = moisture_cycle_assessment(both["absorption"], both["desorption"])
        assert cycle.uts_low > pa66_at_moisture(0.085, "low")[3]

    def test_cycle_plot_is_written(self, both, tmp_path):
        cycle = moisture_cycle_assessment(both["absorption"], both["desorption"])
        path = plot_moisture_cycle(
            both["absorption"], both["desorption"], cycle, tmp_path / "cycle.png"
        )
        assert path.exists() and path.stat().st_size > 1000

    def test_summary_json_carries_the_new_sections(self, both, small_strip, tmp_path):
        history = both["desorption"]
        kt = stress_concentration_factors(small_strip)
        tier1 = screen(geometry=history.geometry, corners=("nominal",))
        comparison = compare_with_tier1(history, tier1, small_strip, kt)
        cycle = moisture_cycle_assessment(both["absorption"], both["desorption"])
        tensile = peak_tensile_assessment(history, case="desorption")

        path = write_summary(
            history,
            comparison,
            free_edge_validity(small_strip),
            tmp_path / "s.json",
            tensile=tensile,
            cycle=cycle,
        )
        data = json.loads(path.read_text())
        assert data["peak_tensile"]["stress_MPa"] == pytest.approx(tensile.peak_stress)
        assert data["peak_tensile"]["stress_free_moisture"] == pytest.approx(0.025)
        assert data["moisture_cycle"]["r_ratio"] == pytest.approx(cycle.r_ratio)
        assert isinstance(data["moisture_cycle"]["is_significant"], bool)

    def test_summary_json_omits_the_sections_when_not_supplied(
        self, both, small_strip, tmp_path
    ):
        history = both["absorption"]
        kt = stress_concentration_factors(small_strip)
        tier1 = screen(geometry=history.geometry, corners=("nominal",))
        path = write_summary(
            history,
            compare_with_tier1(history, tier1, small_strip, kt),
            free_edge_validity(small_strip),
            tmp_path / "s.json",
        )
        data = json.loads(path.read_text())
        assert "peak_tensile" not in data
        assert "moisture_cycle" not in data


@pytest.mark.requires_gmsh
class TestAbsorptionIsNotATensileCase:
    """Guards the distinction the desorption case is defined against."""

    def test_wetting_never_drives_the_hole_edge_into_tension(
        self, small_strip, small_geometry
    ):
        model = DiffusionModel.between(
            small_geometry, MoistureState.DRY, MoistureState.IMMERSED
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=6)
        diffusion = solve_diffusion(small_strip, model, times)
        history = stress_history(
            small_strip,
            diffusion,
            MechanicsModel.from_materials(small_geometry, stress_free_moisture=0.0),
            every=2,
        )
        assessment = peak_tensile_assessment(history, case="absorption")
        assert not assessment.develops_tension
        assert assessment.peak_is_initial_state
        assert history.bulk_sxx[-1] < 0
