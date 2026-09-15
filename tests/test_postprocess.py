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
    Tier1Comparison,
    bulk_element_mask,
    compare_with_tier1,
    free_edge_validity,
    plot_field,
    plot_hole_stress_history,
    plot_stress_vs_uptake,
    plot_tier1_comparison,
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
