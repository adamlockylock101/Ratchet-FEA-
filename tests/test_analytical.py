"""Handbook LEFM: stress concentration, Newman's crack-from-hole fits, validity.

These formulae are the reference the FE model is checked against, so they are
tested against their own textbook limits rather than against the FE.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from ratchet_fea.analytical import (
    EDGE_CRACK_FACTOR,
    critical_crack_length,
    feddersen_width_correction,
    gross_section_stress,
    handbook_k,
    handbook_k_curve,
    kt_gross_from_net,
    kt_hole_in_finite_width_strip,
    lefm_validity,
    mechanical_peak_stress,
    net_section_stress,
    newman_double_crack_factor,
    newman_single_crack_factor,
    row_interaction_note,
    screen,
)
from ratchet_fea.geometry import CrackOrientation
from ratchet_fea.materials import MoistureCondition


class TestStressConcentration:
    def test_reduces_to_kirsch_for_an_infinitely_wide_plate(self):
        assert kt_hole_in_finite_width_strip(0.0) == pytest.approx(3.0)

    def test_falls_as_the_hole_grows_relative_to_the_width(self):
        values = [kt_hole_in_finite_width_strip(x) for x in (0.0, 0.1, 0.2, 0.3, 0.4)]
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_known_value_at_a_quarter_width(self):
        expected = 3.00 - 3.13 * 0.25 + 3.66 * 0.0625 - 1.53 * 0.015625
        assert kt_hole_in_finite_width_strip(0.25) == pytest.approx(expected)

    def test_rejects_arguments_outside_the_fitted_range(self):
        with pytest.raises(ValueError, match="validity range"):
            kt_hole_in_finite_width_strip(0.6)

    def test_gross_conversion(self):
        assert kt_gross_from_net(2.5, 0.2) == pytest.approx(2.5 / 0.8)

    @pytest.mark.parametrize(
        "ratio,expected", [(1.5, "NOT reliable"), (2.5, "mild"), (4.0, "isolated")]
    )
    def test_row_note_reflects_the_pitch(self, ratio, expected):
        assert expected in row_interaction_note(ratio)


class TestFarFieldStress:
    def test_gross_uses_the_full_section(self, geometry):
        assert gross_section_stress(750.0, geometry) == pytest.approx(
            750.0 / geometry.gross_section_area
        )

    def test_net_exceeds_gross(self, geometry):
        assert net_section_stress(750.0, geometry) > gross_section_stress(
            750.0, geometry
        )

    def test_k_is_driven_by_the_gross_stress(self, geometry):
        """Handbook solutions are written in terms of the remote stress, not
        the concentrated stress at the hole -- the concentration is already in
        the geometry factor."""
        k = float(handbook_k(1.0, geometry, force=500.0))
        doubled = float(handbook_k(1.0, geometry, force=1000.0))
        assert doubled == pytest.approx(2.0 * k)

    def test_uncracked_peak_is_kt_times_net(self, geometry):
        peak, kt = mechanical_peak_stress(geometry.service_tension, geometry)
        assert peak == pytest.approx(
            kt * net_section_stress(geometry.service_tension, geometry)
        )


class TestNewmanFits:
    """Both fits are pinned at both asymptotes; that is what makes them usable."""

    def test_single_crack_short_limit_is_an_edge_crack_in_three_sigma(self):
        # 1.1215 * 3 = 3.3645
        assert newman_single_crack_factor(1e-9, 2.0) == pytest.approx(
            EDGE_CRACK_FACTOR * 3.0, rel=0.01
        )

    def test_double_crack_short_limit_is_the_same(self):
        assert newman_double_crack_factor(1e-9, 2.0) == pytest.approx(
            EDGE_CRACK_FACTOR * 3.0, rel=1e-3
        )

    def test_single_crack_long_limit_is_one_over_root_two(self):
        """A hole with one crack behaves as a through crack of length 2r + a,
        so K -> sigma sqrt(pi a / 2) and F -> 1/sqrt(2)."""
        assert newman_single_crack_factor(1e8, 2.0) == pytest.approx(
            1.0 / math.sqrt(2.0), rel=1e-3
        )

    def test_double_crack_long_limit_is_one(self):
        """Two symmetric cracks behave as a central crack of half-length r + a."""
        assert newman_double_crack_factor(1e8, 2.0) == pytest.approx(1.0, rel=1e-3)

    def test_both_fall_monotonically_over_the_usable_range(self):
        """Up to a/r = 20, far beyond anything this strap can hold."""
        a = np.geomspace(1e-4, 40.0, 400)
        for fit in (newman_single_crack_factor, newman_double_crack_factor):
            f = fit(a, 2.0)
            assert np.all(np.diff(f) <= 1e-9)

    def test_the_single_crack_fit_wiggles_only_far_outside_that(self):
        """Pins the documented quartic artefact, so it cannot grow unnoticed.

        The fit dips slightly below its 1/sqrt(2) asymptote at very large a/r
        and returns. Harmless here; worth knowing about before reusing the fit.
        """
        a = np.geomspace(40.0, 1e4, 500)
        f = newman_single_crack_factor(a, 2.0)
        asymptote = 1.0 / math.sqrt(2.0)
        assert np.max(np.abs(f - asymptote)) / asymptote < 0.01

    def test_they_agree_at_short_crack_lengths(self, geometry):
        """Both are just an edge crack in the hole's hoop stress there."""
        a = 1e-4
        assert newman_single_crack_factor(a, 2.0) == pytest.approx(
            newman_double_crack_factor(a, 2.0), rel=0.02
        )

    def test_they_diverge_at_long_crack_lengths(self):
        assert newman_single_crack_factor(100.0, 2.0) < newman_double_crack_factor(
            100.0, 2.0
        )

    def test_depend_only_on_the_ratio(self):
        assert newman_single_crack_factor(2.0, 1.0) == pytest.approx(
            newman_single_crack_factor(4.0, 2.0)
        )

    def test_reject_non_positive_input(self):
        with pytest.raises(ValueError, match="crack length must be positive"):
            newman_single_crack_factor(0.0, 2.0)
        with pytest.raises(ValueError, match="hole radius must be positive"):
            newman_single_crack_factor(1.0, 0.0)


class TestWidthCorrection:
    def test_tends_to_one_for_a_small_flaw(self):
        assert feddersen_width_correction(1e-6, 25.0) == pytest.approx(1.0, abs=1e-9)

    def test_grows_as_the_flaw_consumes_the_section(self):
        values = [feddersen_width_correction(c, 25.0) for c in (2.0, 5.0, 10.0)]
        assert values[0] < values[1] < values[2]

    def test_is_singular_at_the_edge(self):
        with pytest.raises(ValueError, match="singular"):
            feddersen_width_correction(12.5, 25.0)

    def test_matches_the_closed_form(self):
        assert feddersen_width_correction(5.0, 25.0) == pytest.approx(
            math.sqrt(1.0 / math.cos(math.pi * 0.2))
        )


class TestHandbookK:
    def test_rises_with_crack_length(self, geometry):
        a = np.linspace(0.2, 8.0, 40)
        k = handbook_k(a, geometry)
        assert np.all(np.diff(k) > 0)

    def test_units_are_mpa_root_mm(self, geometry):
        """K = F sigma sqrt(pi a) with sigma in MPa and a in mm."""
        a = 2.0
        expected = (
            newman_single_crack_factor(a, geometry.hole_radius)
            * gross_section_stress(geometry.service_tension, geometry)
            * math.sqrt(math.pi * a)
            * feddersen_width_correction(geometry.hole_radius + a, geometry.strap_width)
        )
        assert float(handbook_k(a, geometry)) == pytest.approx(expected)

    def test_finite_width_raises_k(self, geometry):
        assert float(handbook_k(4.0, geometry, finite_width=True)) > float(
            handbook_k(4.0, geometry, finite_width=False)
        )

    def test_is_zero_for_a_longitudinal_crack(self, geometry):
        """No mode I driving force exists for a crack parallel to the load."""
        k = handbook_k(
            np.array([0.5, 2.0, 5.0]),
            geometry,
            orientation=CrackOrientation.LONGITUDINAL,
        )
        assert np.all(k == 0.0)

    def test_curve_stops_short_of_the_ligament(self, geometry):
        a, k = handbook_k_curve(geometry, max_fraction=0.8)
        assert a[-1] <= 0.8 * geometry.side_ligament + 1e-9
        assert len(a) == len(k)


class TestCriticalCrackLength:
    def test_returns_nan_when_k_never_reaches_toughness(self, geometry):
        assert math.isnan(critical_crack_length(geometry, 1e6))

    def test_finds_the_crossing_when_there_is_one(self, geometry):
        target = 40.0
        a_c = critical_crack_length(geometry, target)
        assert not math.isnan(a_c)
        assert float(handbook_k(a_c, geometry)) == pytest.approx(target, rel=0.02)

    def test_a_tougher_material_needs_a_longer_crack(self, geometry):
        assert critical_crack_length(geometry, 60.0) > critical_crack_length(
            geometry, 40.0
        )

    def test_a_higher_load_shortens_it(self, geometry):
        assert critical_crack_length(geometry, 50.0, force=1500.0) < critical_crack_length(
            geometry, 50.0, force=500.0
        )


class TestLefmValidity:
    def test_plastic_zone_grows_with_k(self, geometry):
        small = lefm_validity(2.0, 20.0, geometry)
        large = lefm_validity(2.0, 60.0, geometry)
        assert large.plastic_zone_plane_stress > small.plastic_zone_plane_stress

    def test_plane_strain_zone_is_a_third_of_the_plane_stress_one(self, geometry):
        v = lefm_validity(2.0, 30.0, geometry)
        assert v.plastic_zone_plane_strain == pytest.approx(
            v.plastic_zone_plane_stress / 3.0
        )

    def test_this_strap_is_too_thin_for_plane_strain(self, geometry):
        """A 3 mm section against a ~20 mm ASTM requirement. Worth saying out
        loud: it makes the K_IC comparison conservative, not invalid."""
        v = lefm_validity(2.0, 25.0, geometry)
        assert not v.thickness_is_plane_strain
        assert v.astm_characteristic_size > geometry.strap_thickness

    def test_small_scale_yielding_holds_at_service_load(self, geometry):
        v = lefm_validity(2.0, 25.0, geometry)
        assert v.small_scale_yielding
        assert v.verdict == "plane-stress"

    def test_small_scale_yielding_fails_at_high_k(self, geometry):
        v = lefm_validity(0.2, 120.0, geometry)
        assert not v.small_scale_yielding
        assert v.verdict == "invalid"
        assert "SMALL-SCALE YIELDING DOES NOT HOLD" in v.message()

    def test_message_explains_the_direction_of_the_conservatism(self, geometry):
        assert "CONSERVATIVE" in lefm_validity(2.0, 25.0, geometry).message()


class TestScreen:
    @pytest.fixture(scope="class")
    def result(self, geometry):
        return screen(geometry=geometry)

    def test_defaults_to_the_transverse_crack(self, result):
        assert result.orientation is CrackOrientation.TRANSVERSE

    def test_defaults_to_the_service_condition(self, result):
        assert result.condition is MoistureCondition.RH50

    def test_reports_the_whole_toughness_bracket(self, result):
        assert set(result.toughness) == {"low", "nominal", "high"}
        assert result.toughness["low"] < result.toughness["high"]

    def test_k_does_not_reach_toughness_at_placeholder_values(self, result):
        """The headline handbook finding."""
        assert not result.propagates_at_any_length
        assert not result.propagates_at_observed
        assert all(math.isnan(v) for v in result.critical_lengths.values())

    def test_margin_at_the_observed_crack_is_several_fold(self, result):
        assert result.margin_at_observed("nominal") > 2.0

    def test_a_much_higher_load_does_make_it_propagate(self, geometry):
        """Confirms the screen can say yes, so the no above means something."""
        assert screen(geometry=geometry, force=20000.0).propagates_at_any_length

    def test_the_longitudinal_case_has_no_driving_force(self, geometry):
        result = screen(geometry=geometry, orientation=CrackOrientation.LONGITUDINAL)
        assert result.max_k == 0.0
        assert not result.propagates_at_any_length
        assert "K IS ZERO FOR THIS ORIENTATION" in result.summary()

    def test_a_drier_material_is_more_brittle(self, geometry):
        """Toughness falls with drying even though strength rises."""
        dry = screen(geometry=geometry, condition=MoistureCondition.DAM)
        wet = screen(geometry=geometry, condition=MoistureCondition.SATURATED)
        assert dry.toughness["nominal"] < wet.toughness["nominal"]

    def test_geometry_changes_move_the_answer(self, geometry):
        """Guards against the screen having frozen the placeholder geometry."""
        thin = replace(geometry, strap_thickness=1.0)
        assert screen(geometry=thin).k_at_observed > screen(
            geometry=geometry
        ).k_at_observed

    def test_summary_reports_the_bracket_and_the_units(self, result):
        text = result.summary()
        assert "K_IC" in text
        assert "MPa*sqrt(mm)" in text
        assert "31.62" in text
