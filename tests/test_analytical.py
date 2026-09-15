"""Tier 1 closed-form screening.

These formulae are the reference the Tier 2 FE model is checked against, so
they are tested against their own textbook limits rather than against the FE.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from ratchet_fea.analytical import (
    Tier1Result,
    constrained_swelling_stress,
    fickian_half_time,
    kt_gross_from_net,
    kt_hole_in_finite_width_strip,
    mechanical_peak_stress,
    net_section_stress,
    row_interaction_note,
    screen,
)
from ratchet_fea.materials import PA66_MOISTURE


class TestStressConcentration:
    def test_reduces_to_kirsch_for_an_infinitely_wide_plate(self):
        """The classical result: Kt = 3 for a small hole in a wide plate."""
        assert kt_hole_in_finite_width_strip(0.0) == pytest.approx(3.0)

    def test_falls_as_the_hole_grows_relative_to_the_width(self):
        values = [kt_hole_in_finite_width_strip(x) for x in (0.0, 0.1, 0.2, 0.3, 0.4)]
        assert all(a > b for a, b in zip(values, values[1:]))

    def test_known_value_at_a_quarter_width(self):
        # 3.00 - 3.13(0.25) + 3.66(0.25)^2 - 1.53(0.25)^3
        expected = 3.00 - 3.13 * 0.25 + 3.66 * 0.0625 - 1.53 * 0.015625
        assert kt_hole_in_finite_width_strip(0.25) == pytest.approx(expected)

    def test_stays_in_a_sensible_range_over_its_validity(self):
        for x in (0.0, 0.1, 0.25, 0.4, 0.5):
            assert 2.0 < kt_hole_in_finite_width_strip(x) <= 3.0

    def test_rejects_arguments_outside_the_fitted_range(self):
        with pytest.raises(ValueError, match="validity range"):
            kt_hole_in_finite_width_strip(0.6)
        with pytest.raises(ValueError, match="validity range"):
            kt_hole_in_finite_width_strip(-0.1)

    def test_gross_conversion(self):
        assert kt_gross_from_net(2.5, 0.2) == pytest.approx(2.5 / 0.8)

    def test_gross_always_exceeds_net(self):
        for x in (0.1, 0.25, 0.4):
            kt = kt_hole_in_finite_width_strip(x)
            assert kt_gross_from_net(kt, x) > kt


class TestRowInteraction:
    @pytest.mark.parametrize(
        "ratio,expected", [(1.5, "NOT reliable"), (2.5, "mild"), (4.0, "isolated")]
    )
    def test_note_reflects_the_pitch(self, ratio, expected):
        assert expected in row_interaction_note(ratio)

    def test_note_is_prose_not_a_correction_factor(self):
        """Deliberate: a hand-waved factor must not end up inside a margin."""
        assert isinstance(row_interaction_note(2.5), str)


class TestMechanicalStress:
    def test_net_section_stress_is_force_over_net_area(self, geometry):
        s = net_section_stress(1000.0, geometry)
        assert s == pytest.approx(1000.0 / geometry.net_section_area)

    def test_peak_is_kt_times_net(self, geometry):
        peak, kt = mechanical_peak_stress(geometry.service_tension, geometry)
        assert kt == pytest.approx(kt_hole_in_finite_width_strip(geometry.d_over_W))
        assert peak == pytest.approx(
            kt * net_section_stress(geometry.service_tension, geometry)
        )

    def test_scales_linearly_with_load(self, geometry):
        a, _ = mechanical_peak_stress(100.0, geometry)
        b, _ = mechanical_peak_stress(300.0, geometry)
        assert b == pytest.approx(3.0 * a)

    def test_a_thicker_strap_carries_less_stress(self, geometry):
        thick = replace(geometry, strap_thickness=2 * geometry.strap_thickness)
        thin_peak, _ = mechanical_peak_stress(500.0, geometry)
        thick_peak, _ = mechanical_peak_stress(500.0, thick)
        assert thick_peak == pytest.approx(thin_peak / 2)


class TestConstrainedSwelling:
    def test_uniaxial_is_e_times_free_strain(self):
        s = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.02, constraint="uniaxial")
        assert s == pytest.approx(1000.0 * 0.25 * 0.02)

    def test_biaxial_adds_the_one_minus_nu_factor(self):
        s = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.02, constraint="biaxial")
        assert s == pytest.approx(1000.0 * 0.25 * 0.02 / 0.6)

    def test_triaxial_adds_the_one_minus_two_nu_factor(self):
        s = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.02, constraint="triaxial")
        assert s == pytest.approx(1000.0 * 0.25 * 0.02 / 0.2)

    def test_constraint_modes_are_ordered(self):
        args = (1000.0, 0.4, 0.25, 0.02)
        uni = constrained_swelling_stress(*args, constraint="uniaxial")
        bi = constrained_swelling_stress(*args, constraint="biaxial")
        tri = constrained_swelling_stress(*args, constraint="triaxial")
        assert uni < bi < tri

    def test_returns_a_magnitude_for_either_direction(self):
        """Desorption reverses the sign; the formula reports the magnitude."""
        wet = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.02)
        dry = constrained_swelling_stress(1000.0, 0.4, 0.25, -0.02)
        assert wet == pytest.approx(dry)
        assert wet > 0

    def test_scales_linearly_with_moisture_change(self):
        a = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.01)
        b = constrained_swelling_stress(1000.0, 0.4, 0.25, 0.03)
        assert b == pytest.approx(3.0 * a)

    def test_triaxial_is_singular_at_nu_one_half(self):
        with pytest.raises(ValueError, match="singular"):
            constrained_swelling_stress(1000.0, 0.5, 0.25, 0.02, constraint="triaxial")

    def test_unknown_constraint_is_rejected(self):
        with pytest.raises(ValueError, match="unknown constraint"):
            constrained_swelling_stress(1000.0, 0.4, 0.25, 0.02, constraint="sideways")


class TestFickianHalfTime:
    def test_matches_the_textbook_fourier_number(self):
        assert fickian_half_time(1.0, 1.0) == pytest.approx(0.19685)

    def test_scales_with_the_square_of_thickness(self):
        assert fickian_half_time(2.0, 1.0) == pytest.approx(
            4.0 * fickian_half_time(1.0, 1.0)
        )

    def test_scales_inversely_with_diffusivity(self):
        assert fickian_half_time(1.0, 2.0) == pytest.approx(
            0.5 * fickian_half_time(1.0, 1.0)
        )

    def test_rejects_non_positive_diffusivity(self):
        with pytest.raises(ValueError, match="must be positive"):
            fickian_half_time(1.0, 0.0)

    def test_placeholder_strap_wets_over_weeks_not_seconds_or_years(self, geometry):
        """Sanity check on the units: a 3 mm nylon strap takes weeks."""
        t = fickian_half_time(
            geometry.diffusion_half_thickness(True), PA66_MOISTURE.diffusivity.nominal
        )
        assert 2 * 86400 < t < 60 * 86400


class TestScreen:
    @pytest.fixture(scope="class")
    def result(self, geometry) -> Tier1Result:
        return screen(geometry=geometry)

    def test_covers_both_exposures_at_every_corner(self, result):
        assert len(result.cases) == 6
        assert {c.label for c in result.cases} == {"50% RH", "immersed"}
        assert {c.corner for c in result.cases} == {"low", "nominal", "high"}

    def test_mechanical_load_alone_is_benign(self, result):
        """The Tier 1 headline: the hole plus service load is not the problem."""
        assert result.mechanical_alone_is_benign

    def test_constrained_swelling_reaches_yield(self, result):
        """The other half of the headline, and the reason Tier 2 exists."""
        assert result.swelling_is_significant

    def test_immersed_is_worse_than_50rh(self, result):
        wet = result.case("immersed", "nominal")
        damp = result.case("50% RH", "nominal")
        assert wet.swelling_stress > damp.swelling_stress
        assert wet.moisture_content > damp.moisture_content

    def test_softening_is_included(self, result):
        """More water means more swelling but a softer polymer to resist it."""
        wet = result.case("immersed", "nominal")
        damp = result.case("50% RH", "nominal")
        assert wet.youngs_modulus < damp.youngs_modulus
        assert wet.yield_strength < damp.yield_strength

    def test_swelling_does_not_scale_with_moisture_alone(self, result):
        """Because softening partly cancels it -- worth confirming explicitly."""
        wet = result.case("immersed", "nominal")
        damp = result.case("50% RH", "nominal")
        moisture_ratio = wet.moisture_content / damp.moisture_content
        stress_ratio = wet.swelling_stress / damp.swelling_stress
        assert stress_ratio < moisture_ratio

    def test_utilisations_are_consistent(self, result):
        for c in result.cases:
            assert c.mechanical_utilisation == pytest.approx(
                c.mechanical_peak / c.yield_strength
            )
            assert c.combined_utilisation == pytest.approx(
                c.combined_peak / c.yield_strength
            )

    def test_combined_is_the_sum_of_the_parts(self, result):
        for c in result.cases:
            assert c.combined_peak == pytest.approx(c.mechanical_peak + c.swelling_stress)

    def test_kt_matches_the_geometry(self, result, geometry):
        assert result.kt_net == pytest.approx(
            kt_hole_in_finite_width_strip(geometry.d_over_W)
        )

    def test_half_time_bracket_straddles_the_nominal(self, result):
        lo, hi = result.half_time_bracket
        assert lo < result.half_time_seconds < hi

    def test_through_thickness_beats_in_plane_for_this_strap(self, result):
        """Justifies the product solution in diffusion.py.

        If the in-plane path were the faster one, a plane-stress in-plane model
        would be sufficient on its own.
        """
        assert result.half_time_seconds < result.in_plane_half_time_seconds

    def test_a_different_geometry_gives_a_different_answer(self, geometry):
        """Guards against the screen having frozen the placeholder geometry."""
        wider = replace(geometry, strap_width=40.0, steel_band_width=30.0)
        assert screen(geometry=wider).kt_net != pytest.approx(
            screen(geometry=geometry).kt_net
        )

    def test_corner_selection_restricts_the_cases(self, geometry):
        result = screen(geometry=geometry, corners=("nominal",))
        assert {c.corner for c in result.cases} == {"nominal"}

    def test_summary_reports_the_conclusions(self, result):
        text = result.summary()
        assert "CONCLUSIONS" in text
        assert "Kt" in text

    def test_case_lookup_rejects_an_unknown_case(self, result):
        with pytest.raises(KeyError):
            result.case("underwater", "nominal")
