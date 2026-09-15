"""PA66 and steel property brackets, and the moisture interpolation.

The interpolation is the load-bearing part: the coupled FE model evaluates the
polymer modulus from the local moisture at every quadrature point, so an error
here propagates into every stress in the study.
"""

from __future__ import annotations

import numpy as np
import pytest

from ratchet_fea.materials import (
    PA66_CONDITIONS,
    PA66_DAM,
    PA66_MOISTURE,
    PA66_RH50,
    PA66_SATURATED,
    STEEL_BAND,
    MoistureCondition,
    all_material_objects,
    pa66_at_moisture,
    pa66_properties,
)
from ratchet_fea.provenance import Provenance


class TestBrackets:
    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_elastic_brackets_are_physical(self, grade):
        assert grade.youngs_modulus.low > 0
        assert 0.0 < grade.poisson_ratio.low < 0.5
        assert grade.poisson_ratio.high < 0.5, "plane stress is singular at nu = 0.5"

    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_strength_brackets_are_physical(self, grade):
        assert grade.yield_strength.low > 0
        assert grade.tensile_strength.high >= grade.yield_strength.low

    def test_diffusivity_bracket_is_log_scaled(self):
        """It spans a decade; an arithmetic mean would bias every time scale."""
        assert PA66_MOISTURE.diffusivity.scale == "log"
        assert PA66_MOISTURE.diffusivity.spread == pytest.approx(10.0)

    def test_diffusivity_is_in_mm2_per_second(self):
        """1e-13..1e-12 m^2/s converts to 1e-7..1e-6 mm^2/s.

        A units slip here would move every predicted time by 10^6.
        """
        assert PA66_MOISTURE.diffusivity.low == pytest.approx(1e-7)
        assert PA66_MOISTURE.diffusivity.high == pytest.approx(1e-6)

    def test_immersed_uptake_exceeds_the_50rh_value(self):
        assert PA66_MOISTURE.saturation_immersed.low > PA66_MOISTURE.saturation_50rh.high

    def test_swelling_coefficient_is_plausible(self):
        """0.25 per unit mass fraction means 2.5 wt% gives ~0.6% linear strain."""
        beta = PA66_MOISTURE.swelling_coefficient.nominal
        assert 0.1 < beta < 0.5
        assert beta * 0.025 == pytest.approx(0.00625, rel=0.2)


class TestMoistureDependence:
    def test_modulus_falls_with_moisture(self):
        assert PA66_DAM.youngs_modulus.nominal > PA66_RH50.youngs_modulus.nominal
        assert PA66_RH50.youngs_modulus.nominal > PA66_SATURATED.youngs_modulus.nominal

    def test_conditioning_roughly_halves_the_modulus(self):
        """The classic PA66 fact the whole investigation turns on."""
        ratio = PA66_DAM.youngs_modulus.nominal / PA66_RH50.youngs_modulus.nominal
        assert 1.8 < ratio < 3.0

    def test_yield_strength_falls_with_moisture(self):
        assert PA66_DAM.yield_strength.nominal > PA66_RH50.yield_strength.nominal
        assert PA66_RH50.yield_strength.nominal > PA66_SATURATED.yield_strength.nominal

    def test_poisson_ratio_rises_with_moisture(self):
        assert PA66_DAM.poisson_ratio.nominal < PA66_SATURATED.poisson_ratio.nominal

    def test_moisture_contents_are_ordered(self):
        assert PA66_DAM.moisture_content == 0.0
        assert PA66_DAM.moisture_content < PA66_RH50.moisture_content
        assert PA66_RH50.moisture_content < PA66_SATURATED.moisture_content


class TestInterpolation:
    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_knots_reproduce_their_grade_exactly(self, grade):
        e, nu, sy, su = pa66_at_moisture(grade.moisture_content)
        assert e == pytest.approx(grade.youngs_modulus.nominal)
        assert nu == pytest.approx(grade.poisson_ratio.nominal)
        assert sy == pytest.approx(grade.yield_strength.nominal)
        assert su == pytest.approx(grade.tensile_strength.nominal)

    def test_interpolates_between_knots(self):
        mid = 0.5 * (PA66_DAM.moisture_content + PA66_RH50.moisture_content)
        e, _, _, _ = pa66_at_moisture(mid)
        assert PA66_RH50.youngs_modulus.nominal < e < PA66_DAM.youngs_modulus.nominal

    def test_clamps_above_saturation_rather_than_extrapolating(self):
        """Linear extrapolation would reach a negative modulus soon enough."""
        assert pa66_at_moisture(0.5)[0] == pytest.approx(
            PA66_SATURATED.youngs_modulus.nominal
        )

    def test_clamps_at_zero(self):
        assert pa66_at_moisture(0.0)[0] == pytest.approx(
            PA66_DAM.youngs_modulus.nominal
        )

    def test_modulus_is_monotone_decreasing(self):
        c = np.linspace(0.0, 0.12, 50)
        e = pa66_properties(c)[0]
        assert np.all(np.diff(e) <= 1e-9)

    def test_corner_selection_changes_the_answer(self):
        low = pa66_at_moisture(0.025, corner="low")[0]
        high = pa66_at_moisture(0.025, corner="high")[0]
        assert low < high

    def test_negative_moisture_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            pa66_at_moisture(-0.01)

    def test_vectorised_matches_scalar(self):
        c = np.array([0.0, 0.012, 0.025, 0.06, 0.085, 0.3])
        arrays = pa66_properties(c)
        for i, ci in enumerate(c):
            scalar = pa66_at_moisture(float(ci))
            for j in range(4):
                assert arrays[j][i] == pytest.approx(scalar[j])

    def test_vectorised_preserves_shape(self):
        c = np.zeros((3, 7))
        for arr in pa66_properties(c):
            assert arr.shape == (3, 7)


class TestSteel:
    def test_steel_is_far_stiffer_than_the_polymer(self):
        """This ratio is why constrained swelling matters at all."""
        ratio = (
            STEEL_BAND.youngs_modulus.nominal / PA66_SATURATED.youngs_modulus.nominal
        )
        assert ratio > 100

    def test_steel_poisson_ratio_is_standard(self):
        assert STEEL_BAND.poisson_ratio.nominal == pytest.approx(0.30, abs=0.02)

    def test_steel_condition_is_irrelevant_but_recorded(self):
        assert STEEL_BAND.condition is MoistureCondition.DAM
        assert STEEL_BAND.moisture_content == 0.0


class TestProvenance:
    def test_nothing_is_claimed_as_measured_or_from_a_datasheet(self):
        """Every property is a generic literature range until the grade is known."""
        for obj in all_material_objects():
            for item in obj.verification_items():
                assert item.provenance is Provenance.LITERATURE

    def test_every_material_reports_its_unverified_inputs(self):
        for obj in all_material_objects():
            assert obj.verification_items(), f"{obj} reports nothing to verify"

    def test_polymer_grades_are_labelled_distinctly_in_the_report(self):
        """A row reading only 'ElasticProperties' would not say which state."""
        labels = {g.doc_label for g in PA66_CONDITIONS}
        assert len(labels) == len(PA66_CONDITIONS)
        for grade in PA66_CONDITIONS:
            assert grade.condition.value in grade.doc_label
