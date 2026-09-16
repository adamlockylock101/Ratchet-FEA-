"""PA66 and steel property brackets, and the moisture interpolation.

The interpolation is the load-bearing part: the coupled FE model evaluates the
polymer modulus from the local moisture at every quadrature point, so an error
here propagates into every stress in the study.
"""

from __future__ import annotations

import math

import pytest

from ratchet_fea.materials import (
    MPA_ROOT_M_TO_MPA_ROOT_MM,
    PA66_CONDITIONS,
    PA66_DAM,
    PA66_RH50,
    PA66_SATURATED,
    SERVICE_CONDITION,
    STEEL_BAND,
    MoistureCondition,
    all_material_objects,
    fracture_toughness_mpa_root_mm,
    grade_for,
)
from ratchet_fea.provenance import Provenance


class TestBrackets:
    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_toughness_bracket_is_physical(self, grade):
        assert grade.fracture_toughness.low > 0
        assert grade.fracture_toughness.high > grade.fracture_toughness.low

    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_elastic_brackets_are_physical(self, grade):
        assert grade.youngs_modulus.low > 0
        assert 0.0 < grade.poisson_ratio.low < 0.5
        assert grade.poisson_ratio.high < 0.5, "plane stress is singular at nu = 0.5"

    @pytest.mark.parametrize("grade", PA66_CONDITIONS)
    def test_strength_brackets_are_physical(self, grade):
        assert grade.yield_strength.low > 0
        assert grade.tensile_strength.high >= grade.yield_strength.low


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


class TestFractureToughness:
    def test_conversion_factor_is_root_one_thousand(self):
        """K has dimensions stress * sqrt(length), so m -> mm multiplies by
        sqrt(1000). Getting this wrong scales every margin by 31.6."""
        assert MPA_ROOT_M_TO_MPA_ROOT_MM == pytest.approx(math.sqrt(1000.0))
        assert MPA_ROOT_M_TO_MPA_ROOT_MM == pytest.approx(31.6228, abs=1e-3)

    def test_conversion_applies_the_factor(self):
        grade = PA66_RH50
        assert fracture_toughness_mpa_root_mm(grade, "low") == pytest.approx(
            grade.fracture_toughness.low * MPA_ROOT_M_TO_MPA_ROOT_MM
        )

    def test_brackets_are_quoted_in_datasheet_units(self):
        """Stored in MPa*sqrt(m) so they can be read against a datasheet."""
        for grade in PA66_CONDITIONS:
            assert 1.0 < grade.fracture_toughness.low < 20.0
            assert grade.fracture_toughness.source.units == "MPa*sqrt(m)"

    def test_toughness_rises_with_conditioning(self):
        """Water plasticises PA66: tougher even as it gets weaker.

        The opposite direction to strength, so a single 'worst case' material
        does not exist -- dry is worst for fracture, wet is worst for yield.
        """
        assert (
            PA66_DAM.fracture_toughness.nominal
            < PA66_RH50.fracture_toughness.nominal
            < PA66_SATURATED.fracture_toughness.nominal
        )

    def test_strength_falls_as_toughness_rises(self):
        assert PA66_DAM.yield_strength.nominal > PA66_SATURATED.yield_strength.nominal
        assert PA66_DAM.fracture_toughness.nominal < PA66_SATURATED.fracture_toughness.nominal

    def test_dry_is_the_conservative_corner_for_fracture(self):
        assert min(
            g.fracture_toughness.low for g in PA66_CONDITIONS
        ) == PA66_DAM.fracture_toughness.low


class TestGradeSelection:
    def test_service_condition_is_conditioned(self):
        assert SERVICE_CONDITION is MoistureCondition.RH50

    def test_grade_for_returns_the_matching_set(self):
        for grade in PA66_CONDITIONS:
            assert grade_for(grade.condition) is grade

    def test_grade_for_defaults_to_the_service_state(self):
        assert grade_for() is PA66_RH50

    def test_unknown_condition_is_rejected(self):
        with pytest.raises(ValueError, match="no PA66 properties"):
            grade_for("underwater")
