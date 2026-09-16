"""The K(a) curve, its comparison against K_IC, and the plots."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from ratchet_fea.fracture import sweep_crack_length
from ratchet_fea.geometry import CrackOrientation
from ratchet_fea.materials import MoistureCondition
from ratchet_fea.postprocess import (
    KCurve,
    build_k_curve,
    plot_crack_opening,
    plot_k_vs_a,
    write_summary,
)


def _curve(**overrides) -> KCurve:
    """A constructed curve, so the reading logic is tested without an FE run."""
    from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

    a = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    base = dict(
        geometry=PLACEHOLDER_GEOMETRY,
        orientation=CrackOrientation.TRANSVERSE,
        condition=MoistureCondition.RH50,
        corner="nominal",
        force=500.0,
        crack_lengths=a,
        k_fe=np.array([13.0, 18.0, 21.0, 26.0, 55.0]),
        k_handbook=np.array([15.0, 21.0, 24.0, 30.0, 66.0]),
        toughness={"low": 94.9, "nominal": 134.4, "high": 173.9},
        domain_independence=np.full(5, 1e-4),
        extraction_agreement=np.full(5, 1.01),
        closed=np.zeros(5, dtype=bool),
        partially_closed=np.zeros(5, dtype=bool),
        min_opening=np.full(5, 1e-3),
    )
    base.update(overrides)
    return KCurve(**base)


class TestReadings:
    def test_k_at_interpolates(self):
        curve = _curve()
        assert curve.k_at(1.5) == pytest.approx(19.5)

    def test_k_at_observed_uses_the_geometry(self, geometry):
        curve = _curve()
        assert curve.observed_crack_length == geometry.observed_crack_length
        assert curve.k_at_observed == pytest.approx(21.0)

    def test_max_k_is_the_peak_of_the_curve(self):
        assert _curve().max_k == pytest.approx(55.0)

    def test_does_not_propagate_when_k_stays_below_toughness(self):
        curve = _curve()
        assert not curve.propagates("low")
        assert math.isnan(curve.critical_crack_length("low"))

    def test_propagates_when_k_crosses(self):
        curve = _curve(k_fe=np.array([13.0, 18.0, 21.0, 100.0, 200.0]))
        assert curve.propagates("low")
        a_c = curve.critical_crack_length("low")
        assert 2.0 < a_c < 4.0

    def test_propagation_defaults_to_the_weakest_corner(self):
        """The question is whether it can run at all."""
        curve = _curve(k_fe=np.array([13.0, 18.0, 21.0, 26.0, 100.0]))
        assert curve.propagates()
        assert not curve.propagates("high")

    def test_a_tougher_corner_needs_a_longer_crack(self):
        curve = _curve(k_fe=np.array([50.0, 90.0, 120.0, 150.0, 200.0]))
        assert curve.critical_crack_length("low") < curve.critical_crack_length("high")

    def test_margin_at_observed(self):
        curve = _curve()
        assert curve.margin_at_observed("nominal") == pytest.approx(134.4 / 21.0)

    def test_force_scaling_is_linear_in_k(self):
        """K goes as load, so the load for K_IC is a straight scaling."""
        curve = _curve()
        expected = 500.0 * 94.9 / 21.0
        assert curve.force_to_reach_toughness_at_observed("low") == pytest.approx(
            expected
        )

    def test_force_at_max_is_lower_than_at_the_observed_crack(self):
        """A longer crack needs less load, which is the point of the curve."""
        curve = _curve()
        assert curve.force_to_reach_toughness("low") < (
            curve.force_to_reach_toughness_at_observed("low")
        )

    def test_handbook_ratio_is_finite_where_the_handbook_is(self):
        assert np.all(np.isfinite(_curve().handbook_ratio()))

    def test_handbook_ratio_is_nan_where_the_handbook_is_zero(self):
        curve = _curve(k_handbook=np.zeros(5))
        assert np.all(np.isnan(curve.handbook_ratio()))


class TestClosure:
    def test_a_fully_closed_curve_is_recognised(self):
        curve = _curve(
            closed=np.ones(5, dtype=bool),
            k_fe=np.zeros(5),
            k_handbook=np.zeros(5),
            min_opening=np.full(5, -1e-3),
        )
        assert curve.is_closed_throughout
        assert curve.mostly_closed
        assert "HELD SHUT" in curve.summary()

    def test_a_mostly_closed_curve_is_recognised(self):
        closed = np.array([True, True, True, False, False])
        curve = _curve(
            closed=closed,
            k_fe=np.array([0.0, 0.0, 0.0, 0.5, 1.2]),
            k_handbook=np.zeros(5),
            partially_closed=~closed,
            min_opening=np.full(5, -1e-3),
        )
        assert not curve.is_closed_throughout
        assert curve.mostly_closed
        assert curve.k_is_negligible
        assert "OVER MOST OF THE SWEEP" in curve.summary()

    def test_an_open_curve_is_neither(self):
        curve = _curve()
        assert not curve.is_closed_throughout
        assert not curve.mostly_closed
        assert not curve.k_is_negligible

    def test_a_closed_summary_does_not_print_nan_ratios(self):
        curve = _curve(
            closed=np.ones(5, dtype=bool), k_fe=np.zeros(5), k_handbook=np.zeros(5)
        )
        assert "nan" not in curve.summary().lower()

    def test_an_open_summary_does_not_print_nan_either(self):
        assert "nan" not in _curve().summary().lower()

    def test_extraction_disagreement_ignores_closed_points(self):
        """Comparing two ways of computing zero measures nothing."""
        curve = _curve(
            closed=np.array([True, True, False, False, False]),
            extraction_agreement=np.array([100.0, 50.0, 1.02, 1.01, 1.00]),
        )
        assert curve.worst_extraction_disagreement == pytest.approx(0.02, abs=1e-9)


class TestVerdict:
    def test_no_propagation_says_so_and_quantifies_it(self):
        text = " ".join(_curve().verdict())
        assert "DOES NOT PROPAGATE" in text
        assert "times the assumed service tension" in text

    def test_propagation_reports_the_critical_length(self):
        curve = _curve(k_fe=np.array([13.0, 18.0, 21.0, 100.0, 200.0]))
        text = " ".join(curve.verdict())
        assert "CAN PROPAGATE" in text

    def test_a_closed_crack_explains_why(self):
        curve = _curve(
            closed=np.ones(5, dtype=bool), k_fe=np.zeros(5), k_handbook=np.zeros(5)
        )
        text = " ".join(curve.verdict())
        assert "CANNOT EVEN OPEN" in text
        assert "compressive" in text.lower()

    def test_every_verdict_names_what_is_left_unexplained(self):
        """A 'does not propagate' answer has to say what still might."""
        text = " ".join(_curve().verdict())
        assert "fatigue" in text.lower()


@pytest.mark.requires_gmsh
class TestAgainstTheFE:
    @pytest.fixture(scope="class")
    def curve(self, small_geometry, coarse_controls):
        results = sweep_crack_length(
            small_geometry, [0.4, 1.0, 2.0, 4.0], controls=coarse_controls
        )
        return build_k_curve(results, small_geometry, small_geometry.service_tension)

    def test_curve_shape(self, curve):
        assert len(curve.crack_lengths) == len(curve.k_fe) == len(curve.k_handbook)

    def test_k_rises_with_crack_length(self, curve):
        assert np.all(np.diff(curve.k_fe) > 0)

    def test_fe_sits_below_the_handbook(self, curve):
        """The row of holes shields the cracked one; the handbook has no row."""
        ratio = curve.handbook_ratio()
        assert np.all(ratio < 1.0)
        assert np.all(ratio > 0.7)

    def test_the_headline_finding_holds(self, curve):
        """K never reaches K_IC at the placeholder values and service load."""
        assert not curve.propagates("low")
        assert curve.margin_at_observed("nominal") > 2.0

    def test_a_large_enough_load_does_make_it_propagate(self, small_geometry,
                                                        coarse_controls):
        """Confirms the model can say yes, so the no above means something."""
        from ratchet_fea.mechanics import MechanicsModel

        model = MechanicsModel.from_materials(small_geometry, force=30000.0)
        results = sweep_crack_length(
            small_geometry, [1.0, 2.0, 4.0], model=model, controls=coarse_controls
        )
        curve = build_k_curve(results, small_geometry, 30000.0)
        assert curve.propagates("low")

    def test_numerical_quality_is_reported_and_good(self, curve):
        assert curve.worst_domain_independence < 0.02
        assert curve.worst_extraction_disagreement < 0.1


@pytest.mark.requires_gmsh
class TestOutputs:
    @pytest.fixture(scope="class")
    def bundle(self, small_geometry, coarse_controls):
        results = sweep_crack_length(
            small_geometry, [0.5, 1.5, 3.0], controls=coarse_controls
        )
        curve = build_k_curve(results, small_geometry, small_geometry.service_tension)
        return results, curve

    def test_k_plot_is_written(self, bundle, tmp_path):
        _, curve = bundle
        path = plot_k_vs_a(curve, tmp_path / "k.png")
        assert path.exists() and path.stat().st_size > 1000

    def test_opening_plot_is_written(self, bundle, tmp_path):
        results, _ = bundle
        path = plot_crack_opening(results, tmp_path / "opening.png")
        assert path.exists() and path.stat().st_size > 1000

    def test_summary_json_round_trips(self, bundle, tmp_path):
        _, curve = bundle
        data = json.loads(write_summary(curve, tmp_path / "s.json").read_text())

        assert data["orientation"] == "transverse"
        assert data["propagates_at_low_toughness"] is False
        assert len(data["series"]["crack_length_mm"]) == len(curve.crack_lengths)
        assert data["k_at_observed"] == pytest.approx(curve.k_at_observed)

    def test_summary_records_both_toughness_units(self, bundle, tmp_path):
        """So a reader can check the bracket against a datasheet."""
        _, curve = bundle
        data = json.loads(write_summary(curve, tmp_path / "s.json").read_text())
        assert data["toughness_mpa_root_m"]["low"] == pytest.approx(
            data["toughness_mpa_root_mm"]["low"] / 31.6228
        )

    def test_summary_records_the_geometry_that_produced_it(self, bundle, tmp_path):
        _, curve = bundle
        data = json.loads(write_summary(curve, tmp_path / "s.json").read_text())
        assert data["geometry"]["strap_width"] == curve.geometry.strap_width

    def test_summary_creates_missing_directories(self, bundle, tmp_path):
        _, curve = bundle
        assert write_summary(curve, tmp_path / "a" / "b" / "s.json").exists()
