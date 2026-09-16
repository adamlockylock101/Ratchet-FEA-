"""K extraction from the FE solution.

Three independent things are checked against each other: the J-integral, the
crack-flank displacement extrapolation, and the handbook solution. Two of them
agreeing is the only evidence any of them is right.
"""

from __future__ import annotations


import numpy as np
import pytest

from ratchet_fea.analytical import handbook_k
from ratchet_fea.fracture import (
    crack_opening,
    evaluate_crack,
    j_integral,
    j_integral_domain_independence,
    k_from_opening,
    sweep_crack_length,
)
from ratchet_fea.geometry import CrackGeometry, CrackOrientation
from ratchet_fea.mechanics import MechanicsModel
from ratchet_fea.mechanics import solve as solve_mechanics
from ratchet_fea.mesh import build_strip_mesh

pytestmark = pytest.mark.requires_gmsh


@pytest.fixture(scope="module")
def solved(small_geometry, coarse_controls, transverse_crack):
    strip = build_strip_mesh(small_geometry, coarse_controls, transverse_crack)
    model = MechanicsModel.from_materials(small_geometry)
    return strip, model, solve_mechanics(strip, model)


class TestJIntegral:
    def test_is_positive_for_an_open_crack(self, solved):
        _, _, result = solved
        assert j_integral(result) > 0

    def test_is_path_independent(self, solved):
        """The defining property of J, and the best available error measure."""
        _, _, result = solved
        mean, spread = j_integral_domain_independence(result, n_rings=6)
        assert mean > 0
        assert spread < 0.02, f"J varies by {spread:.1%} between contours"

    def test_individual_contours_agree(self, solved):
        _, _, result = solved
        from ratchet_fea.fracture import _tip_radius_limit

        limit = _tip_radius_limit(result.strip)
        values = [
            j_integral(result, inner=0.3 * f * limit, outer=f * limit)
            for f in (0.4, 0.6, 0.8)
        ]
        assert np.ptp(values) / np.mean(values) < 0.02

    def test_scales_with_the_square_of_the_load(self, small_geometry, coarse_controls,
                                                transverse_crack):
        """J is an energy, so it goes as load squared; K goes as load."""
        strip = build_strip_mesh(small_geometry, coarse_controls, transverse_crack)
        base = MechanicsModel.from_materials(small_geometry)
        single = j_integral(solve_mechanics(strip, base))
        double = j_integral(
            solve_mechanics(strip, base.with_remote_stress(2 * base.remote_stress))
        )
        assert double == pytest.approx(4.0 * single, rel=1e-6)

    def test_rejects_an_annulus_that_reaches_another_surface(self, solved):
        _, _, result = solved
        from ratchet_fea.fracture import _tip_radius_limit

        limit = _tip_radius_limit(result.strip)
        with pytest.raises(ValueError, match="another free surface"):
            j_integral(result, inner=0.5 * limit, outer=3.0 * limit)

    def test_refuses_an_uncracked_mesh(self, small_strip, small_geometry):
        model = MechanicsModel.from_materials(small_geometry)
        with pytest.raises(ValueError, match="no crack"):
            j_integral(solve_mechanics(small_strip, model))

    def test_radius_limit_stays_clear_of_the_hole(self, solved, small_geometry,
                                                  transverse_crack):
        """The domain form assumes only crack flanks inside the annulus."""
        from ratchet_fea.fracture import _tip_radius_limit

        strip, _, _ = solved
        assert _tip_radius_limit(strip) <= transverse_crack.length + 1e-9


class TestCrackOpening:
    def test_a_transverse_crack_opens(self, solved):
        _, _, result = solved
        _, opening = crack_opening(result)
        assert np.all(opening > 0)

    def test_opening_grows_toward_the_mouth(self, solved):
        """Widest where it meets the hole, closing to zero at the tip."""
        _, _, result = solved
        behind_tip, opening = crack_opening(result)
        order = np.argsort(behind_tip)
        assert opening[order][-1] > opening[order][0]

    def test_opening_vanishes_at_the_tip(self, solved):
        _, _, result = solved
        behind_tip, opening = crack_opening(result)
        nearest = np.argmin(behind_tip)
        assert opening[nearest] < 0.1 * np.max(opening)

    def test_near_tip_opening_follows_the_square_root(self, solved):
        """COD ~ sqrt(r) is the LEFM signature; if it does not hold, K is not
        the right parameter and the extraction is meaningless."""
        _, _, result = solved
        r, opening = crack_opening(result)
        window = (r > 0.05 * result.strip.crack.length) & (
            r < 0.5 * result.strip.crack.length
        )
        slope = np.polyfit(np.log(r[window]), np.log(opening[window]), 1)[0]
        assert slope == pytest.approx(0.5, abs=0.08)


class TestKExtraction:
    def test_the_two_extractions_agree(self, solved):
        """J-based and displacement-based K use completely different data."""
        _, model, result = solved
        j = j_integral(result)
        k_j = np.sqrt(model.youngs_modulus * j)
        k_cod, closed, _ = k_from_opening(result)
        assert not closed
        assert k_cod == pytest.approx(k_j, rel=0.06)

    def test_k_scales_linearly_with_load(self, small_geometry, coarse_controls,
                                         transverse_crack):
        strip = build_strip_mesh(small_geometry, coarse_controls, transverse_crack)
        base = MechanicsModel.from_materials(small_geometry)
        k1, _, _ = k_from_opening(solve_mechanics(strip, base))
        k3, _, _ = k_from_opening(
            solve_mechanics(strip, base.with_remote_stress(3 * base.remote_stress))
        )
        assert k3 == pytest.approx(3.0 * k1, rel=1e-4)

    def test_k_rises_with_crack_length(self, small_geometry, coarse_controls):
        results = [
            evaluate_crack(
                small_geometry, CrackGeometry(length=a), controls=coarse_controls
            )
            for a in (0.5, 1.5, 3.0)
        ]
        k = [r.k for r in results]
        assert k[0] < k[1] < k[2]


class TestAgainstHandbook:
    """The load-bearing validation: an independent closed-form solution."""

    def test_isolated_hole_matches_the_handbook(self, isolated_geometry,
                                                coarse_controls):
        """One hole is exactly what Newman's fit describes.

        The row of holes shields the cracked one, so the full geometry should
        NOT match; an isolated hole should, and does.
        """
        for a in (1.0, 2.0):
            result = evaluate_crack(
                isolated_geometry,
                CrackGeometry(length=a),
                controls=coarse_controls.refined(2.0),
            )
            expected = float(handbook_k(a, isolated_geometry))
            assert result.k == pytest.approx(expected, rel=0.10)

    def test_a_row_of_holes_shields_the_cracked_one(self, small_geometry,
                                                    isolated_geometry,
                                                    coarse_controls):
        """The effect the handbook cannot capture and the FE exists to measure."""
        crack = CrackGeometry(length=2.0)
        in_row = evaluate_crack(small_geometry, crack, controls=coarse_controls)
        isolated = evaluate_crack(isolated_geometry, crack, controls=coarse_controls)
        assert in_row.k < isolated.k


class TestClosure:
    """A crack that cannot open has no mode I driving force, whatever K says."""

    def test_a_longitudinal_crack_is_held_shut(self, small_geometry, coarse_controls):
        result = evaluate_crack(
            small_geometry,
            CrackGeometry(length=1.0, orientation=CrackOrientation.LONGITUDINAL),
            controls=coarse_controls,
        )
        assert result.is_closed
        assert result.min_opening < 0
        assert result.k == 0.0

    def test_a_transverse_crack_is_not(self, small_geometry, coarse_controls):
        result = evaluate_crack(
            small_geometry, CrackGeometry(length=1.0), controls=coarse_controls
        )
        assert not result.is_closed
        assert result.min_opening >= -1e-9
        assert result.k > 0

    def test_closure_is_measured_not_assumed(self, small_geometry, coarse_controls):
        """The model reports the flank opening it computed, not a rule.

        If it were hard-coded by orientation, reversing the load would not
        change the answer. It does.
        """
        strip = build_strip_mesh(
            small_geometry,
            coarse_controls,
            CrackGeometry(length=1.0, orientation=CrackOrientation.LONGITUDINAL),
        )
        model = MechanicsModel.from_materials(small_geometry)
        pulled = solve_mechanics(strip, model)
        pushed = solve_mechanics(strip, model.with_remote_stress(-model.remote_stress))
        _, closed_pulled, _ = k_from_opening(pulled)
        _, closed_pushed, _ = k_from_opening(pushed)
        assert closed_pulled
        assert not closed_pushed

    def test_k_is_zero_for_a_closed_crack_even_though_j_is_not(
        self, small_geometry, coarse_controls
    ):
        """J from an interpenetrating solution is a number, but not a real one."""
        result = evaluate_crack(
            small_geometry,
            CrackGeometry(length=1.0, orientation=CrackOrientation.LONGITUDINAL),
            controls=coarse_controls,
        )
        assert result.k == 0.0
        assert result.k_from_j >= 0.0


class TestConvergence:
    def test_k_is_converged_at_the_working_mesh(self, small_geometry,
                                                coarse_controls):
        crack = CrackGeometry(length=2.0)
        coarse = evaluate_crack(small_geometry, crack, controls=coarse_controls)
        fine = evaluate_crack(
            small_geometry, crack, controls=coarse_controls.refined(2.0)
        )
        assert abs(fine.k - coarse.k) / coarse.k < 0.03

    def test_domain_independence_holds_on_the_coarse_mesh(self, small_geometry,
                                                          coarse_controls):
        result = evaluate_crack(
            small_geometry, CrackGeometry(length=2.0), controls=coarse_controls
        )
        assert result.domain_independence < 0.02


class TestSweep:
    def test_returns_one_result_per_length(self, small_geometry, coarse_controls):
        lengths = [0.5, 1.0, 2.0]
        results = sweep_crack_length(
            small_geometry, lengths, controls=coarse_controls
        )
        assert [r.crack_length for r in results] == lengths

    def test_rejects_a_crack_that_does_not_fit(self, small_geometry, coarse_controls):
        with pytest.raises(ValueError, match="does not fit"):
            sweep_crack_length(small_geometry, [50.0], controls=coarse_controls)

    def test_records_the_remaining_ligament(self, small_geometry, coarse_controls):
        (result,) = sweep_crack_length(
            small_geometry, [2.0], controls=coarse_controls
        )
        assert result.remaining_ligament == pytest.approx(
            small_geometry.side_ligament - 2.0
        )
