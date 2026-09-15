"""Transient Fickian moisture diffusion.

The analytic plane-sheet series and the FE solve are checked against each
other and against textbook values, because the diffusion field sets both the
timing and the magnitude of every stress downstream.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from ratchet_fea.diffusion import (
    DiffusionModel,
    MoistureState,
    log_time_grid,
    slab_remaining_fraction,
    slab_uptake_fraction,
    solve,
    solve_in_plane,
)
from ratchet_fea.materials import PA66_MOISTURE


def _long_time_series(tau: float, n_terms: int = 400) -> float:
    """Independent reference implementation of Crank eq. 4.18."""
    total = sum(
        8.0 / ((2 * n + 1) ** 2 * math.pi**2)
        * math.exp(-((2 * n + 1) ** 2) * math.pi**2 * tau / 4.0)
        for n in range(n_terms)
    )
    return 1.0 - total


class TestSlabSeries:
    def test_half_uptake_at_the_textbook_fourier_number(self):
        """M_t/M_inf = 0.5 at D t / h^2 = 0.19685 (Crank)."""
        assert slab_uptake_fraction(0.19685, 1.0, 1.0)[0] == pytest.approx(0.5, abs=1e-3)

    def test_starts_at_zero(self):
        assert slab_uptake_fraction(0.0, 1.0, 1.0)[0] == pytest.approx(0.0)

    def test_approaches_one(self):
        assert slab_uptake_fraction(10.0, 1.0, 1.0)[0] == pytest.approx(1.0, abs=1e-9)

    def test_is_monotone(self):
        u = slab_uptake_fraction(np.geomspace(1e-5, 10.0, 400), 1.0, 1.0)
        assert np.all(np.diff(u) >= -1e-12)

    def test_stays_within_bounds(self):
        u = slab_uptake_fraction(np.geomspace(1e-8, 1e3, 200), 1.0, 1.0)
        assert np.all((u >= 0.0) & (u <= 1.0))

    @pytest.mark.parametrize(
        "tau", [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.2499, 0.25, 0.3, 0.5, 1.0, 3.0]
    )
    def test_agrees_with_an_independent_long_time_series(self, tau):
        """Cross-checks the short-time branch, which is easy to get wrong.

        Pulling the 1/sqrt(pi) outside the bracket of Crank's short-time series
        scales the correction terms by sqrt(pi) -- an error invisible at very
        small tau but obvious at the branch switch. This test catches it.
        """
        assert slab_uptake_fraction(tau, 1.0, 1.0)[0] == pytest.approx(
            _long_time_series(tau), abs=1e-7
        )

    def test_is_continuous_across_the_branch_switch(self):
        below = slab_uptake_fraction(0.2499999, 1.0, 1.0)[0]
        above = slab_uptake_fraction(0.2500001, 1.0, 1.0)[0]
        # Both branches are accurate to ~1e-7 with 20 terms; the residual step
        # at the switch is far below the decade-wide diffusivity uncertainty.
        assert below == pytest.approx(above, abs=1e-6)

    def test_leading_short_time_behaviour_is_sqrt_t(self):
        """M_t/M_inf -> 2 sqrt(D t / pi) / h before the fronts meet."""
        tau = 1e-4
        assert slab_uptake_fraction(tau, 1.0, 1.0)[0] == pytest.approx(
            2.0 * math.sqrt(tau / math.pi), rel=1e-9
        )

    def test_depends_on_the_fourier_number_only(self):
        a = slab_uptake_fraction(100.0, 1e-3, 2.0)[0]
        b = slab_uptake_fraction(400.0, 1e-3, 4.0)[0]
        assert a == pytest.approx(b)

    def test_remaining_fraction_is_the_complement(self):
        t = np.geomspace(1e-3, 5.0, 20)
        assert np.allclose(
            slab_remaining_fraction(t, 1.0, 1.0), 1.0 - slab_uptake_fraction(t, 1.0, 1.0)
        )

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="diffusivity must be positive"):
            slab_uptake_fraction(1.0, 0.0, 1.0)
        with pytest.raises(ValueError, match="half_thickness must be positive"):
            slab_uptake_fraction(1.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="non-negative"):
            slab_uptake_fraction(-1.0, 1.0, 1.0)


class TestTimeGrid:
    def test_starts_at_zero_and_ends_at_the_horizon(self):
        t = log_time_grid(100.0, n_steps=10)
        assert t[0] == 0.0
        assert t[-1] == pytest.approx(100.0)

    def test_is_strictly_increasing(self):
        assert np.all(np.diff(log_time_grid(1e6, n_steps=30)) > 0)

    def test_resolves_the_early_transient(self):
        """Uniform steps would put almost all the effort in the flat tail."""
        t = log_time_grid(1e6, n_steps=30)
        assert t[1] < 1e6 / 1000

    def test_step_count(self):
        assert len(log_time_grid(10.0, n_steps=12)) == 13

    def test_rejects_bad_inputs(self):
        with pytest.raises(ValueError, match="t_end must be positive"):
            log_time_grid(0.0)
        with pytest.raises(ValueError, match="at least 2 steps"):
            log_time_grid(1.0, n_steps=1)
        with pytest.raises(ValueError, match="0 < t_start < t_end"):
            log_time_grid(1.0, t_start=2.0)


class TestDiffusionModel:
    def test_between_reads_the_material_brackets(self, geometry):
        m = DiffusionModel.between(geometry, MoistureState.DRY, MoistureState.IMMERSED)
        assert m.diffusivity == pytest.approx(PA66_MOISTURE.diffusivity.nominal)
        assert m.c_initial == 0.0
        assert m.c_surface == pytest.approx(PA66_MOISTURE.saturation_immersed.nominal)

    def test_corner_selection_changes_the_diffusivity(self, geometry):
        low = DiffusionModel.between(geometry, corner="low")
        high = DiffusionModel.between(geometry, corner="high")
        assert low.diffusivity < high.diffusivity

    def test_half_thicknesses_come_from_the_geometry(self, geometry):
        m = DiffusionModel.between(geometry)
        assert m.half_thickness_band == pytest.approx(
            geometry.diffusion_half_thickness(True)
        )
        assert m.half_thickness_plain == pytest.approx(
            geometry.diffusion_half_thickness(False)
        )

    def test_absorption_and_desorption_are_distinguished(self, geometry):
        wet = DiffusionModel.between(geometry, MoistureState.DRY, MoistureState.IMMERSED)
        dry = DiffusionModel.between(geometry, MoistureState.IMMERSED, MoistureState.RH50)
        assert wet.is_absorption
        assert not dry.is_absorption
        assert wet.moisture_change > 0 > dry.moisture_change

    def test_characteristic_time_uses_the_fastest_path(self, geometry):
        m = DiffusionModel.between(geometry)
        assert m.governing_half_thickness == pytest.approx(
            min(m.half_thickness_band, m.half_thickness_plain)
        )

    def test_characteristic_time_is_weeks_for_this_strap(self, geometry):
        t = DiffusionModel.between(geometry).characteristic_time()
        assert 2 * 86400 < t < 60 * 86400

    def test_a_transition_to_the_same_state_is_rejected(self, geometry):
        with pytest.raises(ValueError, match="no moisture change"):
            DiffusionModel.between(geometry, MoistureState.DRY, MoistureState.DRY)

    def test_rejects_non_positive_diffusivity(self, geometry):
        with pytest.raises(ValueError, match="diffusivity must be positive"):
            replace(DiffusionModel.between(geometry), diffusivity=0.0)

    def test_moisture_state_contents_are_ordered(self):
        m = PA66_MOISTURE
        assert (
            MoistureState.DRY.content(m)
            < MoistureState.RH50.content(m)
            < MoistureState.IMMERSED.content(m)
        )

    def test_summary_names_the_transition(self, geometry):
        text = DiffusionModel.between(geometry).summary()
        assert "absorption" in text
        assert "half-uptake time" in text


class TestInPlaneSolveVerification:
    """Verify the FE solve against the analytic plane-sheet solution.

    A hole-free rectangular strip with the two long edges exposed and the two
    cut ends left at the natural (zero-flux) condition is exactly a 1D plane
    sheet of half-thickness W/2, so the FE answer must reproduce the series.
    """

    @pytest.fixture(scope="class")
    def strip_mesh(self):
        from skfem import MeshTri

        width, length = 2.0, 1.0
        m = MeshTri.init_tensor(
            np.linspace(0.0, length, 9), np.linspace(0.0, width, 41)
        )
        return m.with_boundaries(
            {
                "exposed": lambda x: (x[1] < 1e-9) | (x[1] > width - 1e-9),
                "ends": lambda x: (x[0] < 1e-9) | (x[0] > length - 1e-9),
            }
        )

    def test_matches_the_analytic_series(self, strip_mesh):
        d = 0.05
        times = np.concatenate([[0.0], np.geomspace(0.02, 4.0, 60)])
        u = solve_in_plane(strip_mesh, ["exposed"], d, times)

        areas = _nodal_areas(strip_mesh)
        mean_remaining = (u @ areas) / areas.sum()
        analytic = slab_remaining_fraction(times, d, 1.0)  # half-thickness = W/2

        # 2% is comfortably inside the discretisation error of a 40-element
        # through-width mesh with backward Euler on a log time grid.
        assert np.max(np.abs(mean_remaining - analytic)) < 0.02

    def test_field_is_uniform_along_the_zero_flux_direction(self, strip_mesh):
        """Confirms the cut ends really are natural (no moisture enters there)."""
        times = np.concatenate([[0.0], np.geomspace(0.02, 2.0, 20)])
        u = solve_in_plane(strip_mesh, ["exposed"], 0.05, times)
        p = strip_mesh.p
        mid = np.isclose(p[1], 1.0, atol=1e-9)
        for step in u[1:]:
            # Exactly zero in theory; what is left is the direct solve's
            # round-off, five orders of magnitude below the field itself.
            assert np.ptp(step[mid]) < 1e-4

    def test_solution_stays_within_bounds(self, strip_mesh):
        times = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 40)])
        u = solve_in_plane(strip_mesh, ["exposed"], 0.05, times)
        assert np.all((u >= 0.0) & (u <= 1.0))

    def test_solution_is_monotone_in_time(self, strip_mesh):
        times = np.concatenate([[0.0], np.geomspace(0.01, 5.0, 40)])
        u = solve_in_plane(strip_mesh, ["exposed"], 0.05, times)
        assert np.all(np.diff(u, axis=0) <= 1e-9)

    def test_initial_condition_is_pristine_everywhere(self, strip_mesh):
        """Including on the exposed surfaces: at t = 0 no moisture has entered."""
        times = np.array([0.0, 0.1])
        u = solve_in_plane(strip_mesh, ["exposed"], 0.05, times)
        assert np.allclose(u[0], 1.0)

    def test_boundary_reaches_the_surface_value_after_the_first_step(self, strip_mesh):
        times = np.array([0.0, 0.1])
        u = solve_in_plane(strip_mesh, ["exposed"], 0.05, times)
        edge = strip_mesh.boundaries["exposed"]
        vertices = np.unique(strip_mesh.facets[:, edge].ravel())
        assert np.allclose(u[1][vertices], 0.0)

    def test_rejects_a_grid_not_starting_at_zero(self, strip_mesh):
        with pytest.raises(ValueError, match="must start at t = 0"):
            solve_in_plane(strip_mesh, ["exposed"], 0.05, np.array([1.0, 2.0]))

    def test_rejects_a_non_increasing_grid(self, strip_mesh):
        with pytest.raises(ValueError, match="strictly increasing"):
            solve_in_plane(strip_mesh, ["exposed"], 0.05, np.array([0.0, 2.0, 1.0]))


def _nodal_areas(mesh):
    from ratchet_fea.diffusion import _vertex_areas

    return _vertex_areas(mesh)


class TestNodalAreas:
    def test_sum_to_the_domain_area(self):
        from skfem import MeshTri

        m = MeshTri.init_tensor(np.linspace(0, 3, 7), np.linspace(0, 2, 5))
        assert _nodal_areas(m).sum() == pytest.approx(6.0)

    def test_all_positive(self):
        from skfem import MeshTri

        assert np.all(_nodal_areas(MeshTri().refined(3)) > 0)


@pytest.mark.requires_gmsh
class TestCoupledSolve:
    @pytest.fixture(scope="class")
    def result(self, small_strip, small_geometry):
        model = DiffusionModel.between(
            small_geometry, MoistureState.DRY, MoistureState.IMMERSED
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=14)
        return solve(small_strip, model, times)

    def test_concentration_shape(self, result, small_strip):
        assert result.concentration.shape == (result.n_times, small_strip.n_vertices)

    def test_starts_dry_and_ends_saturated(self, result):
        assert np.allclose(result.concentration[0], result.model.c_initial)
        assert result.concentration[-1] == pytest.approx(
            result.model.c_surface, rel=1e-2
        )

    def test_concentration_stays_between_the_two_states(self, result):
        lo = min(result.model.c_initial, result.model.c_surface)
        hi = max(result.model.c_initial, result.model.c_surface)
        assert result.concentration.min() >= lo - 1e-12
        assert result.concentration.max() <= hi + 1e-12

    def test_uptake_runs_from_zero_to_one(self, result):
        assert result.uptake_fraction[0] == pytest.approx(0.0, abs=1e-9)
        assert result.uptake_fraction[-1] == pytest.approx(1.0, abs=1e-3)

    def test_uptake_is_monotone(self, result):
        assert np.all(np.diff(result.uptake_fraction) >= -1e-9)

    def test_hole_walls_are_exposed_surfaces(self, result, small_strip):
        """A hole wall is a free surface at every depth, so it wets immediately."""
        mesh = small_strip.mesh
        facets = mesh.boundaries[small_strip.hole_boundaries[0]]
        vertices = np.unique(mesh.facets[:, facets].ravel())
        assert result.concentration[1][vertices] == pytest.approx(
            result.model.c_surface, rel=1e-9
        )

    def test_in_plane_ingress_makes_the_part_wet_faster_than_1d_alone(
        self, result, small_geometry
    ):
        """The FE half-time must beat the pure through-thickness estimate."""
        from ratchet_fea.analytical import fickian_half_time

        one_d = fickian_half_time(
            result.model.governing_half_thickness, result.model.diffusivity
        )
        assert result.time_to_uptake(0.5) < one_d

    def test_but_not_by_much_for_this_strap(self, result):
        """Through-thickness dominates, which is why the product solution matters."""
        from ratchet_fea.analytical import fickian_half_time

        one_d = fickian_half_time(
            result.model.governing_half_thickness, result.model.diffusivity
        )
        assert result.time_to_uptake(0.5) > 0.6 * one_d

    def test_disabling_the_thickness_factor_slows_everything_down(
        self, small_strip, small_geometry
    ):
        """Confirms the product solution is doing real work, not decoration.

        Over a horizon long enough for the real strap to saturate completely,
        an in-plane-only model has barely started -- which is exactly the
        mistiming a plane-stress diffusion model would suffer on its own.
        """
        model = DiffusionModel.between(small_geometry, include_through_thickness=False)
        times = log_time_grid(20 * model.characteristic_time(), n_steps=14)
        in_plane_only = solve(small_strip, model, times)
        with_thickness = solve(
            small_strip, replace(model, include_through_thickness=True), times
        )
        assert with_thickness.uptake_fraction[-1] > 0.99
        assert in_plane_only.uptake_fraction[-1] < 0.9

    def test_desorption_runs_the_other_way(self, small_strip, small_geometry):
        model = DiffusionModel.between(
            small_geometry, MoistureState.IMMERSED, MoistureState.RH50
        )
        times = log_time_grid(20 * model.characteristic_time(), n_steps=12)
        result = solve(small_strip, model, times)
        assert np.all(np.diff(result.concentration.mean(axis=1)) <= 1e-12)
        assert result.concentration[-1].mean() == pytest.approx(
            model.c_surface, rel=1e-2
        )

    def test_time_to_uptake_rejects_out_of_range_fractions(self, result):
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            result.time_to_uptake(1.0)

    def test_band_region_wets_faster_than_the_plain_edges(self, result, small_strip):
        """The steel halves the polymer skin depth, so it saturates sooner."""
        band = small_strip.band_vertex_mask()
        early = result.thickness_remaining[2]
        assert early[band].mean() < early[~band].mean()
