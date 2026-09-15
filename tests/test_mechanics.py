"""Plane-stress membrane solve with a moisture swelling eigenstrain.

The constitutive helpers are checked against closed-form identities, then the
assembled solve is checked with patch tests (uniform states it must reproduce
exactly) before anything is asked of it on the real perforated geometry.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from ratchet_fea.analytical import (
    constrained_swelling_stress,
    kt_hole_in_finite_width_strip,
)
from ratchet_fea.geometry import StrapGeometry
from ratchet_fea.materials import PA66_MOISTURE, STEEL_BAND, pa66_at_moisture
from ratchet_fea.mechanics import (
    MechanicsModel,
    plane_stress_moduli,
    principal_stresses,
    recover_polymer_stress,
    solve,
    stress_concentration_factors,
    von_mises,
)
from ratchet_fea.mesh import CUT_END, CUT_START, SIDE_LOWER, SIDE_UPPER, StripMesh


class TestPlaneStressModuli:
    def test_reproduces_the_standard_relations(self):
        e, nu = 2000.0, 0.35
        q11, q12, q66 = plane_stress_moduli(e, nu)
        assert q11 == pytest.approx(e / (1 - nu**2))
        assert q12 == pytest.approx(nu * e / (1 - nu**2))
        assert q66 == pytest.approx(e / (2 * (1 + nu)))

    def test_shear_modulus_identity(self):
        """q66 must equal G, and q66 = (q11 - q12)/2 for an isotropic material."""
        q11, q12, q66 = plane_stress_moduli(1500.0, 0.4)
        assert q66 == pytest.approx(0.5 * (q11 - q12))

    def test_uniaxial_stress_recovers_the_modulus(self):
        """sigma_xx = E eps_xx when the strip is free to contract laterally."""
        e, nu = 1800.0, 0.38
        q11, q12, _ = plane_stress_moduli(e, nu)
        exx = 1e-3
        eyy = -nu * exx  # free lateral contraction
        assert q11 * exx + q12 * eyy == pytest.approx(e * exx)

    def test_vectorises(self):
        q11, q12, q66 = plane_stress_moduli(np.array([1000.0, 2000.0]), 0.3)
        assert q11.shape == (2,)
        assert q11[1] == pytest.approx(2 * q11[0])

    def test_rejects_incompressible_poisson_ratio(self):
        with pytest.raises(ValueError, match="plane stress"):
            plane_stress_moduli(1000.0, 0.5)


class TestStressInvariants:
    def test_principal_stresses_of_a_uniaxial_state(self):
        s1, s2 = principal_stresses(10.0, 0.0, 0.0)
        assert s1 == pytest.approx(10.0)
        assert s2 == pytest.approx(0.0)

    def test_principal_stresses_of_pure_shear(self):
        s1, s2 = principal_stresses(0.0, 0.0, 5.0)
        assert s1 == pytest.approx(5.0)
        assert s2 == pytest.approx(-5.0)

    def test_principal_stresses_of_an_equibiaxial_state(self):
        s1, s2 = principal_stresses(-30.0, -30.0, 0.0)
        assert s1 == pytest.approx(-30.0)
        assert s2 == pytest.approx(-30.0)

    def test_principals_are_ordered(self):
        rng = np.random.default_rng(0)
        sxx, syy, sxy = rng.normal(size=(3, 200)) * 20
        s1, s2 = principal_stresses(sxx, syy, sxy)
        assert np.all(s1 >= s2)

    def test_invariants_are_preserved(self):
        rng = np.random.default_rng(1)
        sxx, syy, sxy = rng.normal(size=(3, 100)) * 10
        s1, s2 = principal_stresses(sxx, syy, sxy)
        assert np.allclose(s1 + s2, sxx + syy)
        assert np.allclose(s1 * s2, sxx * syy - sxy**2)

    def test_von_mises_of_uniaxial_equals_the_stress(self):
        assert von_mises(25.0, 0.0, 0.0) == pytest.approx(25.0)

    def test_von_mises_is_sign_blind(self):
        """Which is why a compressive swelling stress can still yield."""
        assert von_mises(-40.0, 0.0, 0.0) == pytest.approx(von_mises(40.0, 0.0, 0.0))

    def test_von_mises_of_equibiaxial(self):
        assert von_mises(30.0, 30.0, 0.0) == pytest.approx(30.0)

    def test_von_mises_of_pure_shear(self):
        assert von_mises(0.0, 0.0, 10.0) == pytest.approx(10.0 * math.sqrt(3))


class TestPlyStressRecovery:
    def test_rigid_constraint_reproduces_the_tier_1_formula(self):
        """The single most important identity in the coupled model.

        With the membrane strain held at zero -- a perfectly rigid band -- the
        recovered polymer stress must equal Tier 1's biaxial constrained
        swelling stress, ``E beta dc / (1 - nu)``, and must be COMPRESSIVE
        during absorption.
        """
        e, nu, beta, dc = 1400.0, 0.42, 0.25, 0.06
        q11, q12, q66 = plane_stress_moduli(e, nu)
        eps_sw = beta * dc

        sxx, syy, sxy = recover_polymer_stress(0.0, 0.0, 0.0, q11, q12, q66, eps_sw)

        expected = constrained_swelling_stress(e, nu, beta, dc, constraint="biaxial")
        assert sxx == pytest.approx(-expected)
        assert syy == pytest.approx(-expected)
        assert sxy == pytest.approx(0.0)

    def test_free_swelling_produces_no_stress(self):
        """Unconstrained growth is stress-free, whatever the eigenstrain."""
        q11, q12, q66 = plane_stress_moduli(1400.0, 0.42)
        eps_sw = 0.015
        sxx, syy, sxy = recover_polymer_stress(
            eps_sw, eps_sw, 0.0, q11, q12, q66, eps_sw
        )
        assert sxx == pytest.approx(0.0, abs=1e-9)
        assert syy == pytest.approx(0.0, abs=1e-9)

    def test_drying_reverses_the_sign(self):
        """Constrained shrinkage is tensile -- the direction that cracks things."""
        q11, q12, q66 = plane_stress_moduli(1400.0, 0.42)
        wet, _, _ = recover_polymer_stress(0.0, 0.0, 0.0, q11, q12, q66, 0.015)
        dry, _, _ = recover_polymer_stress(0.0, 0.0, 0.0, q11, q12, q66, -0.015)
        assert wet < 0 < dry
        assert wet == pytest.approx(-dry)

    def test_mechanical_strain_superposes(self):
        q11, q12, q66 = plane_stress_moduli(1400.0, 0.42)
        mech, _, _ = recover_polymer_stress(1e-3, 0.0, 0.0, q11, q12, q66, 0.0)
        both, _, _ = recover_polymer_stress(1e-3, 0.0, 0.0, q11, q12, q66, 0.01)
        swell, _, _ = recover_polymer_stress(0.0, 0.0, 0.0, q11, q12, q66, 0.01)
        assert both == pytest.approx(mech + swell)

    def test_shear_is_untouched_by_swelling(self):
        """An isotropic eigenstrain has no shear component."""
        q11, q12, q66 = plane_stress_moduli(1400.0, 0.42)
        _, _, sxy = recover_polymer_stress(0.0, 0.0, 2e-3, q11, q12, q66, 0.02)
        assert sxy == pytest.approx(2.0 * q66 * 2e-3)


class TestMechanicsModel:
    def test_from_materials_reads_the_geometry(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        assert m.strap_thickness == geometry.strap_thickness
        assert m.band_thickness == geometry.steel_band_thickness
        assert m.remote_tension == geometry.service_tension

    def test_from_materials_reads_the_material_brackets(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        assert m.swelling_coefficient == pytest.approx(
            PA66_MOISTURE.swelling_coefficient.nominal
        )
        assert m.band_modulus == pytest.approx(STEEL_BAND.youngs_modulus.nominal)

    def test_polymer_thickness_excludes_the_band(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        assert m.polymer_thickness_in_band == pytest.approx(
            geometry.strap_thickness - geometry.steel_band_thickness
        )

    def test_band_dominates_the_section_stiffness(self, geometry):
        """Which is why the polymer is close to fully constrained."""
        assert MechanicsModel.from_materials(geometry).stiffness_ratio > 10

    def test_no_band_means_infinite_ratio_is_not_reported(self, geometry):
        m = MechanicsModel.from_materials(geometry).without_band()
        assert m.band_thickness == 0.0
        assert m.stiffness_ratio == pytest.approx(0.0)

    def test_line_load_is_force_per_unit_width(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        assert m.remote_line_load == pytest.approx(
            geometry.service_tension / geometry.strap_width
        )

    def test_variants_are_independent(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        assert m.without_tension().remote_tension == 0.0
        assert m.remote_tension != 0.0
        assert m.with_tension(123.0).remote_tension == 123.0

    def test_include_flags(self, geometry):
        assert MechanicsModel.from_materials(geometry, include_band=False).band_thickness == 0.0
        assert (
            MechanicsModel.from_materials(geometry, include_tension=False).remote_tension
            == 0.0
        )

    def test_validation(self, geometry):
        m = MechanicsModel.from_materials(geometry)
        with pytest.raises(ValueError, match="strap_thickness must be positive"):
            replace(m, strap_thickness=0.0)
        with pytest.raises(ValueError, match="band_thickness must be in"):
            replace(m, band_thickness=10.0)

    def test_summary_mentions_the_section(self, geometry):
        assert "PA66" in MechanicsModel.from_materials(geometry).summary()


# ---------------------------------------------------------------------------
# Patch tests: uniform states the solver must reproduce exactly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plain_strip():
    """A hole-free rectangular strip, wrapped so the solver accepts it.

    Patch tests need a domain whose exact answer is known. Reusing the real
    perforated mesh would confound a solver error with a stress concentration.
    """
    from skfem import MeshTri

    from ratchet_fea.mesh import MeshControls

    geometry = StrapGeometry(
        strap_width=25.0,
        strap_thickness=3.0,
        hole_diameter=4.0,
        hole_pitch=10.0,
        n_holes=1,
        end_margin=30.0,
        steel_band_width=25.0,
        steel_band_thickness=0.8,
    )
    length, width = geometry.modelled_length, geometry.strap_width
    mesh = MeshTri.init_tensor(
        np.linspace(0.0, length, 25), np.linspace(0.0, width, 11)
    ).with_boundaries(
        {
            CUT_START: lambda x: x[0] < 1e-9,
            CUT_END: lambda x: x[0] > length - 1e-9,
            SIDE_LOWER: lambda x: x[1] < 1e-9,
            SIDE_UPPER: lambda x: x[1] > width - 1e-9,
        }
    )
    return StripMesh(mesh=mesh, geometry=geometry, controls=MeshControls())


class TestPatchTests:
    def test_uniform_tension_on_a_plain_polymer_strip(self, plain_strip):
        """sigma_xx = F / (W t) everywhere; sigma_yy = 0 (free to contract)."""
        g = plain_strip.geometry
        force = 600.0
        model = (
            MechanicsModel.from_materials(g, moisture_dependent_modulus=False)
            .without_band()
            .with_tension(force)
        )
        result = solve(plain_strip, model, concentration=None)

        expected = force / g.gross_section_area
        assert np.allclose(result.sxx, expected, rtol=1e-8)
        assert np.allclose(result.syy, 0.0, atol=1e-8 * expected)
        assert np.allclose(result.sxy, 0.0, atol=1e-8 * expected)

    def test_unconstrained_swelling_produces_no_stress(self, plain_strip):
        """A free polymer strip that swells uniformly carries nothing."""
        g = plain_strip.geometry
        model = (
            MechanicsModel.from_materials(g, moisture_dependent_modulus=False)
            .without_band()
            .without_tension()
        )
        c = np.full(plain_strip.n_vertices, 0.05)
        result = solve(plain_strip, model, c)
        assert np.max(np.abs(result.sxx)) < 1e-6
        assert np.max(np.abs(result.syy)) < 1e-6

    def test_constrained_swelling_approaches_the_tier_1_limit(self, plain_strip):
        """With a stiff full-width band, the FE must converge on the formula."""
        g = plain_strip.geometry
        model = MechanicsModel.from_materials(
            g, moisture_dependent_modulus=False
        ).without_tension()
        c_value = 0.06
        c = np.full(plain_strip.n_vertices, c_value)
        result = solve(plain_strip, model, c)

        e, nu, _, _ = pa66_at_moisture(model.stress_free_moisture)
        limit = constrained_swelling_stress(
            e, nu, model.swelling_coefficient, c_value, constraint="biaxial"
        )
        # The band is stiff but not rigid, so the FE value sits just below the
        # rigid-constraint limit -- by roughly 1 / (1 + stiffness ratio).
        interior = np.mean(result.sxx)
        assert interior < 0, "constrained swelling must be compressive"
        assert abs(interior) == pytest.approx(limit, rel=0.15)
        assert abs(interior) < limit

    def test_a_stiffer_band_gets_closer_to_the_rigid_limit(self, plain_strip):
        g = plain_strip.geometry
        base = MechanicsModel.from_materials(
            g, moisture_dependent_modulus=False
        ).without_tension()
        c = np.full(plain_strip.n_vertices, 0.06)

        soft = solve(plain_strip, replace(base, band_modulus=20000.0), c)
        stiff = solve(plain_strip, replace(base, band_modulus=2_000_000.0), c)
        assert abs(np.mean(stiff.sxx)) > abs(np.mean(soft.sxx))

    def test_no_band_and_no_moisture_gives_no_stress(self, plain_strip):
        model = (
            MechanicsModel.from_materials(plain_strip.geometry)
            .without_band()
            .without_tension()
        )
        result = solve(plain_strip, model, concentration=None)
        assert np.max(np.abs(result.sxx)) < 1e-9

    def test_moisture_softening_reduces_the_swelling_stress(self, plain_strip):
        """The water that swells the polymer also softens it; both must count."""
        g = plain_strip.geometry
        c = np.full(plain_strip.n_vertices, 0.085)
        fixed = MechanicsModel.from_materials(
            g, moisture_dependent_modulus=False
        ).without_tension()
        varying = replace(fixed, moisture_dependent_modulus=True)
        assert abs(np.mean(solve(plain_strip, varying, c).sxx)) < abs(
            np.mean(solve(plain_strip, fixed, c).sxx)
        )

    def test_superposition_of_tension_and_swelling(self, plain_strip):
        """The problem is linear, so the two load cases must add."""
        g = plain_strip.geometry
        base = MechanicsModel.from_materials(g, moisture_dependent_modulus=False)
        c = np.full(plain_strip.n_vertices, 0.04)

        only_tension = solve(plain_strip, base.without_tension().with_tension(500.0), None)
        only_swelling = solve(plain_strip, base.without_tension(), c)
        both = solve(plain_strip, base.with_tension(500.0), c)
        assert np.allclose(both.sxx, only_tension.sxx + only_swelling.sxx, atol=1e-6)

    def test_rejects_a_concentration_field_of_the_wrong_length(self, plain_strip):
        model = MechanicsModel.from_materials(plain_strip.geometry)
        with pytest.raises(ValueError, match="nodal field of length"):
            solve(plain_strip, model, np.zeros(3))


@pytest.mark.requires_gmsh
class TestStressConcentration:
    @pytest.fixture(scope="class")
    def single_hole_strip(self, plain_geometry):
        from ratchet_fea.mesh import MeshControls, build_strip_mesh

        return build_strip_mesh(plain_geometry, MeshControls(elements_around_hole=48))

    def test_isolated_hole_matches_the_howland_formula(
        self, single_hole_strip, plain_geometry
    ):
        """One hole, no band, no moisture -- exactly what Tier 1 describes.

        This is the load-bearing validation of the FE model: an independent
        closed-form solution for the same problem.
        """
        kt = stress_concentration_factors(single_hole_strip)
        expected = kt_hole_in_finite_width_strip(plain_geometry.d_over_W)
        assert kt["all_mean"] == pytest.approx(expected, rel=0.06)

    def test_kt_converges_with_mesh_refinement(self, plain_geometry):
        from ratchet_fea.mesh import MeshControls, build_strip_mesh

        values = []
        for n in (24, 48):
            strip = build_strip_mesh(
                plain_geometry, MeshControls(elements_around_hole=n)
            )
            values.append(stress_concentration_factors(strip)["all_mean"])
        assert abs(values[1] - values[0]) / values[0] < 0.03

    def test_a_row_of_holes_shields_the_interior_ones(self, small_strip):
        """The effect Tier 1 could not capture and Tier 2 exists to quantify."""
        kt = stress_concentration_factors(small_strip)
        isolated = kt_hole_in_finite_width_strip(small_strip.geometry.d_over_W)
        assert kt["all_mean"] < isolated

    def test_kt_scales_out_of_the_applied_load(self, small_strip):
        """A concentration factor must not depend on the load magnitude."""
        a = stress_concentration_factors(small_strip, force=200.0)["all_mean"]
        b = stress_concentration_factors(small_strip, force=900.0)["all_mean"]
        assert a == pytest.approx(b, rel=1e-6)

    def test_reports_one_factor_per_hole(self, small_strip, small_geometry):
        kt = stress_concentration_factors(small_strip)
        assert set(kt["per_hole"]) == set(small_geometry.hole_labels)


@pytest.mark.requires_gmsh
class TestPerforatedSolve:
    @pytest.fixture(scope="class")
    def wet_result(self, small_strip, small_geometry):
        model = MechanicsModel.from_materials(small_geometry)
        c = np.full(small_strip.n_vertices, 0.085)
        return solve(small_strip, model, c)

    def test_reports_every_hole(self, wet_result, small_geometry):
        assert set(wet_result.hole_edges) == set(small_geometry.hole_labels)

    def test_hole_edge_entries_are_finite(self, wet_result):
        for edge in wet_result.hole_edges.values():
            for key in ("max_principal", "min_principal", "von_mises"):
                assert np.isfinite(edge[key])
            assert edge["max_principal"] >= edge["min_principal"]

    def test_hole_wall_samples_go_all_the_way_round(self, wet_result):
        theta = wet_result.hole_edges["hole_0"]["theta"]
        assert theta.min() < -2.0 and theta.max() > 2.0

    def test_saturated_strap_is_in_compression(self, wet_result):
        """Constrained swelling: compressive, so not a crack driver on its own."""
        assert wet_result.peak_compressive < 0
        assert abs(wet_result.peak_compressive) > wet_result.peak_tensile

    def test_nodal_projections_are_defined_everywhere(self, wet_result, small_strip):
        assert wet_result.nodal_max_principal.shape == (small_strip.n_vertices,)
        assert np.all(np.isfinite(wet_result.nodal_von_mises))

    def test_von_mises_is_non_negative(self, wet_result):
        assert np.all(wet_result.von_mises_field >= 0)

    def test_removing_the_band_removes_the_swelling_stress(
        self, small_strip, small_geometry
    ):
        """Without something to push against, swelling costs nothing."""
        model = MechanicsModel.from_materials(small_geometry).without_tension()
        c = np.full(small_strip.n_vertices, 0.085)
        with_band = solve(small_strip, model, c)
        without = solve(small_strip, model.without_band(), c)
        assert abs(without.peak_compressive) < 0.1 * abs(with_band.peak_compressive)

    @staticmethod
    def _wet_ring(strip, geometry, c=0.085):
        """A wetted ring around one hole, dry elsewhere."""
        p = strip.mesh.p
        cx, cy = geometry.hole_centres[0]
        distance = np.hypot(p[0] - cx, p[1] - cy)
        return np.where(distance < geometry.hole_radius * 1.6, c, 0.0)

    def test_a_moisture_gradient_puts_dry_material_into_tension(
        self, small_strip, small_geometry
    ):
        """The mechanism that can actually open a crack during absorption.

        Wet material near the hole wall swells; still-dry material further in
        holds it back and is pulled into tension in the process. Shown here on
        the unreinforced section, where the mechanism is not masked -- see the
        next test for what the band does to it.
        """
        model = (
            MechanicsModel.from_materials(small_geometry)
            .without_tension()
            .without_band()
        )
        graded = self._wet_ring(small_strip, small_geometry)

        gradient = solve(small_strip, model, graded)
        uniform = solve(small_strip, model, np.zeros(small_strip.n_vertices))
        assert uniform.peak_tensile == pytest.approx(0.0, abs=1e-6)
        assert gradient.peak_tensile > 5.0

    def test_the_band_soaks_up_the_gradient_mismatch(
        self, small_strip, small_geometry
    ):
        """A result, not a check: the steel band suppresses gradient tension.

        The band is ~25x stiffer per unit width than the polymer, so it reacts
        the swelling mismatch itself instead of letting neighbouring polymer do
        it. Differential swelling therefore produces far less tension in a
        reinforced strap than in a plain one -- while the uniform constrained
        swelling compression, which the band causes, gets larger.
        """
        base = MechanicsModel.from_materials(small_geometry).without_tension()
        graded = self._wet_ring(small_strip, small_geometry)

        with_band = solve(small_strip, base, graded)
        without_band = solve(small_strip, base.without_band(), graded)

        assert with_band.peak_tensile < 0.25 * without_band.peak_tensile
        assert abs(with_band.peak_compressive) > abs(without_band.peak_compressive)
