"""Static plane-stress solve.

Constitutive helpers against closed-form identities, then the assembled solve
against patch tests -- uniform states it must reproduce exactly -- before
anything is asked of it near a crack tip.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ratchet_fea.geometry import StrapGeometry
from ratchet_fea.materials import PA66_DAM, MoistureCondition, grade_for
from ratchet_fea.mechanics import (
    MechanicsModel,
    band_load_sharing,
    plane_stress_moduli,
    principal_stresses,
    solve,
    von_mises,
)
from ratchet_fea.mesh import (
    CUT_END,
    CUT_START,
    SIDE_LOWER,
    SIDE_UPPER,
    MeshControls,
    StripMesh,
)


class TestPlaneStressModuli:
    def test_reproduces_the_standard_relations(self):
        e, nu = 2000.0, 0.35
        q11, q12, q66 = plane_stress_moduli(e, nu)
        assert q11 == pytest.approx(e / (1 - nu**2))
        assert q12 == pytest.approx(nu * e / (1 - nu**2))
        assert q66 == pytest.approx(e / (2 * (1 + nu)))

    def test_shear_modulus_identity(self):
        q11, q12, q66 = plane_stress_moduli(1500.0, 0.4)
        assert q66 == pytest.approx(0.5 * (q11 - q12))

    def test_uniaxial_stress_recovers_the_modulus(self):
        e, nu = 1800.0, 0.38
        q11, q12, _ = plane_stress_moduli(e, nu)
        exx = 1e-3
        assert q11 * exx + q12 * (-nu * exx) == pytest.approx(e * exx)

    def test_rejects_incompressible_poisson_ratio(self):
        with pytest.raises(ValueError, match="plane stress"):
            plane_stress_moduli(1000.0, 0.5)


class TestStressInvariants:
    def test_principal_stresses_of_pure_shear(self):
        s1, s2 = principal_stresses(0.0, 0.0, 5.0)
        assert s1 == pytest.approx(5.0)
        assert s2 == pytest.approx(-5.0)

    def test_invariants_are_preserved(self):
        rng = np.random.default_rng(1)
        sxx, syy, sxy = rng.normal(size=(3, 100)) * 10
        s1, s2 = principal_stresses(sxx, syy, sxy)
        assert np.allclose(s1 + s2, sxx + syy)
        assert np.allclose(s1 * s2, sxx * syy - sxy**2)

    def test_von_mises_of_uniaxial_equals_the_stress(self):
        assert von_mises(25.0, 0.0, 0.0) == pytest.approx(25.0)

    def test_von_mises_of_pure_shear(self):
        assert von_mises(0.0, 0.0, 10.0) == pytest.approx(10.0 * math.sqrt(3))


class TestModel:
    def test_from_materials_reads_the_grade(self, geometry):
        model = MechanicsModel.from_materials(geometry)
        grade = grade_for(MoistureCondition.RH50)
        assert model.youngs_modulus == pytest.approx(grade.youngs_modulus.nominal)

    def test_remote_stress_is_force_over_gross_area(self, geometry):
        model = MechanicsModel.from_materials(geometry, force=900.0)
        assert model.remote_stress == pytest.approx(900.0 / geometry.gross_section_area)

    def test_condition_selects_a_different_material(self, geometry):
        dry = MechanicsModel.from_materials(geometry, condition=MoistureCondition.DAM)
        wet = MechanicsModel.from_materials(geometry, condition=MoistureCondition.RH50)
        assert dry.youngs_modulus > wet.youngs_modulus

    def test_validation(self, geometry):
        with pytest.raises(ValueError, match="Young's modulus must be positive"):
            MechanicsModel(youngs_modulus=0.0, poisson_ratio=0.3, remote_stress=1.0)
        with pytest.raises(ValueError, match="Poisson's ratio"):
            MechanicsModel(youngs_modulus=1000.0, poisson_ratio=0.5, remote_stress=1.0)


class TestBandLoadSharing:
    def test_a_steel_band_carries_almost_all_of_it(self, geometry):
        """~50x the stiffness per unit width, so the polymer barely loads.

        The FE model ignores this on purpose; the number quantifies how
        conservative that choice is.
        """
        assert band_load_sharing(geometry) > 0.9

    def test_no_band_means_no_sharing(self, geometry):
        from dataclasses import replace

        plain = replace(geometry, steel_band_width=0.0, steel_band_thickness=0.0)
        assert band_load_sharing(plain) == 0.0

    def test_a_stiffer_polymer_takes_more_of_the_load(self, geometry):
        """Dry PA66 is twice as stiff, so it shields itself slightly better."""
        dry = band_load_sharing(geometry, PA66_DAM)
        wet = band_load_sharing(geometry, grade_for(MoistureCondition.RH50))
        assert dry < wet

    def test_a_thicker_band_takes_more(self, geometry):
        from dataclasses import replace

        thick = replace(geometry, steel_band_thickness=1.5)
        assert band_load_sharing(thick) > band_load_sharing(geometry)


@pytest.fixture(scope="module")
def plain_strip():
    """A hole-free rectangular strip, wrapped so the solver accepts it.

    Patch tests need a domain whose exact answer is known; reusing the
    perforated mesh would confound a solver error with a stress concentration.
    """
    from skfem import MeshTri

    geometry = StrapGeometry(n_holes=1, end_margin=30.0)
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
    def test_uniform_tension_reproduces_the_remote_stress(self, plain_strip):
        """sigma_xx = sigma everywhere; sigma_yy = 0 (free to contract)."""
        model = MechanicsModel.from_materials(plain_strip.geometry, force=600.0)
        result = solve(plain_strip, model)
        assert np.allclose(result.sxx, model.remote_stress, rtol=1e-8)
        assert np.allclose(result.syy, 0.0, atol=1e-8 * model.remote_stress)
        assert np.allclose(result.sxy, 0.0, atol=1e-8 * model.remote_stress)

    def test_stress_is_independent_of_the_modulus(self, plain_strip):
        """A statically determinate patch: stiffness cannot change the stress."""
        from dataclasses import replace

        model = MechanicsModel.from_materials(plain_strip.geometry)
        soft = solve(plain_strip, replace(model, youngs_modulus=500.0))
        stiff = solve(plain_strip, replace(model, youngs_modulus=5000.0))
        assert np.allclose(soft.sxx, stiff.sxx, rtol=1e-8)

    def test_zero_load_gives_zero_stress(self, plain_strip):
        model = MechanicsModel.from_materials(plain_strip.geometry).with_remote_stress(0.0)
        assert np.max(np.abs(solve(plain_strip, model).sxx)) < 1e-12

    def test_stress_scales_linearly_with_load(self, plain_strip):
        model = MechanicsModel.from_materials(plain_strip.geometry)
        single = solve(plain_strip, model)
        triple = solve(plain_strip, model.with_remote_stress(3 * model.remote_stress))
        assert np.allclose(triple.sxx, 3.0 * single.sxx, rtol=1e-8)

    def test_compression_reverses_the_sign(self, plain_strip):
        model = MechanicsModel.from_materials(plain_strip.geometry)
        pushed = solve(plain_strip, model.with_remote_stress(-model.remote_stress))
        assert np.all(pushed.sxx < 0)


@pytest.mark.requires_gmsh
class TestPerforatedSolve:
    def test_the_hole_concentrates_stress(self, small_strip, small_geometry):
        """Roughly Kt times the net-section stress, from an independent route."""
        from ratchet_fea.analytical import (
            kt_hole_in_finite_width_strip,
            net_section_stress,
        )

        model = MechanicsModel.from_materials(small_geometry)
        result = solve(small_strip, model)
        expected = kt_hole_in_finite_width_strip(
            small_geometry.d_over_W
        ) * net_section_stress(small_geometry.service_tension, small_geometry)
        assert result.peak_tensile == pytest.approx(expected, rel=0.2)

    def test_a_crack_raises_the_peak_stress(self, small_strip, cracked_strip,
                                            small_geometry):
        model = MechanicsModel.from_materials(small_geometry)
        assert solve(cracked_strip, model).peak_tensile > solve(
            small_strip, model
        ).peak_tensile

    def test_nodal_projection_covers_every_vertex(self, cracked_strip, small_geometry):
        model = MechanicsModel.from_materials(small_geometry)
        result = solve(cracked_strip, model)
        nodal = result.nodal(result.von_mises_field)
        assert nodal.shape == (cracked_strip.n_vertices,)
        assert np.all(np.isfinite(nodal))

    def test_von_mises_is_non_negative(self, cracked_strip, small_geometry):
        model = MechanicsModel.from_materials(small_geometry)
        assert np.all(solve(cracked_strip, model).von_mises_field >= 0)
