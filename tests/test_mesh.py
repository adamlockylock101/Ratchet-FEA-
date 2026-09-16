"""Mesh generation.

A silently mis-tagged boundary would give a plausible-looking but wrong stress
field, which is the worst outcome for an investigation, so most of these tests
are about the tags rather than the element quality.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ratchet_fea.geometry import StrapGeometry
from ratchet_fea.geometry import CrackGeometry, CrackOrientation
from ratchet_fea.mesh import (
    CRACK_LOWER,
    CRACK_UPPER,
    CUT_END,
    CUT_START,
    SIDE_LOWER,
    SIDE_UPPER,
    MeshControls,
    build_strip_mesh,
)


class TestMeshControls:
    def test_hole_edge_size_follows_the_hole_circumference(self, geometry):
        c = MeshControls(elements_around_hole=32)
        expected = 2 * np.pi * geometry.hole_radius / 32
        assert c.hole_edge_size(geometry) == pytest.approx(expected)

    def test_hole_edge_size_rescales_with_a_measured_hole(self, geometry):
        """Density is relative, so a smaller hole keeps the same resolution."""
        c = MeshControls(elements_around_hole=32)
        smaller = replace(geometry, hole_diameter=2.0)
        assert c.hole_edge_size(smaller) == pytest.approx(
            0.5 * c.hole_edge_size(geometry)
        )

    def test_bulk_size_respects_the_narrowest_ligament(self, geometry):
        tight = replace(geometry, hole_pitch=5.0)
        c = MeshControls(elements_around_hole=8, bulk_size_factor=100.0,
                         min_ligament_divisions=4)
        assert c.bulk_size(tight) <= tight.inter_hole_ligament / 4 + 1e-12

    def test_bulk_is_never_finer_than_the_hole_edge(self, geometry):
        c = MeshControls()
        assert c.bulk_size(geometry) >= c.hole_edge_size(geometry) * 0.999

    def test_refined_increases_resolution(self):
        c = MeshControls(
            elements_around_hole=24, min_ligament_divisions=6, elements_along_crack=20
        )
        r = c.refined(2.0)
        assert r.elements_around_hole == 48
        assert r.min_ligament_divisions == 12
        assert r.elements_along_crack == 40
        assert r.tip_size_fraction == pytest.approx(c.tip_size_fraction / 2)

    def test_crack_sizes_scale_with_the_crack(self):
        """Density is relative, so a short crack is resolved as finely."""
        c = MeshControls(elements_along_crack=20, tip_size_fraction=0.02)
        short = CrackGeometry(length=1.0)
        long = CrackGeometry(length=4.0)
        assert c.crack_size(long) == pytest.approx(4 * c.crack_size(short))
        assert c.tip_size(short) == pytest.approx(0.02)

    def test_refined_can_coarsen(self):
        assert MeshControls(elements_around_hole=32).refined(0.5).elements_around_hole == 16

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"elements_around_hole": 4}, "at least 8"),
            ({"bulk_size_factor": 0.5}, "must be >= 1"),
            ({"min_ligament_divisions": 1}, "must be >= 2"),
            ({"grading": 0.0}, "must be positive"),
            ({"elements_along_crack": 2}, "at least 4 elements along a crack"),
            ({"tip_size_fraction": 0.0}, r"must lie in \(0, 1\)"),
            ({"tip_size_fraction": 1.5}, r"must lie in \(0, 1\)"),
        ],
    )
    def test_validation(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MeshControls(**kwargs)

    def test_refinement_factor_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            MeshControls().refined(0.0)


@pytest.mark.requires_gmsh
class TestBoundaryTagging:
    def test_all_outer_boundaries_are_present(self, small_strip):
        for name in (CUT_START, CUT_END, SIDE_LOWER, SIDE_UPPER):
            assert name in small_strip.mesh.boundaries

    def test_every_hole_is_tagged_separately(self, small_strip, small_geometry):
        assert small_strip.hole_boundaries == small_geometry.hole_labels

    def test_hole_facets_actually_lie_on_the_hole(self, small_strip, small_geometry):
        mesh = small_strip.mesh
        for i, name in enumerate(small_strip.hole_boundaries):
            cx, cy = small_geometry.hole_centres[i]
            vertices = np.unique(mesh.facets[:, mesh.boundaries[name]].ravel())
            radii = np.hypot(mesh.p[0, vertices] - cx, mesh.p[1, vertices] - cy)
            assert np.allclose(radii, small_geometry.hole_radius, rtol=2e-2)

    def test_cut_faces_are_at_the_ends(self, small_strip, small_geometry):
        mesh = small_strip.mesh
        start = np.unique(mesh.facets[:, mesh.boundaries[CUT_START]].ravel())
        end = np.unique(mesh.facets[:, mesh.boundaries[CUT_END]].ravel())
        assert np.allclose(mesh.p[0, start], 0.0, atol=1e-9)
        assert np.allclose(mesh.p[0, end], small_geometry.modelled_length, atol=1e-9)

    def test_side_edges_are_at_the_strap_edges(self, small_strip, small_geometry):
        mesh = small_strip.mesh
        lower = np.unique(mesh.facets[:, mesh.boundaries[SIDE_LOWER]].ravel())
        upper = np.unique(mesh.facets[:, mesh.boundaries[SIDE_UPPER]].ravel())
        assert np.allclose(mesh.p[1, lower], 0.0, atol=1e-9)
        assert np.allclose(mesh.p[1, upper], small_geometry.strap_width, atol=1e-9)

    def test_load_and_restraint_boundaries(self, small_strip):
        assert small_strip.traction_boundary == CUT_END
        assert small_strip.restrained_boundary == CUT_START


@pytest.mark.requires_gmsh
class TestMeshQuality:
    def test_meshed_area_matches_the_geometry(self, small_strip, small_geometry):
        from ratchet_fea.mesh import _element_areas

        area = float(np.sum(_element_areas(small_strip.mesh)))
        expected = (
            small_geometry.modelled_length * small_geometry.strap_width
            - small_geometry.n_holes * np.pi * small_geometry.hole_radius**2
        )
        assert area == pytest.approx(expected, rel=0.02)

    def test_all_elements_have_positive_area(self, small_strip):
        from ratchet_fea.mesh import _element_areas

        assert np.all(_element_areas(small_strip.mesh) > 0)

    def test_no_node_lies_inside_a_hole(self, small_strip, small_geometry):
        p = small_strip.mesh.p
        for cx, cy in small_geometry.hole_centres:
            r = np.hypot(p[0] - cx, p[1] - cy)
            assert np.all(r > small_geometry.hole_radius * 0.98)

    def test_nodes_stay_within_the_outer_envelope(self, small_strip, small_geometry):
        p = small_strip.mesh.p
        assert p[0].min() >= -1e-9
        assert p[0].max() <= small_geometry.modelled_length + 1e-9
        assert p[1].min() >= -1e-9
        assert p[1].max() <= small_geometry.strap_width + 1e-9

    def test_refining_produces_more_elements(self, small_geometry, coarse_controls):
        coarse = build_strip_mesh(small_geometry, coarse_controls)
        fine = build_strip_mesh(small_geometry, coarse_controls.refined(2.0))
        assert fine.n_elements > coarse.n_elements

    def test_elements_are_finer_at_the_hole_than_in_the_bulk(
        self, small_strip, small_geometry
    ):
        from ratchet_fea.mesh import _element_areas

        areas = _element_areas(small_strip.mesh)
        centroids = small_strip.element_centroids()
        cx, cy = small_geometry.hole_centres[0]
        distance = np.hypot(centroids[0] - cx, centroids[1] - cy)
        near = areas[distance < small_geometry.hole_radius * 1.5]
        far = areas[distance > small_geometry.hole_radius * 4]
        assert near.mean() < far.mean()


@pytest.mark.requires_gmsh
class TestReparameterisation:
    """A measured strap must remesh without any code change."""

    def test_a_completely_different_strap_meshes(self, coarse_controls):
        measured = StrapGeometry(
            strap_width=31.7,
            strap_thickness=2.4,
            hole_diameter=5.1,
            hole_pitch=12.6,
            n_holes=3,
            steel_band_width=22.0,
            steel_band_thickness=0.55,
        )
        strip = build_strip_mesh(measured, coarse_controls)
        assert len(strip.hole_boundaries) == 3
        assert strip.n_elements > 0

    def test_hole_count_follows_the_geometry(self, small_geometry, coarse_controls):
        for n in (1, 2, 4):
            g = replace(small_geometry, n_holes=n)
            strip = build_strip_mesh(g, coarse_controls)
            assert len(strip.hole_boundaries) == n

    def test_a_narrow_strap_meshes(self, coarse_controls):
        narrow = StrapGeometry(
            strap_width=10.0,
            strap_thickness=1.5,
            hole_diameter=2.0,
            hole_pitch=5.0,
            n_holes=2,
            steel_band_width=6.0,
            steel_band_thickness=0.3,
        )
        strip = build_strip_mesh(narrow, coarse_controls)
        assert strip.n_elements > 0

    def test_summary_reports_what_was_built(self, small_strip):
        text = small_strip.summary()
        assert "elements" in text
        assert "hole-edge size" in text


@pytest.mark.requires_gmsh
class TestCrackSeeding:
    def test_both_flanks_are_tagged(self, cracked_strip):
        for name in (CRACK_UPPER, CRACK_LOWER):
            assert name in cracked_strip.mesh.boundaries

    def test_the_flanks_have_matching_facet_counts(self, cracked_strip):
        upper = cracked_strip.mesh.boundaries[CRACK_UPPER]
        lower = cracked_strip.mesh.boundaries[CRACK_LOWER]
        assert len(upper) == len(lower) > 0

    def test_the_flanks_are_geometrically_coincident(self, cracked_strip):
        """They are the same line; only the node numbering differs."""
        mesh = cracked_strip.mesh
        mid = mesh.p[:, mesh.facets].mean(axis=1)
        upper = np.sort(mid[:, mesh.boundaries[CRACK_UPPER]], axis=1)
        lower = np.sort(mid[:, mesh.boundaries[CRACK_LOWER]], axis=1)
        assert np.allclose(upper, lower)

    def test_the_flanks_are_free_surfaces(self, cracked_strip):
        """Which is exactly what a traction-free crack is."""
        mesh = cracked_strip.mesh
        boundary = set(mesh.boundary_facets().tolist())
        for name in (CRACK_UPPER, CRACK_LOWER):
            assert set(mesh.boundaries[name].tolist()) <= boundary

    def test_exactly_one_node_sits_at_the_tip(self, cracked_strip):
        """A second node there would mean the crack is cut clean through."""
        mesh = cracked_strip.mesh
        tip = cracked_strip.crack_tip()
        at_tip = np.hypot(mesh.p[0] - tip[0], mesh.p[1] - tip[1]) < 1e-9
        assert at_tip.sum() == 1

    def test_cracking_adds_nodes_but_not_elements(self, small_geometry,
                                                  coarse_controls, transverse_crack):
        """Node splitting duplicates vertices; it does not change the mesh."""
        cracked = build_strip_mesh(small_geometry, coarse_controls, transverse_crack)
        assert cracked.n_vertices > 0
        from ratchet_fea.mesh import _element_areas

        expected = (
            small_geometry.modelled_length * small_geometry.strap_width
            - small_geometry.n_holes * np.pi * small_geometry.hole_radius**2
        )
        assert float(np.sum(_element_areas(cracked.mesh))) == pytest.approx(
            expected, rel=0.02
        )

    def test_both_orientations_mesh(self, small_geometry, coarse_controls):
        for orientation in CrackOrientation:
            strip = build_strip_mesh(
                small_geometry,
                coarse_controls,
                CrackGeometry(length=1.5, orientation=orientation),
            )
            assert strip.is_cracked
            assert CRACK_UPPER in strip.mesh.boundaries

    def test_the_tip_lands_where_the_geometry_says(self, small_geometry,
                                                   coarse_controls):
        crack = CrackGeometry(length=2.0)
        strip = build_strip_mesh(small_geometry, coarse_controls, crack)
        assert np.allclose(strip.crack_tip(), small_geometry.crack_tip(crack))

    def test_an_oversized_crack_is_rejected(self, small_geometry, coarse_controls):
        with pytest.raises(ValueError, match="does not fit"):
            build_strip_mesh(
                small_geometry, coarse_controls, CrackGeometry(length=50.0)
            )

    def test_an_uncracked_mesh_has_no_flanks(self, small_strip):
        assert not small_strip.is_cracked
        assert CRACK_UPPER not in small_strip.mesh.boundaries

    def test_uncracked_mesh_refuses_crack_queries(self, small_strip):
        with pytest.raises(ValueError, match="no crack"):
            small_strip.crack_tip()

    def test_elements_are_finest_at_the_tip(self, cracked_strip):
        from ratchet_fea.mesh import _element_areas

        areas = _element_areas(cracked_strip.mesh)
        centroids = cracked_strip.element_centroids()
        tip = cracked_strip.crack_tip()
        distance = np.hypot(centroids[0] - tip[0], centroids[1] - tip[1])
        near = areas[distance < 0.2]
        far = areas[distance > 3.0]
        assert len(near) > 0
        assert near.mean() < far.mean()

    def test_summary_describes_the_crack(self, cracked_strip):
        text = cracked_strip.summary()
        assert "crack" in text
        assert "ligament ahead" in text
