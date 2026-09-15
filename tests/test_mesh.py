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
from ratchet_fea.mesh import (
    BAND,
    CUT_END,
    CUT_START,
    POLYMER,
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
        c = MeshControls(elements_around_hole=24, min_ligament_divisions=6)
        r = c.refined(2.0)
        assert r.elements_around_hole == 48
        assert r.min_ligament_divisions == 12

    def test_refined_can_coarsen(self):
        assert MeshControls(elements_around_hole=32).refined(0.5).elements_around_hole == 16

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"elements_around_hole": 4}, "at least 8"),
            ({"bulk_size_factor": 0.5}, "must be >= 1"),
            ({"min_ligament_divisions": 1}, "must be >= 2"),
            ({"grading": 0.0}, "must be positive"),
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

    def test_exposed_boundaries_are_the_real_surfaces_only(self, small_strip):
        """The cut faces are model truncation, not surfaces moisture enters."""
        exposed = small_strip.exposed_boundaries
        assert SIDE_LOWER in exposed and SIDE_UPPER in exposed
        assert all(h in exposed for h in small_strip.hole_boundaries)
        assert CUT_START not in exposed
        assert CUT_END not in exposed

    def test_load_and_restraint_boundaries(self, small_strip):
        assert small_strip.traction_boundary == CUT_END
        assert small_strip.restrained_boundary == CUT_START


@pytest.mark.requires_gmsh
class TestSubdomains:
    def test_band_and_polymer_regions_are_tagged(self, small_strip):
        assert BAND in small_strip.mesh.subdomains
        assert POLYMER in small_strip.mesh.subdomains
        assert small_strip.has_band_subdomain

    def test_band_elements_lie_inside_the_band_footprint(self, small_strip, small_geometry):
        mask = small_strip.band_element_mask()
        centroids = small_strip.element_centroids()
        mid = small_geometry.strap_width / 2 + small_geometry.steel_band_offset
        half = small_geometry.steel_band_width / 2
        assert np.all(np.abs(centroids[1, mask] - mid) <= half + 1e-6)
        assert np.all(np.abs(centroids[1, ~mask] - mid) >= half - 1e-6)

    def test_band_mask_covers_a_sensible_area_fraction(self, small_strip, small_geometry):
        mask = small_strip.band_element_mask()
        expected = small_geometry.steel_band_width / small_geometry.strap_width
        assert mask.mean() == pytest.approx(expected, rel=0.15)

    def test_vertex_mask_is_consistent_with_the_element_mask(self, small_strip):
        v_mask = small_strip.band_vertex_mask()
        e_mask = small_strip.band_element_mask()
        # Every vertex of a band element must be inside the band footprint.
        band_vertices = np.unique(small_strip.mesh.t[:, e_mask].ravel())
        assert np.all(v_mask[band_vertices])

    def test_a_full_width_band_covers_everything(self, small_geometry, coarse_controls):
        g = replace(small_geometry, steel_band_width=small_geometry.strap_width)
        strip = build_strip_mesh(g, coarse_controls)
        assert strip.band_element_mask().all()
        assert strip.band_vertex_mask().all()

    def test_no_band_means_no_band_elements(self, small_geometry, coarse_controls):
        g = replace(small_geometry, steel_band_width=0.0, steel_band_thickness=0.0)
        strip = build_strip_mesh(g, coarse_controls)
        assert not strip.band_element_mask().any()


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
