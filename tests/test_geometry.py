"""Strap geometry: derived quantities, validation, and re-parameterisation.

The last group matters most. The whole repository is built on the promise that
replacing the placeholder dimensions with measured ones is a one-line change,
so these tests check that derived quantities actually follow the inputs rather
than having been frozen at their placeholder values somewhere.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, StrapGeometry
from ratchet_fea.provenance import Provenance


class TestDerivedQuantities:
    def test_section_areas(self, geometry):
        assert geometry.gross_section_area == pytest.approx(
            geometry.strap_width * geometry.strap_thickness
        )
        assert geometry.net_section_area == pytest.approx(
            (geometry.strap_width - geometry.hole_diameter) * geometry.strap_thickness
        )
        assert geometry.net_section_area < geometry.gross_section_area

    def test_ratios(self, geometry):
        assert geometry.d_over_W == pytest.approx(
            geometry.hole_diameter / geometry.strap_width
        )
        assert geometry.pitch_over_d == pytest.approx(
            geometry.hole_pitch / geometry.hole_diameter
        )

    def test_hole_radius_is_half_the_diameter(self, geometry):
        assert geometry.hole_radius == pytest.approx(0.5 * geometry.hole_diameter)

    def test_modelled_length_spans_the_row_plus_both_margins(self, geometry):
        expected = (geometry.n_holes - 1) * geometry.hole_pitch + 2 * geometry.end_margin
        assert geometry.modelled_length == pytest.approx(expected)

    def test_hole_centres_are_evenly_spaced_on_the_centreline(self, geometry):
        centres = geometry.hole_centres
        assert centres.shape == (geometry.n_holes, 2)
        assert np.allclose(np.diff(centres[:, 0]), geometry.hole_pitch)
        assert np.allclose(centres[:, 1], geometry.strap_width / 2)
        assert centres[0, 0] == pytest.approx(geometry.end_margin)

    def test_hole_row_offset_shifts_the_centres(self, geometry):
        offset = replace(geometry, hole_row_offset=2.0)
        assert np.allclose(
            offset.hole_centres[:, 1], geometry.strap_width / 2 + 2.0
        )

    def test_ligaments(self, geometry):
        assert geometry.side_ligament == pytest.approx(
            (geometry.strap_width - geometry.hole_diameter) / 2
        )
        assert geometry.inter_hole_ligament == pytest.approx(
            geometry.hole_pitch - geometry.hole_diameter
        )

    def test_offset_row_gives_the_narrower_of_the_two_side_ligaments(self, geometry):
        offset = replace(geometry, hole_row_offset=3.0)
        assert offset.side_ligament < geometry.side_ligament

    def test_labels_and_interior_classification(self, geometry):
        assert geometry.hole_labels == [f"hole_{i}" for i in range(geometry.n_holes)]
        assert not geometry.is_interior_hole(0)
        assert not geometry.is_interior_hole(geometry.n_holes - 1)
        assert geometry.is_interior_hole(1)

    def test_a_single_hole_has_no_interior_holes(self):
        one = replace(PLACEHOLDER_GEOMETRY, n_holes=1)
        assert not one.is_interior_hole(0)


class TestSteelBand:
    def test_band_presence(self, geometry):
        assert geometry.has_steel_band
        assert not replace(geometry, steel_band_thickness=0.0).has_steel_band
        assert not replace(geometry, steel_band_width=0.0).has_steel_band

    def test_polymer_thickness_in_band_excludes_the_steel(self, geometry):
        assert geometry.polymer_thickness_in_band == pytest.approx(
            geometry.strap_thickness - geometry.steel_band_thickness
        )

    def test_band_full_width_detection(self, geometry):
        assert not geometry.band_full_width
        assert replace(geometry, steel_band_width=geometry.strap_width).band_full_width


class TestDiffusionGeometry:
    def test_plain_region_uses_half_the_full_thickness(self, geometry):
        assert geometry.diffusion_half_thickness(in_band_region=False) == pytest.approx(
            geometry.strap_thickness / 2
        )

    def test_band_region_is_a_one_sided_skin(self, geometry):
        """A layer sealed on one face behaves as half a slab of twice its depth."""
        assert geometry.diffusion_half_thickness(in_band_region=True) == pytest.approx(
            geometry.polymer_thickness_in_band / 2
        )

    def test_the_band_makes_the_polymer_skin_wet_faster(self, geometry):
        assert geometry.diffusion_half_thickness(True) < geometry.diffusion_half_thickness(False)

    def test_without_a_band_both_regions_agree(self, geometry):
        plain = replace(geometry, steel_band_thickness=0.0, steel_band_width=0.0)
        assert plain.diffusion_half_thickness(True) == pytest.approx(
            plain.diffusion_half_thickness(False)
        )

    def test_in_plane_length_is_the_shortest_path_to_dry_material(self, geometry):
        assert geometry.in_plane_diffusion_length == pytest.approx(
            min(geometry.side_ligament, geometry.inter_hole_ligament / 2)
        )


class TestEndMargin:
    def test_defaults_to_one_strap_width(self):
        """St Venant: the truncation planes must not reach the first hole."""
        g = StrapGeometry(strap_width=30.0)
        assert g.end_margin == pytest.approx(30.0)

    def test_follows_a_changed_strap_width(self):
        narrow = StrapGeometry(strap_width=12.0, steel_band_width=8.0)
        assert narrow.end_margin == pytest.approx(12.0)

    def test_an_explicit_value_is_respected(self):
        assert StrapGeometry(end_margin=7.0).end_margin == pytest.approx(7.0)


class TestValidation:
    def test_hole_must_fit_within_the_width(self):
        with pytest.raises(ValueError, match="breaks out of the strap width"):
            StrapGeometry(hole_diameter=30.0, hole_pitch=40.0, strap_width=25.0)

    def test_offset_row_must_not_break_out(self):
        with pytest.raises(ValueError, match="breaks out of the strap width"):
            StrapGeometry(hole_row_offset=11.0)

    def test_pitch_must_exceed_the_diameter(self):
        with pytest.raises(ValueError, match="must exceed hole_diameter"):
            StrapGeometry(hole_pitch=3.0, hole_diameter=4.0, n_holes=3)

    def test_a_single_hole_needs_no_pitch_check(self):
        StrapGeometry(hole_pitch=1.0, hole_diameter=4.0, n_holes=1)

    def test_end_margin_must_clear_the_hole(self):
        with pytest.raises(ValueError, match="end_margin must exceed"):
            StrapGeometry(end_margin=1.0, hole_diameter=4.0)

    def test_band_cannot_be_thicker_than_the_strap(self):
        with pytest.raises(ValueError, match="cannot be thicker"):
            StrapGeometry(steel_band_thickness=5.0, strap_thickness=3.0)

    def test_band_cannot_be_wider_than_the_strap(self):
        with pytest.raises(ValueError, match="cannot be wider"):
            StrapGeometry(steel_band_width=40.0, strap_width=25.0)

    def test_at_least_one_hole(self):
        with pytest.raises(ValueError, match="at least 1"):
            StrapGeometry(n_holes=0)

    def test_negative_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="must be positive"):
            StrapGeometry(strap_thickness=-1.0)

    def test_unpierced_band_must_clear_the_hole_row(self):
        """If the holes do not pierce the band, the band must be out of their way.

        This catches the physically incoherent combination where the band is
        declared continuous yet sits right where the holes are punched.
        """
        with pytest.raises(ValueError, match="overlaps the hole row"):
            StrapGeometry(holes_pierce_band=False, steel_band_width=18.0)

    def test_unpierced_band_is_accepted_when_offset_clear(self):
        g = StrapGeometry(
            holes_pierce_band=False,
            steel_band_width=6.0,
            steel_band_offset=8.0,
            hole_diameter=4.0,
        )
        assert not g.holes_pierce_band


class TestReparameterisation:
    """Measured dimensions must propagate; nothing may be frozen internally."""

    def test_every_derived_quantity_follows_a_new_strap(self):
        measured = StrapGeometry(
            strap_width=31.7,
            strap_thickness=2.4,
            hole_diameter=5.1,
            hole_pitch=12.6,
            n_holes=4,
            steel_band_width=20.0,
            steel_band_thickness=0.55,
        )
        assert measured.d_over_W == pytest.approx(5.1 / 31.7)
        assert measured.net_section_area == pytest.approx((31.7 - 5.1) * 2.4)
        assert measured.inter_hole_ligament == pytest.approx(12.6 - 5.1)
        assert measured.side_ligament == pytest.approx((31.7 - 5.1) / 2)
        assert measured.modelled_length == pytest.approx(3 * 12.6 + 2 * 31.7)
        assert measured.hole_centres.shape == (4, 2)
        assert measured.polymer_thickness_in_band == pytest.approx(2.4 - 0.55)
        assert measured.diffusion_half_thickness(False) == pytest.approx(1.2)

    def test_geometry_is_immutable_so_a_study_cannot_drift(self, geometry):
        with pytest.raises(Exception):
            geometry.strap_width = 99.0

    def test_replace_produces_an_independent_geometry(self, geometry):
        other = replace(geometry, hole_diameter=6.0)
        assert geometry.hole_diameter == 4.0
        assert other.hole_diameter == 6.0


class TestProvenanceIsRecorded:
    def test_every_field_has_a_source(self, geometry):
        from dataclasses import fields

        for f in fields(geometry):
            assert f.name in geometry.SOURCES, f"{f.name} has no recorded provenance"

    def test_nothing_is_yet_claimed_as_measured(self, geometry):
        """Guards against a placeholder quietly being promoted to fact."""
        assert all(
            s.provenance is not Provenance.MEASURED for s in geometry.SOURCES.values()
        )

    def test_the_whole_geometry_shows_up_in_the_report(self, geometry):
        from dataclasses import fields

        reported = {it.name for it in geometry.verification_items()}
        assert reported == {f.name for f in fields(geometry)}

    def test_every_source_says_how_to_verify_it(self, geometry):
        for name, source in geometry.SOURCES.items():
            assert source.verify_by, f"{name} does not say how to verify it"

    def test_summary_mentions_the_key_dimensions(self, geometry):
        text = geometry.summary()
        assert "net section" in text
        assert "d/W" in text
