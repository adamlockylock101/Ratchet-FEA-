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

from ratchet_fea.geometry import (
    PLACEHOLDER_GEOMETRY,
    CrackGeometry,
    CrackOrientation,
    StrapGeometry,
)
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


class TestCrackOrientation:
    def test_transverse_is_mode_i_under_axial_tension(self):
        assert CrackOrientation.TRANSVERSE.is_mode_i_under_axial_tension

    def test_longitudinal_is_not(self):
        """It lies parallel to the load, so tension cannot open it."""
        assert not CrackOrientation.LONGITUDINAL.is_mode_i_under_axial_tension

    def test_hoop_stress_factors_come_from_the_kirsch_field(self):
        """+3 sigma at 90 degrees from the load axis, -1 sigma on it."""
        assert CrackOrientation.TRANSVERSE.hole_hoop_stress_factor == 3.0
        assert CrackOrientation.LONGITUDINAL.hole_hoop_stress_factor == -1.0

    def test_the_signs_are_opposite(self):
        """Which is the whole reason the two orientations behave differently."""
        assert (
            CrackOrientation.TRANSVERSE.hole_hoop_stress_factor
            * CrackOrientation.LONGITUDINAL.hole_hoop_stress_factor
            < 0
        )


class TestCrackGeometry:
    def test_direction_and_normal_are_perpendicular(self):
        for orientation in CrackOrientation:
            crack = CrackGeometry(length=1.0, orientation=orientation)
            d, n = np.array(crack.direction), np.array(crack.normal)
            assert d @ n == pytest.approx(0.0)
            assert np.linalg.norm(d) == pytest.approx(1.0)
            assert np.linalg.norm(n) == pytest.approx(1.0)

    def test_transverse_runs_across_the_strap(self):
        assert CrackGeometry(length=1.0).direction == (0.0, 1.0)

    def test_longitudinal_runs_along_it(self):
        crack = CrackGeometry(length=1.0, orientation=CrackOrientation.LONGITUDINAL)
        assert crack.direction == (1.0, 0.0)

    def test_side_flips_the_direction(self):
        assert CrackGeometry(length=1.0, side=-1).direction == (0.0, -1.0)

    def test_length_must_be_positive(self):
        with pytest.raises(ValueError, match="must be positive"):
            CrackGeometry(length=0.0)

    def test_side_must_be_plus_or_minus_one(self):
        with pytest.raises(ValueError, match="must be \\+1 or -1"):
            CrackGeometry(length=1.0, side=2)


class TestCrackPlacement:
    def test_mouth_sits_on_the_hole_wall(self, geometry):
        for orientation in CrackOrientation:
            crack = CrackGeometry(length=2.0, orientation=orientation)
            mouth = geometry.crack_mouth(crack)
            centre = geometry.hole_centres[geometry.default_crack_hole()]
            assert np.hypot(*(mouth - centre)) == pytest.approx(geometry.hole_radius)

    def test_tip_is_one_crack_length_beyond_the_mouth(self, geometry):
        crack = CrackGeometry(length=2.5)
        assert np.hypot(
            *(geometry.crack_tip(crack) - geometry.crack_mouth(crack))
        ) == pytest.approx(2.5)

    def test_default_hole_is_an_interior_one(self, geometry):
        index = geometry.default_crack_hole()
        assert geometry.is_interior_hole(index)

    def test_an_explicit_hole_index_is_honoured(self, geometry):
        crack = CrackGeometry(length=1.0, hole_index=0)
        assert geometry.crack_mouth(crack)[0] == pytest.approx(
            geometry.hole_centres[0][0]
        )

    def test_an_out_of_range_hole_index_is_rejected(self, geometry):
        with pytest.raises(ValueError, match="outside"):
            geometry.crack_mouth(CrackGeometry(length=1.0, hole_index=99))

    def test_transverse_ligament_runs_to_the_free_edge(self, geometry):
        crack = CrackGeometry(length=1.0)
        assert geometry.max_crack_length(crack) == pytest.approx(
            geometry.side_ligament
        )

    def test_longitudinal_ligament_runs_to_the_next_hole(self, geometry):
        crack = CrackGeometry(length=1.0, orientation=CrackOrientation.LONGITUDINAL)
        assert geometry.max_crack_length(crack) == pytest.approx(
            geometry.inter_hole_ligament
        )

    def test_remaining_ligament_shrinks_as_the_crack_grows(self, geometry):
        short = geometry.remaining_ligament(CrackGeometry(length=1.0))
        long = geometry.remaining_ligament(CrackGeometry(length=4.0))
        assert short - long == pytest.approx(3.0)

    def test_admissibility(self, geometry):
        assert geometry.crack_is_admissible(CrackGeometry(length=2.0))
        assert not geometry.crack_is_admissible(CrackGeometry(length=50.0))

    def test_crack_placement_follows_measured_dimensions(self):
        """The one-line-update promise, applied to the crack too."""
        measured = StrapGeometry(
            strap_width=31.7, hole_diameter=5.1, hole_pitch=12.6, n_holes=3,
            steel_band_width=20.0,
        )
        crack = CrackGeometry(length=1.0)
        assert measured.max_crack_length(crack) == pytest.approx((31.7 - 5.1) / 2)
        mouth = measured.crack_mouth(crack)
        assert mouth[1] == pytest.approx(31.7 / 2 + 5.1 / 2)


class TestObservedCrack:
    def test_is_recorded_and_flagged(self, geometry):
        assert geometry.observed_crack_length > 0
        source = geometry.SOURCES["observed_crack_length"]
        assert source.needs_verification
        assert "hole edge" in source.verify_by

    def test_it_fits_the_placeholder_geometry(self, geometry):
        """If it did not, the K(a) curve could not be read at it."""
        for orientation in CrackOrientation:
            crack = CrackGeometry(
                length=geometry.observed_crack_length, orientation=orientation
            )
            assert geometry.crack_is_admissible(crack)
