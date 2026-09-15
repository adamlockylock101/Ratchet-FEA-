"""Provenance tracking: the machinery that stops a placeholder passing for data."""

from __future__ import annotations

import math

import pytest

from ratchet_fea.provenance import (
    Bracket,
    Documented,
    Provenance,
    Source,
    VerificationItem,
    collect_verification_items,
    format_verification_report,
)


def _source(provenance=Provenance.LITERATURE, units="MPa"):
    return Source(provenance, units, "note", "verify by this")


class TestProvenance:
    def test_only_measured_needs_no_verification(self):
        assert not Provenance.MEASURED.needs_verification
        for p in Provenance:
            if p is not Provenance.MEASURED:
                assert p.needs_verification

    def test_rank_orders_by_trustworthiness(self):
        assert Provenance.MEASURED.rank < Provenance.DATASHEET.rank
        assert Provenance.DATASHEET.rank < Provenance.LITERATURE.rank
        assert Provenance.LITERATURE.rank < Provenance.PHOTO_ESTIMATE.rank
        assert Provenance.PHOTO_ESTIMATE.rank < Provenance.ASSUMED.rank


class TestBracket:
    def test_linear_nominal_is_the_arithmetic_mean(self):
        assert Bracket(10.0, 20.0, _source()).nominal == pytest.approx(15.0)

    def test_log_nominal_is_the_geometric_mean(self):
        b = Bracket(1e-7, 1e-5, _source("mm^2/s"), scale="log")
        assert b.nominal == pytest.approx(1e-6)

    def test_log_nominal_differs_from_linear_for_a_wide_bracket(self):
        """A decade-wide bracket must not be averaged arithmetically.

        Diffusivity spans a decade; the arithmetic mean of 1e-7 and 1e-6 is
        5.5e-7, which sits far closer to the top of the range than the middle
        of it and would bias every predicted time scale.
        """
        b = Bracket(1e-7, 1e-6, _source(), scale="log")
        assert b.nominal == pytest.approx(math.sqrt(1e-13))
        assert b.nominal < 0.5 * (1e-7 + 1e-6)

    def test_at_selects_corners(self):
        b = Bracket(2.0, 8.0, _source())
        assert b.at("low") == 2.0
        assert b.at("high") == 8.0
        assert b.at("nominal") == pytest.approx(5.0)

    def test_at_rejects_an_unknown_corner(self):
        with pytest.raises(ValueError, match="bracket corner"):
            Bracket(1.0, 2.0, _source()).at("middling")

    def test_inverted_bracket_is_rejected(self):
        with pytest.raises(ValueError, match="exceeds high"):
            Bracket(5.0, 1.0, _source())

    def test_log_bracket_rejects_non_positive_bounds(self):
        with pytest.raises(ValueError, match="strictly positive"):
            Bracket(0.0, 1.0, _source(), scale="log")

    def test_unknown_scale_is_rejected(self):
        with pytest.raises(ValueError, match="unknown scale"):
            Bracket(1.0, 2.0, _source(), scale="quadratic")

    def test_spread_reports_how_well_pinned_a_property_is(self):
        assert Bracket(1.0, 10.0, _source()).spread == pytest.approx(10.0)
        assert Bracket(1.0, 1.0, _source()).spread == pytest.approx(1.0)

    def test_iterating_yields_the_bounds(self):
        assert list(Bracket(1.0, 3.0, _source())) == [1.0, 3.0]


class _Thing(Documented):
    SOURCES = {
        "guessed": Source(Provenance.PHOTO_ESTIMATE, "mm", "", "calipers"),
        "known": Source(Provenance.MEASURED, "mm", "", ""),
    }

    def __init__(self):
        self.guessed = 4.0
        self.known = 25.0


class TestDocumented:
    def test_measured_values_are_omitted_from_the_report(self):
        items = _Thing().verification_items()
        names = {it.name for it in items}
        assert "guessed" in names
        assert "known" not in names

    def test_value_and_units_are_carried_through(self):
        (item,) = _Thing().verification_items()
        assert item.value == "4 mm"
        assert item.verify_by == "calipers"

    def test_source_for_raises_a_helpful_error_for_an_untracked_field(self):
        with pytest.raises(KeyError, match="traceable"):
            _Thing().source_for("nonexistent")


class TestReport:
    def test_empty_report_says_everything_is_measured(self):
        text = format_verification_report([])
        assert "MEASURED" in text

    def test_report_tabulates_items(self):
        text = format_verification_report(
            [VerificationItem("Owner", "thing", "1 mm", Provenance.ASSUMED, "measure it")]
        )
        for expected in ("Owner", "thing", "1 mm", "assumed", "measure it"):
            assert expected in text
        assert "1 input(s)" in text

    def test_collect_sorts_least_trustworthy_first(self):
        class Mixed(Documented):
            SOURCES = {
                "a": Source(Provenance.LITERATURE, "", "", ""),
                "b": Source(Provenance.ASSUMED, "", "", ""),
            }

            def __init__(self):
                self.a = 1.0
                self.b = 2.0

        items = collect_verification_items(Mixed())
        assert [it.provenance for it in items] == [
            Provenance.ASSUMED,
            Provenance.LITERATURE,
        ]

    def test_collect_walks_containers(self):
        items = collect_verification_items([_Thing(), _Thing()])
        # Identical rows are de-duplicated rather than repeated.
        assert len(items) == 1

    def test_collect_ignores_none(self):
        assert collect_verification_items(None) == []
