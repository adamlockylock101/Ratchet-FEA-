"""The entry points.

Loading the scripts as modules catches wiring breakage that unit tests on the
library alone would miss.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves its module through sys.modules, so the entry has to
    # exist before the module body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sweep():
    return _load("run_lefm_sweep")


@pytest.fixture(scope="module")
def handbook():
    return _load("run_handbook_screen")


class TestParsers:
    def test_sweep_runs_both_orientations_by_default(self, sweep):
        assert sweep.build_parser().parse_args([]).orientation == "both"

    def test_handbook_runs_both_orientations_by_default(self, handbook):
        assert handbook.build_parser().parse_args([]).orientation == "both"

    def test_default_condition_is_the_service_state(self, sweep):
        from ratchet_fea.materials import SERVICE_CONDITION

        args = sweep.build_parser().parse_args([])
        assert sweep.CONDITIONS[args.condition] is SERVICE_CONDITION

    def test_dry_is_selectable_as_the_brittle_corner(self, sweep):
        from ratchet_fea.materials import MoistureCondition

        args = sweep.build_parser().parse_args(["--condition", "dry"])
        assert sweep.CONDITIONS[args.condition] is MoistureCondition.DAM

    def test_geometry_overrides_are_applied(self, sweep):
        args = sweep.build_parser().parse_args(
            ["--width", "31.7", "--tension", "800", "--observed-crack", "3.5"]
        )
        geometry, measured = sweep.geometry_from_args(args)
        assert geometry.strap_width == pytest.approx(31.7)
        assert geometry.service_tension == pytest.approx(800.0)
        assert geometry.observed_crack_length == pytest.approx(3.5)
        assert "strap_width" in measured

    def test_without_overrides_uses_the_placeholders(self, sweep):
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        geometry, measured = sweep.geometry_from_args(
            sweep.build_parser().parse_args([])
        )
        assert geometry is PLACEHOLDER_GEOMETRY
        assert measured == []

    def test_handbook_overrides_are_applied(self, handbook):
        args = handbook.build_parser().parse_args(["--hole-diameter", "4.6"])
        geometry, measured = handbook.geometry_from_args(args)
        assert geometry.hole_diameter == pytest.approx(4.6)
        assert measured == ["hole_diameter"]


class TestSweepRange:
    def test_stays_inside_the_ligament(self, sweep):
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, CrackGeometry, CrackOrientation

        args = sweep.build_parser().parse_args([])
        for orientation in CrackOrientation:
            lengths = sweep.sweep_lengths(PLACEHOLDER_GEOMETRY, orientation, args)
            limit = PLACEHOLDER_GEOMETRY.max_crack_length(
                CrackGeometry(1.0, orientation)
            )
            assert lengths[-1] <= limit * args.max_fraction + 1e-9
            assert lengths[0] > 0

    def test_is_geometrically_spaced(self, sweep):
        """K varies fastest at short cracks, and a short crack is what the part has."""
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, CrackOrientation

        args = sweep.build_parser().parse_args([])
        lengths = sweep.sweep_lengths(
            PLACEHOLDER_GEOMETRY, CrackOrientation.TRANSVERSE, args
        )
        ratios = lengths[1:] / lengths[:-1]
        assert ratios.max() - ratios.min() < 1e-9

    def test_quick_uses_fewer_points(self, sweep):
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, CrackOrientation

        full = sweep.build_parser().parse_args([])
        quick = sweep.build_parser().parse_args(["--quick"])
        assert len(
            sweep.sweep_lengths(PLACEHOLDER_GEOMETRY, CrackOrientation.TRANSVERSE, quick)
        ) < len(
            sweep.sweep_lengths(PLACEHOLDER_GEOMETRY, CrackOrientation.TRANSVERSE, full)
        )


class TestHandbookEntryPoint:
    def test_runs_end_to_end(self, handbook, capsys):
        assert handbook.main([]) == 0
        out = capsys.readouterr().out
        assert "HANDBOOK LEFM SCREEN" in out
        assert "ASSUMPTIONS STILL TO VERIFY" in out

    def test_covers_both_orientations(self, handbook, capsys):
        handbook.main([])
        out = capsys.readouterr().out
        assert "TRANSVERSE" in out
        assert "LONGITUDINAL" in out
        assert "K IS ZERO FOR THIS ORIENTATION" in out

    def test_verification_report_can_be_suppressed(self, handbook, capsys):
        handbook.main(["--no-verification-report"])
        assert "ASSUMPTIONS STILL TO VERIFY" not in capsys.readouterr().out

    def test_a_huge_load_flips_the_conclusion(self, handbook, capsys):
        """Confirms the screen can say yes, so its no means something."""
        handbook.main(["--tension", "30000", "--orientation", "transverse"])
        # The prose is wrapped, so compare on normalised whitespace.
        out = " ".join(capsys.readouterr().out.split())
        assert "unstable under a single pull" in out


@pytest.mark.requires_gmsh
class TestSweepEntryPoint:
    def test_quick_run_covers_both_orientations(self, sweep, tmp_path, capsys):
        assert sweep.main(["--quick", "--out", str(tmp_path)]) == 0
        out = capsys.readouterr().out

        assert "CRACK ORIENTATION: TRANSVERSE" in out
        assert "CRACK ORIENTATION: LONGITUDINAL" in out
        assert "K vs CRACK LENGTH" in out
        assert "DOES THE CRACK RUN UNDER RATCHET LOAD ALONE?" in out
        assert "HOW CONSERVATIVE IS THIS?" in out

        assert (tmp_path / "transverse" / "summary.json").exists()
        assert (tmp_path / "transverse" / "k_vs_crack_length.png").exists()
        assert (tmp_path / "longitudinal" / "crack_opening.png").exists()

    def test_a_single_orientation_runs_alone(self, sweep, tmp_path, capsys):
        sweep.main(
            ["--quick", "--orientation", "transverse", "--out", str(tmp_path)]
        )
        out = capsys.readouterr().out
        assert "CRACK ORIENTATION: LONGITUDINAL" not in out

    def test_convergence_mode_exits_cleanly(self, sweep, capsys):
        assert sweep.main(["--quick", "--convergence"]) == 0
        assert "MESH CONVERGENCE" in capsys.readouterr().out
