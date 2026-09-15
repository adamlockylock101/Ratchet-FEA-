"""The entry points.

Loading the scripts as modules catches wiring breakage that unit tests on the
library alone would miss, and pins the case definitions -- which encode the
physical scenario each run represents.
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
def tier2():
    return _load("run_tier2_fe")


@pytest.fixture(scope="module")
def tier1():
    return _load("run_tier1_analytical")


class TestCaseDefinitions:
    def test_both_directions_are_defined(self, tier2):
        assert set(tier2.CASES) == {"absorption", "desorption"}

    def test_absorption_wets_the_strap(self, tier2):
        from ratchet_fea.materials import PA66_MOISTURE

        case = tier2.CASES["absorption"]
        assert case.start.content(PA66_MOISTURE) < case.end.content(PA66_MOISTURE)

    def test_desorption_dries_it_from_the_conditioned_state(self, tier2):
        """The user-facing scenario: conditioned to equilibrium, then dried out."""
        from ratchet_fea.diffusion import MoistureState
        from ratchet_fea.materials import PA66_MOISTURE

        case = tier2.CASES["desorption"]
        assert case.start is MoistureState.RH50
        assert case.end is MoistureState.DRY
        assert case.start.content(PA66_MOISTURE) > case.end.content(PA66_MOISTURE)

    def test_desorption_assumes_a_relaxed_reference(self, tier2):
        """Otherwise drying could only unload, never pull."""
        assert tier2.CASES["desorption"].relaxation == 1.0

    def test_absorption_starts_at_the_as_moulded_reference(self, tier2):
        assert tier2.CASES["absorption"].relaxation == 0.0

    def test_every_case_describes_itself(self, tier2):
        for case in tier2.CASES.values():
            assert len(case.description) > 40


class TestStressFreeResolution:
    def _args(self, tier2, **overrides):
        args = tier2.build_parser().parse_args([])
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_desorption_default_relaxes_to_the_conditioned_state(self, tier2):
        from ratchet_fea.diffusion import DiffusionModel
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        case = tier2.CASES["desorption"]
        model = DiffusionModel.between(PLACEHOLDER_GEOMETRY, case.start, case.end)
        value = tier2.resolve_stress_free_moisture(case, model, self._args(tier2))
        assert value == pytest.approx(model.c_initial)

    def test_relaxation_flag_overrides_the_case_default(self, tier2):
        from ratchet_fea.diffusion import DiffusionModel
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        case = tier2.CASES["desorption"]
        model = DiffusionModel.between(PLACEHOLDER_GEOMETRY, case.start, case.end)
        args = self._args(tier2, relaxation=0.0)
        assert tier2.resolve_stress_free_moisture(case, model, args) == 0.0

    def test_explicit_stress_free_moisture_wins_over_relaxation(self, tier2):
        from ratchet_fea.diffusion import DiffusionModel
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        case = tier2.CASES["desorption"]
        model = DiffusionModel.between(PLACEHOLDER_GEOMETRY, case.start, case.end)
        args = self._args(tier2, relaxation=0.0, stress_free_moisture=0.04)
        assert tier2.resolve_stress_free_moisture(case, model, args) == 0.04

    def test_absorption_is_stress_free_dry_whatever_the_relaxation(self, tier2):
        from ratchet_fea.diffusion import DiffusionModel
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        case = tier2.CASES["absorption"]
        model = DiffusionModel.between(PLACEHOLDER_GEOMETRY, case.start, case.end)
        for fraction in (0.0, 0.5, 1.0):
            args = self._args(tier2, relaxation=fraction)
            assert tier2.resolve_stress_free_moisture(case, model, args) == 0.0


class TestParsers:
    def test_tier2_defaults_to_running_both_halves_of_the_cycle(self, tier2):
        assert tier2.build_parser().parse_args([]).case == "both"

    def test_tier2_geometry_overrides_are_applied(self, tier2):
        args = tier2.build_parser().parse_args(["--width", "31.7", "--tension", "800"])
        geometry, measured = tier2.geometry_from_args(args)
        assert geometry.strap_width == pytest.approx(31.7)
        assert geometry.service_tension == pytest.approx(800.0)
        assert "strap_width" in measured

    def test_tier2_without_overrides_uses_the_placeholders(self, tier2):
        from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY

        geometry, measured = tier2.geometry_from_args(tier2.build_parser().parse_args([]))
        assert geometry is PLACEHOLDER_GEOMETRY
        assert measured == []

    def test_tier1_geometry_overrides_are_applied(self, tier1):
        args = tier1.build_parser().parse_args(["--hole-diameter", "4.6"])
        geometry, measured = tier1.geometry_from_args(args)
        assert geometry.hole_diameter == pytest.approx(4.6)
        assert measured == ["hole_diameter"]


class TestTier1EntryPoint:
    def test_runs_end_to_end(self, tier1, capsys):
        assert tier1.main(["--corner", "nominal"]) == 0
        out = capsys.readouterr().out
        assert "TIER 1 SCREENING RESULT" in out
        assert "ASSUMPTIONS STILL TO VERIFY" in out

    def test_verification_report_can_be_suppressed(self, tier1, capsys):
        tier1.main(["--no-verification-report"])
        assert "ASSUMPTIONS STILL TO VERIFY" not in capsys.readouterr().out


@pytest.mark.requires_gmsh
class TestTier2EntryPoint:
    def test_quick_run_covers_both_cases_and_the_cycle(self, tier2, tmp_path, capsys):
        assert tier2.main(["--quick", "--out", str(tmp_path)]) == 0
        out = capsys.readouterr().out

        assert "CASE: ABSORPTION" in out
        assert "CASE: DESORPTION" in out
        assert "PEAK TENSILE STRESS AT THE HOLE EDGE" in out
        assert "MOISTURE CYCLING" in out
        assert "WHY THIS IS NOT A FATIGUE ANALYSIS" in out
        assert "HOW MUCH OF THAT RESTS ON THE STRESS-FREE ASSUMPTION" in out

        assert (tmp_path / "absorption" / "summary.json").exists()
        assert (tmp_path / "desorption" / "summary.json").exists()
        assert (tmp_path / "desorption" / "moisture_cycle.png").exists()

    def test_a_single_case_says_the_cycle_cannot_be_formed(
        self, tier2, tmp_path, capsys
    ):
        tier2.main(["--quick", "--case", "absorption", "--out", str(tmp_path)])
        out = capsys.readouterr().out
        assert "Re-run with --case both" in out

    def test_mesh_convergence_mode_exits_cleanly(self, tier2, capsys):
        assert tier2.main(["--mesh-convergence"]) == 0
        assert "MESH CONVERGENCE" in capsys.readouterr().out
