#!/usr/bin/env python3
"""Tier 2 entry point: coupled moisture diffusion + plane-stress FE.

Meshes the perforated strap from the dimensions in geometry.py, solves the
transient Fickian moisture field, solves the plane-stress membrane problem at
each stored time with the moisture mapped to a swelling eigenstrain, and
extracts the hole-edge stress history -- then cross-checks the whole thing
against the Tier 1 closed form.

    python scripts/run_tier2_fe.py                     # absorption, nominal
    python scripts/run_tier2_fe.py --quick             # coarse, for a smoke test
    python scripts/run_tier2_fe.py --case desorption   # drying out
    python scripts/run_tier2_fe.py --case both --corner high
    python scripts/run_tier2_fe.py --mesh-convergence  # Kt convergence only

Outputs go to results/<case>/ as PNG figures plus a summary.json.

REPLACING THE PLACEHOLDER GEOMETRY
    Edit the defaults on StrapGeometry in src/ratchet_fea/geometry.py, or pass
    measured dimensions on the command line. Mesh density, diffusion time
    scales, load normalisation and post-processing all derive from that object,
    so a measured strap re-runs the whole study unchanged.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratchet_fea import diffusion as dif  # noqa: E402
from ratchet_fea import mechanics as mech  # noqa: E402
from ratchet_fea import postprocess as post  # noqa: E402
from ratchet_fea.analytical import screen  # noqa: E402
from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY  # noqa: E402
from ratchet_fea.materials import all_material_objects  # noqa: E402
from ratchet_fea.mesh import MeshControls, build_strip_mesh  # noqa: E402
from ratchet_fea.provenance import (  # noqa: E402
    collect_verification_items,
    format_verification_report,
)

CASES = {
    # (start state, end state). Absorption is the service condition; desorption
    # is what happens between uses, and is the direction that turns the
    # constrained-swelling magnitude into tension near a drying surface.
    "absorption": (dif.MoistureState.DRY, dif.MoistureState.IMMERSED),
    "desorption": (dif.MoistureState.IMMERSED, dif.MoistureState.RH50),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--case",
        choices=(*CASES, "both"),
        default="absorption",
        help="moisture transition to model (default: absorption)",
    )
    p.add_argument(
        "--corner",
        choices=("low", "nominal", "high"),
        default="nominal",
        help="material bracket corner (default: nominal)",
    )
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")

    m = p.add_argument_group("mesh and time resolution")
    m.add_argument("--elements-around-hole", type=int, default=48)
    m.add_argument("--time-steps", type=int, default=32, help="diffusion time steps")
    m.add_argument(
        "--mechanics-every",
        type=int,
        default=2,
        help="solve mechanics on every Nth diffusion step (default: 2)",
    )
    m.add_argument(
        "--horizon",
        type=float,
        default=20.0,
        help="simulated duration as a multiple of the wetting half-time "
        "(default: 20, i.e. well past equilibrium)",
    )
    m.add_argument(
        "--quick",
        action="store_true",
        help="coarse mesh and few steps; for checking the pipeline runs, "
        "NOT for quoting numbers",
    )

    mo = p.add_argument_group("model options")
    mo.add_argument(
        "--fixed-modulus",
        action="store_true",
        help="hold the polymer modulus constant instead of evaluating it from "
        "the local moisture (overstates swelling stress by 2-3x; for "
        "comparison against hand calculations only)",
    )
    mo.add_argument(
        "--no-band",
        action="store_true",
        help="omit the steel band, to isolate its contribution",
    )
    mo.add_argument(
        "--no-tension",
        action="store_true",
        help="omit the mechanical load, to isolate the swelling stress",
    )
    mo.add_argument(
        "--stress-free-moisture",
        type=float,
        default=0.0,
        help="moisture content at which the polymer is stress-free (default 0, "
        "i.e. the strap was assembled around the band dry). Raising this is the "
        "single biggest lever on whether drying produces tension -- see README",
    )
    mo.add_argument(
        "--mesh-convergence",
        action="store_true",
        help="run the Kt mesh convergence study and exit",
    )

    g = p.add_argument_group("measured geometry (overrides the placeholders)")
    g.add_argument("--width", type=float, help="strap width, mm")
    g.add_argument("--thickness", type=float, help="strap thickness, mm")
    g.add_argument("--hole-diameter", type=float, help="perforation diameter, mm")
    g.add_argument("--hole-pitch", type=float, help="perforation pitch, mm")
    g.add_argument("--n-holes", type=int, help="perforations in the modelled window")
    g.add_argument("--band-width", type=float, help="steel band width, mm")
    g.add_argument("--band-thickness", type=float, help="steel band thickness, mm")
    g.add_argument("--tension", type=float, help="peak service tension, N")
    return p


def geometry_from_args(args):
    overrides = {
        "strap_width": args.width,
        "strap_thickness": args.thickness,
        "hole_diameter": args.hole_diameter,
        "hole_pitch": args.hole_pitch,
        "n_holes": args.n_holes,
        "steel_band_width": args.band_width,
        "steel_band_thickness": args.band_thickness,
        "service_tension": args.tension,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if not supplied:
        return PLACEHOLDER_GEOMETRY, []
    return replace(PLACEHOLDER_GEOMETRY, **supplied), sorted(supplied)


def run_mesh_convergence(geometry, base_controls) -> int:
    """Kt against mesh density -- quote nothing that is still moving."""
    from ratchet_fea.analytical import kt_hole_in_finite_width_strip

    print("MESH CONVERGENCE (mechanical Kt at the hole edge)")
    print("=" * 62)
    print(f"  Tier 1 isolated-hole Kt (net section): {kt_hole_in_finite_width_strip(geometry.d_over_W):.4f}")
    print()
    print(f"  {'around hole':>12} {'elements':>9} {'interior Kt':>12} {'end Kt':>9} {'change':>8}")
    print("  " + "-" * 56)

    previous = None
    for factor in (0.5, 1.0, 2.0):
        controls = base_controls.refined(factor)
        strip = build_strip_mesh(geometry, controls)
        kt = mech.stress_concentration_factors(strip)
        interior = kt["interior_mean"]
        ends = np.mean(
            [
                v
                for i, v in enumerate(kt["per_hole"].values())
                if not geometry.is_interior_hole(i)
            ]
        )
        change = "-" if previous is None else f"{100 * abs(interior - previous) / previous:.2f}%"
        print(
            f"  {controls.elements_around_hole:>12} {strip.n_elements:>9} "
            f"{interior:>12.4f} {ends:>9.4f} {change:>8}"
        )
        previous = interior
    print()
    print("  A change under about 2% between the last two rows means the")
    print("  hole-edge stress is converged and safe to quote.")
    return 0


def run_case(case_name, geometry, args, tier1, out_root) -> dict:
    start_state, end_state = CASES[case_name]
    out_dir = out_root / case_name
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = MeshControls(
        elements_around_hole=16 if args.quick else args.elements_around_hole
    )
    n_steps = 10 if args.quick else args.time_steps
    every = 2 if args.quick else args.mechanics_every

    print()
    print("=" * 78)
    print(f"CASE: {case_name.upper()}  ({start_state.value} -> {end_state.value})")
    print("=" * 78)
    print()

    t0 = time.perf_counter()
    strip = build_strip_mesh(geometry, controls)
    print(strip.summary())
    print()

    diffusion_model = dif.DiffusionModel.between(
        geometry, start=start_state, end=end_state, corner=args.corner
    )
    print(diffusion_model.summary())
    print()

    horizon = args.horizon * diffusion_model.characteristic_time()
    times = dif.log_time_grid(horizon, n_steps=n_steps)
    print(
        f"  Simulating {horizon / post.SECONDS_PER_DAY:.1f} days "
        f"in {n_steps} log-spaced steps."
    )
    diffusion_result = dif.solve(strip, diffusion_model, times)
    print(
        f"  Global uptake reaches 50% at "
        f"{diffusion_result.time_to_uptake(0.5) / post.SECONDS_PER_DAY:.2f} d, "
        f"{diffusion_result.uptake_fraction[-1] * 100:.1f}% by the end."
    )
    print()

    mech_model = mech.MechanicsModel.from_materials(
        geometry,
        corner=args.corner,
        include_band=not args.no_band,
        include_tension=not args.no_tension,
        moisture_dependent_modulus=not args.fixed_modulus,
        stress_free_moisture=args.stress_free_moisture,
    )
    print(mech_model.summary())
    print()

    history = post.stress_history(
        strip, diffusion_result, mech_model, every=every, progress=args.quick
    )

    kt_tier2 = mech.stress_concentration_factors(strip)
    comparison = post.compare_with_tier1(history, tier1, strip, kt_tier2)
    verdict = post.free_edge_validity(strip)

    print(comparison.summary())
    print()
    print("MODEL VALIDITY")
    print("--------------")
    for line in _wrap(verdict.message()):
        print(f"  {line}")
    print()

    _print_history_table(history)

    figures = _write_figures(strip, history, comparison, diffusion_result, out_dir)
    summary_path = post.write_summary(
        history, comparison, verdict, out_dir / "summary.json"
    )

    print()
    print(f"  Wrote {summary_path}")
    for f in figures:
        print(f"  Wrote {f}")
    print(f"  Case completed in {time.perf_counter() - t0:.1f} s")

    return {
        "case": case_name,
        "history": history,
        "comparison": comparison,
        "verdict": verdict,
    }


def _write_figures(strip, history, comparison, diffusion_result, out_dir):
    figures = [
        post.plot_hole_stress_history(history, out_dir / "hole_stress_vs_time.png"),
        post.plot_stress_vs_uptake(history, out_dir / "stress_vs_uptake.png"),
        post.plot_tier1_comparison(comparison, out_dir / "tier1_cross_check.png"),
    ]

    # Field maps at the instant of peak hole-edge von Mises, which is where the
    # model says the polymer is working hardest.
    peak = int(np.argmax(history.peak_von_mises_envelope()))
    day = history.times_days[peak]
    figures.append(
        post.plot_field(
            strip,
            diffusion_result.concentration[history.indices[peak]],
            out_dir / "concentration_field.png",
            f"Thickness-averaged moisture content at t = {day:.1f} d",
            "mass fraction",
            cmap="Blues",
        )
    )
    figures.append(
        post.plot_field(
            strip,
            history.final_result.nodal_max_principal,
            out_dir / "stress_field_max_principal.png",
            "PA66 maximum principal stress at equilibrium (tension positive)",
            "MPa",
            cmap="coolwarm",
        )
    )
    figures.append(
        post.plot_field(
            strip,
            history.final_result.nodal_von_mises,
            out_dir / "stress_field_von_mises.png",
            "PA66 von Mises stress at equilibrium",
            "MPa",
            cmap="magma",
        )
    )
    return figures


def _print_history_table(history):
    print("HOLE-EDGE STRESS HISTORY (interior holes, MPa, tension positive)")
    print("-" * 78)
    print(
        f"  {'t (d)':>10} {'uptake':>7} {'max prin':>9} {'min prin':>9} "
        f"{'von Mises':>10} {'vM/yield':>9} {'bulk sxx':>9}"
    )
    print("  " + "-" * 70)
    tens = history.peak_tensile_envelope()
    comp = history.peak_compressive_envelope()
    vm = history.peak_von_mises_envelope()
    util = history.yield_utilisation_envelope()
    step = max(1, len(history.times) // 12)
    rows = sorted(set(list(range(0, len(history.times), step)) + [len(history.times) - 1]))
    for i in rows:
        print(
            f"  {history.times_days[i]:>10.3f} {history.uptake_fraction[i]:>7.3f} "
            f"{tens[i]:>9.2f} {comp[i]:>9.2f} {vm[i]:>10.2f} {util[i]:>9.2f} "
            f"{history.bulk_sxx[i]:>9.2f}"
        )


def _wrap(text, width=74):
    import textwrap

    return textwrap.wrap(text, width=width)


def _print_interpretation(outcomes):
    print()
    print("=" * 78)
    print("WHAT THE MODEL SAYS")
    print("=" * 78)
    print()

    for outcome in outcomes:
        h = outcome["history"]
        tens = float(np.max(h.peak_tensile_envelope()))
        comp = float(np.min(h.peak_compressive_envelope()))
        util = float(np.max(h.yield_utilisation_envelope()))
        print(f"  {outcome['case'].upper()}")
        print(f"    peak hole-edge tension      {tens:8.2f} MPa")
        print(f"    peak hole-edge compression  {comp:8.2f} MPa")
        print(f"    peak von Mises / yield      {util:8.2f}")
        if util >= 1.0:
            print(
                "    -> the polymer reaches its yield strength at the hole edge in\n"
                "       this case."
            )
        else:
            print("    -> stays below yield at the hole edge in this case.")
        print()

    print("  READ THE SIGNS CAREFULLY.")
    for line in _wrap(
        "Constrained swelling during absorption is compressive, and compression "
        "does not open a crack. If the compressive von Mises stress reaches "
        "yield, the polymer flows locally; that plastic strain is then locked "
        "in, and on drying it reappears as tension. That ratcheting over "
        "repeated wet/dry cycles is the mechanism this model points at, and it "
        "is a Tier 3 question -- it needs a plasticity model and a cyclic "
        "moisture history, neither of which is in scope here."
    ):
        print(f"    {line}")
    print()
    print("  THE BIGGEST UNCERTAINTY IS THE STRESS-FREE STATE.")
    for line in _wrap(
        "Every stress above is measured from the moisture content at which the "
        f"polymer is taken to be stress-free, currently "
        f"{outcomes[0]['history'].mechanics_model.stress_free_moisture:g} "
        "(assembled dry). That reference is ASSUMED, not measured, and it moves "
        "the answer more than anything else in the model: if the strap is "
        "instead stress-free at its in-service equilibrium moisture, then "
        "DRYING below that puts the hole edges into tens of MPa of TENSION, "
        "right where the mechanical stress concentration already is. Re-run "
        "with --stress-free-moisture 0.025 to see it. Resolving this is the "
        "highest-value experiment available: anneal a sample and measure the "
        "dimensional change, or measure residual strain by hole drilling."
    ):
        print(f"    {line}")
    print()
    for line in _wrap(
        "This model also averages through the thickness, so it cannot see a "
        "drying skin pulled into tension over a still-wet core. That is the "
        "classic moisture-driven surface cracking mechanism in polyamides and "
        "it needs a through-thickness model. A low tensile stress here is NOT "
        "evidence that moisture is harmless."
    ):
        print(f"    {line}")
    print()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    geometry, measured = geometry_from_args(args)

    print("=" * 78)
    print("STP-RB-001  TIER 2  --  COUPLED MOISTURE DIFFUSION + PLANE-STRESS FE")
    print("=" * 78)
    print()
    if measured:
        print(f"Geometry overridden on the command line for: {', '.join(measured)}")
        print("(Move these into geometry.py so every run picks them up.)")
        print()
    print(geometry.summary())
    print()

    controls = MeshControls(
        elements_around_hole=16 if args.quick else args.elements_around_hole
    )
    if args.mesh_convergence:
        return run_mesh_convergence(geometry, controls)

    if args.quick:
        print("!! --quick: coarse mesh and few time steps. Pipeline check only. !!")
        print()

    tier1 = screen(geometry=geometry, corners=(args.corner,))

    cases = tuple(CASES) if args.case == "both" else (args.case,)
    outcomes = [run_case(name, geometry, args, tier1, args.out) for name in cases]

    _print_interpretation(outcomes)

    print()
    items = collect_verification_items(geometry, *all_material_objects())
    print(format_verification_report(items))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
