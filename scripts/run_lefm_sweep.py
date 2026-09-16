#!/usr/bin/env python3
"""Static LEFM: sweep crack length and compare K against K_IC.

Seeds a crack at a perforation, applies the ratchet tension, and computes the
stress intensity factor at the crack tip over a range of crack lengths. Answers
one question: does the crack, once nucleated, run under normal ratchet load
alone?

    python scripts/run_lefm_sweep.py                    # both orientations
    python scripts/run_lefm_sweep.py --quick            # coarse smoke test
    python scripts/run_lefm_sweep.py --orientation transverse
    python scripts/run_lefm_sweep.py --condition dry    # brittlest material
    python scripts/run_lefm_sweep.py --tension 2000
    python scripts/run_lefm_sweep.py --convergence      # mesh convergence only

THE TWO ORIENTATIONS ARE DIFFERENT PROBLEMS
    transverse    across the strap, from the hole wall toward the free edge.
                  Normal to the load, so mode I, and it starts in the hole's
                  +3 sigma hoop stress. This one can run.
    longitudinal  along the strap, from the hole wall toward the next
                  perforation. Parallel to the load, and it starts in the
                  hole's -1 sigma compressive lobe. Tension presses it shut.

    The failed part is reported to have cracked toward the adjacent hole, which
    is the longitudinal case. Both are run by default so the comparison is on
    the page rather than in an argument.

REPLACING THE PLACEHOLDER GEOMETRY
    Edit the defaults on StrapGeometry in src/ratchet_fea/geometry.py, or pass
    measured dimensions on the command line. Mesh density, crack seeding, load
    normalisation and the sweep range all derive from that object.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratchet_fea import fracture, postprocess as post  # noqa: E402
from ratchet_fea.analytical import screen  # noqa: E402
from ratchet_fea.geometry import (  # noqa: E402
    PLACEHOLDER_GEOMETRY,
    CrackGeometry,
    CrackOrientation,
)
from ratchet_fea.materials import (  # noqa: E402
    MoistureCondition,
    all_material_objects,
)
from ratchet_fea.mechanics import MechanicsModel, band_load_sharing  # noqa: E402
from ratchet_fea.mesh import MeshControls  # noqa: E402
from ratchet_fea.provenance import (  # noqa: E402
    collect_verification_items,
    format_verification_report,
)

CONDITIONS = {
    "dry": MoistureCondition.DAM,
    "conditioned": MoistureCondition.RH50,
    "saturated": MoistureCondition.SATURATED,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--orientation",
        choices=[o.value for o in CrackOrientation] + ["both"],
        default="both",
        help="which way the crack runs out of the hole (default: both)",
    )
    p.add_argument(
        "--condition",
        choices=tuple(CONDITIONS),
        default="conditioned",
        help="PA66 conditioning state (default: conditioned, the service state; "
        "'dry' is the brittlest and the conservative choice for fracture)",
    )
    p.add_argument(
        "--corner",
        choices=("low", "nominal", "high"),
        default="nominal",
        help="material bracket corner for the elastic properties (default: nominal)",
    )
    p.add_argument("--tension", type=float, help="ratchet tension, N")
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory")

    s = p.add_argument_group("sweep and mesh")
    s.add_argument("--points", type=int, default=14, help="crack lengths to evaluate")
    s.add_argument(
        "--max-fraction",
        type=float,
        default=0.85,
        help="sweep up to this fraction of the available ligament (default 0.85; "
        "LEFM stops meaning anything as the ligament vanishes)",
    )
    s.add_argument("--elements-around-hole", type=int, default=32)
    s.add_argument("--elements-along-crack", type=int, default=24)
    s.add_argument(
        "--quick",
        action="store_true",
        help="coarse mesh and few crack lengths; checks the pipeline runs, "
        "NOT for quoting numbers",
    )
    s.add_argument(
        "--convergence",
        action="store_true",
        help="run the mesh convergence study on K and exit",
    )

    g = p.add_argument_group("measured geometry (overrides the placeholders)")
    g.add_argument("--width", type=float, help="strap width, mm")
    g.add_argument("--thickness", type=float, help="strap thickness, mm")
    g.add_argument("--hole-diameter", type=float, help="perforation diameter, mm")
    g.add_argument("--hole-pitch", type=float, help="perforation pitch, mm")
    g.add_argument("--n-holes", type=int, help="perforations in the modelled window")
    g.add_argument(
        "--observed-crack",
        type=float,
        help="crack length measured on the failed part, from the HOLE WALL, mm",
    )
    return p


def geometry_from_args(args):
    overrides = {
        "strap_width": args.width,
        "strap_thickness": args.thickness,
        "hole_diameter": args.hole_diameter,
        "hole_pitch": args.hole_pitch,
        "n_holes": args.n_holes,
        "service_tension": args.tension,
        "observed_crack_length": args.observed_crack,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if not supplied:
        return PLACEHOLDER_GEOMETRY, []
    return replace(PLACEHOLDER_GEOMETRY, **supplied), sorted(supplied)


def controls_from_args(args) -> MeshControls:
    if args.quick:
        return MeshControls(elements_around_hole=16, elements_along_crack=10)
    return MeshControls(
        elements_around_hole=args.elements_around_hole,
        elements_along_crack=args.elements_along_crack,
    )


def sweep_lengths(geometry, orientation, args) -> np.ndarray:
    """Crack lengths to evaluate, from just-nucleated to most of the ligament."""
    probe = CrackGeometry(length=1.0, orientation=orientation)
    limit = geometry.max_crack_length(probe) * args.max_fraction
    n = 6 if args.quick else args.points
    # Geometric spacing: K varies fastest at short crack lengths, and a short
    # crack is what the part actually has.
    return np.geomspace(max(limit / 40.0, 1e-3), limit, n)


def run_convergence(geometry, args) -> int:
    """K against mesh density. Quote nothing that is still moving."""
    print("MESH CONVERGENCE (K at the observed crack length)")
    print("=" * 66)
    a = geometry.observed_crack_length
    crack = CrackGeometry(length=a, orientation=CrackOrientation.TRANSVERSE)
    if not geometry.crack_is_admissible(crack):
        print(f"  the observed {a:g} mm crack does not fit this geometry")
        return 1

    print(f"  transverse crack, a = {a:g} mm")
    print()
    print(
        f"  {'refinement':>11} {'along crack':>12} {'elements':>9} {'K':>10} "
        f"{'domain indep':>13} {'change':>8}"
    )
    print("  " + "-" * 68)

    base = controls_from_args(args)
    model = MechanicsModel.from_materials(geometry)
    previous = None
    for factor in (0.5, 1.0, 2.0):
        controls = base.refined(factor)
        result = fracture.evaluate_crack(geometry, crack, model, controls)
        change = (
            "-"
            if previous is None
            else f"{100 * abs(result.k - previous) / previous:.2f}%"
        )
        print(
            f"  {'x' + format(factor, 'g'):>11} {controls.elements_along_crack:>12} "
            f"{'':>9} {result.k:>10.3f} {result.domain_independence:>12.4%} "
            f"{change:>8}"
        )
        previous = result.k
    print()
    print("  A change under about 2% between the last two rows means K is")
    print("  converged and safe to quote.")
    return 0


def run_orientation(orientation, geometry, args, out_root) -> dict:
    out_dir = out_root / orientation.value
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 78)
    print(f"CRACK ORIENTATION: {orientation.value.upper()}")
    print("=" * 78)
    for line in _wrap(_orientation_blurb(orientation)):
        print(f"  {line}")
    print()

    t0 = time.perf_counter()
    condition = CONDITIONS[args.condition]
    force = geometry.service_tension
    model = MechanicsModel.from_materials(
        geometry, force=force, condition=condition, corner=args.corner
    )
    print(model.summary())
    print()

    controls = controls_from_args(args)
    lengths = sweep_lengths(geometry, orientation, args)
    print(
        f"  Sweeping {len(lengths)} crack lengths, "
        f"{lengths[0]:.3f} to {lengths[-1]:.3f} mm "
        f"(ligament available: "
        f"{geometry.max_crack_length(CrackGeometry(1.0, orientation)):.2f} mm)"
    )
    results = fracture.sweep_crack_length(
        geometry, lengths, orientation, model, controls, progress=True
    )

    curve = post.build_k_curve(
        results, geometry, force, orientation, condition, args.corner
    )
    print()
    print(curve.summary())
    print()

    figures = [post.plot_k_vs_a(curve, out_dir / "k_vs_crack_length.png")]
    if orientation is CrackOrientation.LONGITUDINAL or np.any(curve.min_opening < 0):
        figures.append(
            post.plot_crack_opening(
                results,
                out_dir / "crack_opening.png",
                f"Crack flank opening -- {orientation.value} crack",
            )
        )
    summary_path = post.write_summary(curve, out_dir / "summary.json")

    print(f"  Wrote {summary_path}")
    for f in figures:
        print(f"  Wrote {f}")
    print(f"  Completed in {time.perf_counter() - t0:.1f} s")

    return {"orientation": orientation, "curve": curve, "results": results}


def _orientation_blurb(orientation) -> str:
    if orientation is CrackOrientation.TRANSVERSE:
        return (
            "Across the strap, from the hole wall toward the free side edge. The "
            "crack plane is normal to the ratchet tension, so it is loaded in "
            "mode I, and its mouth sits in the +3 sigma hoop stress at the hole. "
            "This is the classical crack-from-a-hole configuration and the one "
            "that would separate the strap."
        )
    return (
        "Along the strap, from the hole wall toward the next perforation. The "
        "crack plane is parallel to the ratchet tension and its mouth sits in "
        "the hole's -1 sigma compressive lobe. This is the direction the failed "
        "part is reported to have cracked in, so it is modelled rather than "
        "argued about: the FE measures the opening between the crack faces and "
        "reports what it finds."
    )


def _wrap(text, width=74):
    import textwrap

    return textwrap.wrap(text, width=width)


def _print_verdict(outcomes, geometry, args):
    print()
    print("=" * 78)
    print("DOES THE CRACK RUN UNDER RATCHET LOAD ALONE?")
    print("=" * 78)
    print()
    for outcome in outcomes:
        print(f"  {outcome['orientation'].value.upper()}")
        for paragraph in outcome["curve"].verdict():
            for line in _wrap(paragraph, width=72):
                print(f"    {line}")
            print()

    sharing = band_load_sharing(geometry)
    if sharing > 0:
        print("  HOW CONSERVATIVE IS THIS?")
        for line in _wrap(
            f"The FE model puts the whole ratchet tension into the polymer. If the "
            f"steel band is real and bonded, it would carry about {sharing:.0%} of "
            f"it, and the polymer stress -- and therefore K -- would be roughly "
            f"{1 / (1 - sharing):.0f} times smaller than reported above. The band's "
            f"existence is inferred from photographs rather than observed, so "
            f"ignoring it is deliberate. It means every 'does not propagate' "
            f"verdict here has a large margin behind it, and any 'propagates' "
            f"verdict should be re-checked with the band included.",
            width=72,
        ):
            print(f"    {line}")
        print()


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    geometry, measured = geometry_from_args(args)

    print("=" * 78)
    print("STP-RB-001  --  STATIC LINEAR ELASTIC FRACTURE MECHANICS")
    print("=" * 78)
    print()
    if measured:
        print(f"Geometry overridden on the command line for: {', '.join(measured)}")
        print("(Move these into geometry.py so every run picks them up.)")
        print()
    print(geometry.summary())
    print()

    if args.convergence:
        return run_convergence(geometry, args)

    if args.quick:
        print("!! --quick: coarse mesh and few crack lengths. Pipeline check only. !!")
        print()

    # Handbook screen first: seconds, no FE, and it frames what follows.
    print(screen(geometry=geometry, orientation=CrackOrientation.TRANSVERSE).summary())
    print()

    orientations = (
        tuple(CrackOrientation)
        if args.orientation == "both"
        else (CrackOrientation(args.orientation),)
    )
    outcomes = [run_orientation(o, geometry, args, args.out) for o in orientations]

    _print_verdict(outcomes, geometry, args)

    print()
    items = collect_verification_items(geometry, *all_material_objects())
    print(format_verification_report(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
