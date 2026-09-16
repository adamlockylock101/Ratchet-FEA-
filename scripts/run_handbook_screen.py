#!/usr/bin/env python3
"""Handbook LEFM screen: K vs K_IC from closed-form solutions, no FE stack.

Run this first. In under a second, with nothing installed beyond numpy, it says
whether a crack at a perforation can reach the toughness of PA66 under the
ratchet tension -- and prints every assumption that still needs checking
against a physical part.

    python scripts/run_handbook_screen.py
    python scripts/run_handbook_screen.py --condition dry
    python scripts/run_handbook_screen.py --tension 2000
    python scripts/run_handbook_screen.py --observed-crack 3.5

scripts/run_lefm_sweep.py then does the same job with a finite element model
that includes the row of holes and the finite width, and checks itself against
these numbers.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratchet_fea.analytical import screen  # noqa: E402
from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY, CrackOrientation  # noqa: E402
from ratchet_fea.materials import (  # noqa: E402
    MoistureCondition,
    all_material_objects,
)
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
        help="PA66 conditioning state (default: conditioned; 'dry' is the "
        "brittlest and the conservative choice for a fracture check)",
    )
    g = p.add_argument_group("measured geometry (overrides the placeholders)")
    g.add_argument("--width", type=float, help="strap width, mm")
    g.add_argument("--thickness", type=float, help="strap thickness, mm")
    g.add_argument("--hole-diameter", type=float, help="perforation diameter, mm")
    g.add_argument("--hole-pitch", type=float, help="perforation pitch, mm")
    g.add_argument("--tension", type=float, help="ratchet tension, N")
    g.add_argument(
        "--observed-crack",
        type=float,
        help="crack length on the failed part, from the HOLE WALL, mm",
    )
    p.add_argument(
        "--no-verification-report",
        action="store_true",
        help="suppress the list of unverified assumptions (not recommended)",
    )
    return p


def geometry_from_args(args):
    overrides = {
        "strap_width": args.width,
        "strap_thickness": args.thickness,
        "hole_diameter": args.hole_diameter,
        "hole_pitch": args.hole_pitch,
        "service_tension": args.tension,
        "observed_crack_length": args.observed_crack,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if not supplied:
        return PLACEHOLDER_GEOMETRY, []
    return replace(PLACEHOLDER_GEOMETRY, **supplied), sorted(supplied)


def _wrap(text, width=74):
    import textwrap

    return textwrap.wrap(text, width=width)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    geometry, measured = geometry_from_args(args)
    condition = CONDITIONS[args.condition]

    print("=" * 78)
    print("STP-RB-001  --  HANDBOOK LEFM SCREEN")
    print("=" * 78)
    print()
    if measured:
        print(f"Geometry overridden on the command line for: {', '.join(measured)}")
        print("(Move these into geometry.py so every run picks them up.)")
        print()
    print(geometry.summary())
    print()

    orientations = (
        tuple(CrackOrientation)
        if args.orientation == "both"
        else (CrackOrientation(args.orientation),)
    )
    screens = []
    for orientation in orientations:
        result = screen(geometry=geometry, orientation=orientation, condition=condition)
        screens.append(result)
        print(result.summary())
        print()

    print("READING THIS")
    print("------------")
    for result in screens:
        print(f"  {result.orientation.value.upper()}")
        for line in _wrap(_reading(result), width=72):
            print(f"    {line}")
        print()

    print("  These are infinite-plate solutions with a width correction bolted on.")
    for line in _wrap(
        "They know nothing about the row of holes either side of the cracked one, "
        "which shields it. scripts/run_lefm_sweep.py resolves that with a finite "
        "element model and checks itself against these numbers.",
        width=72,
    ):
        print(f"    {line}")
    print()

    if not args.no_verification_report:
        items = collect_verification_items(geometry, *all_material_objects())
        print(format_verification_report(items))
    return 0


def _reading(result) -> str:
    if not result.orientation.is_mode_i_under_axial_tension:
        return (
            "No mode I driving force exists for this orientation at any crack "
            "length: the crack plane is parallel to the ratchet tension and its "
            "mouth sits in the hole's compressive hoop lobe, so pulling the strap "
            "presses the crack shut rather than opening it. If the part really "
            "cracked this way, ratchet tension is not the driver."
        )
    if result.propagates_at_any_length:
        return (
            f"K reaches the weakest end of the toughness bracket somewhere on the "
            f"ligament. Crack lengths from about "
            f"{min(v for v in result.critical_lengths.values() if v == v):.2f} mm are "
            f"unstable under a single pull at {result.force:g} N."
        )
    return (
        f"K never reaches even the weakest plausible K_IC at any crack length the "
        f"strap can hold. Peak K over the whole ligament is {result.max_k:.1f} "
        f"MPa*sqrt(mm) against a bracket of "
        f"{result.toughness['low']:.0f} to {result.toughness['high']:.0f}. At the "
        f"observed {result.observed_crack_length:g} mm crack the margin is "
        f"{result.margin_at_observed('nominal'):.1f}x. A single pull does not run "
        f"this crack, so something other than static overload is keeping it going."
    )


if __name__ == "__main__":
    raise SystemExit(main())
