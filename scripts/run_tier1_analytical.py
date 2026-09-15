#!/usr/bin/env python3
"""Tier 1 entry point: closed-form screening of the STP-RB-001 strap.

Run this first. It answers, in a second and with no FE stack installed,
whether the mechanical service load or moisture swelling is the more likely
driver of the cracking -- and prints every assumption that still needs
checking against a physical part.

    python scripts/run_tier1_analytical.py
    python scripts/run_tier1_analytical.py --corner high
    python scripts/run_tier1_analytical.py --tension 800

REPLACING THE PLACEHOLDER GEOMETRY
    Edit the defaults on StrapGeometry in src/ratchet_fea/geometry.py, or pass
    the measured dimensions on the command line. Nothing downstream hardcodes a
    dimension, so both Tier 1 and Tier 2 pick the change up automatically.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ratchet_fea.analytical import screen  # noqa: E402
from ratchet_fea.geometry import PLACEHOLDER_GEOMETRY  # noqa: E402
from ratchet_fea.materials import all_material_objects  # noqa: E402
from ratchet_fea.provenance import (  # noqa: E402
    collect_verification_items,
    format_verification_report,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--corner",
        choices=("low", "nominal", "high", "all"),
        default="all",
        help="material bracket corner(s) to evaluate (default: all)",
    )
    g = p.add_argument_group("measured geometry (overrides the placeholders)")
    g.add_argument("--width", type=float, help="strap width, mm")
    g.add_argument("--thickness", type=float, help="strap thickness, mm")
    g.add_argument("--hole-diameter", type=float, help="perforation diameter, mm")
    g.add_argument("--hole-pitch", type=float, help="perforation pitch, mm")
    g.add_argument("--band-width", type=float, help="steel band width, mm")
    g.add_argument("--band-thickness", type=float, help="steel band thickness, mm")
    g.add_argument("--tension", type=float, help="peak service tension, N")
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
        "steel_band_width": args.band_width,
        "steel_band_thickness": args.band_thickness,
        "service_tension": args.tension,
    }
    supplied = {k: v for k, v in overrides.items() if v is not None}
    if not supplied:
        return PLACEHOLDER_GEOMETRY, []
    return replace(PLACEHOLDER_GEOMETRY, **supplied), sorted(supplied)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    geometry, measured = geometry_from_args(args)
    corners = ("low", "nominal", "high") if args.corner == "all" else (args.corner,)

    print("=" * 78)
    print("STP-RB-001  TIER 1  --  CLOSED-FORM SCREENING")
    print("=" * 78)
    print()

    if measured:
        print(f"Geometry overridden on the command line for: {', '.join(measured)}")
        print("(Move these into geometry.py so every run picks them up.)")
        print()

    print(geometry.summary())
    print()

    result = screen(geometry=geometry, corners=corners)
    print(result.summary())
    print()

    print("READING THIS")
    print("------------")
    if result.mechanical_alone_is_benign and result.swelling_is_significant:
        print(
            "  The mechanical service load, even with the hole's stress\n"
            "  concentration applied, stays below yield across the whole material\n"
            "  bracket. Constrained moisture swelling against the steel band does\n"
            "  not. Moisture is the mechanism worth modelling -- which is what\n"
            "  Tier 2 does.\n"
        )
        print(
            "  Tier 1 cannot resolve two things, and Tier 2 exists for both:\n"
            "    1. the holes are a ROW, not one isolated hole;\n"
            "    2. the swelling field is TRANSIENT and non-uniform, and the\n"
            "       gradient matters more than the final uniform state.\n"
        )
    elif not result.mechanical_alone_is_benign:
        print(
            "  The mechanical load alone reaches yield somewhere in the material\n"
            "  bracket. Pin down the modulus and strength of the actual grade\n"
            "  before reading anything into the moisture mechanism.\n"
        )
    else:
        print(
            "  Neither mechanism reaches yield across the bracket. Either the\n"
            "  placeholder dimensions understate the real stress, the service\n"
            "  load is higher than assumed, or the failure is driven by something\n"
            "  outside this model: fatigue, creep rupture, a moulding defect, UV\n"
            "  or thermal ageing, or chemical attack.\n"
        )

    print(
        "  NOTE ON SIGN: constrained swelling during absorption is COMPRESSIVE.\n"
        "  Compression does not open a crack. What produces tension is a moisture\n"
        "  GRADIENT, or drying after the polymer has already yielded in\n"
        "  compression. Tier 2 resolves both; do not read the magnitude above as\n"
        "  a tensile stress.\n"
    )

    if not args.no_verification_report:
        print()
        items = collect_verification_items(geometry, *all_material_objects())
        print(format_verification_report(items))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
