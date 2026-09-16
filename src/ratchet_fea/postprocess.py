"""Assemble the K(a) curve, compare it against K_IC, and say what it means.

The deliverable is a curve, not a number: ``K`` rises with crack length, so the
only useful question is whether it ever crosses the toughness anywhere the part
can hold a crack, and if so at what length.

Three things are reported for every run:

* the FE K(a) curve, with the handbook curve over it as an independent check;
* the toughness bracket, and the crack length at which K first reaches each end
  of it (or the fact that it never does);
* if K never reaches K_IC, the load that WOULD be needed to make it do so --
  because "the crack does not run" is only useful if you know by how much.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .analytical import handbook_k, lefm_validity
from .geometry import CrackOrientation, StrapGeometry
from .materials import (
    SERVICE_CONDITION,
    MoistureCondition,
    fracture_toughness_mpa_root_mm,
    grade_for,
)

__all__ = [
    "KCurve",
    "build_k_curve",
    "plot_k_vs_a",
    "plot_crack_opening",
    "write_summary",
]

CORNERS = ("low", "nominal", "high")


def _force_text(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{value:.0f} N"


def _range_text(values) -> str:
    """``lo .. hi`` over the finite entries, or a dash if there are none."""
    finite = np.asarray(values)[np.isfinite(values)]
    if len(finite) == 0:
        return "- (handbook K is zero for this orientation)"
    return f"{finite.min():.3f} .. {finite.max():.3f}"


@dataclass
class KCurve:
    """K against crack length, with everything needed to read it."""

    geometry: StrapGeometry
    orientation: CrackOrientation
    condition: MoistureCondition
    corner: str
    force: float
    crack_lengths: np.ndarray
    k_fe: np.ndarray
    k_handbook: np.ndarray
    #: ``corner -> K_IC`` in MPa*sqrt(mm).
    toughness: dict
    domain_independence: np.ndarray
    extraction_agreement: np.ndarray
    closed: np.ndarray
    partially_closed: np.ndarray
    min_opening: np.ndarray

    # ------------------------------------------------------------- readings
    @property
    def observed_crack_length(self) -> float:
        return self.geometry.observed_crack_length

    @property
    def is_closed_throughout(self) -> bool:
        """The crack never opens anywhere on the sweep."""
        return bool(np.all(self.closed))

    @property
    def mostly_closed(self) -> bool:
        """Shut over most of the sweep, even if the tip opens somewhere.

        A longitudinal crack can open slightly once its tip reaches across the
        ligament into the NEXT hole's tensile lobe, while the rest of it is
        still pressed shut. That is a real effect, not an artefact, but it does
        not make the crack a mode I problem.
        """
        return bool(np.mean(self.closed) > 0.5)

    @property
    def k_is_negligible(self) -> bool:
        """Peak K is a rounding error against the weakest toughness."""
        return self.max_k < 0.05 * min(self.toughness.values())

    @property
    def max_k(self) -> float:
        return float(np.max(self.k_fe)) if len(self.k_fe) else 0.0

    def k_at(self, crack_length: float) -> float:
        """FE K interpolated to a crack length, MPa*sqrt(mm)."""
        if len(self.crack_lengths) < 2:
            return float(self.k_fe[0]) if len(self.k_fe) else 0.0
        return float(np.interp(crack_length, self.crack_lengths, self.k_fe))

    @property
    def k_at_observed(self) -> float:
        return self.k_at(self.observed_crack_length)

    def critical_crack_length(self, corner: str = "nominal") -> float:
        """Crack length at which the FE K first reaches K_IC, mm.

        ``nan`` when it never does -- which is the interesting answer.
        """
        kic = self.toughness[corner]
        above = self.k_fe >= kic
        if not np.any(above):
            return float("nan")
        index = int(np.argmax(above))
        if index == 0:
            return float(self.crack_lengths[0])
        return float(
            np.interp(
                kic,
                [self.k_fe[index - 1], self.k_fe[index]],
                [self.crack_lengths[index - 1], self.crack_lengths[index]],
            )
        )

    def propagates(self, corner: str = "low") -> bool:
        """True if K reaches K_IC anywhere on the sweep.

        Defaults to the LOW toughness corner: the question is whether the crack
        can run at all, so the weakest plausible material is the right one to
        ask it of.
        """
        return self.max_k >= self.toughness[corner]

    def margin_at_observed(self, corner: str = "nominal") -> float:
        """``K_IC / K`` at the observed crack length. Above 1 means it holds."""
        k = self.k_at_observed
        return float("inf") if k <= 0 else self.toughness[corner] / k

    def force_to_reach_toughness(self, corner: str = "low") -> float:
        """Ratchet tension that would make ``K = K_IC`` at the peak of the curve.

        K is linear in load, so this is a straight scaling. It converts a
        margin into something physical: how much harder than assumed the strap
        would have to be pulled for static fracture to explain the failure.
        """
        if self.max_k <= 0:
            return float("inf")
        return self.force * self.toughness[corner] / self.max_k

    def force_to_reach_toughness_at_observed(self, corner: str = "low") -> float:
        k = self.k_at_observed
        return float("inf") if k <= 0 else self.force * self.toughness[corner] / k

    @property
    def worst_domain_independence(self) -> float:
        return float(np.max(self.domain_independence)) if len(self.domain_independence) else 0.0

    @property
    def worst_extraction_disagreement(self) -> float:
        """Largest gap between the J-based and COD-based K, as a fraction.

        Only counted where the crack is actually open and K is large enough for
        the ratio to mean anything: comparing two ways of computing a number
        near zero measures nothing.
        """
        threshold = 0.02 * max(self.max_k, 1e-12)
        usable = (
            np.isfinite(self.extraction_agreement) & (~self.closed) & (self.k_fe > threshold)
        )
        if not np.any(usable):
            return float("nan")
        return float(np.max(np.abs(self.extraction_agreement[usable] - 1.0)))

    def handbook_ratio(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.k_handbook > 0, self.k_fe / self.k_handbook, np.nan)

    # -------------------------------------------------------------- reporting
    def summary(self) -> str:
        g = self.geometry
        lines = [
            "K vs CRACK LENGTH",
            "=================",
            "",
            f"  orientation        {self.orientation.value}",
            f"  material           PA66, {self.condition.value} [{self.corner}]",
            f"  ratchet tension    {self.force:g} N",
            f"  crack lengths      {self.crack_lengths[0]:.2f} .. "
            f"{self.crack_lengths[-1]:.2f} mm "
            f"({len(self.crack_lengths)} points)",
            f"  observed crack     {self.observed_crack_length:g} mm",
            "",
        ]

        if self.is_closed_throughout or self.mostly_closed:
            everywhere = self.is_closed_throughout
            lines += [
                "  THE CRACK IS HELD SHUT"
                + (" AT EVERY LENGTH ON THE SWEEP." if everywhere
                   else " OVER MOST OF THE SWEEP."),
                "",
                "  The FE model measures the opening between the two crack faces",
                "  directly, and finds them overlapping"
                + (" at every length" if everywhere else " over most of the range")
                + ": the faces are being",
                "  pressed together, because under axial tension this crack plane sits in the",
                f"  hole's compressive hoop lobe ({self.orientation.hole_hoop_stress_factor:+.0f} x the far field).",
                "",
                f"  Most negative opening on the sweep: {np.min(self.min_opening):.2e} mm",
                "  (a linear model has no contact, so the faces are free to pass",
                "   through each other; that overlap is the signature of closure).",
                "",
            ]
            if not everywhere:
                opened = np.where(~self.closed)[0]
                lines += [
                    f"  It does open slightly from a = "
                    f"{self.crack_lengths[opened[0]]:.2f} mm, where the tip has",
                    "  reached far enough across the ligament to feel the NEXT hole's",
                    f"  tensile lobe. Peak K there is {self.max_k:.2f} MPa*sqrt(mm),"
                    f" against a",
                    f"  toughness of {min(self.toughness.values()):.0f} at the weakest"
                    f" corner -- a factor of",
                    f"  {min(self.toughness.values()) / max(self.max_k, 1e-12):.0f}"
                    f" below. Still nothing that runs a crack.",
                    "",
                ]
            lines += [
                f"  {'a (mm)':>8} {'K_FE':>10} {'opening':>12} {'state':>9}",
                "  " + "-" * 43,
            ]
            for i in range(len(self.crack_lengths)):
                state = (
                    "shut" if self.closed[i]
                    else ("partial" if self.partially_closed[i] else "open")
                )
                lines.append(
                    f"  {self.crack_lengths[i]:>8.3f} {self.k_fe[i]:>10.2f} "
                    f"{self.min_opening[i]:>12.2e} {state:>9}"
                )
            lines.append("")
            return "\n".join(lines)

        lines += [
            f"  {'a (mm)':>8} {'K_FE':>10} {'K_handbook':>11} {'FE/handbook':>12} "
            f"{'closed':>7}",
            "  " + "-" * 52,
        ]
        step = max(1, len(self.crack_lengths) // 12)
        rows = sorted(
            set(list(range(0, len(self.crack_lengths), step)) + [len(self.crack_lengths) - 1])
        )
        ratio = self.handbook_ratio()
        for i in rows:
            flag = "yes" if self.closed[i] else ("partial" if self.partially_closed[i] else "no")
            ratio_text = "-" if not np.isfinite(ratio[i]) else f"{ratio[i]:.3f}"
            lines.append(
                f"  {self.crack_lengths[i]:>8.3f} {self.k_fe[i]:>10.2f} "
                f"{self.k_handbook[i]:>11.2f} {ratio_text:>12} {flag:>7}"
            )

        lines += [
            "",
            "  K in MPa*sqrt(mm). Divide by 31.62 for MPa*sqrt(m).",
            "",
            "  AGAINST THE TOUGHNESS BRACKET",
            "  ('N at obs a' is the tension needed to fracture the crack the part",
            "   actually has; 'N at max a' the tension needed once the crack has",
            f"   grown to {self.crack_lengths[-1]:.1f} mm, most of the ligament.)",
            "",
            f"  {'corner':<9} {'K_IC':>22} {'critical a':>12} {'margin at obs':>14} "
            f"{'N at obs a':>12} {'N at max a':>12}",
            "  " + "-" * 88,
        ]
        for corner in CORNERS:
            kic = self.toughness[corner]
            critical = self.critical_crack_length(corner)
            critical_text = "never" if math.isnan(critical) else f"{critical:.2f} mm"
            at_obs = self.force_to_reach_toughness_at_observed(corner)
            at_max = self.force_to_reach_toughness(corner)
            lines.append(
                f"  {corner:<9} {kic:>8.1f} MPa*sqrt(mm)"
                f" ({kic / 31.6228:>4.1f} MPa*sqrt(m)) {critical_text:>12} "
                f"{self.margin_at_observed(corner):>14.2f} "
                f"{_force_text(at_obs):>12} {_force_text(at_max):>12}"
            )

        validity = lefm_validity(
            self.observed_crack_length,
            self.k_at_observed,
            g,
            grade_for(self.condition),
            corner=self.corner,
            orientation=self.orientation,
        )
        lines += [
            "",
            "  NUMERICAL QUALITY",
            f"    J-integral domain independence   worst {self.worst_domain_independence:.3%}",
            f"    J vs displacement-extrapolation  worst {self.worst_extraction_disagreement:.1%} apart",
            f"    FE vs handbook                   {_range_text(ratio)}",
            "",
            "  LEFM APPLICABILITY",
            f"    {validity.message()}",
        ]
        return "\n".join(lines)

    def verdict(self) -> list:
        """The plain answer, as paragraphs."""
        if self.mostly_closed and not self.is_closed_throughout:
            opened = np.where(~self.closed)[0]
            return [
                "THE CRACK DOES NOT PROPAGATE UNDER RATCHET TENSION, BECAUSE IT IS "
                "HELD SHUT FOR ALMOST ALL OF ITS LIFE. A crack running from one "
                "perforation toward the next lies parallel to the load, and its "
                "mouth sits where the hole concentrates a COMPRESSIVE hoop stress. "
                f"The FE model measures the flank opening directly and finds the "
                f"faces overlapping for every crack length below "
                f"{self.crack_lengths[opened[0]]:.2f} mm.",
                f"Beyond that the tip has reached far enough across the ligament to "
                f"feel the next hole's tensile lobe and the crack cracks open a "
                f"little, but peak K over the whole sweep is only {self.max_k:.2f} "
                f"MPa*sqrt(mm) against a toughness of "
                f"{min(self.toughness.values()):.0f} at the weakest corner -- a "
                f"factor of {min(self.toughness.values()) / max(self.max_k, 1e-12):.0f} "
                f"below. Pulling the strap harder does not change that ratio, "
                f"because both scale with load.",
                "So if the observed crack really does run hole-to-hole, ratchet "
                "tension is not what is driving it. The mechanism has to be "
                "something this model does not contain: bearing load from the "
                "ratchet pawl pressing on the hole edge, bending or twisting around "
                "the buckle, residual stress from moulding, or an environmental "
                "mechanism. Measuring which way the crack actually runs on the "
                "failed part is the first thing to settle.",
            ]
        if self.is_closed_throughout:
            return [
                "THE CRACK DOES NOT PROPAGATE UNDER RATCHET TENSION, BECAUSE IT "
                "CANNOT EVEN OPEN. A crack running from one perforation toward the "
                "next lies parallel to the load, and its mouth sits where the hole "
                "concentrates a COMPRESSIVE hoop stress. Pulling the strap harder "
                "presses this crack shut, it does not open it. No crack length and "
                "no load within this model changes that.",
                "So if the observed crack really does run hole-to-hole, ratchet "
                "tension is not what is driving it, and the mechanism has to be "
                "something this model does not contain: bearing load from the "
                "ratchet pawl pressing on the hole edge, bending or twisting of the "
                "strap around the buckle, a residual stress from moulding, or an "
                "environmental mechanism. Measuring which way the crack actually "
                "runs on the failed part is the first thing to settle.",
            ]

        out = []
        kic_low = self.toughness["low"]
        if not self.propagates("low"):
            out.append(
                f"THE CRACK DOES NOT PROPAGATE UNDER RATCHET TENSION ALONE. Peak K "
                f"over the whole ligament is {self.max_k:.1f} MPa*sqrt(mm), against "
                f"a toughness bracket of {kic_low:.1f} to "
                f"{self.toughness['high']:.1f}. K never reaches even the weakest "
                f"plausible K_IC at any crack length the strap can hold, so a "
                f"single pull at {self.force:g} N cannot run this crack."
            )
            at_observed = self.force_to_reach_toughness_at_observed("low")
            worst = self.force_to_reach_toughness("low")
            longest = float(self.crack_lengths[-1])
            out.append(
                f"To break the strap at the {self.observed_crack_length:g} mm crack "
                f"it actually has would take about {at_observed:.0f} N, "
                f"{at_observed / self.force:.1f} times the assumed service tension. "
                f"Even with the crack grown to {longest:.1f} mm -- most of the way "
                f"across the ligament -- it would still take {worst:.0f} N, "
                f"{worst / self.force:.1f} times service. Both figures use the "
                f"WEAKEST end of the toughness bracket and ignore the steel band's "
                f"load sharing, so both are already generous to crack growth."
            )
        else:
            critical = self.critical_crack_length("low")
            out.append(
                f"THE CRACK CAN PROPAGATE. K reaches the weakest end of the "
                f"toughness bracket ({kic_low:.1f} MPa*sqrt(mm)) at a crack length "
                f"of {critical:.2f} mm. Beyond that, a single pull at "
                f"{self.force:g} N drives it."
            )

        margin = self.margin_at_observed("nominal")
        out.append(
            f"At the observed {self.observed_crack_length:g} mm crack, K is "
            f"{self.k_at_observed:.1f} MPa*sqrt(mm) -- a factor of {margin:.1f} below "
            f"the nominal toughness. Since K rises with crack length and the crack "
            f"has evidently stopped rather than run, the part is behaving "
            f"consistently with this result."
        )
        out.append(
            "WHAT THAT LEAVES. A crack that will not run in one pull can still grow "
            "a little on every pull. Static LEFM has ruled out overload; it says "
            "nothing about fatigue crack growth under repeated ratcheting, creep "
            "crack growth under sustained load, or environmental attack. Those are "
            "the remaining candidates, and they need cyclic and time-dependent "
            "data this model does not use."
        )
        return out


def build_k_curve(
    results: list,
    geometry: StrapGeometry,
    force: float,
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE,
    condition: MoistureCondition = SERVICE_CONDITION,
    corner: str = "nominal",
) -> KCurve:
    """Assemble a :class:`KCurve` from a list of
    :class:`~ratchet_fea.fracture.CrackTipResult`."""
    if not results:
        raise ValueError("no crack results to build a curve from")

    a = np.array([r.crack_length for r in results])
    grade = grade_for(condition)
    handbook = np.array(
        [float(handbook_k(length, geometry, force, orientation)) for length in a]
    )

    return KCurve(
        geometry=geometry,
        orientation=orientation,
        condition=condition,
        corner=corner,
        force=force,
        crack_lengths=a,
        k_fe=np.array([r.k for r in results]),
        k_handbook=handbook,
        toughness={
            c: fracture_toughness_mpa_root_mm(grade, c) for c in CORNERS
        },
        domain_independence=np.array([r.domain_independence for r in results]),
        extraction_agreement=np.array([r.extraction_agreement for r in results]),
        closed=np.array([r.is_closed for r in results]),
        partially_closed=np.array([r.partially_closed for r in results]),
        min_opening=np.array([r.min_opening for r in results]),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def _figure(figsize=(9.5, 6.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt, plt.figure(figsize=figsize)


def plot_k_vs_a(curve: KCurve, path: Path) -> Path:
    """The headline figure: K against crack length, against the K_IC bracket."""
    plt, fig = _figure()
    ax = fig.add_subplot(111)
    a = curve.crack_lengths

    ax.plot(a, curve.k_fe, "-o", ms=3.5, color="tab:blue", label="K, finite element")
    ax.plot(
        a,
        curve.k_handbook,
        "--",
        color="tab:grey",
        label="K, handbook (isolated hole, Newman + Feddersen)",
    )

    ax.axhspan(
        curve.toughness["low"],
        curve.toughness["high"],
        color="firebrick",
        alpha=0.15,
        label="K_IC bracket, PA66",
    )
    for corner, style in (("low", ":"), ("nominal", "-"), ("high", ":")):
        ax.axhline(
            curve.toughness[corner],
            color="firebrick",
            ls=style,
            lw=1.2 if corner == "nominal" else 0.9,
        )
    ax.annotate(
        f"K_IC {curve.toughness['low'] / 31.6228:.1f}-"
        f"{curve.toughness['high'] / 31.6228:.1f} MPa*sqrt(m)",
        xy=(a[0], curve.toughness["nominal"]),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
        color="firebrick",
    )

    observed = curve.observed_crack_length
    if a[0] <= observed <= a[-1]:
        ax.axvline(observed, color="0.35", ls="-.", lw=1.0)
        ax.annotate(
            f"observed crack\n{observed:g} mm",
            xy=(observed, 0.06),
            xycoords=ax.get_xaxis_transform(),
            xytext=(5, 0),
            textcoords="offset points",
            fontsize=8,
            color="0.3",
        )

    top = max(curve.toughness["high"], curve.max_k) * 1.12
    ax.set_ylim(0, top)
    ax.set_xlim(a[0], a[-1])
    ax.set_xlabel("crack length from the hole wall, a (mm)")
    ax.set_ylabel("stress intensity factor K (MPa*sqrt(mm))")
    ax.set_title(
        f"K vs crack length -- {curve.orientation.value} crack at a perforation\n"
        f"{curve.force:g} N ratchet tension, PA66 {curve.condition.value}, "
        "steel band ignored (conservative)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_crack_opening(results: list, path: Path, title: str = "") -> Path:
    """Crack flank opening, which is how closure is diagnosed."""
    from .fracture import crack_opening  # noqa: F401  (documented cross-reference)

    plt, fig = _figure(figsize=(9.0, 5.0))
    ax = fig.add_subplot(111)

    a = np.array([r.crack_length for r in results])
    ax.plot(a, np.array([r.min_opening for r in results]) * 1e3, "-o", ms=4,
            color="tab:purple", label="most negative flank opening")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.fill_between(
        a, ax.get_ylim()[0] * np.ones_like(a), 0.0, color="firebrick", alpha=0.08
    )
    ax.annotate(
        "below zero = faces overlap = crack held shut\n"
        "(a linear model has no contact to stop them)",
        xy=(0.5, 0.12),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="0.3",
    )
    ax.set_xlabel("crack length from the hole wall, a (mm)")
    ax.set_ylabel("flank opening (micrometres)")
    ax.set_title(title or "Crack flank opening")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------------


def write_summary(curve: KCurve, path: Path) -> Path:
    """Dump the headline numbers as JSON, for diffing between runs."""
    payload = {
        "orientation": curve.orientation.value,
        "condition": curve.condition.value,
        "bracket_corner": curve.corner,
        "force_N": curve.force,
        "geometry": {
            k: v
            for k, v in curve.geometry.__dict__.items()
            if isinstance(v, (int, float, bool))
        },
        "toughness_mpa_root_mm": curve.toughness,
        "toughness_mpa_root_m": {
            c: v / 31.6228 for c, v in curve.toughness.items()
        },
        "observed_crack_length_mm": curve.observed_crack_length,
        "k_at_observed": curve.k_at_observed,
        "max_k": curve.max_k,
        "is_closed_throughout": bool(curve.is_closed_throughout),
        "propagates_at_low_toughness": bool(curve.propagates("low")),
        "critical_crack_length_mm": {
            c: curve.critical_crack_length(c) for c in CORNERS
        },
        "margin_at_observed": {c: curve.margin_at_observed(c) for c in CORNERS},
        "force_for_k_ic_N": {c: curve.force_to_reach_toughness(c) for c in CORNERS},
        "numerical_quality": {
            "worst_domain_independence": curve.worst_domain_independence,
            "worst_extraction_disagreement": curve.worst_extraction_disagreement,
        },
        "series": {
            "crack_length_mm": curve.crack_lengths.tolist(),
            "k_fe": curve.k_fe.tolist(),
            "k_handbook": curve.k_handbook.tolist(),
            "closed": curve.closed.tolist(),
            "min_opening_mm": curve.min_opening.tolist(),
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path
