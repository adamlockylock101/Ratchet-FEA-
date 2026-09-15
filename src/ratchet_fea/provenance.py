"""Provenance tracking for every number that enters the model.

STP-RB-001 is an investigation into a *physical* part that nobody has yet put
calipers or a DSC on.  Almost every number in this repository is therefore a
placeholder.  The single most dangerous failure mode for this kind of work is
forgetting which numbers are real and which were read off a photograph.

Every geometric dimension and material property carries a :class:`Provenance`
tag.  ``scripts/run_tier1_analytical.py`` and ``scripts/run_tier2_fe.py`` both
print a verification report listing everything that is not yet ``MEASURED``, so
the caveat travels with the result instead of living in someone's memory.

Grep for ``VERIFY:`` across the source tree to find every assumption that needs
checking against a physical part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dc_fields, is_dataclass, replace as dc_replace
from enum import Enum
from typing import ClassVar, Iterable, Iterator, Mapping


class Provenance(str, Enum):
    """Where a number came from, in decreasing order of trustworthiness."""

    #: Taken off a physical sample with an instrument.  Trustworthy.
    MEASURED = "measured"
    #: Datasheet for the *specific* grade used in the part.
    DATASHEET = "datasheet"
    #: Published property range for the generic polymer family.  Bracketed,
    #: because unfilled PA66 properties move by 2-3x between dry-as-moulded and
    #: conditioned, and grades vary on top of that.
    LITERATURE = "literature"
    #: Scaled off a photograph.  Good to maybe +/-20%.
    PHOTO_ESTIMATE = "photo-estimate"
    #: A modelling decision with no measurement behind it at all.
    ASSUMED = "assumed"

    @property
    def needs_verification(self) -> bool:
        """True for anything that is not yet tied to the real part."""
        return self is not Provenance.MEASURED

    @property
    def rank(self) -> int:
        order = [
            Provenance.MEASURED,
            Provenance.DATASHEET,
            Provenance.LITERATURE,
            Provenance.PHOTO_ESTIMATE,
            Provenance.ASSUMED,
        ]
        return order.index(self)


@dataclass(frozen=True)
class Source:
    """Metadata attached to a single named quantity.

    This deliberately holds no value -- it describes where the value *came
    from*.  Values live as plain floats on the dataclasses so that arithmetic
    stays readable (``geom.strap_width / 2``, not ``geom.strap_width.value / 2``).
    """

    provenance: Provenance
    units: str = ""
    note: str = ""
    #: What action would upgrade this to ``MEASURED``.
    verify_by: str = ""

    @property
    def needs_verification(self) -> bool:
        return self.provenance.needs_verification


@dataclass(frozen=True)
class Bracket:
    """A low/high range for a property that is not known to a single value.

    Tier 1 deliberately works in brackets rather than point values: the answer
    to "does this part crack?" is only meaningful if it is robust across the
    plausible range of PA66 properties.  Anything that flips conclusion between
    ``low`` and ``high`` is a measurement that needs taking.
    """

    low: float
    high: float
    source: Source
    #: ``"linear"`` averages arithmetically; ``"log"`` geometrically.  Use
    #: ``"log"`` for quantities that span decades, such as diffusivity.
    scale: str = "linear"

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"bracket low ({self.low}) exceeds high ({self.high})")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"unknown scale {self.scale!r}")
        if self.scale == "log" and self.low <= 0.0:
            raise ValueError("log-scaled bracket requires strictly positive bounds")

    @property
    def nominal(self) -> float:
        """Central value, respecting the bracket's scale."""
        if self.scale == "log":
            return math.sqrt(self.low * self.high)
        return 0.5 * (self.low + self.high)

    @property
    def spread(self) -> float:
        """``high / low``.  A spread near 1 means the property is well pinned."""
        if self.low == 0.0:
            return math.inf
        return self.high / self.low

    def at(self, which: str) -> float:
        """Select ``"low"``, ``"nominal"`` or ``"high"`` by name.

        Lets an entire study be re-run at a bracket corner from one CLI flag
        rather than by editing numbers.
        """
        try:
            return {"low": self.low, "nominal": self.nominal, "high": self.high}[which]
        except KeyError:
            raise ValueError(
                f"bracket corner must be 'low', 'nominal' or 'high', got {which!r}"
            ) from None

    def __iter__(self) -> Iterator[float]:
        yield self.low
        yield self.high

    def __repr__(self) -> str:  # pragma: no cover - display only
        u = f" {self.source.units}" if self.source.units else ""
        return f"Bracket({self.low:g}..{self.high:g}{u}, {self.source.provenance.value})"


@dataclass(frozen=True)
class VerificationItem:
    """One row of the "things still to check against a real part" report."""

    owner: str
    name: str
    value: str
    provenance: Provenance
    verify_by: str
    note: str = ""


def _format_value(value: object, units: str) -> str:
    if isinstance(value, Bracket):
        shown = f"{value.low:g}..{value.high:g}"
    elif isinstance(value, bool):
        shown = str(value)
    elif isinstance(value, float):
        shown = f"{value:g}"
    else:
        shown = str(value)
    return f"{shown} {units}".strip()


class Documented:
    """Mixin for objects whose inputs carry :class:`Source` metadata.

    There are two ways a value becomes traceable:

    1.  A plain float field named in the class-level ``SOURCES`` mapping.  This
        keeps numerics readable -- ``geom.strap_width / 2``, not
        ``geom.strap_width.value / 2`` -- while still recording provenance.
    2.  A :class:`Bracket` field, which carries its own :class:`Source`.

    :meth:`verification_items` walks both, and recurses into nested
    ``Documented`` attributes, so one call on a top-level object reports every
    unverified input beneath it.
    """

    SOURCES: ClassVar[Mapping[str, Source]] = {}

    @property
    def doc_label(self) -> str:
        """Name shown in the ``WHERE`` column of the verification report.

        Override where the class name alone is ambiguous -- several PA66
        conditioning states share the same container classes, and a report row
        reading ``ElasticProperties`` would not say *which* state.
        """
        return type(self).__name__

    def source_for(self, name: str) -> Source:
        value = getattr(self, name, None)
        if isinstance(value, Bracket):
            return value.source
        try:
            return self.SOURCES[name]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} has no provenance recorded for {name!r}; "
                "every model input must be traceable -- add it to SOURCES"
            ) from None

    def _documented_attribute_names(self) -> list[str]:
        names = list(self.SOURCES)
        if is_dataclass(self):
            names += [f.name for f in dc_fields(self) if f.name not in self.SOURCES]
        return names

    def verification_items(self) -> list[VerificationItem]:
        items: list[VerificationItem] = []
        seen: set[int] = set()
        for name in self._documented_attribute_names():
            value = getattr(self, name, None)

            # Nested documented objects report on themselves.
            if isinstance(value, Documented):
                if id(value) not in seen:
                    seen.add(id(value))
                    for child in value.verification_items():
                        # Re-own anonymous container classes onto the parent so
                        # the report says which grade a property belongs to.
                        if child.owner == type(value).__name__:
                            child = dc_replace(child, owner=self.doc_label)
                        items.append(child)
                continue

            if isinstance(value, Bracket):
                source = value.source
            elif name in self.SOURCES:
                source = self.SOURCES[name]
            else:
                continue

            if not source.needs_verification:
                continue
            items.append(
                VerificationItem(
                    owner=self.doc_label,
                    name=name,
                    value=_format_value(value, source.units),
                    provenance=source.provenance,
                    verify_by=source.verify_by,
                    note=source.note,
                )
            )
        return items


def collect_verification_items(*objects: object) -> list[VerificationItem]:
    """Gather verification rows from any mix of :class:`Documented` objects.

    Also walks containers of ``Bracket`` values (e.g. a material property set)
    so that a single call covers geometry and materials together.
    """
    items: list[VerificationItem] = []
    for obj in objects:
        if obj is None:
            continue
        if isinstance(obj, Documented):
            items.extend(obj.verification_items())
        if isinstance(obj, (list, tuple, set)):
            items.extend(collect_verification_items(*obj))
    unique: dict[tuple, VerificationItem] = {}
    for it in items:
        unique.setdefault((it.owner, it.name, it.value), it)
    ordered = list(unique.values())
    ordered.sort(key=lambda it: (-it.provenance.rank, it.owner, it.name))
    return ordered


def format_verification_report(
    items: Iterable[VerificationItem], title: str = "ASSUMPTIONS STILL TO VERIFY"
) -> str:
    """Render verification rows as a fixed-width table for the run scripts."""
    items = list(items)
    if not items:
        return f"{title}\n  (none -- every input is MEASURED)\n"

    headers = ("WHERE", "QUANTITY", "VALUE", "PROVENANCE", "VERIFY BY")
    rows = [
        (it.owner, it.name, it.value, it.provenance.value, it.verify_by or "-")
        for it in items
    ]
    widths = [
        max(len(headers[i]), max(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    rule = "  ".join("-" * w for w in widths)
    lines = [title, "=" * len(title), ""]
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append(rule)
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    lines.append("")
    lines.append(
        f"{len(rows)} input(s) are not yet tied to a physical measurement. "
        "Treat all results as screening-level only."
    )
    return "\n".join(lines) + "\n"
