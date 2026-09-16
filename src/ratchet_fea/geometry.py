"""Geometry of the STP-RB-001 ratchet belt strap.

UNITS (consistent throughout the whole repository -- see README.md):
    length          mm
    force           N
    stress          MPa  (N/mm^2)
    stress intensity  MPa*sqrt(mm)   (1 MPa*sqrt(m) = 31.6228 MPa*sqrt(mm))

=============================================================================
!! EVERY DIMENSION IN :data:`PLACEHOLDER_GEOMETRY` IS A VISUAL ESTIMATE !!
=============================================================================
They were scaled off photographs of the failed part, not measured.  Replacing
them is a one-line change -- construct a :class:`StrapGeometry` with the real
numbers and pass it to the Tier 1 and Tier 2 entry points.  Nothing downstream
hardcodes a dimension; mesh density, crack seeding, load normalisation and
post-processing all derive from this object.

Coordinate system for the 2D (plane-stress) models
--------------------------------------------------
    x  along the strap (the load direction)      0 .. modelled_length
    y  across the strap width                    0 .. strap_width
    z  through the thickness (out of plane)      0 .. strap_thickness

The FE model represents a *window* cut out of a longer strap, containing
``n_holes`` perforations plus ``end_margin`` of plain material at each end so
the truncation planes sit a St Venant distance away from the first and last
hole.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .provenance import Documented, Provenance, Source

__all__ = [
    "CrackOrientation",
    "CrackGeometry",
    "StrapGeometry",
    "PLACEHOLDER_GEOMETRY",
]


class CrackOrientation(str, Enum):
    """Which way a crack runs out of a perforation.

    The two are completely different fracture problems, because a circular hole
    in a uniaxial tension field has hoop stress

        sigma_theta = sigma (1 + 2 cos 2(theta - 90 deg))   on the hole wall

    which is ``+3 sigma`` at the point 90 degrees round from the load axis and
    ``-sigma`` at the point on the load axis itself.
    """

    #: Across the strap, from the hole wall toward the free side edge. The
    #: crack plane is NORMAL to the ratchet tension, so it is loaded in mode I
    #: and starts life in the ``+3 sigma`` hoop stress at the hole. This is the
    #: classical crack-from-a-hole configuration, and the one that separates
    #: the strap if it runs.
    TRANSVERSE = "transverse"

    #: Along the strap, from the hole wall toward the next perforation. The
    #: crack plane is PARALLEL to the ratchet tension, and it starts in the
    #: ``-sigma`` compressive lobe of the hole. Tension alone cannot open it.
    #: Modelled anyway, because it is the direction the failed part is reported
    #: to have cracked in, and the model should say so rather than assume it.
    LONGITUDINAL = "longitudinal"

    @property
    def is_mode_i_under_axial_tension(self) -> bool:
        return self is CrackOrientation.TRANSVERSE

    @property
    def hole_hoop_stress_factor(self) -> float:
        """Hoop stress at the crack mouth, as a multiple of the far field.

        ``+3`` where a transverse crack starts, ``-1`` where a longitudinal one
        does. The sign is the whole story: a crack cannot open in a compressive
        hoop stress.
        """
        return 3.0 if self is CrackOrientation.TRANSVERSE else -1.0


@dataclass(frozen=True)
class CrackGeometry:
    """A single crack seeded at one perforation.

    ``length`` is measured from the HOLE WALL, not the hole centre -- the
    convention every crack-from-a-hole handbook solution uses, and the one a
    measurement off the failed part naturally gives.
    """

    #: Crack length from the hole wall, mm. Conventionally ``a``.
    length: float
    orientation: CrackOrientation = CrackOrientation.TRANSVERSE
    #: Which perforation the crack starts from. Defaults to the middle one,
    #: which is the only kind representative of a long strap.
    hole_index: int | None = None
    #: Which side of the hole. +1 is toward increasing y (transverse) or
    #: increasing x (longitudinal).
    side: int = 1

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError("crack length must be positive")
        if self.side not in (-1, 1):
            raise ValueError("crack side must be +1 or -1")

    @property
    def direction(self) -> tuple:
        """Unit vector the crack runs along, in the FE frame."""
        if self.orientation is CrackOrientation.TRANSVERSE:
            return (0.0, float(self.side))
        return (float(self.side), 0.0)

    @property
    def normal(self) -> tuple:
        """Unit normal to the crack plane -- the direction it opens in."""
        dx, dy = self.direction
        return (-dy, dx)


@dataclass(frozen=True)
class StrapGeometry(Documented):
    """Dimensions of the perforated, steel-reinforced PA66 strap.

    The strap is modelled as a through-thickness laminate: a PA66 matrix with
    a continuous steel band buried in it.

    The band matters to the fracture problem only through load sharing: if it
    is real and bonded, it carries most of the ratchet tension and shields the
    polymer.  The LEFM model defaults to ignoring it -- see
    :mod:`ratchet_fea.mechanics` -- because the band's existence is itself
    inferred rather than observed, and ignoring it is the conservative
    assumption.
    """

    # --- outer envelope -----------------------------------------------------
    strap_width: float = 25.0
    strap_thickness: float = 3.0

    # --- perforation row ----------------------------------------------------
    hole_diameter: float = 4.0
    hole_pitch: float = 10.0
    n_holes: int = 5
    #: Offset of the hole centreline from the strap centreline (0 = centred).
    hole_row_offset: float = 0.0
    #: Plain material modelled beyond the first/last hole centre.
    #: ``None`` means one full strap width, the usual St Venant distance, so it
    #: rescales automatically when the placeholder dimensions are replaced.
    end_margin: float | None = None

    # --- internal steel reinforcing band ------------------------------------
    steel_band_width: float = 18.0
    steel_band_thickness: float = 0.8
    #: Offset of the band centreline from the strap centreline (0 = centred).
    steel_band_offset: float = 0.0
    #: Whether the perforations are punched through the steel band as well as
    #: the polymer.  This changes the constraint model *qualitatively* and is
    #: the single highest-value thing to check on a real sample.
    holes_pierce_band: bool = True

    # --- service load -------------------------------------------------------
    #: Peak longitudinal tension the strap sees in normal use.
    service_tension: float = 500.0

    # --- observed damage ----------------------------------------------------
    #: Length of the crack seen on the failed part, measured from the HOLE EDGE
    #: (not from the hole centre). The question this repository exists to answer
    #: is whether K exceeds K_IC at or below this length.
    observed_crack_length: float = 2.0

    SOURCES = {
        "strap_width": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Scaled off failure photographs against the buckle for reference.",
            "Calipers across the strap, away from the failure.",
        ),
        "strap_thickness": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Sets the section area and therefore the far-field stress, which "
            "K is directly proportional to.",
            "Micrometer on an unperforated section.",
        ),
        "hole_diameter": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Sets d/W and hence the stress concentration factor.",
            "Pin gauge or optical comparator on an undamaged hole.",
        ),
        "hole_pitch": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Sets hole-to-hole interaction, which is the main thing Tier 2 "
            "adds over the Tier 1 single-hole estimate.",
            "Calipers centre-to-centre over 10 holes, divide by 9.",
        ),
        "n_holes": Source(
            Provenance.ASSUMED,
            "-",
            "Number of holes in the modelled window, not in the whole strap. "
            "Enough to show the interior (periodic) hole behaviour.",
            "No measurement needed; check convergence by raising it.",
        ),
        "hole_row_offset": Source(
            Provenance.ASSUMED,
            "mm",
            "Assumed the perforation row is on the strap centreline.",
            "Calipers hole edge to each strap edge; any difference is 2x this.",
        ),
        "end_margin": Source(
            Provenance.ASSUMED,
            "mm",
            "Model truncation distance, not a physical dimension. Defaults to "
            "one strap width (St Venant).",
            "No measurement needed; check the end stress field is uniform.",
        ),
        "steel_band_width": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "VERIFY: the band's existence and width are inferred, not seen. "
            "Sets how much of the ratchet tension bypasses the polymer.",
            "Section and etch, or X-ray / magnet test on a scrap length.",
        ),
        "steel_band_thickness": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Controls the stiffness ratio and hence how much of the load the "
            "polymer actually carries.",
            "Section and measure under a microscope.",
        ),
        "steel_band_offset": Source(
            Provenance.ASSUMED,
            "mm",
            "Assumed the band is on the strap centreline. An off-centre band "
            "would add bending that this membrane model cannot represent.",
            "Section and measure to each face.",
        ),
        "holes_pierce_band": Source(
            Provenance.ASSUMED,
            "-",
            "If the band is continuous through the hole region it bridges a "
            "crack in the polymer, which is a layered-cracking problem this "
            "plane-stress model cannot represent at all.",
            "Section through a hole, or look for steel at the hole wall.",
        ),
        "observed_crack_length": Source(
            Provenance.PHOTO_ESTIMATE,
            "mm",
            "Crack length on the failed part, from the hole edge. The whole "
            "LEFM conclusion is read off the K(a) curve at this value, so an "
            "error here moves the answer directly.",
            "Measure from the hole edge to the crack tip on the failed part, "
            "under magnification. Note which way the crack runs: toward the "
            "adjacent hole (along the strap) or toward the strap edge (across "
            "it). They are completely different fracture problems.",
        ),
        "service_tension": Source(
            Provenance.ASSUMED,
            "N",
            "Peak hand-ratcheted tension. Tier 1 found the answer is not "
            "sensitive to this (mechanical stress is far below strength), but "
            "it sets the mechanical half of the combined load case.",
            "Load cell in line with the strap during normal tightening.",
        ),
    }

    def __post_init__(self) -> None:
        if self.end_margin is None:
            # One strap width of plain material each end. Verified empirically:
            # below about this distance the first and last hole's stress
            # concentration is still moving with the truncation planes, and the
            # two ends disagree with each other, which is a sure sign the
            # boundary condition is leaking into the result.
            object.__setattr__(self, "end_margin", self.strap_width)

        if self.n_holes < 1:
            raise ValueError("n_holes must be at least 1")
        if self.hole_diameter <= 0 or self.strap_width <= 0 or self.strap_thickness <= 0:
            raise ValueError("strap dimensions must be positive")
        if self.end_margin <= self.hole_radius:
            raise ValueError("end_margin must exceed the hole radius")
        if self.steel_band_thickness >= self.strap_thickness:
            raise ValueError("steel band cannot be thicker than the strap")
        if self.steel_band_thickness < 0 or self.steel_band_width < 0:
            raise ValueError("steel band dimensions must be non-negative")
        if self.steel_band_width > self.strap_width + 1e-12:
            raise ValueError("steel band cannot be wider than the strap")

        # The hole row must actually fit inside the strap, with a ligament left.
        top = self.hole_row_offset + self.hole_radius
        bottom = self.hole_row_offset - self.hole_radius
        if top >= self.strap_width / 2 or bottom <= -self.strap_width / 2:
            raise ValueError(
                "hole row breaks out of the strap width; check hole_diameter, "
                "hole_row_offset and strap_width"
            )

        if self.hole_pitch <= self.hole_diameter and self.n_holes > 1:
            raise ValueError(
                f"hole_pitch ({self.hole_pitch}) must exceed hole_diameter "
                f"({self.hole_diameter}) or the holes merge"
            )

        if not self.holes_pierce_band and self.has_steel_band:
            # The band must then clear the holes entirely in y.
            band_lo = self.steel_band_offset - self.steel_band_width / 2
            band_hi = self.steel_band_offset + self.steel_band_width / 2
            if band_lo < top and band_hi > bottom:
                raise ValueError(
                    "holes_pierce_band is False but the band footprint overlaps "
                    "the hole row; either the band is pierced, or it is split / "
                    "offset away from the holes -- resolve against a real sample"
                )

    # ------------------------------------------------------------------ basic
    @property
    def hole_radius(self) -> float:
        return 0.5 * self.hole_diameter

    @property
    def has_steel_band(self) -> bool:
        return self.steel_band_thickness > 0.0 and self.steel_band_width > 0.0

    @property
    def modelled_length(self) -> float:
        """x-extent of the FE window."""
        return (self.n_holes - 1) * self.hole_pitch + 2 * self.end_margin

    @property
    def polymer_thickness_in_band(self) -> float:
        """PA66 thickness remaining where the steel band is present."""
        return self.strap_thickness - self.steel_band_thickness

    @property
    def band_full_width(self) -> bool:
        """True if the band spans the whole strap (no polymer-only side rails)."""
        return self.has_steel_band and self.steel_band_width >= self.strap_width - 1e-9

    # -------------------------------------------------------------- sections
    @property
    def gross_section_area(self) -> float:
        """Full cross-sectional area of the strap, mm^2."""
        return self.strap_width * self.strap_thickness

    @property
    def net_section_area(self) -> float:
        """Cross-sectional area through a hole, mm^2."""
        return (self.strap_width - self.hole_diameter) * self.strap_thickness

    @property
    def d_over_W(self) -> float:
        """Hole diameter to strap width ratio; the argument to Kt charts."""
        return self.hole_diameter / self.strap_width

    @property
    def pitch_over_d(self) -> float:
        """Hole spacing ratio; controls hole-to-hole interaction."""
        return self.hole_pitch / self.hole_diameter

    @property
    def side_ligament(self) -> float:
        """Narrowest material path from hole edge to strap edge, mm.

        With a centred row this is the same on both sides.  It is the shortest
        path a transverse crack has to cross, and the ligament that carries
        the net-section load.
        """
        upper = self.strap_width / 2 - (self.hole_row_offset + self.hole_radius)
        lower = (self.hole_row_offset - self.hole_radius) + self.strap_width / 2
        return min(upper, lower)

    @property
    def inter_hole_ligament(self) -> float:
        """Edge-to-edge gap between adjacent holes along the strap, mm."""
        return self.hole_pitch - self.hole_diameter

    # ----------------------------------------------------------- hole layout
    @property
    def hole_centres(self) -> np.ndarray:
        """``(n_holes, 2)`` array of hole centre coordinates in the FE frame."""
        xs = self.end_margin + self.hole_pitch * np.arange(self.n_holes, dtype=float)
        ys = np.full(self.n_holes, self.strap_width / 2 + self.hole_row_offset)
        return np.column_stack([xs, ys])

    @property
    def hole_labels(self) -> list[str]:
        """Stable names used for mesh boundary tags and result dictionaries."""
        return [f"hole_{i}" for i in range(self.n_holes)]

    def is_interior_hole(self, index: int) -> bool:
        """True for holes with a neighbour on both sides.

        Only interior holes are representative of a long strap. The first and
        last hole in the model genuinely carry a higher stress concentration,
        because they are not shielded by a neighbour on one side -- but in the
        real strap that applies only to the two holes at the very ends of the
        perforation row, not to the many in between. Quote interior values, and
        read the end holes as a separate case.
        """
        return 0 < index < self.n_holes - 1

    # ----------------------------------------------------------- crack setup
    def default_crack_hole(self) -> int:
        """Index of the hole a crack is seeded at unless told otherwise.

        The middle one. End holes see the model truncation planes and carry a
        higher stress concentration because nothing shields them on one side,
        so they are not representative of a long strap.
        """
        return self.n_holes // 2

    def crack_mouth(self, crack) -> np.ndarray:
        """Point on the hole wall where the crack starts, in the FE frame."""
        index = self.default_crack_hole() if crack.hole_index is None else crack.hole_index
        if not 0 <= index < self.n_holes:
            raise ValueError(f"hole index {index} outside 0..{self.n_holes - 1}")
        centre = self.hole_centres[index]
        dx, dy = crack.direction
        return np.array(
            [centre[0] + dx * self.hole_radius, centre[1] + dy * self.hole_radius]
        )

    def crack_tip(self, crack) -> np.ndarray:
        """Crack tip position in the FE frame."""
        mouth = self.crack_mouth(crack)
        dx, dy = crack.direction
        return np.array([mouth[0] + dx * crack.length, mouth[1] + dy * crack.length])

    def remaining_ligament(self, crack) -> float:
        """Uncracked material ahead of the tip, mm.

        For a transverse crack that is the distance to the free side edge; for
        a longitudinal one, the distance to the next hole's wall. Zero or
        negative means the crack has run out of material, which LEFM cannot
        describe.
        """
        return self.max_crack_length(crack) - crack.length

    def max_crack_length(self, crack) -> float:
        """Longest crack this geometry can hold before it breaks through, mm."""
        if crack.orientation is CrackOrientation.TRANSVERSE:
            centre_y = self.strap_width / 2 + self.hole_row_offset
            edge = self.strap_width if crack.side > 0 else 0.0
            return abs(edge - centre_y) - self.hole_radius
        index = (
            self.default_crack_hole() if crack.hole_index is None else crack.hole_index
        )
        neighbour = index + crack.side
        if not 0 <= neighbour < self.n_holes:
            # No neighbouring hole that way: the crack runs to the cut face,
            # which is a model boundary rather than a physical limit.
            return self.end_margin - self.hole_radius
        return self.inter_hole_ligament

    def crack_is_admissible(self, crack) -> bool:
        """Whether the crack fits, with ligament left ahead of the tip."""
        return 0.0 < crack.length < self.max_crack_length(crack)

    # ------------------------------------------------------------- reporting
    def summary(self) -> str:
        band = (
            f"{self.steel_band_width:g} x {self.steel_band_thickness:g} mm"
            f" ({'pierced by holes' if self.holes_pierce_band else 'clear of holes'})"
            if self.has_steel_band
            else "none"
        )
        return "\n".join(
            [
                "STRAP GEOMETRY (STP-RB-001)",
                "---------------------------",
                f"  strap                {self.strap_width:g} x {self.strap_thickness:g} mm",
                f"  perforations         {self.n_holes} x dia {self.hole_diameter:g} mm "
                f"at {self.hole_pitch:g} mm pitch",
                f"  d/W                  {self.d_over_W:.3f}",
                f"  pitch/d              {self.pitch_over_d:.2f}",
                f"  side ligament        {self.side_ligament:g} mm",
                f"  inter-hole ligament  {self.inter_hole_ligament:g} mm",
                f"  gross section        {self.gross_section_area:g} mm^2",
                f"  net section          {self.net_section_area:g} mm^2",
                f"  steel band           {band}",
                f"  modelled window      {self.modelled_length:g} x {self.strap_width:g} mm",
                f"  service tension      {self.service_tension:g} N",
            ]
        )


#: Default geometry used by both entry points.
#:
#: VERIFY: every dimension here is a visual estimate from photographs.  Build a
#: ``StrapGeometry(...)`` with measured values and the whole study re-runs.
PLACEHOLDER_GEOMETRY = StrapGeometry()
