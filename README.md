# STP-RB-001 — PA66 ratchet belt strap, cracking in normal use

A static linear elastic fracture mechanics (LEFM) investigation: a crack is
seeded at a perforation, the ratchet tension is applied, and the stress
intensity factor at the crack tip is compared against the PA66 K_IC bracket
over a sweep of crack lengths.

**One question: once a crack has nucleated at a hole, does it run under normal
ratchet load alone?**

> ### ⚠️ Read this before quoting any number
>
> **Every dimension here is a visual estimate scaled off photographs. Every
> material property is a published range for generic unfilled PA66, not a
> datasheet for the grade in the part.** Nobody has yet put calipers, a
> micrometer or a DSC on a physical sample.
>
> The code is built so that fixing this is a one-line change — see
> [Replacing the placeholders](#replacing-the-placeholders). Until then, both
> entry points print every unverified input and what it would take to measure
> it, and all results are screening-level only.

---

## Contents

- [The answer](#the-answer)
  - [The two crack orientations are different problems](#the-two-crack-orientations-are-different-problems)
  - [Transverse crack: K never reaches K_IC](#transverse-crack-k-never-reaches-k_ic)
  - [Longitudinal crack: it cannot even open](#longitudinal-crack-it-cannot-even-open)
  - [What that leaves](#what-that-leaves)
- [Units](#units)
- [Installation](#installation)
- [Running it](#running-it)
- [Replacing the placeholders](#replacing-the-placeholders)
- [How the model works](#how-the-model-works)
- [Cross-checks and verification](#cross-checks-and-verification)
- [Assumptions that still need a physical part](#assumptions-that-still-need-a-physical-part)
- [What this model cannot do](#what-this-model-cannot-do)
- [Repository layout](#repository-layout)
- [Development](#development)

---

## The answer

At the placeholder geometry, 500 N ratchet tension and PA66 conditioned to 50 %
RH: **no. The crack does not propagate under ratchet load alone, in either
orientation, and it is not close.**

### The two crack orientations are different problems

The perforations run **along** the strap and the ratchet tension runs along it
too. That makes the direction a crack runs out of a hole decisive, because the
hoop stress on a hole wall in a uniaxial field is

```
sigma_theta = sigma (1 + 2 cos 2(theta - 90°))
```

— **+3σ** at 90° from the load axis, **−σ** on the axis itself.

| | **transverse** | **longitudinal** |
|---|---|---|
| runs | across the strap, toward the free edge | along the strap, toward the next hole |
| crack plane vs load | normal — mode I | parallel — not mode I |
| hoop stress at the mouth | **+3σ** | **−σ** (compressive) |
| ligament available | 10.5 mm | 6.0 mm |
| can it open? | yes | **no, it is pressed shut** |

The failed part is reported to have cracked **toward the adjacent hole**, which
is the longitudinal case. Both are modelled and reported, so the comparison is
on the page rather than in an argument.

### Transverse crack: K never reaches K_IC

| | value |
|---|---|
| K at the observed 2 mm crack | **20.7 MPa·√mm** (0.66 MPa·√m) |
| Peak K over the whole 10.5 mm ligament | **56.2 MPa·√mm** |
| K_IC bracket, PA66 at 50 % RH | **94.9 – 173.9 MPa·√mm** (3.0 – 5.5 MPa·√m) |
| Crack length at which K reaches K_IC | **never** |
| Margin at the observed crack | **6.5×** (nominal), **4.6×** (weakest toughness) |

K rises with crack length but flattens out well below the toughness band — even
with the crack grown to 8.9 mm, most of the way across the ligament. Converting
that margin into something physical:

| | to fracture at the **observed 2 mm** crack | to fracture at a **near-breakthrough 8.9 mm** crack |
|---|---|---|
| weakest toughness corner | **2293 N** (4.6× service) | **845 N** (1.7× service) |
| nominal | 3249 N (6.5×) | 1197 N (2.4×) |

Checked at the **dry-as-moulded** corner too — the brittlest state, K_IC
2.5 MPa·√m — and it still never reaches it: margin 3.8× at the observed crack.

Both figures already ignore the steel band. If the band is real and bonded it
would carry about **98 %** of the tension, making K roughly **56× smaller**
still. Every "does not propagate" verdict here has a large margin behind it.

### Longitudinal crack: it cannot even open

The FE model measures the opening between the two crack faces directly rather
than assuming anything. For a crack running hole-to-hole it finds the faces
**overlapping** at every length below 2.89 mm — they are being pressed
together, because that crack plane sits in the hole's compressive lobe. A
linear model has no contact to stop them, and that interpenetration is the
signature of closure.

Past 2.89 mm the tip has reached far enough across the ligament to feel the
*next* hole's tensile lobe and the crack cracks open slightly, but peak K over
the whole sweep is **1.23 MPa·√mm** against a toughness of 94.9 — a factor of
**77** below. Pulling harder does not change that ratio: both scale with load.

### What that leaves

Static overload is ruled out. What is not:

1. **Which way the crack actually runs.** If it really is hole-to-hole, ratchet
   tension is not the driver at all, and the mechanism has to be something this
   model does not contain — bearing load from the ratchet pawl pressing on the
   hole edge, bending or twisting around the buckle, residual stress from
   moulding, or an environmental mechanism. **Measuring the crack direction on
   the failed part is the single highest-value thing to do next**, because it
   decides which of two completely different investigations is the right one.
2. **Fatigue.** A crack that will not run in one pull can still grow a little
   on every pull. Needs S-N or Paris-law data for the grade, and a duty cycle.
3. **Creep crack growth** under sustained tension, which a monotonic K_IC says
   nothing about.
4. **Environmental attack** — a chemical or UV mechanism that lowers the
   effective toughness far below the bracket used here.

---

## Units

One consistent system. The single exception is documented and isolated.

| quantity | unit |
|---|---|
| length | mm |
| force | N |
| stress, modulus | MPa (N/mm²) |
| **stress intensity, K** | **MPa·√mm** |
| J-integral | N/mm |

**The one conversion.** K_IC brackets in `materials.py` are stored in
**MPa·√m**, because that is what every datasheet and paper quotes, and a
bracket nobody can read against a datasheet is a bracket nobody will maintain.
`materials.fracture_toughness_mpa_root_mm()` is the only place the two meet:

```
1 MPa·√m = √1000 MPa·√mm = 31.6228 MPa·√mm
```

Getting that factor wrong scales every fracture margin in the study by 31.6,
which is why it is a named function with a test on it rather than a
multiplication scattered through the code.

---

## Installation

The handbook screen needs only numpy and scipy. The FE model needs a mesher and
a solver.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[fe,dev]"      # or: pip install -r requirements.txt
```

### gmsh needs system libraries even when headless

The `gmsh` wheel links against X11/GL shared objects and raises `OSError` at
import time without them. On Debian/Ubuntu:

```bash
sudo apt-get install -y libglu1-mesa libxft2 libxinerama1 \
                        libxcursor1 libxrender1 libxfixes3 libfontconfig1
```

`mesh.build_strip_mesh` catches this and names the missing packages. The test
suite detects it and skips every gmsh-dependent test automatically.

---

## Running it

### Handbook screen — seconds, no FE stack

```bash
python scripts/run_handbook_screen.py
python scripts/run_handbook_screen.py --condition dry      # brittlest state
python scripts/run_handbook_screen.py --tension 2000
python scripts/run_handbook_screen.py --observed-crack 3.5
```

### FE sweep

```bash
python scripts/run_lefm_sweep.py                      # both orientations, ~21 s
python scripts/run_lefm_sweep.py --quick              # coarse smoke test
python scripts/run_lefm_sweep.py --orientation transverse
python scripts/run_lefm_sweep.py --condition dry
python scripts/run_lefm_sweep.py --convergence        # mesh convergence only
```

Outputs land in `results/<orientation>/`:

| file | what it shows |
|---|---|
| `k_vs_crack_length.png` | **the headline figure** — FE K(a), the handbook curve over it, and the K_IC band |
| `crack_opening.png` | flank opening against crack length; below zero means held shut |
| `summary.json` | every headline number, for diffing between runs |

---

## Replacing the placeholders

**This is the one-line change.** Edit the defaults on `StrapGeometry` in
[`src/ratchet_fea/geometry.py`](src/ratchet_fea/geometry.py):

```python
PLACEHOLDER_GEOMETRY = StrapGeometry(
    strap_width=25.0,            # ← measured value here
    strap_thickness=3.0,
    hole_diameter=4.0,
    hole_pitch=10.0,
    observed_crack_length=2.0,   # ← from the hole WALL, not the hole centre
    ...
)
```

…and set that field's `SOURCES` entry to `Provenance.MEASURED`, so it drops out
of the verification report.

Everything downstream follows, because nothing else holds a dimension:

- **mesh density** is relative — elements around a hole's circumference,
  elements along the crack, tip size as a fraction of crack length — so a
  measured hole gets the same resolution the placeholder did;
- **the crack** is placed from the hole wall of an interior hole, and the sweep
  range comes from the ligament available in that direction;
- the **modelled window length** and **end margin** derive from the strap width
  (one width each end, the St Venant distance);
- **load normalisation** derives from the gross section area;
- the **J-integral contour radii** derive from the distance from the tip to the
  nearest other free surface.

For a one-off without editing source, both scripts take the key dimensions on
the command line (`--width`, `--thickness`, `--hole-diameter`, `--hole-pitch`,
`--tension`, `--observed-crack`).

`tests/test_geometry.py::TestReparameterisation` and
`tests/test_mesh.py::TestReparameterisation` exist to catch a dimension that
has been frozen somewhere internally.

---

## How the model works

### `analytical.py` — handbook solutions

Newman's collocation fits (NASA TN D-6376, 1971) for a radial crack from a
circular hole in an infinite plate under remote tension, with Feddersen's
secant correction for finite width. Both fits are **exact at both asymptotes**,
which is what makes them trustworthy enough to check an FE model against:

| | a → 0 | a → ∞ |
|---|---|---|
| one crack | 3.39 | **1/√2** — hole + crack behaves as a through crack of length 2r + a |
| two symmetric cracks | 3.36 | **1** — behaves as a central crack of half-length r + a |

The short-crack limit is the same for both and has a clean reading: an edge
crack (1.1215) sitting in the hole's 3σ hoop stress — 1.1215 × 3 = 3.3645.

Also here: `lefm_validity()`, which reports whether a K-based argument is
admissible at all (see [below](#is-lefm-even-applicable)).

### `mesh.py` — seeding a real crack

A crack is a surface, not a thin slot, so it cannot be cut out of the geometry.
It is made in three steps:

1. the crack line is **imprinted** onto the plate with an OCC fragment, so mesh
   edges lie exactly along it;
2. the mesh is graded down toward the tip;
3. the nodes along the crack line are **duplicated**, and the elements on one
   side remapped onto the duplicates — everywhere except the tip, which stays
   shared. That single connected node is what makes it a crack tip rather than
   a cut all the way through, and it is checked explicitly.

The flanks then become ordinary traction-free boundaries with no special
treatment, which is exactly what a crack is.

### `mechanics.py` — static plane stress

One load case, no eigenstrain, no time dependence. P2 (quadratic) vector
elements, so strain varies linearly within each element — that is what lets a
crack-tip field converge on a mesh of this density.

**The steel band is deliberately not in the model.** Its existence is inferred
from photographs, not observed, and ignoring it is conservative: a bonded band
would carry most of the tension and shield the polymer.
`mechanics.band_load_sharing()` reports how much it would take (≈98 %), so the
size of that conservatism is visible rather than hidden.

### `fracture.py` — two independent K extractions

| | uses | self-check |
|---|---|---|
| **J-integral**, domain form | the energy field | J must not depend on which annulus it is evaluated over — measured on every run |
| **Displacement extrapolation** | crack-flank opening | none, but it uses completely different information from J |

The domain form is written with the crack-direction vector explicit rather than
in crack-local axes, so both orientations go through the same code with no
rotation step to get wrong:

```
J = ∫_A [ σ_ij ∂u_j/∂x_k d_k − W d_i ] ∂q/∂x_i dA
```

For plane stress the flank opening is `COD(r) = (8 K_I / E) √(r/2π)` —
independent of Poisson's ratio — so `K_I(r)` is constant near the tip and
extrapolates cleanly to r = 0.

**Closure is measured, not assumed.** A linear elastic model has no contact and
will happily let crack faces interpenetrate, reporting a confident and
meaningless K. `k_from_opening()` checks the sign of the opening and refuses to
report a mode I K when the faces overlap. Reversing the load sign flips the
verdict, which is how the tests confirm it is measuring rather than pattern-
matching on orientation.

---

## Cross-checks and verification

Every FE run prints its own numerical quality, and at the production mesh:

| check | result |
|---|---|
| J-integral domain independence | **0.02 %** across five annuli |
| J-based K vs displacement-extrapolated K | **< 2 %** apart |
| FE vs handbook, **isolated** hole | **1 – 6 %** |
| FE vs handbook, **row of 5 holes** | FE is **13 – 16 % lower** |
| mesh convergence | **0.03 %** between refinement levels |

That last row is a result, not an error: the neighbouring holes shield the
cracked one. The same shielding shows up independently in the uncracked stress
concentration factor — FE Kt 2.23 against the isolated-hole 2.59, 13.8 % lower
— which is a satisfying consistency check between two different quantities.

The 298-test suite additionally pins:

- both Newman fits at **both asymptotes**, including the 1.1215 × 3 reading of
  the short-crack limit;
- `COD ∝ √r` near the tip — the LEFM signature. If that does not hold, K is not
  the right parameter and the extraction means nothing;
- J scaling as load² and K as load;
- **patch tests** — uniform tension on a plain strip reproduces `F/(W t)` to
  1e-8, stress is independent of modulus in a statically determinate patch;
- the `√1000` toughness unit conversion;
- that a large enough load *does* make the crack propagate, so the "does not"
  verdict means something.

```bash
pytest -q                            # all 298, ~25 s
pytest -q -m "not requires_gmsh"     # on a machine without gmsh
```

### Is LEFM even applicable?

Reported automatically against whatever dimensions are supplied:

- **Small-scale yielding holds.** The plane-stress plastic zone is 0.03 mm
  against a 2 mm crack and an 8.5 mm ligament, so K does characterise the tip.
- **The section is far too thin for plane strain.** ASTM E399 wants
  2.5(K_IC/σ_y)² ≈ 20 mm; the strap is 3 mm. The part is in **plane stress**,
  where the effective toughness is higher than the plane-strain K_IC, often by
  a factor of two or more.

That second point cuts one way only, and the code says so: comparing against
K_IC is **conservative**, so a "does not propagate" verdict is *strengthened* by
it and a "propagates" verdict would not be.

---

## Assumptions that still need a physical part

Both entry points print the full table. In priority order:

| # | Assumption | Why it matters | How to settle it |
|---|---|---|---|
| 1 | **Which way the crack runs** | Decides whether this is a mode I problem at all. Transverse and longitudinal give completely different answers | Look at the failed part under magnification and note the direction relative to the strap axis |
| 2 | **Observed crack length 2.0 mm** | The K(a) curve is read at this value | Measure from the hole wall to the crack tip |
| 3 | **PA66 grade and its K_IC** | The toughness bracket spans 3.0–5.5 MPa·√m, and datasheets rarely quote it for polyamides | ASTM D5045 SENB or compact tension on razor-notched specimens, conditioned to the service state |
| 4 | **Service tension 500 N** | K is linear in it, so the margin scales directly | Load cell in line with the strap during normal tightening |
| 5 | **Strap thickness 3.0 mm** | Sets the section area and therefore the far-field stress | Micrometer on an unperforated section |
| 6 | **Hole diameter and pitch** | Set the stress concentration and the hole-to-hole shielding | Pin gauge; calipers centre-to-centre over 10 holes |
| 7 | **Steel band exists, 18 × 0.8 mm** | Ignoring it is conservative, but if it is real the margins are ~56× larger still | Section and etch, X-ray, or a magnet on a scrap length |

Grep for `VERIFY:` across `src/` to find every assumption flagged in place.

### Modelling assumptions (not measurements)

- **Linear elasticity, no plasticity.** Justified here by the small-scale
  yielding check, but it is a check that could fail at a higher load.
- **Plane stress.** Right for a 3 mm section, and it means the K_IC comparison
  is conservative.
- **The crack is straight, sharp and traction-free**, and grows in its own
  plane. A real crack in a tough polymer blunts and may turn.
- **A single crack at one hole.** No interaction with cracks at neighbouring
  holes, which would raise K.
- **Monotonic, room-temperature, short-term properties.**

---

## What this model cannot do

In the order they would change the answer:

1. **Say why the crack keeps growing.** Static LEFM rules overload out; it says
   nothing about what does. Fatigue crack growth under repeated ratcheting is
   the obvious next candidate and needs `da/dN` data for the grade.
2. **Handle a crack driven by anything other than axial tension** — pawl
   bearing load on the hole edge, bending around the buckle, residual stress.
   Any of these could open a longitudinal crack that tension cannot.
3. **Represent the steel band's effect on a crack.** A polymer crack bridged by
   an intact band is a layered-cracking problem, not a plane-stress one.
4. **Crack turning, branching or blunting.**
5. **Large-scale yielding**, if a higher load or a tougher/weaker grade pushes
   the plastic zone out of the small-scale regime — then J or the essential
   work of fracture is needed, not K.

---

## Repository layout

```
src/ratchet_fea/
  provenance.py    where every number came from, and what would replace it
  geometry.py      StrapGeometry + CrackGeometry (PLACEHOLDER values)
  materials.py     PA66 and steel property brackets, including K_IC
  analytical.py    Kt, Newman crack-from-hole K, LEFM validity     ┐ handbook
  mesh.py          perforated strip with a seeded crack            ┐
  mechanics.py     static plane-stress elastic solve               │ FE
  fracture.py      J-integral, K extraction, crack closure         │
  postprocess.py   K(a) curve, K_IC comparison, plots              ┘
scripts/
  run_handbook_screen.py
  run_lefm_sweep.py
tests/             one module per source module, 298 tests
```

The FE modules are imported lazily from `ratchet_fea/__init__.py`, so the
handbook screen works without scikit-fem, gmsh or matplotlib installed.

---

## Development

```bash
pytest -q                            # everything (298 tests, ~25 s)
pytest -q -m "not requires_gmsh"     # skipping anything needing gmsh
pytest tests/test_fracture.py -q     # one module
```

Conventions worth keeping:

- **No number is hardcoded in the FE code.** Every geometric and material value
  reaches `mesh`/`mechanics`/`fracture` through `StrapGeometry`,
  `CrackGeometry` or a `materials` bracket. New inputs go in those modules with
  a `Source` recording their provenance, or `Documented.source_for` will raise.
- **Properties are brackets, not values**, and conclusions are checked at both
  ends. Anything that flips between `low` and `high` is a measurement request,
  not a result. Note that the brackets do not share a worst case: **dry PA66 is
  worst for fracture, wet is worst for yield.**
- **Flag assumptions in place** with a `VERIFY:` comment as well as in
  `SOURCES`.
- **Every new physics path gets an independent check** — a closed form, an
  asymptote, or a patch test — not just a regression value.

### History

This repository previously held a coupled moisture-diffusion / swelling study
of the same part. That was a different hypothesis and has been removed rather
than left half-finished alongside this one; `geometry.py` and `materials.py`
survive from it, with the moisture transport properties stripped out. See the
git history if that line of investigation is ever picked back up.
