# STP-RB-001 — PA66 ratchet belt strap, cracking in normal use

A tiered fracture-mechanics investigation into why a perforated, steel-reinforced
PA66 ratchet belt strap cracks during ordinary service.

> ### ⚠️ Read this before quoting any number
>
> **Every dimension in this repository is a visual estimate scaled off
> photographs. Every material property is a published range for generic
> unfilled PA66, not a datasheet for the grade in the part.** Nobody has yet
> put calipers, a micrometer or a DSC on a physical sample.
>
> The code is built so that fixing this is a one-line change — see
> [Replacing the placeholders](#replacing-the-placeholders). Until then, both
> entry points print a table of every unverified input and what it would take
> to measure it, and all results are screening-level only.

---

## Contents

- [What the investigation has found so far](#what-the-investigation-has-found-so-far)
- [Tier structure](#tier-structure)
- [Units](#units)
- [Installation](#installation)
- [Running it](#running-it)
- [Replacing the placeholders](#replacing-the-placeholders)
- [How the Tier 2 model works](#how-the-tier-2-model-works)
- [Cross-checks and verification](#cross-checks-and-verification)
- [Assumptions that still need a physical part](#assumptions-that-still-need-a-physical-part)
- [What this model cannot do — Tier 3 scope](#what-this-model-cannot-do--tier-3-scope)
- [Repository layout](#repository-layout)
- [Development](#development)

---

## What the investigation has found so far

All numbers below come from the **placeholder** geometry and the **nominal**
corner of the literature property brackets. They are the shape of an argument,
not a result.

### Tier 1 — the mechanical load is not the problem

| | value |
|---|---|
| Stress concentration at a hole, `Kt` (net section, isolated hole) | 2.59 |
| Net-section stress at 500 N service tension | 7.9 MPa |
| Peak mechanical stress at the hole edge | 20.5 MPa |
| PA66 yield strength, 50 % RH conditioned | 40–55 MPa |
| PA66 yield strength, water saturated | 30–45 MPa |

The mechanical service load, even with the hole's concentration applied and the
steel band's load-sharing ignored, stays below yield at **every** corner of the
material bracket. Constrained moisture swelling does not: at saturation it
reaches 17–54 MPa, which spans the conditioned yield strength.

**That is why Tier 2 exists.** Tier 1 cannot resolve two things: the holes are a
*row*, not one isolated hole, and the swelling field is *transient and
non-uniform*.

### Tier 2 — the row shields itself, and the sign matters

| quantity | Tier 1 | Tier 2 (FE) | reading |
|---|---|---|---|
| `Kt` at an interior hole | 2.587 | **2.229** | holes in a line along the load shield each other; Tier 1 is 14 % conservative |
| `Kt` at an end hole | — | **2.417** | the first and last hole of the row genuinely carry more |
| Constrained swelling stress, reinforced bulk | 31.2 MPa | **30.5 MPa** | agrees to 2 % — the formula and the FE solve the same problem here |
| Time to half moisture uptake | 8.7 d | **8.4 d** | in-plane ingress adds a little to the through-thickness path |

**The sign of the swelling stress is the crux, and it is easy to get wrong.**
Constrained *swelling* during absorption is **compressive** — and compression
does not open a crack. At nominal properties the hole edge sits at about
−31 MPa compressive, 0.81 of the moisture-softened yield strength, and the
tensile stress from the service load is swamped by it.

So on the assumptions as they currently stand, this model does **not** show a
tensile crack driver. What it shows instead is three candidate routes to one,
in descending order of how much evidence would be needed to settle them:

1. **The stress-free reference.** Every stress is measured from the moisture
   content at which the polymer is taken to be stress-free. That is currently
   assumed to be dry-as-moulded, and it is the single largest lever in the
   whole model:

   | stress-free moisture | absorption: peak hole-edge tension | desorption: peak hole-edge tension |
   |---|---|---|
   | 0.000 (assembled dry — current default) | +1.4 MPa | −12.6 MPa (compression throughout) |
   | 0.025 (stress-free at 50 % RH) | **+29.7 MPa** | +1.3 MPa |
   | 0.085 (stress-free saturated) | **+100 MPa** (beyond strength — unphysical, shown to bound the lever) |

   If the strap is stress-free anywhere near its in-service moisture, then
   **drying below that** — a hot car, a dry winter, a wash-and-dry cycle — puts
   tens of MPa of tension exactly where the mechanical stress concentration
   already is. Reproduce with `--stress-free-moisture 0.025`.

2. **Compressive yielding and ratcheting.** At the high bracket corner the
   hole-edge von Mises stress reaches 1.42 × yield during absorption (0.42 × at
   the low corner — the conclusion flips across the literature range). Yielding
   in compression locks in plastic strain that reappears as tension on drying.
   Modelling that needs plasticity and a cyclic moisture history — Tier 3.

3. **Through-thickness gradients.** This is a membrane model fed a
   thickness-averaged concentration, so it is blind to the drying skin pulled
   into tension over a still-wet core — the classic moisture-driven surface
   cracking mechanism in polyamides. A low tensile stress here is **not**
   evidence that moisture is harmless.

### One more Tier 2 result worth knowing

The steel band *suppresses* differential-swelling tension. A wetted ring around
a hole produces ~11.5 MPa of tension in the surrounding dry polymer with no
band, but only ~0.3 MPa with it: the band is ~25× stiffer per unit width, so it
reacts the mismatch itself. The band makes the *uniform* constrained-swelling
compression worse and the *gradient* tension better.

---

## Tier structure

| Tier | Question | Cost | Status |
|---|---|---|---|
| **1** | Is any mechanism close enough to strength to be worth modelling? | seconds, numpy only | built, tested |
| **2** | Do the holes interact, and what does the transient moisture field actually do? | ~20 s per case | built, tested |
| **3** | Plasticity, cyclic moisture, through-thickness gradients, fatigue | not started | [scoped below](#what-this-model-cannot-do--tier-3-scope) |

Tier 1 is deliberately independent of Tier 2 — it needs no FE stack — and Tier 2
is [cross-checked against it](#cross-checks-and-verification) on every run.

---

## Units

One consistent system throughout. There is no unit conversion anywhere in the
code, because there is nothing to convert.

| quantity | unit |
|---|---|
| length | mm |
| force | N |
| stress, modulus | MPa (N/mm²) |
| time | s |
| diffusivity | mm²/s |
| moisture content | dimensionless mass fraction (kg water / kg dry polymer) |
| coefficient of moisture expansion | linear strain per unit mass fraction |

Diffusivity is the trap: literature quotes water in PA66 as 10⁻¹³–10⁻¹² m²/s,
which is **10⁻⁷–10⁻⁶ mm²/s**. A slip here moves every predicted time by 10⁶.

---

## Installation

Tier 1 needs only numpy and scipy. Tier 2 additionally needs an FE stack.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[tier2,dev]"      # or: pip install -r requirements.txt
```

### gmsh needs system libraries even when headless

The `gmsh` wheel links against X11/GL shared objects and raises `OSError` at
import time without them. On Debian/Ubuntu:

```bash
sudo apt-get install -y libglu1-mesa libxft2 libxinerama1 \
                        libxcursor1 libxrender1 libxfixes3 libfontconfig1
```

`mesh.build_strip_mesh` catches this and says exactly which packages are
missing. The test suite detects it and skips every gmsh-dependent test
automatically (`-m "not requires_gmsh"` does the same on demand), so the rest
of the suite still runs on a machine without them.

---

## Running it

### Tier 1 — closed-form screening

```bash
python scripts/run_tier1_analytical.py                 # all bracket corners
python scripts/run_tier1_analytical.py --corner high
python scripts/run_tier1_analytical.py --tension 800 --hole-diameter 4.6
```

### Tier 2 — coupled FE

```bash
python scripts/run_tier2_fe.py                          # absorption, nominal
python scripts/run_tier2_fe.py --quick                  # ~9 s pipeline check
python scripts/run_tier2_fe.py --case both --corner high
python scripts/run_tier2_fe.py --stress-free-moisture 0.025
python scripts/run_tier2_fe.py --mesh-convergence       # Kt convergence only
```

Outputs land in `results/<case>/`:

| file | what it shows |
|---|---|
| `hole_stress_vs_time.png` | peak principal stress at every hole edge against time (log) |
| `stress_vs_uptake.png` | hole-edge stress and yield utilisation against moisture uptake |
| `tier1_cross_check.png` | the three Tier 1 ↔ Tier 2 comparisons |
| `concentration_field.png` | thickness-averaged moisture at peak hole-edge von Mises |
| `stress_field_max_principal.png`, `stress_field_von_mises.png` | equilibrium PA66 stress |
| `summary.json` | every headline number, for diffing between runs |

Useful flags for isolating one effect at a time: `--no-band`, `--no-tension`,
`--fixed-modulus`.

---

## Replacing the placeholders

**This is the one-line change.** Edit the defaults on `StrapGeometry` in
[`src/ratchet_fea/geometry.py`](src/ratchet_fea/geometry.py):

```python
PLACEHOLDER_GEOMETRY = StrapGeometry(
    strap_width=25.0,          # ← measured value here
    strap_thickness=3.0,
    hole_diameter=4.0,
    hole_pitch=10.0,
    ...
)
```

…and update the `SOURCES` entry for each one to
`Provenance.MEASURED`, so it drops out of the verification report.

Everything downstream follows automatically, because nothing else holds a
dimension:

- **mesh density** is expressed relative to the hole circumference and the
  ligament widths, so a measured hole gets the same number of elements round it
  as the placeholder did;
- the **modelled window length** and the **end margin** derive from the strap
  width (one width each end, the St Venant distance — verified empirically:
  below that the end holes' stress is still moving with the truncation planes);
- **diffusion time scales** derive from the strap and band thicknesses;
- **load normalisation** derives from the net section area;
- **post-processing clearances** derive from the strap thickness.

For a quick one-off without editing source, both scripts take the key
dimensions on the command line (`--width`, `--thickness`, `--hole-diameter`,
`--hole-pitch`, `--band-width`, `--band-thickness`, `--tension`). They print a
reminder to move the values into `geometry.py`.

`tests/test_geometry.py::TestReparameterisation` and
`tests/test_mesh.py::TestReparameterisation` exist specifically to catch a
dimension that has been frozen somewhere internally.

---

## How the Tier 2 model works

### `mesh.py` — the perforated strip

Builds the strap window in gmsh's OCC kernel via pygmsh: a rectangle, fragmented
along the steel band's footprint so elements conform to the stiffness jump, with
the perforations subtracted. Boundaries and subdomains are tagged by **position**
(entity centre of mass and bounding box) rather than by entity number, which
survives gmsh renumbering entities after a boolean operation.

Named boundaries: `cut_start`, `cut_end` (model truncation planes — *not*
physical surfaces), `side_lower`, `side_upper` (real free edges), and
`hole_0 … hole_{n-1}`, each tagged separately so post-processing can report a
peak stress per hole.

Generated meshes are validated before being returned: all expected boundaries
present, one tag per hole, meshed area matching rectangle-minus-holes to 2 %,
and no element straddling the band edge. A silently mis-tagged boundary would
give a plausible-looking but wrong stress field, which is the worst possible
outcome for an investigation.

> **Note:** `skfem.io.meshio` returns *subdomain* indices offset by the
> preceding line-cell block, so `mesh.subdomains[...]` does not index elements
> directly. Boundary (facet) tags are unaffected. `band_element_mask()`
> therefore classifies geometrically from `StrapGeometry` — exact, because the
> band footprint is a rectangle and the mesh conforms to its edge.

### `diffusion.py` — transient Fickian moisture

Solves `∂c/∂t = D ∇²c` on the mesh by backward Euler on a log-spaced time grid
(diffusion transients are self-similar in √t; uniform steps would spend almost
all their effort on the flat tail). The mass matrix is row-sum lumped, which
restores the discrete maximum principle and keeps the solution bounded and
monotone despite the discontinuous initial condition.

**The through-thickness correction is the important part.** A plane-stress model
lives in the mid-plane, so a 2D solve only sees moisture entering through the
side edges and hole walls — but the strap's two big faces are exposed too, and
much closer to the interior (≈1.1–1.5 mm against ≈3 mm in-plane). In-plane-only
would predict the strap wetting through about **seven times too slowly**.

This is handled exactly, not fudged. For a prism with uniform initial
concentration and the same fixed concentration on every surface, the Fickian
solution separates (Crank, *The Mathematics of Diffusion*, §2.5.4):

```
(c_s − c) / (c_s − c₀)  =  u_2D(x, y, t) · u_1D(z, t)
```

so the code solves the in-plane problem by FE, evaluates the through-thickness
plane-sheet solution analytically (short-time √t series below a Fourier number
of 0.25, long-time exponential series above — both to better than 1e-6 with 20
terms), and multiplies. Averaging over the thickness leaves exactly what a
plane-stress model needs.

The band is treated as impermeable, so the polymer inside its footprint is a
skin exposed on one face and sealed on the other, which behaves as half a slab
of twice its depth.

### `mechanics.py` — plane stress with a swelling eigenstrain

The strap is a through-thickness laminate, so rather than model it in 3D the
section is condensed to a membrane using the **A-matrix of classical laminate
theory**. Every layer shares the in-plane strain, so

```
A    = Σ t_k Q_k          (section stiffness)
N_sw = Σ t_k Q_k ε_sw,k   (section force from the eigenstrain)
```

with `ε_sw = 0` in the steel, because steel neither absorbs water nor swells.
That single fact *is* the mechanism.

**Ply-level recovery is what matters.** The smeared membrane stress `N/t`
averages polymer and steel together and cannot be compared to a PA66 strength.
After solving for the membrane strain the code recovers the polymer's own
stress, `σ_PA66 = Q_PA66 : (ε − ε_sw)`. In the rigid-constraint limit (`ε → 0`)
this reduces exactly to Tier 1's `E β Δc / (1 − ν)` — which is
[cross-checked on every run](#cross-checks-and-verification) and pinned by a
unit test.

Displacement uses P2 (quadratic) vector elements, so strain varies linearly
within each element; that is what lets the hole-edge stress converge on a mesh
of this density. Hole-edge stresses are sampled on the facets themselves rather
than read from a smoothed nodal field — nodal averaging across a hole edge pulls
the value toward the interior and systematically under-reports `Kt`.

The polymer modulus is evaluated pointwise from the local concentration, so the
water that drives the swelling also softens the polymer resisting it. Holding it
fixed overstates the swelling stress by 2–3× (`--fixed-modulus` to see).

### `postprocess.py` — extraction, plots, cross-check

Loops the mechanics solve over the stored concentration fields, collects the
per-hole stress envelope, compares against Tier 1, and reports whether the
membrane model is even usable at the hole edges (see below).

---

## Cross-checks and verification

Two independent routes to the same number is the only cheap defence against a
silently wrong FE model. Every Tier 2 run prints:

| comparison | expected relationship |
|---|---|
| `Kt` at a hole | FE slightly **below** the Howland/Peterson isolated-hole value — that difference *is* the row-interaction result |
| constrained swelling stress in the reinforced bulk | FE **≈ equal** to `E β Δc / (1 − ν)`; both solve the same problem there |
| time to half uptake | FE slightly **faster** than the plane-sheet value, because in-plane ingress adds to it |

Anything outside a factor of 2 is flagged as a disagreement. The comparison uses
the same bracket corner as the FE run, because the diffusivity bracket spans a
decade.

Beyond that, the test suite (**325 tests**) verifies:

- the FE `Kt` against the **Howland/Peterson closed form** for a single hole in
  a finite-width strip (agrees to 6 %, converged to 0.4 %);
- the FE diffusion solve against the **analytic plane-sheet series** for a
  hole-free strip (agrees to 2 %);
- the short-time and long-time moisture series against **each other** across the
  branch switch (an easy place to lose a √π);
- **patch tests** — uniform tension on a plain strip reproduces `F/(W t)` to
  1e-8; unconstrained swelling produces exactly zero stress; tension and
  swelling superpose;
- the rigid-constraint limit reproducing **Tier 1's formula** exactly;
- that derived geometry actually follows its inputs, so a measured strap
  re-runs the study unchanged.

```bash
pytest -q                            # all 325, ~8 s
pytest -q -m "not requires_gmsh"     # 244 of them, on a machine without gmsh
```

### A limitation the results forced into the open

Classical laminate theory enforces the *resultant* traction on a free edge, not
the traction on each layer, so at a hole wall it lets the polymer and steel
plies carry equal and opposite self-equilibrating stress. Reality relaxes that
mismatch to zero through interlaminar shear over a boundary layer roughly one
laminate thickness wide.

For the placeholder dimensions that boundary layer is ~3 mm against a 6 mm
inter-hole ligament — a ratio of 0.5, which the code reports as **marginal**.
The *swelling* component of the hole-edge stress should therefore be read as an
**upper bound**. The mechanical stress concentration and the reinforced bulk
stress are unaffected.

`postprocess.free_edge_validity()` recomputes this against whatever dimensions
are supplied, so the caveat updates itself when real measurements arrive.

---

## Assumptions that still need a physical part

Both entry points print the full table. In priority order, the ones that would
change a conclusion:

| # | Assumption | Why it matters | How to settle it |
|---|---|---|---|
| 1 | **Stress-free moisture content = 0** (`mechanics.py`) | Largest single lever in the model. Decides whether the hole edges ever see tension at all. | Anneal a sample and measure the dimensional change; or hole-drilling residual strain |
| 2 | **The holes pierce the steel band** (`geometry.py`) | If the band is instead split or interrupted around the holes, the longitudinal constraint at the hole is released and the mechanism weakens qualitatively | Section through a hole; look for steel at the hole wall |
| 3 | **A steel band exists, 18 × 0.8 mm** | Its existence is *inferred*, not observed. No band means no constrained swelling at all | Section and etch, X-ray, or a magnet on a scrap length |
| 4 | **PA66 grade and its properties** | Yield strength brackets span 30–55 MPa; the yielding conclusion flips across them. Glass fill would change everything | Moulder's part record; FTIR + DSC + ash test |
| 5 | **Diffusivity 1e-7…1e-6 mm²/s** | Spans a decade, so all *timing* is order-of-magnitude only | Gravimetric sorption on a coupon of known thickness; fit √t |
| 6 | **Swelling coefficient 0.20–0.30** | Multiplies straight through to the swelling stress | Measure a coupon dry and conditioned; strain ÷ mass uptake |
| 7 | **Strap thickness 3.0 mm** | Time-to-saturation goes as thickness², so a 20 % error is a 44 % error in timing | Micrometer on an unperforated section |
| 8 | **Service tension 500 N** | Sets the mechanical half of the load case (which is *not* currently governing) | Load cell in line during normal tightening |

Lower-priority items — hole diameter and pitch, band offset, Poisson's ratios —
are in the printed table with their own verification routes.

Grep for `VERIFY:` across `src/` to find every assumption flagged in place.

### Modelling assumptions (not measurements)

- **Fick's law with constant D.** Real PA66 sorption is often slightly
  non-Fickian and `D` rises with temperature and concentration. The decade-wide
  bracket dwarfs that error, but it means timing is order-of-magnitude only.
- **Linear elasticity.** No plasticity, no viscoelasticity, no creep — all three
  are real in PA66 at these stresses and timescales.
- **The band is perfectly bonded and impermeable.** A debonded interface would
  wick moisture along the band far faster than this model allows, which is a
  qualitatively different and worse case.
- **Monotonic room-temperature strengths.** The strap is failing under
  *repeated* use, so the governing limit is more likely fatigue or creep rupture
  well below yield.
- **Piecewise-linear modulus vs moisture** through three conditioned states. The
  real curve is sigmoidal, steepest as the wet Tg crosses room temperature near
  2–3 % uptake.

---

## What this model cannot do — Tier 3 scope

In the order they would change the answer:

1. **Through-thickness moisture gradients.** A membrane model fed a
   thickness-averaged concentration is blind to a drying skin in tension over a
   wet core — the classic polyamide surface-cracking mechanism. Needs a 3D or
   layered through-thickness model.
2. **Plasticity and moisture ratcheting.** Compressive yielding during wetting
   locks in plastic strain that reappears as tension on drying. Needs an
   elastic-plastic model and a cyclic wet/dry history.
3. **The laminate free-edge boundary layer** at the hole walls (see above) —
   needs interlaminar shear, so 3D.
4. **Fatigue and creep rupture.** The part fails in repeated service, not under
   a monotonic pull. Needs cyclic data at the right moisture state.
5. **Fracture mechanics proper.** Once a crack exists, `K`/`J` at the hole edge
   and a growth law — which is where the "fracture mechanics" in the title
   eventually has to land.

---

## Repository layout

```
src/ratchet_fea/
  provenance.py    where every number came from, and what would replace it
  geometry.py      StrapGeometry — all dimensions (PLACEHOLDER)     ┐ Tier 1
  materials.py     PA66 + steel property brackets (LITERATURE)      │
  analytical.py    Kt, constrained swelling, Fickian timing         ┘
  mesh.py          perforated strip mesh via pygmsh/gmsh            ┐
  diffusion.py     transient Fickian field + product solution       │ Tier 2
  mechanics.py     plane stress + swelling eigenstrain (CLT)        │
  postprocess.py   extraction, plots, Tier 1 cross-check            ┘
scripts/
  run_tier1_analytical.py
  run_tier2_fe.py
tests/             one module per source module, 325 tests
```

Tier 2 modules are imported lazily from `ratchet_fea/__init__.py`, so Tier 1
works without scikit-fem, gmsh or matplotlib installed.

---

## Development

```bash
pytest -q                            # everything (325 tests, ~8 s)
pytest -q -m "not requires_gmsh"     # 244, skipping anything needing gmsh
pytest tests/test_mechanics.py -q    # one module
```

The `requires_gmsh` marker gates mesh generation only; the patch tests and the
analytic diffusion verification still need scikit-fem, since they run on
hand-built meshes.

Conventions worth keeping:

- **No number is hardcoded in the FE code.** Every geometric and material value
  reaches `mesh`/`diffusion`/`mechanics` through `StrapGeometry` or a
  `materials` bracket. New inputs go in those two modules with a `Source`
  recording their provenance, or `Documented.source_for` will raise.
- **Properties are brackets, not values,** and conclusions are checked at both
  ends. Anything that flips between `low` and `high` is a measurement request,
  not a result.
- **Flag assumptions in place** with a `VERIFY:` comment as well as in the
  `SOURCES` table.
- **Every new physics path gets an independent check** — a closed form, an
  analytic series, or a patch test — not just a regression value.

### Provenance history

Tier 1 (`geometry.py`, `materials.py`, `analytical.py`,
`run_tier1_analytical.py`) was **reconstructed** to the described interface when
Tier 2 was built, because the repository was empty at that point. It reproduces
the Tier 1 conclusions it was specified to have — mechanical load benign,
constrained swelling reaching yield — but if an earlier Tier 1 exists elsewhere,
diff the placeholder dimensions and the property brackets against it before
trusting the absolute numbers.
