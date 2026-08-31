# Vevo AI Color Matching — UFO White PE

Predicts an ink **recipe** (which colorants, and at what percentage) from a
**measured reflectance spectrum**, for the UFO White PE translucent ink
system. Given a spectrophotometer reading of a film, the model outputs a
colorant list with percentages that should reproduce that color.

The production path takes a reading over **both** a light and a dark backing.
Single-backing models exist alongside it for callers who only have one —
see [Backing modes](#backing-modes-light-dark-or-both).

There is also a **browser app** (`python app.py`) that wraps all of this: paste
or upload a spectrum, get a recipe, and see how far the sample sits from a named
Pantone standard. See [Usage](#usage).

### Quick map

| If you want to… | Read |
|---|---|
| Understand the model | [Pipeline](#pipeline), [Results](#results) |
| Know how good it really is | [How accuracy is measured](#how-accuracy-is-measured) |
| Run it | [Usage](#usage) |
| Feed it one backing instead of two | [Backing modes](#backing-modes-light-dark-or-both) |
| Go the other direction (colour → recipe) | [The Pantone linkage](#the-pantone-linkage) |
| Know when a result needs a human | [Formulation rules](#formulation-rules-flagged-not-enforced) |
| Know what to distrust | [Known limitations](#known-limitations--next-steps) |
| Know what was already tried | [What didn't work](#what-didnt-work-and-why) |

## Data

Two X-Rite CxF3 colour-exchange files, which turned out to be a **paired**
dataset rather than two unrelated colour sets — see
[The Pantone linkage](#the-pantone-linkage):

| File | Contents | CxF `ObjectType` |
|---|---|---|
| `NPXXXXCPWNUF UFO White PE.cxf` | 2372 UFO formulas actually mixed and measured, with recipes | `Trial` |
| `Pantone_Coated_V_4.cxf` | 2135 Pantone Coated standards — the colours those formulas were *aiming* at | `Target` |

Cross-referenced against `All recipes and components UFO White PE.xlsx`.

Each formula in the file was measured twice — once over a light backing and
once over a dark backing — because the film is **translucent**, not opaque.
That pairing is what makes it possible to fit real Kubelka-Munk optics
later, though this project currently ships without that layer (see
[What got removed](#what-got-removed-and-why)).

- 2372 total formula entries parsed from the CxF file
- 1988 have **both** light- and dark-backing measurements (`has_light &
  has_dark`) — everything else is dropped by `train.py`, since the production
  pipeline's features need both. The 384 light-only formulas are not waste:
  the light-only backing mode trains on them too, which is worth about a
  point of accuracy (see [Backing modes](#backing-modes-light-dark-or-both))
- **Zero formulas are dark-only.** A dark-only model is trainable from the dark
  half of paired formulas, but no dark-only submission has ever actually
  occurred in this dataset — worth confirming with the client that the mode is
  a real requirement rather than a symmetry assumption
- 36-point reflectance spectrum per measurement, 380–730nm in 10nm steps
- 14 possible colorants (fixed list, see `src/cxf_parser.py:COLORANT_NAMES`)
- Recipes are fractions of the 14 colorants, ~summing to 1.0 -- **17 rows
  have corrupted/partial recipe data (fractions summing to ~0.1-0.14, not
  ~1.0) and are dropped from both training and evaluation.** This used to
  only be done for the (now-removed) physics fit; it wasn't applied to
  Stage 1/2 until one such row was found to be single-handedly dragging
  purple's Stage 2 R² from 0.83 down to -2.31 (purple only has 8-9 active
  test examples, so one corrupted label dominated the whole statistic) --
  see [What didn't work](#what-didnt-work-and-why) for the full story.
- Split: **first** 85% train / 15% test (`random_state=42`) on the full
  data, **then** the corrupted rows are dropped from each side separately
  -- filtering before the split would silently reshuffle sklearn's whole
  train/test partition rather than just removing a few rows, risking
  leakage of formulas the model actually trained on into what would look
  like a "clean" test set. Final counts: 1674 train / 297 test.

There is **no masstone (single-colorant) measurement** in this dataset —
every colorant's spectral signature has to be inferred from mixtures, never
measured in isolation. There is also no film-thickness or applicator-gap
metadata anywhere in the CxF file (`PhysicalAttributes` is empty on every
record). That second gap turns out to matter more than it looks — see
[transparent white is underdetermined](#known-limitations--next-steps).

## The Pantone linkage

*Found 2026-08-17. `src/pantone.py`.*

The two CxF files are the same dataset seen from both ends. **Every UFO formula
name encodes the Pantone colour it was formulated to match:**

```
NP0325CPWNUF   <->   PANTONE 325 C
NP7469CPWNUF   <->   PANTONE 7469 C

NP + <Pantone code, zero-padded to 4> + PWNUF
```

The source filename's own placeholder — `NPXXXXCPWNUF UFO White PE.cxf` — *is*
that code slot. It had been sitting in the filename the whole time.

**Confirmed colorimetrically, not just by string pattern.** Across 1817 formulas
having both a target and both-backing measurements:

| | Mean ΔE00 |
|---|---|
| Name-matched pairs | **0.75** |
| Same data, pairing shuffled | 39.5 |

A 53× separation. If the naming convention were coincidence, those two numbers
would be the same. All 45 apparently-uncovered Pantone entries are the base
colorants themselves plus the grey series — not misses.

### Why it matters

The dataset stopped being *"spectra with recipe labels"* and became
**target → attempt → recipe** triples. Three things follow:

1. **A ground truth for colour, which this project previously had none of.**
   Recipe accuracy was always measurable; whether a recipe *looks* right was
   not, because the Kubelka-Munk forward model was never trustworthy enough
   (see [What got removed](#what-got-removed-and-why)). Now a measured sample
   can be compared against the standard it was aiming at directly.
2. **An off-target warning.** A recipe reproduces *the sample it was measured
   from*, drift included. If the drawdown missed its target, the prediction
   faithfully reproduces the miss. `predict.py --target "325 C"` and the app's
   target field now say so explicitly instead of leaving it implied.
3. **The inverse direction becomes ordinary supervised learning** — see below.

### Which backing does a Pantone standard resemble? (measured)

Neither, strictly — **a Pantone standard has no backing at all.** The UFO file
tags every measurement with a `PaperType` of `Over light` (2372) or `Over dark`
(1988); the Pantone file carries **zero** such tags and exactly one spectrum per
standard (2135 objects, 2135 spectra). Backing is not a meaningful property of
an opaque swatch on coated stock — nothing shows through, so what sits behind it
cannot change the reading. That asymmetry is the whole reason UFO White PE needs
two measurements and Pantone needs one.

Functionally, though, the light backing is the correct analogue, and this is
measured rather than argued (`eval_pantone_backing.py`, 1817 matched formulas):

| CIEDE2000 from the Pantone standard | mean | median | p90 |
|---|---|---|---|
| to the UFO **light** backing | **0.75** | 0.45 | 1.65 |
| to the UFO **dark** backing | 23.34 | 3.78 | 63.67 |

Light is closer on **93.2%** of formulas; RMS curve difference is 0.045 vs 0.182.

Splitting by film opacity shows the mechanism rather than just the outcome:

| Opacity quartile (mean \|R_light − R_dark\|) | ΔE00 light | ΔE00 dark | light wins |
|---|---|---|---|
| Q1 0.002–0.011 (most opaque) | 0.69 | 1.68 | 89.2% |
| Q2 0.011–0.025 | 0.86 | 2.48 | 89.0% |
| Q3 0.025–0.239 | 0.77 | 28.06 | 94.7% |
| Q4 0.239–0.734 (most transparent) | 0.68 | **61.04** | **100.0%** |

The light-backing agreement is essentially flat across all four quartiles
(0.68–0.86) — the light substrate matches Pantone's white stock regardless of
how transparent the ink is. The dark-backing figure explodes by a factor of 36
from Q1 to Q4, which is exactly the predicted behaviour: an opaque film hides
its substrate and reads similarly over either backing, while a transparent one
lets the black substrate through and lands nowhere near the standard.

So: comparisons use the UFO **light-backing** spectrum. Comparing against the
dark backing would measure the substrate, not the formulation.

**Noise floor: ΔE00 1.0.** The Pantone file mixes measurement modes (2069 M0,
66 M2 of 2135), which differ in UV content, so part of any residual difference
is instrument disagreement rather than formulation error. `NOISE_FLOOR_DE`
encodes this; sub-1.0 deltas are not reported as real misses.

### Inverse direction: colour → recipe (`src/inverse.py`, `train_inverse.py`)

The deployed pipeline runs *forward*: you already mixed something and it tells
you what is in it. The question a customer actually asks is the reverse — *here
is the Pantone colour I need, what do I mix?*

That looked blocked, and **an earlier claim in this project that it needed a
forward model plus an iterative search was wrong.** The linkage removes the
blocker outright: ~1800 direct `(target spectrum → known recipe)` pairs already
exist across the two files. No forward model, no search — ordinary supervised
regression.

Two structural differences from the forward pipeline:

- **Single backing.** A Pantone standard has one reflectance curve, not a
  light/dark pair, so this uses `build_single_spectrum_features` (142 features)
  rather than `build_feature_matrix` (285). The `contrast` group that encodes
  opacity does not exist here.
- **Leak-safe splitting.** Adjacent Pantone codes are frequently the same shade
  to the eye (2745 C vs 2746 C), so a random split would put one in train and
  its twin in test and report a flattering number. `purged_split` takes a random
  test set and then *drops* training rows within a ΔE00 buffer of any test
  colour. This replaced an earlier connected-components grouping that proved
  degenerate — at ΔE00 5.0 it collapsed 2000 of 2072 colours into a single
  group, leaving 12 usable test rows. See
  [What didn't work](#what-didnt-work-and-why).

### Can a Pantone spectrum be used as INPUT to the forward model? No.

*`eval_pantone_input.py`. Measured on 60 held-out formulas.*

A natural client question: both are 36-point reflectance curves, and a UFO light
measurement sits a mean ΔE00 0.72 from its Pantone standard — so can the
standard be fed in place of a measurement? Same model, same known recipes, only
the input swapped:

| Input | Exact-match | Per-decision | Active-cell MAE |
|---|---|---|---|
| UFO light measurement (designed input) | **85.00%** | 98.81% | 0.0126 |
| Pantone standard spectrum (substituted) | **21.67%** | 88.57% | 0.0939 |

**A 63-point collapse** — worse than dark-only mode. The two inputs are
colorimetrically near-identical (median ΔE00 0.47, well inside the ΔE00 2.0
"commercial match" band): *to the eye they are the same colour, to the model
they are not the same measurement.* Stage 1 reads spectral curve shape, and an
opaque swatch on coated stock is a different physical object from a translucent
film over a light substrate. This is the same effect that made
[Pantone training augmentation](#what-didnt-work-and-why) cost 1.35%, now
measured directly.

#### Why a ΔE00 0.45 match still breaks it

*`eval_pantone_gap.py`, 1817 matched formulas.* Three mechanisms, in order of
severity:

**1. The UFO light spectra are edge-padded and the Pantone spectra are not.**
The UFO light-backing readings fill the 380–730 grid by *repeating* the edge
values, where Pantone carries real data across the full range:

| Spectrum source | 380 = 390 = 400 nm |
|---|---|
| UFO **light** measurement | **99.96%** (1 exception in 2372) |
| UFO **dark** measurement | **0.35%** |
| Pantone standard | 4.82% |

**This is channel-specific, and an earlier draft of this section got it wrong by
saying "100% of UFO rows".** Only the light channel is padded; dark readings
carry real, varying data at 380–400 nm. Any guard built on this must apply to
the light channel alone — applied to both, it would reject 99.65% of genuine
dark measurements.

A useful corollary: that 99.96%-vs-0.35% split separates the two channels far
better than brightness does (`mean(R_light) > mean(R_dark)` holds on only 87.5%
of paired formulas), so the padding signal *can* identify which backing a
spectrum came from — see [Known limitations](#known-limitations--next-steps).
Consequence: **10 of the 142 features have exactly zero variance across every
training row** — the derivatives at those padded positions. The model has never
once seen them vary. A Pantone input makes them non-zero on **94.8%** of
formulas. This is a structural incompatibility, not a subtle domain shift, and
it was found by accident when a variance-normalised metric divided by zero.

**2. Curve-shape features move far more than colour does.** Displacement from
the UFO measurement to its Pantone standard, in units of the training data's own
standard deviation:

| Feature family | mean shift | worst feature |
|---|---|---|
| raw reflectance `R` | 0.13 SD | 0.61 SD |
| Kubelka-Munk `K/S` | 0.10 SD | 0.82 SD |
| `dR` (curve slope) | **0.44 SD** | 1.95 SD |
| `dK/S` | 0.28 SD | 1.91 SD |

The derivatives — the features that carry *shape*, which is what distinguishes
spectrally similar colorants — move three to four times further than the raw
values. And **52.6% of the shift is systematic** rather than random, so it is a
consistent domain offset the model cannot average away.

K/S compounds this because it is `(1-R)²/2R`, which diverges as `R → 0`: at
`R = 0.010`, a change of 0.005 — utterly invisible — is a **34% change in K/S**.

**3. ΔE00 is a 12:1 lossy compression, by design.** It is computed from Lab:
36 reflectance numbers integrated through 3 CIE colour-matching functions,
deliberately discarding everything the eye cannot see. Of the 1817 formulas,
**1529 are visually identical to their standard (ΔE00 < 1.0) and still sit
0.25 SD away in feature space**, and the correlation between ΔE00 and actual
feature displacement is only **0.450** — colour agreement explains about a fifth
of the variance in what the model actually sees.

That last point generalises past Pantone: **a colour-difference metric cannot
validate model input for a model built to see past colour.** Two inks of
identical colour and different composition is metamerism — the exact problem
this project exists to solve, which is why Stage 1 reads all 36 points and their
transforms rather than three Lab coordinates.

Two consequences:

- The app must not be handed Pantone spectra as measurements. It currently
  accepts them without complaint — see the OOD gap in
  [Known limitations](#known-limitations--next-steps). Note that a cheap and
  highly reliable guard falls out of mechanism 1: a genuine UFO **light**
  measurement has repeated values at 380/390/400 nm, and a Pantone spectrum
  almost never does. Measured as a rule — *reject a light-box spectrum unless
  its first three values are identical* — that is a **0.042% false-rejection
  rate catching 95.2% of Pantone spectra**, far better than a generic novelty
  score (RMS-to-5th-nearest-neighbour scores AUROC 0.659 and catches only 22.7%).
- Answering "what recipe hits this Pantone colour" is the **inverse** model's
  job, below — a model actually trained on target spectra — not the forward
  model fed the wrong input.

**Honest scope:** for the ~2090 Pantone codes already formulated, the right
answer is a database lookup, not a prediction — and `lookup_baseline` in
`src/inverse.py` is exactly that baseline, which any model here has to beat. The
model exists for targets that were *never* formulated: newer Pantone ranges, or
arbitrary customer colours. **It is not currently wired into the app** (see
[Known limitations](#known-limitations--next-steps)).

### Training gate

`train.py` now applies `TARGET_DELTA_MAX_DE = 8.0`: a formula whose measured
spectrum sits more than ΔE00 8 from the Pantone standard it claims to be is a
bad drawdown, and training on it teaches the model that a wrong colour maps to
that recipe. The gate is applied **per-side, after the split** — filtering
before the split would reshuffle sklearn's whole partition rather than removing
a few rows, which is the exact leakage bug documented in
[What didn't work](#what-didnt-work-and-why).

## Pipeline

Two stages, sequential:

```
reflectance spectrum (R_light, R_dark)
        │
        ▼
┌───────────────────┐
│  Stage 1           │   which colorants are active?
│  multi-label        │   (14 independent yes/no decisions)
│  classification     │
└─────────┬──────────┘
          ▼
┌───────────────────┐
│  Stage 2           │   how much of each active colorant?
│  concentration      │   (normalized to sum to 100%)
│  regression         │
└─────────┬──────────┘
          ▼
   colorant → % recipe
```

### Features (`src/features.py`)

285 engineered features per spectrum, built from the raw 36-point light and
dark reflectance curves:

| Group | Count | What it captures |
|---|---|---|
| `R_light`, `R_dark` | 36 + 36 | raw reflectance |
| `KS_light`, `KS_dark` | 36 + 36 | single-constant Kubelka-Munk K/S = (1-R)²/2R, a standard color-science linearization of the concentration↔color relationship |
| `contrast` | 36 | R_light − R_dark per wavelength — directly encodes hiding power/opacity |
| `deriv_light` | 35 | first difference of the light-backing *reflectance* curve — shape/slope, more distinctive between similar colorants than raw levels |
| `deriv_KS_light`, `deriv_KS_dark` | 35 + 35 | first difference of the light/dark **K/S-transformed** curves — a different, and separately useful, view of curve shape (see below) |

A larger 354-feature version (+ dark-backing *reflectance* derivative, + 2nd
derivatives, + light/dark ratio) was tried and measured worse. But the K/S-
derivative pair above, tested separately, is a genuine (if modest) win:
exact-match 80.27% → 80.60%, macro-F1 0.920 → 0.930 — because K/S is
specifically designed to linearize the concentration/color relationship, so
its slope isolates pigment-specific absorption changes more cleanly than
raw reflectance's slope does. See
[What didn't work](#what-didnt-work-and-why) for the full feature-engineering
history.

### Stage 1 — colorant-set classification (`src/stage1_classifier.py`)

**Model:** TabICLv2 (Soda team, Inria) — a pretrained tabular foundation
model that does in-context learning (no traditional weight training; it
computes predictions by attending over the full training set as "context"
at inference time), wrapped in a single `ClassifierChain` for multi-label
support.

- Replaces the earlier 4-architecture GBM stack (HistGradientBoosting +
  CatBoost + LightGBM + XGBoost, combined via a per-colorant logistic
  meta-learner, 82.5–82.8% exact-match) after a session dedicated to
  pushing past that plateau tested essentially every remaining lever: SVM,
  ExtraTrees, PLS-DA, k-NN, a custom deep-learning MLP/CNN, extensive
  hyperparameter search — none beat the GBM stack. Two pretrained tabular
  foundation models did: **TabPFN and TabICL**, tested independently, both
  landing in the same range. **TabICL** (BSD-3-Clause, no version-dependent
  licensing complexity) is what shipped. **TabPFN's licensing is more
  nuanced than either of two earlier claims in this README** — corrected
  twice now (2026-08-12), see [Known limitations](#known-limitations--next-steps)
  for the full, source-verified breakdown: the client code is permissive,
  but which *model weights* you actually get is version-dependent, and the
  package's default version carries a **non-commercial-only** license. An
  older weights version is confirmed commercially usable, but isn't what
  you get by just calling `TabPFNClassifier()` — see
  [What didn't work](#what-didnt-work-and-why) for TabPFN's results.
- Validated result: **87.2% exact-match, macro-F1 0.950** — ahead of the
  GBM stack on every metric, with the standout being **Red 032's F1 jumping
  from 0.67 to 0.91**, the largest single improvement found anywhere in
  this project's history, on the colorant every other technique (including
  the old stack) struggled with most. See [Results](#stage-1) for the full
  per-colorant table.
- **Hard requirement: a CUDA GPU.** In-context learning means every
  prediction re-processes the training set as context, which is
  prohibitively slow on CPU — a single prediction took ~26 minutes on this
  project's CPU-only environment, vs. ~87 seconds on an RTX 5060 Laptop
  GPU (sm_120/Blackwell). Still not instant: the Flask app shows an
  elapsed-time loading state rather than assuming a fast response. This is
  a **server-side** requirement only — individual users of the deployed app
  don't need a GPU, since the client just sends a spectrum and receives a
  recipe over HTTP.
- Colorants present below **2% of the recipe** are treated as trace/noise,
  not a real active ingredient (`ACTIVE_THRESHOLD = 0.02`) — amounts that
  small barely affect the measured spectrum and are effectively
  undetectable from it anyway. Unchanged from the previous architecture.
- If no colorant clears its probability threshold for a row, the model
  falls back to the single most-probable one (every real recipe uses at
  least one colorant). Unchanged from the previous architecture.
- The replaced GBM-stack code (`StackedStage1`, `Stage1Ensemble`,
  `fit_stacked_core`, the 4-architecture factory, isotonic calibration) is
  recoverable from git/zip history if TabICL's GPU dependency ever becomes
  a blocker — see [What didn't work](#what-didnt-work-and-why) for why
  each of the non-foundation-model alternatives tested in that final push
  didn't make the cut.

### Stage 2 — concentration regression (`src/stage2_regressor.py`, `src/stage2_joint.py`)

**Model:** per colorant, whichever of **four** competing approaches
cross-validates better:

1. **Per-colorant regressor**, trained only on rows where that colorant is
   active, with the model *type* itself chosen by cross-validation among:
   `DummyRegressor(mean)`, `Ridge` (α = 1, 10, 100), `PLSRegression`
   (partial least squares — the standard chemometrics tool for
   more-features-than-samples spectral regression), and
   `HistGradientBoostingRegressor`.
2. **Masked-softmax joint neural net** (PyTorch, hidden layers 128→64),
   trained *once* on every formula, predicting all 14 fractions
   simultaneously from a shared representation. The active-colorant mask is
   fed in as an extra input *and* used to mask the final softmax (inactive
   colorants' logits are forced to −∞ before the softmax), so the output is
   guaranteed to be a valid composition — every active fraction ≥ 0,
   summing to exactly 1 — by construction, with no post-hoc clipping or
   renormalizing needed. Trained on a **blended loss** (`MAE + 0.1 × relative
   error`, the relative term computed only over active cells with a clamped
   denominator) rather than plain MAE — plain MAE let trace colorants
   (under 10% concentration) sit at ~83% average relative error, since a
   fixed-size absolute mistake barely moves the loss on a small colorant but
   dominates the loss on a large one. The blend cut trace relative error to
   ~52% with dominant-colorant accuracy essentially unchanged (a small,
   swept, deliberately bounded weight — see
   [What didn't work](#what-didnt-work-and-why) for the sweep and the
   reasoning behind not overweighting it).
3. **ILR (isometric log-ratio) regressor**: fractions are transformed into
   ILR coordinates — the standard statistical representation for "parts of
   a whole" data, since ordinary regression on raw percentages doesn't
   respect the sum-to-one constraint — and a gradient-boosted model
   (one per ILR coordinate) predicts those coordinates from the spectral
   features. The inverse ILR transform (itself just a softmax) maps back to
   a valid composition, the same guarantee as approach 2 but via a
   classical-statistics route instead of a neural net.
4. **TabICLRegressor** (same GPU foundation model as Stage 1), trained
   per-colorant on active rows only — added after Stage 1's TabICL win
   prompted testing the regressor variant here too. Validated: **overall
   MAE 0.0085 vs. 0.0129 for masked-softmax (~34% better) on 13 of 14
   colorants**, but **worse specifically on Red 032** (0.0671 vs. ~0.028) —
   as an isolated per-colorant model it doesn't get the cross-colorant
   information-sharing that lets masked-softmax borrow signal for
   data-starved colorants. Added as a **4th competing candidate rather than
   a wholesale replacement**, specifically so cross-validation keeps
   masked-softmax where it's genuinely better instead of forcing one
   family to win everywhere.

Why four: a single fixed regressor (the original approach) badly overfits
colorants with only a handful of active training examples (R² as low as
**−2.99** for one with just 6 active rows) while doing fine on
well-supported ones. The masked-softmax MLP fixes the sparse cases. The ILR
regressor never won a single colorant in this dataset — a real, tested
negative result — but is kept in the competition since a different dataset
or feature set could tip that balance, and the added training cost is
small. TabICL wins big on well-supported colorants but loses on the most
data-starved one (Red 032), for the same reason the masked-softmax MLP beat
plain per-colorant regressors in the first place: joint models can borrow
cross-colorant signal that an isolated per-colorant model — even a strong
one — cannot. No single family wins everywhere, so cross-validation still
picks per colorant rather than committing globally to one. See
[Results](#stage-2) for the actual numbers.

At inference, each active colorant's raw predicted fraction is clipped to
≥0, and the row is renormalized so the active fractions sum to 1.0 (a
no-op for the two joint approaches, which already guarantee this).

### Backing modes: light, dark, or both

*`src/backing_modes.py`, `train_backing_modes.py`, `eval_backing_modes.py`.
Measured 2026-08-19.*

The client asked for a tool that accepts a light-backing spectrum alone, a
dark-backing spectrum alone, or both. **The production both-backing path is
deliberately untouched by this work** — the single-backing models are new
artifacts under `models/backing_modes/`, and `MODE_BOTH` resolves to the
existing, already-validated production files byte-for-byte.

Measured on the production split with the deployed Stage 1 architecture
(TabICL `ClassifierChain`), 297-formula test set:

| Mode | Features | Exact-match | vs both |
|---|---|---|---|
| **both backings** (production) | 285 | **87.21%** | — |
| light only + recovered rows | 142 | 86.53% | −0.67% |
| light only (paired rows only) | 142 | 85.52% | −1.68% |
| dark only | 142 | 74.75% | **−12.46%** |

Every figure in that table is scored at the project's 2% presence threshold.
**The light-only row above is now historical** — light-only is served from a
different model, retrained at the client's real 0.1% minimum. See
[The 0.1% retrain](#the-01-retrain-wolves_v1) immediately below.

#### The 0.1% retrain (wolves_v1)

*`eval_wolves_v1.py`, `models/wolves_v1/`. Measured 2026-08-25.*

The 86.53% above answers "does the model get every colorant **above 2%**
right". The client formulates down to **0.1%**, so that is the wrong question,
and the honest way to ask the right one is to relabel the same 297-row test set
at 0.1% and rescore. Doing so:

| Light-only Stage 1 model | vs 2% labels | vs 0.1% labels |
|---|---|---|
| incumbent, trained at 0.02 | 86.53% | 59.26% |
| **wolves_v1, trained at 0.001** | — | **88.22%** |
| *ceiling for the incumbent on 0.1% labels* | | *66.33%* |

Two things to read off it.

**The incumbent's 59.26% is not a tuning failure, and 88.22% is not a 29-point
modelling breakthrough.** A model trained to treat sub-2% colorants as absent
cannot report them at any decision threshold. Only 66.33% of test rows have an
*identical* active set under both labellings, so 66.33% is the best the
incumbent could score against the 0.1% question even if it were perfect at its
own — it is 7 points below that ceiling, which is ordinary error. The gain
comes from asking the model the question the client actually asks.

**Stage 2 did not move, and could not have.** Stage 2 masks on `> 0` and never
reads `ACTIVE_THRESHOLD`, so both Stage 2 models solve an identical problem;
`eval_wolves_v1.py` reports a threshold-invariant Stage 2 MAE (true-active
mask) alongside the headline precisely so a retraining-noise difference there
cannot be mistaken for a threshold effect. Only Stage 1 can genuinely move.

**Serving is threshold-independent, so no environment variable is set, read,
or needed at serve time.** The 0.1% decision is baked into the saved
`stage1.joblib`. Traced through the whole serving path:

- `predict_active()` thresholds *probabilities* (default 0.5), not
  concentrations.
- `predict_fractions()` masks on the boolean array Stage 1 handed it.
- Of the eight modules the prediction path imports, `ACTIVE_THRESHOLD` appears
  in exactly one executable place: `build_labels()` in
  `src/stage1_classifier.py`, which is a **training** function and is never
  called from `predict.py` or `app.py`.

`MODE_LIGHT` is pinned to this variant through `DEPLOYED_VARIANT` in
`src/backing_modes.py`, and `predict.serving_model_paths()` resolves it. That
registry is deliberately **separate** from `model_paths()`, which still
describes where training *writes*: keeping them apart means a future
`train_backing_modes.py --force` rebuilds the default tree and cannot silently
overwrite the pinned model. The app's startup banner prints the resolved
directory, variant, training threshold and expected accuracy per mode, so what
a running process serves is never a guess.

**One worked example**, `test_spectra/01_NP2281CPWNUF.json` (PANTONE 2281 C),
run through the deployed light-only path:

| Colorant | True | Predicted |
|---|---|---|
| transp. white | 90.15% | 89.76% |
| yellow | 8.21% | 8.54% |
| green | **1.64%** | **1.70%** |

Green is the whole argument in one row. At 1.64% it sits *below* the 2%
convention and *above* the client's 0.1% one, so the 2% labelling for this
formula is `{white, yellow}` and the 0.1% labelling is `{white, yellow,
green}`. A model trained on 2% labels was trained to call this green absent.
The served model reports it, to within 0.06pp — an exact set match against the
labelling the client actually works to. (One spectrum is an illustration, not
evidence; the 297-row figures above are the evidence. The superseded model was
not re-run on this row.)

**88.22% and 87.21% are not comparable, and the code refuses to compare them.**
They score different labellings of different tasks. `describe_mode()` therefore
returns `cost_vs_both = None` for any mode whose
`MODE_STAGE1_LABEL_THRESHOLD` differs from the both-backing one, and the UI
names the labelling instead. Subtracting them would have displayed *"light
backing only: +1.0% vs both backings"* — i.e. that supplying **less** data
improves accuracy.

Three findings drove the design:

1. **Light-only is nearly free.** TabICL degrades far more gracefully with fewer
   features than a plain per-colorant classifier does — a CPU proxy had
   predicted −7%; the truth is −0.67%.
2. **Dark-only is expensive, and the reason is physical, not modelling.** Over a
   dark backing, **5.00% of *different* recipe pairs collapse to within ΔE00 1.0
   of each other, versus 0.00% over light**, and mean chroma halves (39.2 →
   21.7). When two different recipes produce the same reading, no model can
   separate them — this is a measurement-information limit, and no amount of
   modelling effort will recover it.
3. **Light-only mode can train on the 384 formulas `train.py` discards** for
   having no dark measurement — worth +1.01%.

The mode is **detected from what you supply**, not selected by hand
(`detect_mode`). Every prediction returns the mode it used and that mode's
expected accuracy, so a reduced-input answer can never be presented as a
full-confidence one.

**Memory note.** All three modes resident at once was measured at ~9.4 GB RSS
against roughly 14 GB free on the target machine, so `predict.py` keeps a
bounded cache: `MODE_BOTH` stays pinned, at most one single-backing model sits
beside it, and requesting a third evicts the older.

#### Formulation rules: flagged, not enforced

*`src/constraints.py`, `predict.review_flags()`.*

The client's rules are "at most three non-white colorants" and "nothing below
0.1%". `predict_recipe_flexible(enforce_constraints=True)` will apply them, and
**it is off by default and stays off.**

The measurement: across the 297-row test set only **2 predictions** exceed three
non-white colorants, worth at most **+0.67pp** if every one were a genuine error
— below this project's documented run-to-run noise. Against that,
`enforce_max_colorants` would *destroy* a correct answer on the ~0.3% of real
formulas whose truth genuinely does contain four, because it drops the
lowest-probability colorant whether or not that colorant belongs. Silently
rewriting a correct recipe to satisfy a rule the recipe legitimately breaks is a
worse failure than reporting the rule was broken.

So the rule is surfaced, not applied. `predict.review_flags()` returns the
reasons a recipe should be seen by a human, and the API returns them under
`review_flags`:

| id | Raised when | Why a human, not the code |
|---|---|---|
| `red_032` | recipe contains Red 032 | data-starved colorant, spectrally near-indistinguishable from four other reds — needs a physical drawdown |
| `colorant_cap` | more than 3 non-white colorants | outside the client's standard formulation rule; may still be correct, so confirm rather than auto-correct |

The two are **independent** — a recipe can trip either, both, or neither — and
are kept as separate entries rather than one boolean because they call for
different actions. The legacy `needs_review` / `review_colorant` fields still
carry the Red 032 flag unchanged for any existing caller.

## How accuracy is measured

*`src/evaluation.py`. Read this before quoting 87.2% to anyone.*

Stage 1's headline metric is **exact colorant-set match**: a row counts only if
all 14 presence decisions are simultaneously right. Presence is defined by a
threshold on the true fraction — `ACTIVE_THRESHOLD` in `src/cxf_parser.py`,
overridable per run via the `VEVO_ACTIVE_THRESHOLD` environment variable. At the
project default of `0.02`, a formula containing 1.96% of a colorant is labelled
ABSENT and one containing 2.06% is labelled PRESENT.

That default is **this project's own modelling convention, not a client
specification and not something written in the CxF file.** It was raised from 0%
to 2% because sub-2% amounts barely affect the measured spectrum and are
effectively undetectable from it.

**The threshold is now mode-dependent, so every exact-match figure must be
quoted with the threshold it was measured at.** The light-only model is trained
and scored at `0.001`, the client's real minimum working concentration; both-
backing and dark-only remain at `0.02`. `MODE_STAGE1_LABEL_THRESHOLD` in
`src/backing_modes.py` records which is which, and is the reason
`describe_mode()` will not subtract one mode's accuracy from another's. The rest
of this section describes the `0.02` convention specifically — the deadband
analysis, the 87.205% headline and the "half the errors are threshold artifacts"
finding are all measurements of the both-backing production model at `0.02`, and
none of them transfer to the 0.1% light-only figure. See
[The 0.1% retrain](#the-01-retrain-wolves_v1).

Two consequences follow, and they pull in opposite directions.

**First, the metric is harsh by construction.** Fourteen simultaneous decisions
at 98.94% per-decision accuracy gives 0.98942¹⁴ ≈ 87.2%. The headline number is
not "the model is wrong 13% of the time" — it is "the model gets all fourteen
right, at once, on 87% of formulas."

**Second, roughly half the remaining errors are threshold artifacts, not real
mistakes.** Re-running the deployed classifier and classifying its own errors
(288 of 297 rows, 43 wrong decisions — that subset reproduces the headline at
87.153% vs 87.205%): **21 of the 43 (49%) involve a trace amount straddling the
0.02 line** — rubine red at 0.0196 vs 0.0224, black at 0.0199 vs 0.0206, process
blue at 0.0193 vs 0.0202. The model is being scored wrong for failing to resolve
one part in a thousand of ink, a distinction no formulator can act on.

*(This section previously said "25 of 44 (57%)". That figure was carried from an
earlier note rather than measured against the deployed model; corrected
2026-08-20. The corrected split leaves ~22 genuinely wrong decisions, which is
the number used everywhere else in this README.)*

So `src/evaluation.py` reports a **deadband** metric alongside exact-match: a
presence disagreement is forgiven when the true fraction lies within
`DEFAULT_DEADBAND = 0.01` of the threshold — i.e. in [0.01, 0.03], the region
where "present" vs "absent" is a labelling convention rather than a fact about
the ink.

| Reading | Value |
|---|---|
| Strict exact-match (headline, comparable to all project history) | **87.205%** |
| Deadband — threshold artifacts forgiven | 94.276% |
| Deadband, excluding transparent white | 96.633% |

That leaves roughly **22 genuinely wrong decisions out of 4,158**. Red 032,
black, process blue and rhodamine red have **zero** real errors — every one of
their apparent misses is a trace straddling the line.

**Three cautions on this metric:**

- **Report it alongside exact-match, never instead of it.** Every historical
  figure in this README is in strict terms, and the comparison only works if
  the headline stays strict.
- **It is not threshold calibration.** The three failed attempts in
  [What didn't work](#what-didnt-work-and-why) tuned the classifier's *decision*
  thresholds to score better against an unchanged metric. This changes what
  counts as an error and tunes nothing — there is no parameter fit to any data.
- **`DEFAULT_TOLERANCE = 0.03` is stricter, not laxer.** The whole-recipe
  agreement test requires the *quantities* to be right too, where exact-match
  only checks presence.

## Results

Evaluated on the 297-row held-out test set (299 rows minus 2 with
corrupted recipe labels — see [Data](#data)).

### Stage 1

| Metric | Value |
|---|---|
| Macro-F1 (colorant presence) | **0.950** |
| Exact colorant-set match rate | **87.2%** |

Per-colorant precision / recall / F1 (clean test set):

| Colorant | Support | Precision | Recall | F1 |
|---|---|---|---|---|
| transp. white | 274 | 0.98 | 0.98 | 0.98 |
| yellow | 125 | 1.00 | 1.00 | **1.00** |
| orange 021 | 36 | 1.00 | 0.92 | 0.96 |
| warm red | 47 | 1.00 | 0.94 | 0.97 |
| rubine red | 85 | 0.96 | 0.96 | 0.96 |
| rhodamine red | 17 | 0.94 | 1.00 | 0.97 |
| red 032 | 6 | 1.00 | 0.83 | **0.91** |
| purple | 5 | 0.67 | 0.80 | 0.73 |
| violet | 33 | 1.00 | 0.94 | 0.97 |
| reflex blue | 42 | 1.00 | 0.93 | 0.96 |
| process blue | 97 | 0.97 | 0.99 | 0.98 |
| blue 072 | 9 | 1.00 | 1.00 | **1.00** |
| green | 18 | 0.86 | 1.00 | 0.92 |
| black | 171 | 0.99 | 0.98 | 0.99 |

**Red 032's F1 jumped 0.67 → 0.91** — the single largest per-colorant
improvement found anywhere in this project, on the colorant every earlier
technique (including the 4-GBM stack below) struggled with most. **Purple
is now the weakest colorant (F1 0.73)**, but on only 5 test examples that's
close to noise — its Stage 2 quantity estimate (below) is excellent, so
this doesn't look like a real modeling gap the way Red 032's used to be.

**A wide model search was run on Stage 1 across two separate pushes.** The
first (targeting >80%) tried extended features, two threshold-calibration
strategies, Label Powerset, PLS-DA, SVM, ExtraTrees, L1 regularization on
Stage 2, and a multi-task joint network — all reverted except a
K/S-curve-derivative feature (+0.3 points) and **stacking four
gradient-boosting architectures with a trained meta-learner** (+2 points,
80.6% → 82.6%, the biggest win of that push), plus dropping
corrupted-label rows (+0.6 more, → 83.2%). The **second push** (this
session, chasing >82%) tested k-NN, a custom deep-learning MLP/CNN, and
extensive hyperparameter search on the GBM stack — none worked — before
landing on **TabICL**, a pretrained tabular foundation model, which beat
the entire 4-architecture GBM stack outright: **83.2% → 87.2%**, with
Red 032 as the standout individual win. A **third push** (2026-08-13,
targeting 90%) tried three different ensembling techniques on top of
production TabICL — more internal ensembling, chain-order averaging (the
exact technique that worked for the old GBM stack), and a TabICL+TabPFN
cross-model ensemble — and all three came back flat or worse, consistently
enough to treat the current single-chain default-settings TabICL as a
genuine local optimum rather than an under-tuned starting point. See
[What didn't work](#what-didnt-work-and-why) for the full results table
from all three pushes, and the [Pipeline](#stage-1--colorant-set-classification-srcstage1_classifierpy)
section above for why TabICL specifically (vs. TabPFN, whose licensing turned
out to be more nuanced than a one-line answer — see the correction in
[Known limitations](#known-limitations--next-steps)).

### Stage 2

Model generations, all measured on the same held-out test set:

| Metric | v1: fixed GBR | v2: hybrid (per-colorant + sklearn joint MLP) | v3: masked-softmax + ILR competition | + K/S-derivative features | + stacked Stage 1 | + mask-fix + MAPE loss | + corrupted-row cleanup | + TabICL as 4th family (current) |
|---|---|---|---|---|---|---|---|---|
| MAE, all cells (incl. trivial true-zero) | 0.0073 | 0.0058 | 0.0046 | 0.0047 | 0.0047 | 0.0047 | 0.0045 | 0.0021 |
| **MAE, active cells only** (the real number, unaffected by Stage 1) | 0.0242 | 0.0222 | 0.0175 | 0.0181 | 0.0181 | 0.0179 | 0.0165 | **0.0081** |
| End-to-end MAE (Stage 1's predicted set → Stage 2) | — | 0.0273 | 0.0241 | 0.0225 | 0.0233 | 0.0241 | 0.0225 | **0.0128** |

The second-to-last-but-one column fixed a real bug found while validating
the MAPE-blend change: the masked-softmax model's input had the
active-colorant mask concatenated onto it twice (once by the caller, once
again inside the network's own `forward()`), silently making the deployed
model bigger and worse-behaved than anything that had actually been tested
— see [What didn't work](#what-didnt-work-and-why). The corrupted-row
cleanup column comes from something much simpler: **17 rows across the
dataset have corrupted recipe labels** (colorant fractions summing to
~0.1-0.14 instead of ~1.0 — partial/broken records, not real formulas) that
were being used to train and evaluate Stage 1/2 without anyone checking for
it. Dropping them (2 from the test set, 15 from training) is what actually
fixed purple, not any of the modeling changes before it. The **last
column** — adding TabICLRegressor as a 4th cross-validated candidate
per colorant, alongside the existing per-colorant/masked-softmax/ILR
competition — is the biggest jump in the table: active-cell MAE **0.0165 →
0.0081** (51% better), end-to-end **0.0225 → 0.0128** (43% better). Same
mechanism as Stage 1's win: TabICL simply predicts better than the
alternatives for well-supported colorants. See the per-colorant table below
for exactly which colorants switched, and
[Pipeline](#stage-2--concentration-regression-srcstage2_regressorpy-srcstage2_jointpy)
above for why it was added as a competing family rather than a replacement.

Per-colorant (active cells only, true active mask — isolates Stage 2's
regressor quality from Stage 1's classification error), current version:

| Colorant | n (test) | MAE | R² | Model used |
|---|---|---|---|---|
| transp. white | 274 | 0.0100 | 0.997 | TabICL |
| yellow | 141 | 0.0055 | 0.998 | TabICL |
| orange 021 | 39 | 0.0090 | 0.994 | TabICL |
| warm red | 51 | 0.0160 | 0.872 | TabICL |
| rubine red | 104 | 0.0057 | 0.993 | TabICL |
| rhodamine red | 18 | 0.0171 | 0.990 | TabICL |
| red 032 | 6 | **0.0977** | **0.466** | masked-softmax MLP |
| purple | 8 | 0.0159 | 0.955 | masked-softmax MLP |
| violet | 36 | 0.0091 | 0.978 | TabICL |
| reflex blue | 46 | 0.0075 | 0.992 | TabICL |
| process blue | 119 | 0.0055 | 0.995 | TabICL |
| blue 072 | 11 | 0.0090 | 0.984 | TabICL |
| green | 21 | 0.0058 | 0.998 | TabICL |
| black | 217 | 0.0048 | 0.996 | TabICL |

**TabICL won cross-validated selection for 12 of 14 colorants.**
Cross-validation correctly kept the older masked-softmax MLP for the two
most data-scarce colorants (red 032, purple) — TabICL was tested for both
and lost (see [What didn't work](#what-didnt-work-and-why)), confirming
the reasoning above: joint models can borrow cross-colorant signal that an
isolated per-colorant model, however strong, cannot.

Purple's R² here (0.955) is noticeably better than the 0.83 reported after
the corrupted-row cleanup, and red 032's (0.466) is noticeably *worse* than
that generation's 0.63 — both are the **same model family** (masked-softmax
MLP) as before, just retrained. With only 6-8 active test examples for
either colorant, a full retrain's random initialization alone can swing R²
by this much; treat any single run's red032/purple numbers as having a wide
error bar, not a precise measurement. This is the same lesson the purple
corrupted-label investigation already taught: don't over-trust a metric
computed from a handful of examples. See
[Known limitations](#known-limitations--next-steps) for what that means for
red 032 specifically.

### Bottom line

Stage 1 gets the right *set* of colorants correct on roughly 7 out of every
8 test formulas (87.2%). When it does, Stage 2 estimates each active
colorant's percentage very accurately for anything with reasonable training
data — active-cell MAE is under 0.01 (under a single percentage point) for
11 of the 14 colorants, with R² above 0.97 for most of them. That now
includes **purple** (R²=0.955) and blue 072 (84 examples, R²=0.98) despite
both looking superficially "rare" in raw test-set support numbers.
**Red 032** (32 examples total, R²=0.47) is the one colorant left with a
genuine, unresolved data-scarcity problem — the best colorant-identification
result in the project (F1 0.67→0.91) but still the worst quantity estimate
by a wide margin, and every modeling technique tried on the quantity side
(including two different TabICL setups this session) has failed to close
that gap. See [Known limitations](#known-limitations--next-steps) for the
full breakdown.

**Read 87.2% in context.** It is fourteen simultaneous correct decisions per
formula, and most of what it counts as failure is a trace amount straddling a
2% labelling threshold rather than a wrong answer. Forgiving those, the model
makes roughly **22 genuinely wrong decisions out of 4,158** — and the largest
single remaining contributor, transparent white, is
[provably underdetermined by the data](#known-limitations--next-steps) rather
than badly modelled. See
[How accuracy is measured](#how-accuracy-is-measured). What is *not* validated
is whether a predicted recipe would physically look right; that needs drawdowns.

## What got removed, and why

An earlier version of this pipeline had a **Stage 3**: a two-constant
Kubelka-Munk physics model (`K` and `S` fit separately per colorant, plus
the light/dark backing reflectances) that forward-predicted a spectrum from
a candidate recipe and refined Stage 2's fractions against the measured
spectrum via nonlinear least squares. It also drove a CIEDE2000 color-
accuracy evaluation.

It's gone now. Reasons:

- **Stage 3 never won.** On every test run, Stage 2's ML output had lower
  recipe MAE than the physics-refined version. The physics layer was dead
  weight in the actual prediction path.
- **The physics floor was too high to trust.** Even feeding the forward
  model the *true* recipe, it only reconstructed the measured spectrum to
  ~8% reflectance MAE (down from ~10% for a cruder single-constant
  approximation) — nowhere near clean enough for the CIEDE2000 numbers it
  produced to mean anything. Mean dE2000 sat around 14 with under 1% of
  test formulas in the "commercially acceptable" (dE2000 < 3) range,
  the entire time, regardless of which recipe fed it.
- **A specific, well-motivated fix didn't pan out.** Residual analysis found
  a strong correlation (0.71) between reconstruction error and transparent-
  white loading — the textbook signature of pigment "crowding" (scattering
  efficiency dropping off at high loading). A saturating correction for
  exactly that was implemented and tested with a proper warm-started
  parameter search (to rule out an optimization artifact). It never beat
  "no correction" on total reconstruction error. Real correlation, but this
  functional form didn't explain it — likely because the per-wavelength fit
  already has enough free parameters to absorb the pattern some other way.
- **No way to independently validate it anyway.** There's no masstone data
  and no film-thickness metadata in this dataset, so the physics model was
  always going to be under-constrained. The only real way to validate
  color accuracy is to physically draw down a handful of predicted recipes
  and measure them — not something achievable from this dataset alone.

Given all that, keeping ~6-8 minutes of nonlinear-least-squares fitting in
the training loop for a stage that never actually contributed to the
shipped output wasn't worth it. Recipe accuracy (Stage 1 exact-match, Stage
2 MAE) is now the sole target metric, which is also more honest — it's what
Stage 2's output is actually evaluated against here, unfiltered by a shaky
forward model.

The removed modules (`src/km2.py`, `src/stage3_refine.py`,
`src/fingerprint.py`, `src/color_science.py`) are recoverable from git/zip
history if the physics direction is ever picked back up — the two-constant
KM equations and the crowding-correction experiment are documented there in
detail.

## What didn't work, and why

Tried, in the order attempted — all reverted except those marked
**Worked** (kept, genuine wins):

| Attempt | Result | Why |
|---|---|---|
| Extended features (354, +dark/2nd derivatives/ratio) + 8-chain ensemble | Exact-match unchanged (80.3%), macro-F1 dropped 0.920→0.901 | Redundant with existing features; just added noise for a tree ensemble at this dataset size |
| Per-colorant threshold calibration (optimize F1) | Exact-match dropped to 76.6% | Optimizing per-colorant F1 trades exact-match away for rare-colorant recall — different objective |
| Per-colorant threshold calibration (optimize exact-match directly) | Calib split: 77%→82%. Test set: stuck at 78% (= the calibration-split data-loss floor) | Overfit the small calibration split; didn't generalize |
| Label Powerset (predict the whole colorant combo as one multiclass label) | 68.2% vs 80.3% baseline | 52 of 172 distinct training combos appear only once; 3.7% of test rows have a combo *never seen in training* — structurally unsolvable for a pure multiclass approach |
| Single-constant Kubelka-Munk (original physics) | Physics floor 10.2% reflectance MAE | Assumes full opacity; can't represent the light/dark backing difference this translucent film shows |
| Two-constant Kubelka-Munk | Physics floor 8.1% reflectance MAE | Better, but still too high to trust; see [What got removed](#what-got-removed-and-why) |
| Pigment-crowding correction on top of two-constant KM | No improvement over "no correction," verified with warm-started re-optimization | Real correlation (0.71) in the residuals, but this parametric fix didn't reduce total reconstruction error |
| ILR (isometric log-ratio) transform regressor for Stage 2 | Lost cross-validation for all 14 colorants to the masked-softmax MLP | Kept in the codebase anyway (see `src/stage2_joint.py`) — cheap to keep competing, and a different dataset could tip the balance; just documenting that it doesn't win here |
| PLS-DA as Stage 1's base classifier (swapped into the same ClassifierChain ensemble) | 22.4%–46.5% exact-match across a components sweep (3→20), vs. 80.3% baseline; still climbing but with steep diminishing returns | PLS-DA is a *linear* method; the actual signal in this data looks non-linear (specific curve shapes/thresholds), which is exactly what tree-based models are built to capture and linear methods structurally can't |
| SVM (RBF kernel) as Stage 1's base classifier | Peaked at 69.9% (C=100), *declined* at higher C (300: 69.2%, 1000: 67.2%) — a real, found optimum, not a cut-off-early result | Kernel methods rely on distances across all features at once and get noisier as irrelevant/redundant dimensions pile up (215+ engineered, correlated features); trees split on individual thresholds instead, which handles this better here |
| ExtraTrees (Extremely Randomized Trees) as Stage 1's base classifier | Flat 65.2%–65.9% across 5 very different configs (tree count, depth, leaf size, class-balancing) | A bagging-style ensemble (independent randomized trees, averaged) rather than boosting's sequential error-correction; the flatness across configs (not a tuning issue) suggests boosting's sequential correction is doing real, load-bearing work on this specific data |
| Extended features: K/S-transformed curve derivative (`deriv_KS_light`/`deriv_KS_dark`, 285 features total) | **Worked**: exact-match 80.27%→80.60%, macro-F1 0.920→0.930 | Not reverted — kept. See [Features](#features-srcfeaturespy) above |
| L1 weight regularization added to the Stage 2 masked-softmax MLP's loss (meant to "snap hallucinated percentages to 0") | Monotonically worse with every increase: MAE 0.0181 (none) → 0.0197 → 0.0325 → 0.1480 → 0.1676 (strongest) | The premise didn't apply: inactive colorants are already forced to *exactly* 0.0 by the existing softmax masking (logits set to −∞ before the softmax runs), so there was nothing left to "clean up." L1-on-weights instead just shrinks the weights needed to fit the genuinely active colorants, damaging every colorant uniformly, not just noisy ones |
| Multi-task network: single PyTorch model, shared trunk, Sigmoid classification head + Softmax regression head, trained jointly (weighted loss, swept reg_weight 0.2→5.0) | Worse on **both** tasks at every weighting: best case (reg_weight=5.0) was 70.9% exact-match (vs. 80.6% for the current classifier alone) and 0.0297 end-to-end MAE (vs. 0.0225 for the current two-stage pipeline) | Both tasks improved *together* as reg_weight increased rather than trading off, suggesting the small shared trunk simply lacks the capacity to serve two different task types (classification needs a 5-model chain ensemble on its own to reach 80%) rather than the tasks conflicting; a much larger network/more tuning might close some gap but is unlikely to catch two already-specialized, separately-tuned systems |
| Naive soft-vote averaging across 4 gradient-boosting architectures (HGB + CatBoost + LightGBM + XGBoost) for Stage 1 | 80.6% — no better than the single best architecture (81.3%), worse than the eventual stacked result | Equal-weight averaging lets the weakest architecture (CatBoost, 78.3% alone) drag the ensemble down as much as the stronger ones pull it up — the same "equal-weight penalty" problem hard voting has, just softened slightly |
| **Stacking** the same 4 architectures via a per-colorant logistic-regression meta-learner (trained on 5-fold OOF probabilities) instead of naive voting | **Worked**: 82.6% exact-match, +2 points over baseline — the single largest Stage 1 gain found | Not reverted — kept, now the production Stage 1 architecture. See `src/stage1_classifier.py` |
| Per-colorant threshold optimization (Optuna, 500 trials) against exact-match on **pooled 5-fold OOF predictions from the full stacked model** (~1689 rows, not a single ~253-row split like the earlier attempts) | OOF exact-match improved (77.6%→79.3%), but the **held-out test set got worse** (82.6%→78.6%) | Third failed threshold-tuning attempt, with three different methodologies. Fixing the calibration-data-volume concern didn't fix it, which points past "not enough data" to something more fundamental: exact-match is a joint, all-14-decisions-correct constraint, and directly optimizing 14 free thresholds against it gives the optimizer room to find combinations that fit whatever data it's shown without the gains being genuine, transferable signal. A flat 0.5 cutoff generalizes better precisely because it isn't fit to any particular slice of data |
| Beam search decoding (width=5) for a single Classifier Chain, replacing greedy top-1-per-step decoding | Identical exact-match to greedy (82.3% both), macro-F1 slightly lower (0.926 vs 0.935) | The chain's per-step classifiers are already confident/well-calibrated enough that the greedy top choice matches the beam-search optimum on nearly every row — there wasn't a meaningful cascading-early-error problem on this data for beam search to fix |
| RAkEL (Random k-Labelsets, k=4, 40 random subsets, LightGBM multiclass base, majority-vote aggregation) | 79.9% vs. 82.3% for a plain single chain | Random subsets don't reliably capture the *specific* colorant correlations that actually matter (e.g. black+process blue); a random group of 4 is as likely to bundle unrelated labels together, paying label-powerset's sparsity cost without reliably getting the local-correlation benefit |
| K/S-curve **2nd** derivative feature (as opposed to the 1st derivative above, which worked) | +2 points exact-match on a cheap single-architecture proxy (80.6%→82.6%), but **purple's F1 dropped 0.83→0.67 and red 032 stayed flat** | Caught before spending hours validating against the full stack: the row-level diff showed the gain concentrated entirely in already-easy, well-supported colorants (white, black, process blue, rubine red) with zero benefit to red 032 and active harm to purple -- inflating exact-match through volume on easy cases rather than solving the trace-colorant problem it was meant to address. Not integrated |
| MAPE-blended loss for Stage 2's masked-softmax MLP (`MAE + λ·relative_error`, λ swept 0-0.5, settled on 0.1) | **Worked**: overall MAE 0.0181→0.0179, trace-colorant relative error ~83%→~53%, dominant-colorant accuracy essentially unchanged | Not reverted — kept. Found and fixed a real bug in the process: the first full-pipeline validation attempt came back *worse* (0.0192) than an isolated single-model test of the same change (0.0179) — investigating the gap turned up a pre-existing wiring bug (see next row) that was making the deployed model bigger and worse-behaved than what had actually been validated |
| (bug fix, found via the above) Masked-softmax MLP was receiving the active-colorant mask concatenated onto its input **twice** — once by the caller (`X_aug = hstack([X, mask])`), once again inside `_MaskedSoftmaxNet.forward()`, which already concatenates it internally | Fixing it alone (holding the loss function at plain MAE) closed the gap between the isolated-test and full-pipeline numbers | Pre-existing since the masked-softmax model was first built, not introduced by the MAPE work -- just never surfaced because no one had directly compared an isolated single-model test against the full pipeline's numbers before. `src/stage2_regressor.py` now passes the plain (un-augmented) `X` to the masked-softmax calls and reserves the pre-concatenated `X_aug` for ILR, which has no internal mask handling of its own |
| Purple "spectral collision" hypothesis test: check whether Stage 1 was substituting a red+blue mix for purple on the true-purple test rows (the proposed mechanism), oversample purple's rows, and/or add a non-spectral tie-breaker feature | **Premise didn't hold**: Stage 1's purple recall was already 5/6 on the dirty test set, and the one miss substituted nothing (just under-predicted) -- the real damage was in Stage 2's regression (R²=−2.31), not classification | Redirected the investigation to Stage 2's actual per-row predictions instead, which is what found the row 293 issue below. Neither of the proposed fixes (oversampling, extra feature) was tried, since the root cause turned out to be something else entirely |
| **Root cause found**: one test row (colorant fractions summing to 0.138, not ~1.0 -- a corrupted/partial recipe label) was solely responsible for purple's R²=−2.31 (8-9 active test examples is too few for one bad label not to dominate). 17 such rows exist across the full dataset (2 in the original test split, 15 in training) and were being used everywhere despite already being excluded from the old physics fit | **Worked, and by a lot**: dropping these rows raised purple's R² from −2.31 to 0.83 and improved overall Stage 2 MAE (0.0179→0.0165) and Stage 1 exact-match (82.6%→83.2%) | Not reverted — kept. `train.py` now splits train/test *first* (matching every other number in this project) and drops corrupted rows from each side *afterward*. The first attempt at this fix filtered before splitting instead, which reshuffled sklearn's entire train/test partition and produced a suspiciously huge jump (82.6%→94.9%) that turned out to be a data-leakage artifact -- caught by comparing against the original partition before trusting the number. Lesson generalized: with n=6-9 test examples per rare colorant, always check for a data-quality explanation before accepting a modeling one, and never trust a metric change that's larger than the change you made could plausibly explain |
| Red 032 (32 examples total, the most data-starved colorant): oversampling with Gaussian jitter (factors 3×/5×/8× on Stage 1, 3×/5× on Stage 2) and moderate inverse-frequency loss weighting (2×/5×/10× on Stage 2's absolute-error term) | **Neither helped.** Stage 2 R² never beat the unweighted/non-oversampled baseline (0.55) at any setting tested (0.42-0.54); Stage 1 recall moved by exactly one flipped prediction at one factor (3×) and reverted to baseline at 5× and 8× -- noise on a 6-example statistic, not a trend | Both techniques work by making the model pay more attention to the *same* ~26 real training examples (duplication and reweighting are mechanically similar levers); both failing independently points at the real bottleneck being insufficient *distinguishing information* in those examples (red 032 correlates 0.95-0.99 with warm red, orange, rubine red, and rhodamine red), not insufficient attention paid to them. Empirically confirms real additional lab data is the only lever left for red 032 |
| Mitra (AWS AutoGluon pretrained tabular foundation model) as a further Stage 1/2 candidate | Failed to train: `Not enough memory to safely train model. Estimated to require 26.197 GB out of 11.204 GB available` | Fixable via an AutoGluon memory-ratio override, but that override carries real OOM/system-instability risk; dropped per explicit decision not to force it rather than a modeling failure |
| TabPFN (Prior Labs) as a Stage 1 base classifier, on GPU, default settings (`TabPFNClassifier(device="cuda")`, no version pinned) | 85.9% exact-match, macro-F1 0.936, red032-F1 0.909 — close to but slightly below TabICL (87.2%/0.950/0.91) | Not deployed. Licensing history on this row, corrected twice: first called non-commercial-only (wrong), then called freely commercial (also wrong/incomplete). **Verified 2026-08-12 by reading the actual license file for the specific weights `model_path="auto"` downloads**: default settings pull **TabPFN-3** (`Prior-Labs/tabpfn_3` on HF), licensed under `tabpfn-3-license-v1.0`, which explicitly states *"the model, its derivatives, and its outputs cannot be used for any commercial or production purpose"* — production use requires Prior Labs' separate paid Commercial Enterprise License. This test's numbers are from that non-commercial-licensed model; see the next row for the version confirmed commercially usable |
| **TabPFN pinned to `ModelVersion.V2`** (`Prior-Labs/TabPFN-v2-clf`/`-reg`) — the older weights, checked directly and confirmed under the permissive Prior Labs License (commercial use OK with attribution) — Stage 1 classifier chain | 85.185% exact-match, macro-F1 **0.9549** (edges out TabICL's 0.950), red032-F1 0.9091 (ties TabICL) — accuracy is competitive to slightly ahead of TabICL. **But predict took 2217s (~37 min) for the 297-row test set**, vs. TabICL's ~87s for a *single* prediction in production | Not deployed, despite matching/beating TabICL on every accuracy metric: the inference cost makes it impractical for a live app regardless of license or accuracy — v2's per-step cost is dramatically higher than v3's (which took 595s for the same test) or TabICL's. This is the only TabPFN variant confirmed license-safe, and it still doesn't win on the metric that actually matters for deployment |
| Same v2-pinned TabPFN, as an isolated per-colorant **Stage 2** regressor (same convention as the isolated TabICL regressor test above) | Overall active-cell MAE **0.0096** (vs. production 0.0081, isolated TabICL 0.0085 — slightly worse). **Red 032 MAE 0.0624** — surprisingly *better* than production's masked-softmax (0.0977) and better than isolated TabICL (0.0671-0.0737), despite being an isolated model with no cross-colorant sharing. Total time 247.6s for all 14 colorants (fit+predict), far more practical than Stage 1's chain result above | Red 032's result is a genuinely interesting anomaly worth flagging, not a settled win: it's measured on only 6 test examples, and this project already has a documented lesson (the purple corrupted-label investigation) that statistics this small swing heavily run-to-run. Not integrated as a 5th Stage 2 family without first putting it through the same honest cross-validated OOF-MAE comparison the other 4 families already go through — a single train/test split isn't enough evidence given the sample size |
| **TabICL** (Soda/Inria pretrained tabular foundation model, in-context learning, GPU) replacing the 4-GBM stack as Stage 1's classifier | **Worked**: 83.2%→87.2% exact-match, macro-F1 0.921→0.950, **Red 032 F1 0.67→0.91** (the single largest per-colorant gain in this project) | Not reverted — kept, now the production Stage 1 architecture. Only after k-NN, a custom deep-learning MLP/CNN, and further hyperparameter search on the existing GBM stack all failed to move past 82.6% did pretrained foundation models get tried; unlike the GBM stack's boosted trees, TabICL's in-context learning effectively lets every prediction draw on the *entire* training set as reference examples rather than compressing it into fixed learned weights, which matters most exactly where training data is thinnest (red 032). See `src/stage1_classifier.py` |
| TabICLRegressor as a **full replacement** for Stage 2's per-colorant/joint-softmax/ILR competition (single family, no per-colorant selection) | Overall active-cell MAE 0.0085 vs. 0.0129 for masked-softmax (~34% better) on 13 of 14 colorants, but **worse on red 032** (0.0671 vs. ~0.028) | As an isolated per-colorant model, TabICL doesn't get the cross-colorant information-sharing that lets the joint masked-softmax MLP borrow signal for data-starved colorants — the same limitation a plain per-colorant regressor has, just with a stronger base model. Not deployed as a full replacement; instead added as a **4th competing candidate** in the existing per-colorant CV-selection framework (see next row) |
| **TabICLRegressor as a 4th candidate family**, competing per-colorant against per-colorant-regressor/masked-softmax/ILR exactly like the existing three | **Worked**: active-cell MAE 0.0165→0.0081 (51% better), end-to-end 0.0225→0.0128 (43% better). Wins 12 of 14 colorants; CV correctly keeps masked-softmax for red 032 and purple | Not reverted — kept, now in production. Confirms the "why four" reasoning above: letting CV pick per colorant captures TabICL's strength on well-supported colorants without losing masked-softmax's cross-colorant information-sharing advantage on the two data-starved ones. See `src/stage2_regressor.py` |
| Chained TabICL for red 032 specifically: `sklearn.multioutput.RegressorChain(TabICLRegressor)` over all 14 colorants jointly, ordered most- to least-supported (red 032 last) so its regressor could condition on every other colorant's fraction, active-mask included as an input feature | **Worse, not better**: red 032 MAE 0.1053 — worse than both production (0.0557 CV / 0.0977 held-out, masked-softmax) and the isolated per-colorant TabICL attempt above (0.0737) | `RegressorChain` trains with teacher forcing (each regressor sees prior colorants' *true* values while fitting) but predicts using the chain's own *predicted* values in sequence — for a target this data-starved (6 active examples), the compounded prediction error from 13 upstream regressors overwhelmed whatever cross-colorant signal chaining was meant to contribute. Masked-softmax avoids this because it predicts all 14 fractions from one shared hidden layer in a single forward pass, not sequentially. This was the last untested modeling lever for red 032's Stage 2 number; with it ruled out, every reasonable technique (oversampling, reweighting, per-colorant/joint/isolated-TabICL/chained-TabICL) has now been tried — treat red 032's Stage 2 accuracy as a closed investigation a second time over, see [Known limitations](#known-limitations--next-steps) |
| Push Stage 1 past 87.2% toward a 90% target (2026-08-13), attempt 1: sweep TabICL's internal `n_estimators` (default 8, which is itself internal ensembling over shuffled feature/class variants) up to 16 and 32 | n_estimators=16: exact-match unchanged at 87.205%, macro-F1 up slightly (0.9498→0.9573), red032-F1 unchanged, but predict time 2.4x slower. **n_estimators=32: worse on every metric** — exact-match 86.195%, macro-F1 0.9414 (below the n_estimators=8 baseline), red032-F1 0.80 (down from 0.91) — at 4.5x the baseline predict time | More internal ensembling isn't free: past a point it destabilizes rather than smooths predictions, hitting the most data-starved colorant hardest. Not adopted — production stays at the default `n_estimators=8` |
| Attempt 2: chain-order ensembling for TabICL Stage 1 — fit 5 `ClassifierChain`s with different random label orders (mirroring the *old* GBM architecture's proven 5-chain-average design, which was one of that architecture's real wins) and average their probabilities | **Worse from the first additional chain on, never recovered**: 1 chain (=production) 87.205%/0.9498/0.91 → 2 chains 86.195%/0.9396/0.80 → settled at 5 chains: 86.195%/0.9393/0.80. Red 032's F1 dropped to 0.80 as soon as a 2nd chain was averaged in and stayed there. Cost: 720s to build vs. 221s for one chain | The old GBM architecture's chains were individually weak, high-variance base learners, so order-diversity averaging genuinely reduced variance. TabICL is already a strong, internally-ensembled model on its own — averaging multiple already-good chains just dilutes a confident correct answer with differently-ordered, less-confident answers, hurting the most fragile colorant (red 032) most. A technique proven to work for the old architecture does not transfer to TabICL |
| **Augmenting light-only training with the 1796 Pantone target spectra** (they are single-backing, so they fit the light-only feature shape exactly — nearly doubling the training set for free) | **Worse: −1.35% exact-match.** A leakage guard was asserted first, so this is not a measurement artifact | Pantone standards are *opaque swatches on coated stock*; UFO light-backing readings are a *translucent film over a light substrate*. They are close enough to compare (mean ΔE00 0.75) but not close enough to train on interchangeably — the model learns a slightly different measurement population and pays for it on the real one. A negative result worth recording because the idea is obviously attractive |
| Grouped train/test split for the inverse model via connected components at ΔE00 5.0 (intended to keep near-twin Pantone shades out of the test set) | **Degenerate**: collapsed 2000 of 2072 colours into a single group, leaving 12 usable test rows | Transitive chaining — Pantone's coated library is dense enough that A≈B and B≈C links almost everything into one component even when A and C are plainly different colours. Replaced by `purged_split` (random test set, then *drop* training rows within the buffer), which gets the same leak-safety without the collapse. See `src/inverse.py` |
| Attempt 3 (research-only, not deployment-practical given TabPFN's latency): cross-model ensemble — average TabICL's and TabPFN v2's (commercially-licensed, pinned) Stage 1 probabilities together, since they're genuinely different architectures/pretraining rather than copies of the same model | Essentially a null result: TabICL solo 87.205%/0.9498/0.91, TabPFN v2 solo 85.185%/0.9549/0.91, **ensemble 87.205%/0.9494/0.91** — exact-match identical to TabICL alone, macro-F1 slightly *below* TabICL alone despite TabPFN's higher individual score, red032 unchanged | Simple probability averaging didn't combine complementary strengths — it diluted whichever model was confidently correct on a given row with the other model's less-confident answer. Three different, individually well-motivated ensembling techniques (this one plus the two above) all failed to beat the single production TabICL configuration — a consistent enough pattern to conclude the current single-chain, default-settings TabICL setup is a genuine local optimum, and 90%+ likely isn't reachable through model-level ensembling tricks. Same root cause as red 032's Stage 2 problem: not enough distinguishing data for the hardest cases, which no amount of ensembling fixes |

## Usage

**Requires a CUDA GPU.** Both stages now use TabICL, a pretrained tabular
foundation model that does in-context learning — training and prediction
both re-process the training set as context, which is only practical on
GPU (a single CPU prediction takes ~26 minutes vs. ~87 seconds on an RTX
5060 Laptop). This is a training/serving-machine requirement, not a
per-client one — a deployed Flask app's users don't need a GPU themselves.

```bash
pip install -r requirements.txt

python train.py
# trains Stage 1 + Stage 2, prints metrics, saves to models/

python predict.py --input sample_spectrum.json
# loads the trained models, prints a predicted recipe

python predict.py --input sample_spectrum.json --target "325 C"
# ...and reports how far the measured sample sits from that Pantone standard
```

### The browser app

```bash
python app.py          # then open http://localhost:5000
```

Paste or upload a spectrum, get a recipe. The app routes automatically to the
right backing mode and states which one it used along with that mode's expected
accuracy. Inference is **60–90s** — the wait is a designed state showing which
stage is running and elapsed time, not a frozen page.

**The app does not compare results against Pantone** (removed 2026-08-20 by
request). It predicts a recipe and reports the input mode, nothing else. The
Pantone machinery is untouched everywhere else in the codebase — `src/pantone.py`
still backs `train.py`'s target-delta training gate and the `eval_pantone_*`
scripts — it is simply not part of what the tool reports. What this drops is the
off-target warning: the app no longer tells you when a drawdown has drifted from
its intended colour, so remember that **a recipe reproduces the sample it was
measured from, drift included.**

### Input format

```json
{
  "R_light": [0.15, 0.15, ... 36 values, 380-730nm in 10nm steps ...],
  "R_dark":  [0.15, 0.14, ... 36 values ...]
}
```

Either key may be omitted — supply `R_light` alone, `R_dark` alone, or both.
The app's Paste tab takes bare comma/space/newline-separated numbers into
labelled light and dark boxes, so no JSON needs to be hand-written.

**The backing cannot be inferred from the numbers.** Guessing by brightness was
tested and is wrong 12% of the time, so the caller has to say which backing a
spectrum came from — hence two labelled boxes rather than one box and a guess.

Example inputs live in `test_spectra/` (14 files, each carrying its true recipe
as `_true_recipe_percent` for checking predictions against).

### Other scripts

```bash
python train_backing_modes.py --modes light dark   # build the single-backing models
python eval_backing_modes.py                       # what each mode costs
python eval_forward_leakage.py                     # is exact-match inflated by near-duplicate twins?
python train_inverse.py                            # colour -> recipe direction
python backup.py "label"                           # snapshot before risky work
python backup.py --list  /  --restore <name>
```

`backup.py` SHA256-verifies every copy, and `--restore` takes a safety snapshot
of the current state before overwriting anything.

## Project layout

```
train.py                      end-to-end training + evaluation (production, both backings)
predict.py                    load trained models, predict a recipe from a spectrum
app.py                        Flask browser app
requirements.txt

train_backing_modes.py        build the single-backing models
eval_backing_modes.py         measure what each backing mode costs
eval_forward_leakage.py       test whether exact-match is inflated by near-duplicate twins
train_inverse.py              inverse direction: target colour -> recipe
backup.py                     verified snapshot/restore of code (and optionally data)

src/
  cxf_parser.py               parses the CxF3 file into a wide DataFrame; owns ACTIVE_THRESHOLD
                              (default 0.02, override with VEVO_ACTIVE_THRESHOLD)
  features.py                 285-feature matrix (R, K/S, contrast, derivatives)
                              + 142-feature single-spectrum variant
  stage1_classifier.py        Stage 1: TabICL in a ClassifierChain, GPU-only
  stage2_regressor.py         Stage 2: per-colorant / joint-MLP / ILR / TabICL competition, picked per colorant by CV
  stage2_joint.py             the two joint (whole-composition) Stage 2 models: masked-softmax MLP (PyTorch) + ILR regressor
  backing_modes.py            light / dark / both routing, feature building, expected accuracy per mode
  colorimetry.py              reflectance -> XYZ -> Lab -> CIEDE2000, numpy only
  pantone.py                  the Pantone target library and its linkage to UFO formula names
  evaluation.py               deadband + tolerance metrics, error taxonomy
  inverse.py                  colour -> recipe pairs, leak-safe splitting, lookup baseline

templates/index.html          app markup
static/app.js, static/style.css   app behaviour and styling

models/                       production artifacts (stage1_classifier.joblib, stage2_regressors.joblib)
  backing_modes/light/, dark/   single-backing artifacts (production files are NOT duplicated here)
test_spectra/                 14 example inputs, each with its true recipe embedded

NPXXXXCPWNUF UFO White PE.cxf raw spectral + recipe data (source of truth, ObjectType=Trial)
Pantone_Coated_V_4.cxf        Pantone Coated standards (ObjectType=Target)
All recipes and components UFO White PE.xlsx   recipe/component reference sheet
sample_spectrum.json          example input for predict.py
```

### Why `colorimetry.py` reimplements CIEDE2000

The CIE tables and the ΔE00 formula are written out by hand in numpy rather than
pulling in `colour-science`, so the handoff package keeps slim dependencies —
this ships as an offline-capable desktop tool. It is verified against a
reference implementation: Lab agrees to 1.45e-05, and **CIEDE2000 agrees to
exactly 0.0** across 6000 random pairs plus neutrals and hue-wrap edge cases,
with the Sharma test vector reproducing 2.0425.

## Known limitations / next steps

- **Correction to an earlier claim in this README:** it previously said
  red 032, purple, and blue 072 each had "well under 20 active examples in
  the entire dataset." That was wrong — those were counts in the 299-row
  *test set*, not the full dataset. The real full-dataset counts are red
  032: 32, purple: 73, blue 072: 84. Only red 032 is genuinely data-starved;
  purple and blue 072 have a reasonable amount of training data.
- **Corrected again: purple's earlier "spectral confusability" diagnosis was
  overstated.** Purple's reflectance curve does genuinely correlate
  0.79–0.86 with rhodamine red and violet, colorants it's frequently mixed
  with — that correlation is real. But it turned out **not to be the
  dominant cause** of purple's R²=−2.31: a single corrupted training/test
  label (recipe fractions summing to 0.138, not 1.0) was doing almost all
  of the damage, given only 8-9 active test examples. Dropping that one
  row raised purple's R² to 0.83 — see
  [What didn't work](#what-didnt-work-and-why) for the full story,
  including a near-miss with a data-leakage bug while fixing it. Blue 072
  has an even closer spectral twin (reflex blue, r=0.987) but that twin is
  essentially never combined with it in real recipes, so it was never
  actually a problem in practice. **Red 032** is the one colorant where
  data scarcity (32 examples total) remains a genuine, unresolved issue.
- **General lesson from the purple investigation**: with 6-9 active test
  examples for the rarest colorants, always rule out a data-quality
  explanation before trusting a modeling one, and be suspicious of any
  metric swing bigger than what the change you made could plausibly
  produce -- that's usually a sign of leakage or a measurement artifact,
  not a real result.
- **Red 032 update: Stage 1 is essentially solved, Stage 2 is not, and the
  Stage 2 gap is now a closed investigation twice over.** Switching Stage 1
  to TabICL took red 032's F1 from 0.67 to 0.91 — no longer a weak point in
  colorant *identification*. Its Stage 2 *quantity* estimate, however,
  remains the worst in the model (MAE 0.0977, R²=0.47 on the held-out test
  set) despite two additional TabICL-based attempts this session: an
  isolated per-colorant TabICL regressor (lost — no cross-colorant
  information sharing) and a chained TabICL `RegressorChain` explicitly
  designed to give red 032 access to the other 13 colorants' predicted
  fractions (lost even worse — compounded chain-prediction error outweighed
  the added signal). Combined with the oversampling/reweighting/per-colorant
  regressor attempts from earlier in the project, essentially every
  reasonable modeling technique has now been tried on red 032's Stage 2
  number and failed. The root cause hasn't changed: only 6 active training
  examples, correlating 0.95–0.99 with several other colorants — genuinely
  insufficient distinguishing information, not insufficient model capacity
  or attention. **The only remaining lever is more real lab measurements
  containing red 032.** See [What didn't work](#what-didnt-work-and-why)
  for the full evidence trail.
- **TabPFN licensing — corrected twice on 2026-08-12, here's the actual,
  source-verified state.** This README first said TabPFN's weights were
  non-commercial-only (the stated reason TabICL shipped instead). That was
  corrected to "TabPFN is freely commercial" — but that correction was
  itself incomplete. The real picture, checked by reading the license file
  for each specific artifact rather than a general description:
  - The `tabpfn` **client code** (PyPI package) is under the Prior Labs
    License (Apache 2.0 derivative) — permissive, commercial-friendly.
  - **Which model weights you get depends on version, and this matters a
    lot.** Calling `TabPFNClassifier()`/`TabPFNRegressor()` with no
    arguments downloads **TabPFN-3** by default (confirmed by reading the
    installed package's own version-resolution code) — and TabPFN-3's
    weights, hosted at `Prior-Labs/tabpfn_3` on Hugging Face, are licensed
    under `tabpfn-3-license-v1.0`, which states outright: *"the model, its
    derivatives, and its outputs cannot be used for any commercial or
    production purpose."* Commercial use requires Prior Labs' separate,
    paid Commercial Enterprise License.
  - The **older TabPFN v2 weights** (`Prior-Labs/TabPFN-v2-clf`/`-reg`),
    checked the same direct way, **are** under the permissive Prior Labs
    License — commercial use OK with attribution. This is the only TabPFN
    variant confirmed safe to deploy, and it must be explicitly pinned
    (`ModelVersion.V2`) — the package will not give you this by default.
  - **Lesson, generalized**: a pretrained-model package's client-library
    license does not automatically apply to whatever weights it downloads,
    and "the default settings" can silently point at a *different*,
    more restrictive license than what you checked. Verify the license of
    the specific weights file actually being loaded, not just the package's
    top-level license field — and re-check after any package upgrade, since
    what counts as "default" can change between versions.
  - This project still isn't the place to get a final legal answer for a
    commercial deployment — verify with counsel regardless of which version
    is used.
- **Transparent white is provably underdetermined, and it dominates the
  remaining error.** White is the single largest contributor to what is left of
  the strict metric — the deadband number jumps 94.276% → 96.633% when it is
  excluded. Among formulas targeting the *same* colour, white's loading is
  **3.81× more variable** than the other colorants'.

  The mechanism is **missing film thickness**, not spectral confusability. The
  clearest case: `NP0102C` is 18.74% white / 81.01% yellow, and `NPYELL` is 0%
  white / 100% yellow — but their *yellow shares of the coloured portion* are
  0.997 and 1.000. These are the same recipe at two dilutions. Transparent white
  is an extender: adding it dilutes the film, and a diluted film laid down
  thicker reads much like an undiluted film laid down thinner. The CxF file has
  no film-thickness or applicator-gap metadata (`PhysicalAttributes` is empty on
  every record), so the one variable that would separate them is absent.

  **This is a data limitation, not a modelling one** — no model can recover a
  quantity the measurement does not encode. Practical consequence: consider
  quoting the headline metric *excluding* white rather than trying to model it,
  and treat white percentages as the least reliable part of any prediction. An
  earlier framing of this as "colour is blind to extender" was an
  oversimplification; thickness is the actual missing variable.
- **No masstone (single-colorant) measurements exist.** Every colorant's
  spectral signature is inferred from mixtures only, which limits how
  cleanly any physics-based approach (if revisited) can isolate individual
  colorant behavior.
- **No independent colour-accuracy validation — partially improved, not
  closed.** Recipe accuracy (MAE, exact-match) is validated against held-out
  data. Whether a *predicted* recipe would actually look right is still not
  validated: that needs a trustworthy forward model, and this project's
  Kubelka-Munk layer was not (see
  [What got removed](#what-got-removed-and-why)).

  What the [Pantone linkage](#the-pantone-linkage) *did* close is the adjacent
  question — how far a **measured sample** sits from the standard it was aiming
  at. That is now directly reportable. The remaining gap is specifically
  predicted-recipe colour, and **the only way to close it is to physically draw
  down and measure a sample of predicted recipes.** Until then the app's footer
  says so, and output should be treated as a starting point, not a final
  formulation.
- **The app is forward-only, by decision.** `src/inverse.py` and
  `train_inverse.py` exist and are evaluated, but nothing in `app.py` calls
  them, and the Pantone comparison was removed from the app on 2026-08-20 by
  request. Both remain available in the codebase for CLI and analysis use.
  Measured on held-out colour groups, the inverse model scores **22.19%**
  exact-match against a **22.83%** nearest-lookup baseline — it ties on colorant
  identification and beats the baseline on quantities (active-cell MAE 0.0708 vs
  0.1000). If it is ever surfaced, the honest framing is a starting point for
  the bench, not a formulation; and for the 97.9% of Pantone codes already
  formulated it should return the recorded recipe rather than a prediction.
- **There is no automated test suite.** Correctness is currently established by
  the eval scripts and by module-level assertions (e.g. `src/evaluation.py`
  asserts its presence rule reproduces `build_labels`). A regression suite is
  the largest piece of engineering hygiene still missing.
- **No out-of-distribution check.** If a user submits a spectrum unlike anything
  in the training set, the model returns a confident recipe anyway. The
  app reports the nearest Pantone standards, which gives a human a *hint*, but
  there is no novelty score and nothing refuses to answer. This has been
  proposed twice and not built.
