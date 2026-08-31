"""Predict an ink recipe from a measured reflectance spectrum.

Usage:
    python predict.py --input spectrum.json

Where spectrum.json looks like:
{
  "R_light": [0.15, 0.15, ... 36 values, 380-730nm in 10nm steps ...],
  "R_dark":  [0.15, 0.14, ... 36 values ...]
}

Prints the predicted recipe (colorant -> percentage).
"""
from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd

from src.backing_modes import (
    ALL_MODES, DEPLOYED_VARIANT, MODE_BOTH, MODE_STAGE1_EXACT,
    MODELS_SUBDIR, MODE_STAGE1_LABEL_THRESHOLD, describe_mode, detect_mode,
    features_for_mode, serving_model_paths,
)
from src.constraints import apply_min_concentration, enforce_max_colorants
from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS
from src.features import build_feature_matrix
from src.pantone import assess_against_target, load_pantone_library, nearest_targets
from src.stage1_classifier import load_stage1, predict_active
from src.stage2_regressor import load_stage2, predict_fractions

ROOT = pathlib.Path(__file__).parent
MODELS_DIR = ROOT / "models"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"

# Red 032 is the model's most data-starved colorant (32 training examples,
# Stage 2 R²=0.63, spectrally near-indistinguishable from 4 other reds -- see
# README "Known limitations"). Oversampling and loss-reweighting fixes were
# tried and failed; until real additional lab data resolves it, any recipe
# involving it should be checked against a physical drawdown, not auto-approved.
RED_032 = "UFO30032 red 032"

# The client's standard formulation rule: at most three non-white colorants
# (transparent white does not count). The model is NOT constrained to obey it --
# see src/constraints.py and README "Formulation rules: flagged, not enforced"
# for why forcing compliance was measured and rejected. Instead a prediction that
# breaks the rule is flagged for a human, exactly as Red 032 is: on the 297-row
# test set only 2 predictions exceed the cap, so this is a rare exception to
# look at, not a routine warning to learn to ignore.
MAX_NON_WHITE_COLORANTS = 3
TRANSPARENT_WHITE = "UFO00061 transp. white"


def load_artifacts():
    stage1_model = load_stage1(str(MODELS_DIR / "stage1_classifier.joblib"))
    stage2_models = load_stage2(str(MODELS_DIR / "stage2_regressors.joblib"))
    return stage1_model, stage2_models


def predict_recipe(
    r_light: list[float], r_dark: list[float], artifacts: tuple | None = None
) -> dict[str, float]:
    """Run the spectrum -> recipe pipeline (Stage 1 -> Stage 2) for a single spectrum.

    artifacts: optional pre-loaded (stage1_model, stage2_models) tuple, so a
    long-lived caller (e.g. a web server) can load the models once instead of
    re-reading the 77MB Stage 1 classifier from disk on every call. Defaults
    to loading fresh, for the CLI's one-shot use case.
    """
    if len(r_light) != len(WAVELENGTHS) or len(r_dark) != len(WAVELENGTHS):
        raise ValueError(f"Expected {len(WAVELENGTHS)} values per spectrum (380-730nm, 10nm steps)")

    stage1_model, stage2_models = artifacts if artifacts is not None else load_artifacts()

    row = {"formula": "input"}
    for i, wl in enumerate(WAVELENGTHS):
        row[f"R_light_{wl}"] = r_light[i]
        row[f"R_dark_{wl}"] = r_dark[i]
    df = pd.DataFrame([row])

    X = build_feature_matrix(df)
    active = predict_active(stage1_model, X)
    fractions = predict_fractions(stage2_models, X, active)

    result = {}
    for j, cname in enumerate(COLORANT_NAMES):
        pct = fractions[0, j]
        if pct > 1e-4:
            result[cname] = round(pct * 100, 2)

    return dict(sorted(result.items(), key=lambda kv: -kv[1]))


# Holding every mode resident is not affordable. The artifacts are ~445MB on
# disk but TabICL expands substantially once loaded -- all three at once was
# measured at ~9.4GB RSS, against roughly 14GB free on the target machine. So
# the both-backing (production) model may stay pinned, and at most one
# single-backing model is kept beside it; requesting a third evicts the older.
_MODE_CACHE: dict = {}
_PINNED = {MODE_BOTH}
MAX_CACHED_SINGLE_MODES = 1


def load_mode_artifacts(mode: str) -> tuple:
    """Load (stage1, stage2) for one backing mode, with a bounded cache.

    Lazy rather than eager: most sessions only ever touch one mode. Startup
    stays fast, and the first prediction in a given mode pays the load, which is
    negligible beside the ~60-90s inference it precedes.

    MODE_BOTH resolves to the deployed production files, so the full-accuracy
    path is byte-for-byte the model already validated at 87.21%.

    MODE_LIGHT resolves to the pinned wolves_v1 variant rather than the default
    backing_modes/ tree -- see DEPLOYED_VARIANT in src/backing_modes.py.
    """
    if mode not in _MODE_CACHE:
        s1_path, s2_path = serving_model_paths(MODELS_DIR, mode)
        if not s1_path.is_file() or not s2_path.is_file():
            # Name the directory actually looked in. For a pinned variant the
            # retrain command below would build into the DEFAULT tree, which is
            # not where serving reads from, so it is deliberately not offered.
            variant = DEPLOYED_VARIANT.get(mode)
            if variant:
                raise FileNotFoundError(
                    f"No model found for '{mode}' mode at {s1_path.parent}. "
                    f"This mode is pinned to the '{variant}' variant "
                    f"(DEPLOYED_VARIANT in src/backing_modes.py), so the missing "
                    f"artifacts must be restored to that directory. Retraining "
                    f"with train_backing_modes.py would write to "
                    f"{MODELS_DIR / MODELS_SUBDIR / mode} instead and would NOT "
                    f"be picked up here."
                )
            raise FileNotFoundError(
                f"No model built for '{mode}' mode at {s1_path.parent}. "
                f"Run: python train_backing_modes.py --modes {mode}"
            )

        # Evict an unpinned mode before adding another, oldest first.
        evictable = [m for m in _MODE_CACHE if m not in _PINNED]
        while len(evictable) >= MAX_CACHED_SINGLE_MODES:
            del _MODE_CACHE[evictable.pop(0)]

        _MODE_CACHE[mode] = (load_stage1(str(s1_path)), load_stage2(str(s2_path)))
    return _MODE_CACHE[mode]


def available_modes() -> list[str]:
    """Which modes actually have trained artifacts on disk."""
    out = []
    for mode in ALL_MODES:
        s1, s2 = serving_model_paths(MODELS_DIR, mode)
        if s1.is_file() and s2.is_file():
            out.append(mode)
    return out


def describe_serving(mode: str) -> dict:
    """What this process would actually load for `mode`, for the startup log.

    Separate from available_modes() so that function keeps returning a plain
    list of mode names -- app.py is its only caller, but the shape is simple
    and worth leaving alone.
    """
    s1, s2 = serving_model_paths(MODELS_DIR, mode)
    return {
        "mode": mode,
        "directory": str(s1.parent),
        "variant": DEPLOYED_VARIANT.get(mode) or "production",
        "trained_at_threshold": MODE_STAGE1_LABEL_THRESHOLD[mode],
        "expected_exact_match": MODE_STAGE1_EXACT[mode],
        "present": s1.is_file() and s2.is_file(),
    }


def serving_report(modes=None) -> str:
    """One line per mode naming the artifacts served, for the startup banner.

    This log is the only record of what a running process is actually serving.
    Printing just the mode names cannot distinguish the pinned wolves_v1 light
    model from the superseded one it replaced -- same mode name, different
    directory, 29 points of exact-match between them.
    """
    lines = []
    for mode in (available_modes() if modes is None else modes):
        d = describe_serving(mode)
        lines.append(
            "  %-6s variant=%-11s trained@%.3f  exact-match %5.2f%%  <- %s"
            % (d["mode"], d["variant"], d["trained_at_threshold"],
               d["expected_exact_match"] * 100, d["directory"])
        )
    return "\n".join(lines)


def predict_recipe_flexible(
    r_light: list[float] | None = None,
    r_dark: list[float] | None = None,
    artifacts_by_mode=None,
    enforce_constraints: bool = False,
) -> dict:
    """Predict a recipe from whichever backing spectra are supplied.

    Accepts light alone, dark alone, or both, routing to the model trained for
    that input. Returns the recipe together with a description of the mode used
    and its expected accuracy -- a reduced-input answer must never be
    presentable as a full-confidence one.

    enforce_constraints: apply the client's formulation rules from
        src/constraints.py -- at most 3 non-white colorants (+ transparent
        white), nothing below 0.1%. OFF by default, so this function and app.py
        behave exactly as they did before the option existed. Turning it on
        changes the recipe, so it is an explicit choice, never a silent one.

    The original predict_recipe() is untouched and still requires both spectra;
    existing callers are unaffected.
    """
    mode = detect_mode(r_light, r_dark)

    if artifacts_by_mode is not None and mode in artifacts_by_mode:
        stage1_model, stage2_models = artifacts_by_mode[mode]
    else:
        stage1_model, stage2_models = load_mode_artifacts(mode)

    X = features_for_mode(mode, r_light=r_light, r_dark=r_dark)
    active = predict_active(stage1_model, X)

    # The two constraints straddle Stage 2 and the order is NOT interchangeable
    # (see src/constraints.py): the colorant cap must land BEFORE Stage 2 so its
    # masked renormalisation re-solves the surviving amounts in one consistent
    # pass, and the concentration floor must land AFTER it, since only then do
    # amounts exist to floor.
    if enforce_constraints:
        proba = stage1_model.predict_proba_matrix(X)
        active = enforce_max_colorants(active, proba, COLORANT_NAMES)

    fractions = predict_fractions(stage2_models, X, active)

    if enforce_constraints:
        fractions = apply_min_concentration(fractions)

    # Cast to plain float: numpy scalars leak into the JSON layer otherwise,
    # and not every serialiser handles them.
    recipe = {}
    for j, cname in enumerate(COLORANT_NAMES):
        pct = float(fractions[0, j])
        if pct > 1e-4:
            recipe[cname] = round(pct * 100, 2)
    recipe = dict(sorted(recipe.items(), key=lambda kv: -kv[1]))

    return {"recipe": recipe, "mode": describe_mode(mode)}


def predict_recipes_batch(mode, specs, artifacts_by_mode=None,
                          enforce_constraints: bool = False) -> list[dict]:
    """Predict MANY spectra of the same backing mode in ONE model call.

    Why this exists: TabICL re-processes the entire 2051-row training set on
    every call, so cost is almost all fixed. Measured on the production light
    model: 1 spectrum 58.8s, 297 spectra 61.5s -- the marginal cost of an extra
    spectrum is about 9 milliseconds.

    Served one at a time, five concurrent users wait 5 x ~40s. Batched, they
    ride in one call and all finish in ~40s. This is what makes the tool usable
    by a team rather than by one person at a time.

    `specs` is a list of (r_light, r_dark) tuples, ALL of the same mode --
    different modes load different artifacts and cannot share a call. Returns
    one result dict per input, in the same order, shaped exactly like
    predict_recipe_flexible's so callers cannot tell which path produced it.
    """
    if not specs:
        return []

    if artifacts_by_mode is not None and mode in artifacts_by_mode:
        stage1_model, stage2_models = artifacts_by_mode[mode]
    else:
        stage1_model, stage2_models = load_mode_artifacts(mode)

    # Feature building is cheap (milliseconds); only the model call is not.
    X = pd.concat(
        [features_for_mode(mode, r_light=l, r_dark=d) for l, d in specs],
        ignore_index=True,
    )
    active = predict_active(stage1_model, X)

    # Same straddle order as the single-spectrum path: cap before Stage 2 so
    # the masked renormalisation re-solves in one pass, floor after it.
    if enforce_constraints:
        proba = stage1_model.predict_proba_matrix(X)
        active = enforce_max_colorants(active, proba, COLORANT_NAMES)

    fractions = predict_fractions(stage2_models, X, active)

    if enforce_constraints:
        fractions = apply_min_concentration(fractions)

    described = describe_mode(mode)
    out = []
    for i in range(len(specs)):
        recipe = {}
        for j, cname in enumerate(COLORANT_NAMES):
            pct = float(fractions[i, j])
            if pct > 1e-4:
                recipe[cname] = round(pct * 100, 2)
        out.append({
            "recipe": dict(sorted(recipe.items(), key=lambda kv: -kv[1])),
            "mode": described,
        })
    return out


def needs_manual_review(recipe: dict[str, float]) -> bool:
    """Whether a predicted recipe should be routed to a human instead of auto-approved.

    Unchanged: this is the Red 032 flag, and existing callers still get exactly
    what they always did. review_flags() below is the fuller answer.
    """
    return RED_032 in recipe


def exceeds_colorant_cap(recipe: dict[str, float]) -> bool:
    """Whether the recipe uses more non-white colorants than the client's rule allows."""
    return sum(1 for c in recipe if c != TRANSPARENT_WHITE) > MAX_NON_WHITE_COLORANTS


def review_flags(recipe: dict[str, float]) -> list[dict]:
    """Every reason this recipe should be seen by a human before use.

    Two INDEPENDENT checks, kept as separate entries rather than collapsed into
    one boolean, because they call for different actions: Red 032 means the
    model is guessing between spectrally near-identical reds and the answer
    needs a physical drawdown; a fourth non-white colorant means the recipe is
    unusual for this client's process and needs confirming that it is intended.
    A recipe can trip both, one, or neither.
    """
    flags = []
    if needs_manual_review(recipe):
        flags.append({
            "id": "red_032",
            "colorant": RED_032,
            "message": (
                "Recipe includes Red 032, the model's most data-starved colorant "
                "and spectrally near-indistinguishable from four other reds. "
                "Verify against a physical drawdown before use -- do not "
                "auto-approve."
            ),
        })
    if exceeds_colorant_cap(recipe):
        flags.append({
            "id": "colorant_cap",
            "colorant": None,
            "message": (
                f"Recipe uses more than {MAX_NON_WHITE_COLORANTS} non-white "
                f"colorants, outside the client's standard formulation rule; "
                f"confirm before use."
            ),
        })
    return flags


def load_pantone_targets():
    """The Pantone standard library, or {} if the CxF file isn't present."""
    return load_pantone_library(PANTONE_PATH)


def describe_colour(r_light, target_code=None, library=None) -> dict:
    """Where a measured sample sits in colour space, independent of its recipe.

    The recipe prediction answers "what is in this sample". It does NOT answer
    "is this sample the colour I wanted" -- if the drawdown drifted, the recipe
    faithfully reproduces the drift. This adds that second answer:

      - `target`: distance from a named Pantone standard, when the caller says
        which one they were aiming for.
      - `nearest`: the closest Pantone standards regardless, so an unlabelled
        sample still gets identified.

    Uses the light-backing spectrum, the only fair comparison to Pantone's
    opaque coated swatches (see src/pantone.py).
    """
    library = load_pantone_targets() if library is None else library
    out: dict = {"target": None, "nearest": []}
    if not library:
        return out

    if target_code:
        out["target"] = assess_against_target(r_light, target_code, library)
    out["nearest"] = [
        {"name": t.name, "code": t.code, "delta_e_2000": round(d, 2)}
        for t, d in nearest_targets(r_light, library, n=3)
    ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ink recipe from a reflectance spectrum")
    parser.add_argument("--input", required=True, help="Path to a JSON file with R_light and R_dark arrays")
    parser.add_argument("--target", help="Pantone code you were aiming for, e.g. '325 C' -- "
                                        "reports how far the measured sample is from it")
    args = parser.parse_args()

    with open(args.input) as f:
        spectrum = json.load(f)

    r_light = spectrum.get("R_light")
    r_dark = spectrum.get("R_dark")

    # Route on whatever was supplied. Both -> the production model, unchanged.
    out = predict_recipe_flexible(r_light, r_dark)
    recipe, mode = out["recipe"], out["mode"]

    print(f"Input: {mode['label']}")
    if mode["is_degraded"]:
        # cost_vs_both is None when this mode was scored against a different
        # presence threshold than the both-backing figure; naming the
        # labelling is then the only honest thing to print.
        qualifier = (f"{mode['cost_vs_both']:+.1%} vs both backings"
                     if mode["cost_vs_both"] is not None else
                     f"measured at the {mode['measured_at_threshold']:.1%} "
                     f"presence threshold; not directly comparable to the "
                     f"both-backing figure")
        print(f"  Expected accuracy {mode['expected_exact_match']:.1%} "
              f"({qualifier})")
        print(f"  {mode['advice']}")
    print()

    print("Predicted recipe:")
    for colorant, pct in recipe.items():
        print(f"  {colorant}: {pct}%")

    # Colour comparison needs a light-backing spectrum; skip it for dark-only.
    colour = (describe_colour(r_light, args.target) if r_light
              else {"target": None, "nearest": []})

    if args.target:
        t = colour["target"]
        print()
        if t is None:
            print(f"Target '{args.target}' not found in the Pantone library.")
        else:
            print(f"Against {t['target_name']}: dE00 {t['delta_e_2000']} "
                  f"({t['interpretation']})")
            if not t["on_target"]:
                print("  NOTE: the measured sample is off this target, so the recipe above")
                print("  reproduces the SAMPLE, not the target colour.")
    elif colour["nearest"]:
        print()
        print("Closest Pantone standards to this sample:")
        for n in colour["nearest"]:
            print(f"  {n['name']}: dE00 {n['delta_e_2000']}")

    flags = review_flags(recipe)
    if flags:
        print()
        print("MANUAL REVIEW REQUIRED:")
        for f in flags:
            print(f"  - {f['message']}")


if __name__ == "__main__":
    main()
