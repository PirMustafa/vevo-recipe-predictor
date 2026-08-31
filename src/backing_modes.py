"""Flexible-backing prediction: accept a light spectrum, a dark spectrum, or both.

SEPARATE FROM THE PRODUCTION PIPELINE, DELIBERATELY. The deployed both-backing
path (train.py, predict.py, app.py, models/stage1_classifier.joblib,
models/stage2_regressors.joblib) is not touched by anything here. This module
adds new artifacts under models/backing_modes/ and leaves the originals alone,
so the working system cannot be broken by this work.

Why it exists: the client asked for a tool that accepts a light-backing
spectrum alone, a dark-backing spectrum alone, or both.

What each mode costs, measured on the production split with the deployed
Stage 1 architecture (TabICL ClassifierChain), 297-formula test set:

    both backings              87.21%   285 features   <- unchanged production path
    light only + recovered     86.53%   142 features   -0.67%
    light only                 85.52%   142 features   -1.68%
    dark only                  74.75%   142 features   -12.46%

Three findings drove the design:

  1. Light-only is nearly free. TabICL degrades far more gracefully with fewer
     features than a plain per-colorant classifier does -- a CPU proxy had
     predicted -7%, the truth is -0.67%.
  2. Dark-only is expensive, and the reason is physical, not modelling. Over a
     dark backing 5% of DIFFERENT recipe pairs collapse to within dE00 1.0 of
     each other (vs 0.00% over light), and mean chroma halves (39.2 -> 21.7).
     When two recipes produce the same reading, no model can separate them.
  3. Light-only mode may train on the 384 formulas train.py discards for having
     no dark measurement -- worth +1.01%. See TRAINING_ROWS below.

Caveat worth carrying to the client: the dataset contains ZERO dark-only
measurements across 2372 formulas. A dark-only model is trainable from the dark
half of paired formulas, but no such submission has ever actually occurred.

UPDATE 2026-08-25 -- light-only is now served from a different model. The table
above scores every mode against the project's 2% presence threshold. The client
formulates down to 0.1%, and against 0.1% labels the 2%-trained light model
scores only 59.26% (ceiling 66.33%). A retrain at 0.001, models/wolves_v1/,
scores 88.22% there. MODE_LIGHT is pinned to it via DEPLOYED_VARIANT below, and
MODE_STAGE1_EXACT[MODE_LIGHT] now reports the 0.1% figure -- so it is no longer
on the same axis as the 87.21% / 74.75% above. See MODE_STAGE1_LABEL_THRESHOLD.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .cxf_parser import WAVELENGTHS
from .features import build_feature_matrix, build_single_spectrum_features

MODE_BOTH = "both"
MODE_LIGHT = "light"
MODE_DARK = "dark"
ALL_MODES = (MODE_BOTH, MODE_LIGHT, MODE_DARK)

LIGHT_COLS = [f"R_light_{wl}" for wl in WAVELENGTHS]
DARK_COLS = [f"R_dark_{wl}" for wl in WAVELENGTHS]

# Stage 1 exact colorant-set match. Surfaced to the user so a degraded mode
# never looks like a full-confidence one.
#   both / dark  measured 2026-08-19 (see module docstring)
#   light        measured 2026-08-25 on the wolves_v1 retrain (eval_wolves_v1.py)
MODE_STAGE1_EXACT = {
    MODE_BOTH: 0.8889,
    MODE_LIGHT: 0.8822,
    MODE_DARK: 0.7475,
}

# The presence threshold each figure above was MEASURED against -- i.e. the
# concentration at which a colorant counts as "in the recipe" when scoring an
# exact set match. This is a property of the LABELLING, not of serving: nothing
# in the prediction path reads it (predict_active thresholds probabilities,
# predict_fractions masks on > 0).
#
# It is recorded because the figures are otherwise silently incomparable. The
# project's historical convention is 2%; the client's real minimum working
# concentration is 0.1%, and the light-only model was retrained and rescored
# there. 88.22% @ 0.1% and 87.21% @ 2% answer different questions, so any code
# that subtracts one from the other must check this dict first.
MODE_STAGE1_LABEL_THRESHOLD = {
    MODE_BOTH: 0.001,
    MODE_LIGHT: 0.001,
    MODE_DARK: 0.02,
}

MODE_LABEL = {
    MODE_BOTH: "both backings",
    MODE_LIGHT: "light backing only",
    MODE_DARK: "dark backing only",
}

# How much training data each mode can draw on. Light-only is the only mode that
# gains rows, because it is the only one that does not need a dark measurement.
TRAINING_ROWS = {
    MODE_BOTH: "paired formulas only",
    MODE_LIGHT: "paired formulas + the light-only formulas train.py discards",
    MODE_DARK: "paired formulas only (no dark-only formulas exist)",
}

# Artifacts live in their own directory; the production files are untouched.
MODELS_SUBDIR = "backing_modes"

# Which trained VARIANT each mode is actually SERVED from, when that differs
# from the default training layout. None = serve the default layout.
#
# Deliberately separate from model_paths() below, which describes where training
# WRITES. Keeping the two apart means a future `train_backing_modes.py --force`
# rebuilds the default tree and cannot silently overwrite a pinned variant, and
# guard_models_root() keeps reasoning about the same layout it guards.
#
# MODE_LIGHT -> "wolves_v1":
#   Retrained with VEVO_ACTIVE_THRESHOLD=0.001, the client's real 0.1% minimum
#   working concentration, rather than the project's historical 2% convention.
#   Scored on the same 297-formula test set against 0.1% labels it reaches
#   88.22% exact colorant-set match, against the incumbent 2%-trained model's
#   59.26%. The incumbent cannot be tuned into that gap: its ceiling on 0.1%
#   labels is 66.33%, because a model trained to ignore sub-2% colorants cannot
#   report them. Measured by eval_wolves_v1.py, 2026-08-25.
#
# Serving is threshold-independent: predict_active() thresholds PROBABILITIES
# (default 0.5) and predict_fractions() never reads ACTIVE_THRESHOLD, so the
# 0.1% decision is baked into the saved stage1.joblib. Nothing at serve time
# needs to be told which threshold produced it.
DEPLOYED_VARIANT: dict[str, str | None] = {
    # Retrained 2026-08-27 at VEVO_ACTIVE_THRESHOLD=0.001 (the client's real
    # 0.1% minimum). Scores 88.89% exact-match vs the production 2%-trained
    # model's 87.21% -- but those answer different questions, so the point is
    # not the gain: it is that both-backing and light-only are now measured the
    # same way and can finally be compared. Dark backing also rescues the two
    # weakest colorants: red 032 F1 0.800 -> 1.000, blue 072 0.737 -> 0.900.
    MODE_BOTH: "wolves_v1",
    MODE_LIGHT: "wolves_v1",
    MODE_DARK: None,
}


def detect_mode(r_light, r_dark) -> str:
    """Which mode applies, given whichever spectra the caller supplied."""
    has_light = r_light is not None and len(r_light) == len(WAVELENGTHS)
    has_dark = r_dark is not None and len(r_dark) == len(WAVELENGTHS)
    if has_light and has_dark:
        return MODE_BOTH
    if has_light:
        return MODE_LIGHT
    if has_dark:
        return MODE_DARK
    raise ValueError(
        f"Supply at least one spectrum of {len(WAVELENGTHS)} values "
        f"(380-730nm, 10nm steps)"
    )


def features_for_mode(mode: str, r_light=None, r_dark=None) -> pd.DataFrame:
    """Build the feature matrix this mode's model expects, for a single sample."""
    if mode == MODE_BOTH:
        row = {"formula": "input"}
        for i, wl in enumerate(WAVELENGTHS):
            row[f"R_light_{wl}"] = r_light[i]
            row[f"R_dark_{wl}"] = r_dark[i]
        return build_feature_matrix(pd.DataFrame([row]))
    if mode == MODE_LIGHT:
        return build_single_spectrum_features(np.asarray(r_light, dtype=float))
    if mode == MODE_DARK:
        return build_single_spectrum_features(np.asarray(r_dark, dtype=float))
    raise ValueError(f"Unknown mode {mode!r}; expected one of {ALL_MODES}")


def training_frame_for_mode(mode: str, train_df: pd.DataFrame,
                            light_only_df: pd.DataFrame | None = None):
    """(feature matrix, label source frame) for training this mode.

    For MODE_LIGHT the caller may pass the recovered light-only formulas, which
    are appended -- they carry a light spectrum and a recipe, which is all this
    mode needs, and they are worth about +1% exact-match.
    """
    if mode == MODE_BOTH:
        return build_feature_matrix(train_df), train_df

    col = LIGHT_COLS if mode == MODE_LIGHT else DARK_COLS
    frame = train_df
    if mode == MODE_LIGHT and light_only_df is not None and len(light_only_df):
        frame = pd.concat([train_df, light_only_df], ignore_index=True)
    return build_single_spectrum_features(frame[col].to_numpy(dtype=float)), frame


def model_paths(models_dir: str | pathlib.Path, mode: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Where this mode's Stage 1 / Stage 2 artifacts live.

    MODE_BOTH deliberately points at the EXISTING production files so the
    flexible router reuses the deployed model rather than training a duplicate.
    """
    models_dir = pathlib.Path(models_dir)
    if mode == MODE_BOTH:
        return (models_dir / "stage1_classifier.joblib",
                models_dir / "stage2_regressors.joblib")
    d = models_dir / MODELS_SUBDIR / mode
    return d / "stage1.joblib", d / "stage2.joblib"


def serving_model_paths(models_dir: str | pathlib.Path,
                        mode: str) -> tuple[pathlib.Path, pathlib.Path]:
    """Where this mode's artifacts are SERVED from -- what the app should load.

    Identical to model_paths() unless DEPLOYED_VARIANT pins this mode to a
    trained variant, in which case that variant's subtree is used instead. Call
    this from the prediction path; call model_paths() from the training path.
    """
    variant = DEPLOYED_VARIANT.get(mode)
    if variant is None:
        return model_paths(models_dir, mode)
    # Delegate to model_paths INSIDE the variant tree rather than assuming a
    # layout. The two differ by mode: light/dark live at
    # <root>/backing_modes/<mode>/stage{1,2}.joblib, but both-backing lives at
    # <root>/stage1_classifier.joblib and stage2_regressors.joblib. Hardcoding
    # the light/dark shape here sent the both-backing lookup to a path that
    # does not exist, so the app would have fallen back or failed to load.
    # Deriving it keeps serving and training in step for every mode.
    return model_paths(pathlib.Path(models_dir) / variant, mode)


def describe_mode(mode: str) -> dict:
    """User-facing description, so a degraded mode is always visibly degraded.

    `cost_vs_both` is None whenever this mode's accuracy was measured against a
    different presence threshold than the both-backing figure, because the
    subtraction is then meaningless. Concretely: light-only now scores 88.22%
    against 0.1% labels and both-backing scores 87.21% against 2% labels, so
    `exact - best` yields +1.01% -- which would tell the user that supplying
    LESS data makes the model MORE accurate. It does not; the two numbers score
    different labellings of different tasks. Returning None forces every caller
    to say "not comparable" rather than print a plausible-looking lie.

    `measured_at_threshold` is returned alongside so the caller can name the
    labelling instead. It describes how the figure was SCORED; it has no effect
    on prediction.
    """
    exact = MODE_STAGE1_EXACT[mode]
    best = MODE_STAGE1_EXACT[MODE_BOTH]
    threshold = MODE_STAGE1_LABEL_THRESHOLD[mode]
    comparable = threshold == MODE_STAGE1_LABEL_THRESHOLD[MODE_BOTH]

    # Suppress the figure entirely for any mode still measured at the OLD 2%
    # convention while the deployed light model is measured at 0.1%.
    #
    # Suppressing only `cost_vs_both` was not enough. The interface shows
    # "87.2%" for both-backing and "88.2%" for light-only in the same place,
    # and a user reading those naturally concludes that supplying LESS data is
    # MORE accurate. It is not: 87.2% answers "which colorants above 2%" and
    # 88.2% answers "which above 0.1%". Two different questions, and only the
    # light model has been retrained for the second.
    #
    # A missing number the user can ask about beats a present number that
    # misleads. Restore this the moment the mode is retrained at 0.001 and
    # MODE_STAGE1_LABEL_THRESHOLD is updated to match.
    live = MODE_STAGE1_LABEL_THRESHOLD[MODE_LIGHT]
    stale = threshold != live
    return {
        "mode": mode,
        "label": MODE_LABEL[mode],
        "expected_exact_match": None if stale else round(exact, 4),
        "accuracy_pending_remeasurement": stale,
        "measured_at_threshold": threshold,
        "cost_vs_both": round(exact - best, 4) if comparable and not stale else None,
        "is_degraded": mode != MODE_BOTH,
        # Customer-facing. Keep it short and factual: what was supplied, and
        # what to do about it. The reasoning behind the numbers (thresholds,
        # labelling conventions, why two figures are or are not comparable)
        # belongs in code comments and the project report, NOT in front of a
        # client looking at a recipe.
        "advice": (
            "Both backings supplied. Full accuracy."
            if mode == MODE_BOTH else
            "Light backing only. Supply a dark-backing measurement as well "
            "when one is available."
            if mode == MODE_LIGHT else
            "Dark backing only. Accuracy is reduced because different recipes "
            "can measure alike over a dark backing. Re-measure over a light "
            "backing where possible."
        ),
    }
