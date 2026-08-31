"""Train the flexible-backing models (light-only and dark-only).

    python train_backing_modes.py                 # train any mode not yet built
    python train_backing_modes.py --force         # retrain regardless
    python train_backing_modes.py --modes light   # just one mode

    # retrain at a different presence threshold, into a SEPARATE tree.
    # The env var MUST be set in the shell before the process starts -- it is
    # read at import time (see src/cxf_parser.py).
    VEVO_ACTIVE_THRESHOLD=0.001 python train_backing_modes.py \\
        --models-root models/exp_t0001 --modes light --force

THIS DOES NOT TOUCH THE PRODUCTION MODEL. The deployed both-backing artifacts
(models/stage1_classifier.joblib, models/stage2_regressors.joblib) are read for
comparison and never written. Everything new lands under
<models-root>/backing_modes/<mode>/, which defaults to models/. Deleting that
directory reverts this work completely.

Any run at a non-default VEVO_ACTIVE_THRESHOLD MUST be given its own
--models-root: the artifacts are not interchangeable, and writing them over the
0.02 baseline would destroy the only validated model. That is enforced, not
merely advised -- see the guard in main().

The router (src/backing_modes.py) points MODE_BOTH at the existing production
files, so the both-backing path continues to use the model already validated at
87.21% rather than a retrained duplicate.

Training data per mode:
  light  paired formulas + the light-only formulas train.py discards (+1.01%)
  dark   paired formulas only -- the dataset contains no dark-only formulas
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.backing_modes import (
    MODE_BOTH, MODE_DARK, MODE_LIGHT, MODE_STAGE1_EXACT, TRAINING_ROWS,
    model_paths, training_frame_for_mode,
)
from src.cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES, WAVELENGTHS, load_dataset
from src.pantone import load_pantone_library, target_deltas
from src.stage1_classifier import build_labels, predict_active, save_stage1, train_stage1
from src.stage2_regressor import predict_fractions, save_stage2, train_stage2

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"
MODELS_DIR = ROOT / "models"

# The threshold every published figure in this repo was measured at. Anything
# else is an experiment and must not share a models directory with it.
BASELINE_THRESHOLD = 0.02
AT_BASELINE_THRESHOLD = abs(ACTIVE_THRESHOLD - BASELINE_THRESHOLD) < 1e-12

TARGET_DELTA_MAX_DE = 8.0          # kept in step with train.py
CC = [f"colorant_{c}" for c in COLORANT_NAMES]
LIGHT_COLS = [f"R_light_{w}" for w in WAVELENGTHS]


def production_partition():
    """train.py's exact sequence: split on dirty data, then clean each side."""
    df = load_dataset(str(CXF_PATH))
    both = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)

    train_df, test_df = train_test_split(both, test_size=0.15, random_state=42)
    train_df, test_df = train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    def drop_corrupt(d):
        t = d[CC].sum(axis=1)
        return d[(t - 1.0).abs() < 0.05].reset_index(drop=True)

    train_df, test_df = drop_corrupt(train_df), drop_corrupt(test_df)

    library = load_pantone_library(PANTONE_PATH)
    if library:
        def drop_off_target(d):
            dd = target_deltas(d["formula"], d[LIGHT_COLS].to_numpy(float), library)
            return d[~(~np.isnan(dd) & (dd > TARGET_DELTA_MAX_DE))].reset_index(drop=True)
        train_df, test_df = drop_off_target(train_df), drop_off_target(test_df)

    light_only = df[df["has_light"] & ~df["has_dark"]].reset_index(drop=True)
    light_only = drop_corrupt(light_only)
    return train_df, test_df, light_only


def train_mode(mode, train_df, test_df, light_only, models_dir=None, force=False):
    """Train one backing mode into `models_dir`/backing_modes/<mode>/.

    models_dir defaults to the production models/ tree only so that existing
    callers keep working; main() always passes --models-root explicitly.
    """
    models_dir = MODELS_DIR if models_dir is None else pathlib.Path(models_dir)
    s1_path, s2_path = model_paths(models_dir, mode)
    if s1_path.exists() and s2_path.exists() and not force:
        print(f"[{mode}] already built at {s1_path.parent} -- skipping (use --force)")
        return None
    s1_path.parent.mkdir(parents=True, exist_ok=True)

    X_train, frame = training_frame_for_mode(
        mode, train_df, light_only if mode == MODE_LIGHT else None
    )
    Y_train = build_labels(frame)
    X_test, _ = training_frame_for_mode(mode, test_df, None)
    Y_test = build_labels(test_df)

    print(f"\n[{mode}] training rows {len(X_train)}  features {X_train.shape[1]}")
    print(f"[{mode}] data: {TRAINING_ROWS[mode]}")

    print(f"[{mode}] Stage 1...")
    stage1 = train_stage1(X_train, Y_train)
    save_stage1(stage1, str(s1_path))

    pred = predict_active(stage1, X_test)
    yt = Y_test.to_numpy()
    exact = float((pred == yt).all(axis=1).mean())
    macro = float(f1_score(yt, pred, average="macro", zero_division=0))
    per_colorant = [float(v) for v in
                    f1_score(yt, pred, average=None, zero_division=0)]

    if AT_BASELINE_THRESHOLD:
        expected = MODE_STAGE1_EXACT[mode]
        print(f"[{mode}] Stage 1 exact-match {exact:.2%}  macro-F1 {macro:.4f}   "
              f"(expected ~{expected:.2%})")
        if abs(exact - expected) > 0.02:
            print(f"[{mode}] NOTE: differs from the measured figure by "
                  f"{exact - expected:+.2%} -- worth checking before relying on it")
    else:
        # MODE_STAGE1_EXACT was measured at 0.02. Comparing a run at another
        # threshold to it would be comparing two different tasks: the labels
        # themselves changed, so a lower number is not necessarily a worse
        # model. Say so rather than printing a misleading delta.
        print(f"[{mode}] Stage 1 exact-match {exact:.2%}  macro-F1 {macro:.4f}   "
              f"(ACTIVE_THRESHOLD {ACTIVE_THRESHOLD:g})")
        print(f"[{mode}] no reference figure exists at this threshold; "
              f"see the baseline table.")

    print(f"[{mode}] per-colorant F1:")
    for name, f1 in zip(COLORANT_NAMES, per_colorant):
        print(f"    {name:<26} {f1:.4f}")

    print(f"[{mode}] Stage 2...")
    stage2 = train_stage2(X_train, frame)
    save_stage2(stage2, str(s2_path))

    true_f = test_df[CC].to_numpy(dtype=float)
    active_true = true_f > 0
    frac = predict_fractions(stage2, X_test, active_true)
    err = np.abs(frac - true_f)
    print(f"[{mode}] Stage 2 MAE active cells {err[active_true].mean():.4f}")

    frac_e2e = predict_fractions(stage2, X_test, pred)
    err_e2e = np.abs(frac_e2e - true_f)
    mask = active_true | pred.astype(bool)
    print(f"[{mode}] End-to-end MAE {err_e2e[mask].mean():.4f}")
    print(f"[{mode}] saved -> {s1_path.parent}")
    return {"mode": mode, "exact": exact, "macro_f1": macro,
            "mae_active": float(err[active_true].mean()),
            "threshold": ACTIVE_THRESHOLD,
            "f1_per_colorant": per_colorant,
            "models_dir": str(s1_path.parent)}


def resolve_models_root(raw: str) -> pathlib.Path:
    """--models-root as an absolute path; relative paths resolve against ROOT."""
    root = pathlib.Path(raw).expanduser()
    if not root.is_absolute():
        root = ROOT / root
    return root.resolve()


def guard_models_root(models_root: pathlib.Path) -> None:
    """Refuse to write experiment artifacts over the validated baseline.

    models/backing_modes/light/ holds the 86.53% light-only model, measured at
    ACTIVE_THRESHOLD 0.02. A run at any other threshold produces artifacts that
    are neither comparable to it nor interchangeable with it, and there is no
    version control in this directory to undo an overwrite. So the combination
    is refused outright rather than warned about.
    """
    if models_root != MODELS_DIR.resolve() or AT_BASELINE_THRESHOLD:
        return

    # The banner goes to stdout and this goes to stderr; without the flush the
    # refusal can appear above the banner it is refusing.
    sys.stdout.flush()

    env = os.environ.get("VEVO_ACTIVE_THRESHOLD", "<unset>")
    suffix = str(ACTIVE_THRESHOLD).replace(".", "")
    print(
        "REFUSING TO RUN.\n"
        f"  ACTIVE_THRESHOLD is {ACTIVE_THRESHOLD:g} "
        f"(VEVO_ACTIVE_THRESHOLD={env}), not the baseline {BASELINE_THRESHOLD:g},\n"
        "  but --models-root resolves to the production tree:\n"
        f"    {models_root}\n"
        "  Training would overwrite models/backing_modes/<mode>/ -- the validated\n"
        "  86.53% light-only baseline -- with artifacts trained on different labels,\n"
        "  and there is no git history here to recover it from.\n"
        "\n"
        "  Give the experiment its own tree, e.g.:\n"
        f"    --models-root models/exp_t{suffix}",
        file=sys.stderr,
    )
    sys.exit(2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the light-only / dark-only models")
    ap.add_argument("--modes", nargs="+", default=[MODE_LIGHT, MODE_DARK],
                    # MODE_BOTH is allowed so the both-backing model can be
                    # retrained at the client's 0.1% threshold. It is safe only
                    # because guard_models_root() refuses to write into the
                    # production models/ tree at any non-default threshold: a
                    # both-backing run therefore lands in a variant directory
                    # (e.g. models/wolves_v1/) and never overwrites the
                    # validated 87.21% production artifacts.
                    choices=[MODE_BOTH, MODE_LIGHT, MODE_DARK],
                    help="which modes to build (both-backing reuses the production model)")
    ap.add_argument("--force", action="store_true", help="retrain even if artifacts exist")
    ap.add_argument("--models-root", default="models",
                    help="directory to write <root>/backing_modes/<mode>/ under "
                         "(default: models). A run at a non-default "
                         "VEVO_ACTIVE_THRESHOLD must point this somewhere else.")
    args = ap.parse_args()

    models_root = resolve_models_root(args.models_root)

    # Every run log must state which threshold produced it: a set of accuracy
    # numbers with no threshold beside them is not interpretable afterwards.
    print("=" * 66)
    print("train_backing_modes")
    print(f"  ACTIVE_THRESHOLD          : {ACTIVE_THRESHOLD:g}"
          f"{'  (baseline default)' if AT_BASELINE_THRESHOLD else '  *** NON-DEFAULT ***'}")
    print(f"  VEVO_ACTIVE_THRESHOLD env : "
          f"{os.environ.get('VEVO_ACTIVE_THRESHOLD', '<unset>')}")
    print(f"  models-root               : {models_root}")
    print(f"  modes                     : {', '.join(args.modes)}")
    print("=" * 66)

    guard_models_root(models_root)

    print("Loading data and reproducing train.py's partition...")
    train_df, test_df, light_only = production_partition()
    print(f"  train {len(train_df)}   test {len(test_df)}   "
          f"recoverable light-only {len(light_only)}")
    print("\nThe production both-backing model is NOT retrained or overwritten.")

    results = [r for m in args.modes
               if (r := train_mode(m, train_df, test_df, light_only,
                                   models_root, args.force))]

    if results:
        print("\n" + "=" * 66)
        if AT_BASELINE_THRESHOLD:
            print("SUMMARY (production both-backing reference: 87.21% exact-match)")
        else:
            print(f"SUMMARY at ACTIVE_THRESHOLD {ACTIVE_THRESHOLD:g}")
            print(f"The 87.21% / 86.53% reference figures were measured at "
                  f"{BASELINE_THRESHOLD:g} and do NOT")
            print("apply here; see the baseline table.")
        for r in results:
            print(f"  {r['mode']:<8} exact {r['exact']:7.2%}   "
                  f"macro-F1 {r['macro_f1']:.4f}   Stage 2 MAE {r['mae_active']:.4f}   "
                  f"T={r['threshold']:g}")
        print(f"  written under {models_root}")


if __name__ == "__main__":
    main()
