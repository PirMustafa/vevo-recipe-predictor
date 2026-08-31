"""End-to-end training & evaluation for the spectrum -> recipe pipeline.

Stage 1 (classify active colorants) -> Stage 2 (estimate fractions).

Usage:
    python train.py
"""
from __future__ import annotations

import pathlib

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES, WAVELENGTHS, load_dataset
from src.evaluation import format_summary, summarise
from src.features import build_feature_matrix
from src.pantone import load_pantone_library, target_deltas
from src.stage1_classifier import build_labels, train_stage1, predict_active, save_stage1
from src.stage2_regressor import train_stage2, predict_fractions, save_stage2

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Second, independent label-quality gate: every formula was made to match a
# specific Pantone standard (see src/pantone.py), so a drawdown sitting wildly
# far from its own target is evidence of a bad measurement or a mis-recorded
# recipe -- something the colorant-sum check below cannot detect.
#
# The threshold is deliberately loose. Measured on the 1817 checkable formulas,
# tightening it costs examples from exactly the colorants this project is most
# starved on:
#     > 3.0  drops  76 rows -- including 13 of 101 blue 072 (13%), 9 of 87
#            purple (10%), 26 of 236 violet (11%). Unacceptable: those are
#            hard-to-formulate colours, not corrupt records.
#     > 5.0  drops  20 rows -- still takes 2 of red 032's 37 examples (5.4%),
#            and red 032 data scarcity is the project's headline limitation.
#     > 8.0  drops   2 rows -- both in data-rich colorants. Clear anomalies
#            only (the worst is PANTONE 7680 C at 16.63, next 8.51).
# So 8.0 removes genuine outliers without paying for it where it hurts most.
# Raise/lower only with the per-colorant loss table in hand.
TARGET_DELTA_MAX_DE = 8.0


def main() -> None:
    # This script writes the PRODUCTION artifacts (models/stage1_classifier.joblib
    # and models/stage2_regressors.joblib) and it is threshold-sensitive, because
    # build_labels() below reads ACTIVE_THRESHOLD.
    #
    # The specific accident this prevents: in PowerShell -- this project's shell --
    # `$env:VEVO_ACTIVE_THRESHOLD='0.001'` persists for the REST OF THE SHELL
    # SESSION. Run a threshold experiment, then run `python train.py` in the same
    # window, and the production model is silently relabelled and overwritten with
    # no warning and no version control to undo it. train_backing_modes.py's guard
    # never sees this path.
    #
    # Experiments belong in train_backing_modes.py --models-root <dir>.
    if ACTIVE_THRESHOLD != 0.02:
        raise SystemExit(
            f"REFUSING TO RUN: VEVO_ACTIVE_THRESHOLD is set "
            f"(ACTIVE_THRESHOLD={ACTIVE_THRESHOLD}, baseline is 0.02).\n"
            f"train.py writes the production models, which are validated at 0.02.\n"
            f"Retraining them at a different threshold would overwrite them with\n"
            f"artifacts that are not comparable to any published figure.\n\n"
            f"  PowerShell: Remove-Item Env:VEVO_ACTIVE_THRESHOLD\n"
            f"  bash:       unset VEVO_ACTIVE_THRESHOLD\n\n"
            f"To train at another threshold, use:\n"
            f"  python train_backing_modes.py --modes light --models-root models/<name>\n"
            f"Backups of the production models: "
            f"_backup_pre_stage12_experiments/models_backup_*/"
        )

    print("Loading and parsing CxF data...")
    df = load_dataset(str(CXF_PATH))
    df = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)
    print(f"Usable formulas (both light & dark): {len(df)}")

    # Split FIRST, on the full (still-dirty) data, so this partition matches
    # every other number in this project's history (same random_state=42
    # applied to the same-sized input -- filtering before the split would
    # change sklearn's shuffle outcome entirely, silently reshuffling which
    # formulas land in train vs test rather than just removing a few rows,
    # with a real risk of leaking formulas the deployed Stage 1 model was
    # actually trained on into what looks like a "new" test set).
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # THEN drop corrupted-label rows (colorant fractions don't sum close to
    # 1.0 -- partial/broken recipes, not real formulas) from each side
    # separately, preserving the original partition. This was already done
    # for the (now-removed) physics fit but never applied to Stage 1/2
    # training or evaluation, which let a single such row (colorant sum
    # 0.138, not ~1.0) single-handedly drag purple's Stage 2 R^2 from 0.98
    # down to -2.31 in testing -- purple only has 9 active test examples, so
    # one corrupted label dominates the whole statistic. See README.
    colorant_cols_check = [f"colorant_{c}" for c in COLORANT_NAMES]

    def _drop_corrupted(d):
        totals = d[colorant_cols_check].sum(axis=1)
        return d[(totals - 1.0).abs() < 0.05].reset_index(drop=True)

    n_train_before, n_test_before = len(train_df), len(test_df)
    train_df = _drop_corrupted(train_df)
    test_df = _drop_corrupted(test_df)
    print(f"Dropped {n_train_before - len(train_df)} corrupted-label rows from train, "
          f"{n_test_before - len(test_df)} from test")
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Second gate: distance from each formula's own Pantone target. Applied
    # per-side after the split for the same partition-stability reason as above.
    # Degrades to a no-op if the Pantone file is absent.
    pantone_library = load_pantone_library(PANTONE_PATH)
    if not pantone_library:
        print(f"No Pantone library at {PANTONE_PATH.name} -- skipping target-delta check")
    else:
        light_cols = [f"R_light_{wl}" for wl in WAVELENGTHS]

        def _drop_off_target(d, label):
            deltas = target_deltas(d["formula"], d[light_cols].to_numpy(dtype=float),
                                   pantone_library)
            checkable = ~np.isnan(deltas)
            # NaN (no target for this formula) must be KEPT, not dropped
            bad = checkable & (deltas > TARGET_DELTA_MAX_DE)
            if bad.any():
                worst = np.argsort(np.where(checkable, deltas, -1))[::-1][:int(bad.sum())]
                for i in worst:
                    print(f"    dropping {d['formula'].iloc[i]} "
                          f"(dE00 {deltas[i]:.2f} from its Pantone target)")
            print(f"  {label}: {int(checkable.sum())} of {len(d)} checkable against a "
                  f"target, {int(bad.sum())} beyond dE00 {TARGET_DELTA_MAX_DE}")
            return d[~bad].reset_index(drop=True)

        print(f"Target-delta check (threshold dE00 > {TARGET_DELTA_MAX_DE}):")
        train_df = _drop_off_target(train_df, "train")
        test_df = _drop_off_target(test_df, "test")

    print(f"Train: {len(train_df)}   Test: {len(test_df)}")

    print("\n[Stage 1] Building features & training multi-label classifier...")
    # Current: a stacked ensemble across 4 gradient-boosting architectures
    # (HGB/CatBoost/LightGBM/XGBoost), combined via a per-colorant logistic
    # meta-learner -- see src/stage1_classifier.py's module docstring for
    # why (naive voting gained nothing; stacking gained +2 points). Many
    # other things were tried and reverted along the way (extended
    # features, threshold calibration, Label Powerset, single alternative
    # architectures like PLS-DA/SVM/ExtraTrees, L1 regularization, joint
    # multi-task learning) -- see README.md for the full ablation history.
    X_train = build_feature_matrix(train_df)
    X_test = build_feature_matrix(test_df)
    Y_train = build_labels(train_df)
    Y_test = build_labels(test_df)

    stage1_model = train_stage1(X_train, Y_train)
    save_stage1(stage1_model, str(MODELS_DIR / "stage1_classifier.joblib"))

    active_pred = predict_active(stage1_model, X_test)
    f1_macro = f1_score(Y_test.to_numpy(), active_pred, average="macro", zero_division=0)
    exact_match = (active_pred == Y_test.to_numpy()).all(axis=1).mean()
    print(f"  Stage 1 macro-F1 (colorant presence): {f1_macro:.3f}")
    print(f"  Stage 1 exact colorant-set match rate: {exact_match:.3%}")

    print("\n[Stage 2] Training per-colorant concentration regressors...")
    stage2_models = train_stage2(X_train, train_df)
    save_stage2(stage2_models, str(MODELS_DIR / "stage2_regressors.joblib"))

    # Evaluated using the TRUE active set (not Stage 1's predictions) so this
    # isolates Stage 2's regressor quality from Stage 1's classification
    # error -- see README.md for the end-to-end (Stage1-predicted-set) numbers.
    colorant_cols = [f"colorant_{c}" for c in COLORANT_NAMES]
    true_fractions = test_df[colorant_cols].to_numpy(dtype=float)
    active_true = (test_df[colorant_cols].to_numpy(dtype=float) > 0)

    stage2_fractions_true_active = predict_fractions(stage2_models, X_test, active_true)
    err = np.abs(stage2_fractions_true_active - true_fractions)
    print(f"  Stage 2 MAE, all cells (incl. trivial true-zero cells): {err.mean():.4f}")
    print(f"  Stage 2 MAE, active cells only (true fraction > 0):     {err[active_true].mean():.4f}")

    # End-to-end: Stage 1's predicted active set feeding Stage 2.
    stage2_fractions_pred_active = predict_fractions(stage2_models, X_test, active_pred)
    err_e2e = np.abs(stage2_fractions_pred_active - true_fractions)
    mask_either = active_true | active_pred
    print(f"  End-to-end MAE (Stage1-predicted set -> Stage2), union(true,pred) active cells: "
          f"{err_e2e[mask_either].mean():.4f}")

    # Exact-match penalises the model for failing to resolve differences far
    # below what a formulator can act on -- most of its errors are trace
    # amounts straddling ACTIVE_THRESHOLD. Report both so the strict number
    # stays comparable to history while the tolerant one shows what the
    # pipeline actually delivers. See src/evaluation.py.
    print("\n[Evaluation] Strict vs tolerance-aware")
    print(format_summary(summarise(
        stage2_fractions_pred_active, true_fractions, pred_active=active_pred
    )))

    print(f"\nArtifacts saved to {MODELS_DIR}")


if __name__ == "__main__":
    main()
