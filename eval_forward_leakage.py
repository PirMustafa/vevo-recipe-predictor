"""Is the forward pipeline's reported exact-match inflated by near-duplicate leakage?

    python eval_forward_leakage.py                 # Stage 1 only (default, faster)
    python eval_forward_leakage.py --purge-de 3    # different buffer
    python eval_forward_leakage.py --stage2        # also refit Stage 2 (much slower)

The concern: this dataset is a Pantone reproduction library, and adjacent Pantone
codes are frequently the same shade to the eye (2745 C vs 2746 C). train.py
splits rows at random, so a test formula can easily keep a near-identical twin in
training -- which would make the headline exact-match number optimistic, because
the model gets to recognise a colour it has effectively already seen.

Method: hold the test set fixed, and compare training on
  (A) everything else                                    -- what train.py does
  (B) everything else MINUS colours too close to any test colour
If (B) drops sharply, the original number was leaning on twins.

"Too close" here means the WHOLE input is nearly duplicated, i.e. both backings
match: max(dE00_light, dE00_dark) < buffer. A formula whose light backing matches
a test colour but whose dark backing does not has a materially different feature
vector and is not a twin.

Note the asymmetry from train_inverse.py: there, purging shrinks the training set
AND the baseline depends on twins, so both move. Here only the model moves, which
makes this a cleaner read on leakage specifically.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.colorimetry import delta_e_2000, reflectance_to_lab
from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS, load_dataset
from src.features import build_feature_matrix
from src.pantone import load_pantone_library, target_deltas
from src.stage1_classifier import build_labels, predict_active, train_stage1

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"

# Kept in sync with train.py deliberately -- this script must reproduce the
# production partition exactly, or the comparison is meaningless.
TARGET_DELTA_MAX_DE = 8.0
COLORANT_COLS = [f"colorant_{c}" for c in COLORANT_NAMES]

LIGHT_COLS = [f"R_light_{wl}" for wl in WAVELENGTHS]
DARK_COLS = [f"R_dark_{wl}" for wl in WAVELENGTHS]


def load_production_partition():
    """Reproduce train.py's data prep exactly, including both label gates."""
    df = load_dataset(str(CXF_PATH))
    df = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    def _drop_corrupted(d):
        totals = d[COLORANT_COLS].sum(axis=1)
        return d[(totals - 1.0).abs() < 0.05].reset_index(drop=True)

    train_df = _drop_corrupted(train_df)
    test_df = _drop_corrupted(test_df)

    library = load_pantone_library(PANTONE_PATH)
    if library:
        def _drop_off_target(d):
            deltas = target_deltas(d["formula"], d[LIGHT_COLS].to_numpy(dtype=float), library)
            bad = ~np.isnan(deltas) & (deltas > TARGET_DELTA_MAX_DE)
            return d[~bad].reset_index(drop=True)
        train_df = _drop_off_target(train_df)
        test_df = _drop_off_target(test_df)

    return train_df, test_df


def purge_twins(train_df, test_df, buffer_de: float):
    """Drop training rows whose light AND dark colour both sit within buffer of a test row."""
    tr_l = reflectance_to_lab(train_df[LIGHT_COLS].to_numpy(dtype=float))
    tr_d = reflectance_to_lab(train_df[DARK_COLS].to_numpy(dtype=float))
    te_l = reflectance_to_lab(test_df[LIGHT_COLS].to_numpy(dtype=float))
    te_d = reflectance_to_lab(test_df[DARK_COLS].to_numpy(dtype=float))

    closest = np.full(len(tr_l), np.inf)
    for i in range(len(te_l)):
        d_light = delta_e_2000(np.repeat(te_l[i:i + 1], len(tr_l), axis=0), tr_l)
        d_dark = delta_e_2000(np.repeat(te_d[i:i + 1], len(tr_d), axis=0), tr_d)
        closest = np.minimum(closest, np.maximum(d_light, d_dark))

    keep = closest >= buffer_de
    return train_df[keep].reset_index(drop=True), int((~keep).sum()), closest


def evaluate_stage1(train_df, test_df, label):
    X_train, X_test = build_feature_matrix(train_df), build_feature_matrix(test_df)
    Y_train, Y_test = build_labels(train_df), build_labels(test_df)

    model = train_stage1(X_train, Y_train)
    pred = predict_active(model, X_test)
    yt = Y_test.to_numpy()

    exact = float((pred == yt).all(axis=1).mean())
    macro_f1 = float(f1_score(yt, pred, average="macro", zero_division=0))
    per_colorant = {
        c: float(f1_score(yt[:, j], pred[:, j], zero_division=0))
        for j, c in enumerate(COLORANT_NAMES)
    }
    print(f"  {label:<34} n_train {len(train_df):>5}   "
          f"exact-match {exact:7.2%}   macro-F1 {macro_f1:.4f}")
    return {"exact": exact, "macro_f1": macro_f1, "per_colorant": per_colorant,
            "n_train": len(train_df)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Test the forward model for near-duplicate leakage")
    ap.add_argument("--purge-de", type=float, default=3.0,
                    help="dE00 buffer; a training colour closer than this to any test colour "
                         "(on BOTH backings) is dropped. Default 3.0")
    ap.add_argument("--stage2", action="store_true",
                    help="also refit Stage 2 on the purged set (much slower)")
    args = ap.parse_args()

    print("Reproducing train.py's production partition...")
    train_df, test_df = load_production_partition()
    print(f"  train {len(train_df)}   test {len(test_df)}")

    purged_df, n_purged, closest = purge_twins(train_df, test_df, args.purge_de)
    print(f"\nTwin analysis (buffer dE00 {args.purge_de}, both backings):")
    print(f"  training rows that are near-twins of a test row: {n_purged} "
          f"({100 * n_purged / len(train_df):.1f}%)")
    print(f"  distance to nearest test colour -- median {np.median(closest):.2f}, "
          f"p10 {np.percentile(closest, 10):.2f}, min {closest.min():.2f}")
    for b in (1.0, 2.0, 3.0, 5.0):
        print(f"    within dE00 {b:>3.0f}: {int((closest < b).sum()):>5} training rows")

    print(f"\nStage 1, identical test set ({len(test_df)} rows), differing only in training data:")
    # ASCII-only labels: the Windows console codepage mangles non-ASCII here
    full = evaluate_stage1(train_df, test_df, "A - train on everything")
    purged = evaluate_stage1(purged_df, test_df, f"B - twins purged (dE00 <{args.purge_de})")

    drop_exact = full["exact"] - purged["exact"]
    drop_f1 = full["macro_f1"] - purged["macro_f1"]
    print(f"\n  exact-match delta: {-drop_exact:+.2%}   macro-F1 delta: {-drop_f1:+.4f}")
    print(f"  training rows removed: {n_purged} "
          f"({100 * n_purged / len(train_df):.1f}% of the training set)")

    print("\n  Reading:")
    if drop_exact > 0.05:
        print("    A meaningful share of the headline number depends on near-twins being")
        print("    present in training. On genuinely novel colours, expect the purged figure.")
    elif drop_exact > 0.02:
        print("    Mild dependence on near-twins. The headline number is slightly")
        print("    optimistic for novel colours but not badly so.")
    else:
        print("    The model holds up with twins removed, so the headline number is NOT")
        print("    materially inflated by leakage. Note some of any drop is simply the")
        print("    smaller training set, not leakage -- the two cannot be fully separated")
        print("    without refitting at matched training sizes.")

    worst = sorted(
        ((c, full["per_colorant"][c] - purged["per_colorant"][c]) for c in COLORANT_NAMES),
        key=lambda kv: -kv[1],
    )[:6]
    print("\n  Colorants most dependent on near-twins (per-colorant F1 drop):")
    for c, d in worst:
        print(f"    {c:<32} {full['per_colorant'][c]:.3f} -> "
              f"{purged['per_colorant'][c]:.3f}  ({-d:+.3f})")

    if args.stage2:
        from src.stage2_regressor import predict_fractions, train_stage2
        print("\nStage 2 on the purged training set (slow)...")
        X_train_p, X_test = build_feature_matrix(purged_df), build_feature_matrix(test_df)
        models = train_stage2(X_train_p, purged_df)
        true_f = test_df[COLORANT_COLS].to_numpy(dtype=float)
        active_true = true_f > 0
        pred_f = predict_fractions(models, X_test, active_true)
        err = np.abs(pred_f - true_f)
        print(f"  Stage 2 MAE all cells:    {err.mean():.4f}")
        print(f"  Stage 2 MAE active cells: {err[active_true].mean():.4f}")
        print("  (compare against train.py's reported figures on the unpurged set)")


if __name__ == "__main__":
    main()
