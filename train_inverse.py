"""Train & evaluate the inverse model: target colour -> recipe.

    python train_inverse.py            # sklearn stack, CPU, ~seconds
    python train_inverse.py --tabicl   # TabICL stack, needs a CUDA GPU, slow

The forward pipeline (train.py) answers "what did I make?". This answers "what
should I make?" -- see src/inverse.py for why the Pantone linkage makes that an
ordinary supervised problem rather than a search over a physics model.

Evaluation is deliberately pessimistic. Splitting is by colour GROUP, not by
row, so a test target never has a near-identical twin in training (adjacent
Pantone codes are often the same shade). Every result is reported against a
nearest-neighbour lookup baseline, because for any colour already in the
library, lookup is what a formulator would actually do -- the model only earns
its place on colours that are NOT already formulated.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from src.cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES, load_dataset
from src.inverse import (
    COLORANT_COLS,
    build_pairs,
    features_for,
    lookup_baseline,
    predicted_recipe_dict,
    purged_split,
    recipes_for,
    score,
)
from src.pantone import load_pantone_library

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"
MODELS_DIR = ROOT / "models"

# Fractions below this are treated as "not in the recipe" when reconstructing a
# recipe from the regressor. Imported rather than redeclared: this must mean the
# same thing as Stage 1's presence cutoff, and a second literal would let the two
# drift apart silently the first time either is retuned.


def _fit_sklearn(X_train, Y_train, R_train):
    """Two-stage sklearn stack: presence classifier, then fraction regressor.

    Mirrors the forward pipeline's shape (classify which colorants, then how
    much) with CPU models, so the whole experiment runs in seconds and the
    plumbing can be validated without waiting on a GPU.
    """
    presence = []
    for j in range(len(COLORANT_NAMES)):
        y = (R_train[:, j] > 0).astype(int)
        if y.min() == y.max():
            presence.append(("constant", int(y[0])))    # colorant never/always used
            continue
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                             random_state=42)
        clf.fit(X_train, y)
        presence.append(("model", clf))

    amount = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
    )
    amount.fit(X_train, R_train)
    return presence, amount


def _predict_sklearn(model, X):
    presence, amount = model
    active = np.zeros((len(X), len(COLORANT_NAMES)), dtype=bool)
    for j, (kind, obj) in enumerate(presence):
        active[:, j] = bool(obj) if kind == "constant" else obj.predict(X).astype(bool)

    frac = np.clip(amount.predict(X), 0.0, None)
    frac[~active] = 0.0
    frac[frac < ACTIVE_THRESHOLD] = 0.0
    # Recipes are compositional: renormalise so each row sums to 1.0
    totals = frac.sum(axis=1, keepdims=True)
    return np.divide(frac, totals, out=np.zeros_like(frac), where=totals > 0)


def _fit_tabicl(X_train, Y_train, R_train):
    from src.stage1_classifier import train_stage1
    import pandas as pd
    Y = pd.DataFrame((R_train > 0).astype(int), columns=COLORANT_NAMES, index=X_train.index)
    stage1 = train_stage1(X_train, Y)
    amount = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42)
    )
    amount.fit(X_train, R_train)
    return stage1, amount


def _predict_tabicl(model, X):
    from src.stage1_classifier import predict_active
    stage1, amount = model
    active = predict_active(stage1, X).astype(bool)
    frac = np.clip(amount.predict(X), 0.0, None)
    frac[~active] = 0.0
    frac[frac < ACTIVE_THRESHOLD] = 0.0
    totals = frac.sum(axis=1, keepdims=True)
    return np.divide(frac, totals, out=np.zeros_like(frac), where=totals > 0)


def _report(name, metrics, extra=""):
    print(f"  {name:<26} exact-match {metrics['exact_match']:7.2%}   "
          f"MAE(all) {metrics['mae_all']:.4f}   MAE(active) {metrics['mae_active']:.4f}{extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the target-colour -> recipe model")
    ap.add_argument("--tabicl", action="store_true",
                    help="use the TabICL classifier for colorant presence (needs a CUDA GPU)")
    ap.add_argument("--save", action="store_true", help="save the fitted model to models/")
    ap.add_argument("--purge-de", type=float, default=None,
                    help="dE00 buffer between train and test colours (default: inverse.GROUP_DE). "
                         "0 disables purging -- optimistic, near-twins may leak")
    args = ap.parse_args()

    print("Loading CxF data and Pantone library...")
    library = load_pantone_library(PANTONE_PATH)
    if not library:
        raise SystemExit(f"Pantone library not found at {PANTONE_PATH}")
    df = load_dataset(str(CXF_PATH))

    pairs = build_pairs(df, library)
    print(f"  UFO formulas parsed:                {len(df)}")
    print(f"  Pantone standards:                  {len(library)}")
    print(f"  (target -> recipe) supervised pairs: {len(pairs)}")
    both = int((df['has_light'] & df['has_dark']).sum())
    print(f"  note: the forward pipeline can only use {both} both-backing formulas;")
    print(f"        this direction needs no backing pair, so it uses more.")

    from src.inverse import GROUP_DE
    purge_de = GROUP_DE if args.purge_de is None else args.purge_de
    train_pairs, test_pairs, info = purged_split(pairs, threshold=purge_de)
    print(f"\nLeak-safe purged split (no train colour within dE00 "
          f"{info['threshold']} of any test colour):")
    print(f"  test pairs:            {info['n_test']}")
    print(f"  train pairs:           {info['n_train']} "
          f"({info['n_purged']} purged as too close to a test colour, "
          f"from {info['n_train_before_purge']})")
    print(f"  closest surviving train-test pair: dE00 {info['min_train_test_de']:.2f}")

    X_train, X_test = features_for(train_pairs), features_for(test_pairs)
    R_train, R_test = recipes_for(train_pairs), recipes_for(test_pairs)
    print(f"  features per sample: {X_train.shape[1]} (single-backing)")

    print("\nResults on held-out colour groups:")

    base_pred, base_d = lookup_baseline(train_pairs, test_pairs)
    _report("nearest-lookup baseline", score(base_pred, R_test),
            f"   median dE00 to neighbour {np.median(base_d):.2f}")

    if args.tabicl:
        print("  [TabICL] fitting presence classifier on GPU, this is slow...")
        model = _fit_tabicl(X_train, None, R_train)
        pred = _predict_tabicl(model, X_test)
        label = "inverse model (TabICL)"
    else:
        model = _fit_sklearn(X_train, None, R_train)
        pred = _predict_sklearn(model, X_test)
        label = "inverse model (sklearn)"

    metrics = score(pred, R_test)
    _report(label, metrics)

    gain = metrics["exact_match"] - score(base_pred, R_test)["exact_match"]
    print(f"\n  exact-match vs baseline: {gain:+.2%}")
    mae_base = score(base_pred, R_test)["mae_active"]
    print(f"  MAE(active) vs baseline: {metrics['mae_active']:.4f} vs {mae_base:.4f} "
          f"({'better' if metrics['mae_active'] < mae_base else 'WORSE'})")

    print("\nPer-colorant presence recall on held-out groups:")
    print(f"  {'colorant':<32} {'n_test':>6} {'recall':>8}")
    for j, c in enumerate(COLORANT_NAMES):
        t = R_test[:, j] > 0
        if t.sum() == 0:
            continue
        rec = float(((pred[:, j] > 0) & t).sum() / t.sum())
        print(f"  {c:<32} {int(t.sum()):>6} {rec:>8.2%}")

    print("\nExample predictions (held-out targets):")
    for i in range(min(4, len(test_pairs))):
        print(f"  {test_pairs['pantone_name'][i]}")
        print(f"    true: {predicted_recipe_dict(R_test[i])}")
        print(f"    pred: {predicted_recipe_dict(pred[i])}")

    if args.save:
        import joblib
        MODELS_DIR.mkdir(exist_ok=True)
        out = MODELS_DIR / ("inverse_tabicl.joblib" if args.tabicl else "inverse_sklearn.joblib")
        joblib.dump({"model": model, "kind": "tabicl" if args.tabicl else "sklearn"}, out)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
