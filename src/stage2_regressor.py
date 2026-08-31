"""Stage 2: concentration estimation for colorants flagged active by Stage 1.

Four families of model compete per colorant, and whichever cross-validates
better wins:

  1. A per-colorant regressor (trained only on rows where that colorant is
     active), model TYPE chosen via CV among a few candidates spanning a
     range of capacity/regularization. Diagnostics showed a single fixed
     gradient-boosted-tree regressor badly overfits colorants with very few
     active training examples (R^2 as low as -2.99 for one with only 6
     active rows), while doing fine (R^2 0.9+) on well-supported ones.
  2. A masked-softmax joint neural net (src/stage2_joint.py), trained ONCE
     on every formula and predicting all 14 fractions together from a
     shared hidden representation, with the softmax guaranteeing the output
     sums to 1 over exactly the Stage-1-active colorants by construction.
     Sharing a representation lets data-starved colorants borrow signal
     from the well-supported ones instead of being fit in isolation.
  3. A joint ILR-transform regressor (src/stage2_joint.py): fractions are
     mapped to ILR (isometric log-ratio) coordinates -- the standard
     statistical tool for regressing on "parts of a whole" data -- and a
     gradient-boosted model predicts those coordinates; the inverse
     transform (a softmax) maps back to a valid composition. Also a joint,
     whole-composition model, but a classical-statistics take rather than a
     neural net.
  4. TabICLv2's regressor (GPU, per-colorant, active rows only -- the same
     foundation model used for Stage 1's classifier). Validated: overall
     MAE 0.0085 vs 0.0129 for masked-softmax (~34% better) on 13 of 14
     colorants, but WORSE specifically on Red 032 (0.0671 vs ~0.028) --
     as an isolated per-colorant model it doesn't get the cross-colorant
     information-sharing that lets masked-softmax borrow signal for
     data-starved colorants (see family 2 above). Added as a 4th candidate
     rather than a replacement precisely so CV keeps masked-softmax where
     it's genuinely better, matching how per-colorant vs. joint models
     already compete below.

Why four: a single fixed regressor (the original approach) badly overfits
sparse colorants. The masked-softmax MLP fixes the sparse cases but is
worse in aggregate than per-colorant selection on the well-supported ones.
TabICL wins big on well-supported colorants but loses on the most
data-starved one. No single family wins everywhere, so cross-validation
picks per colorant rather than committing globally to one.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tabicl import TabICLRegressor

from .cxf_parser import COLORANT_NAMES
from .stage2_joint import (
    cv_ilr,
    cv_masked_softmax,
    fit_ilr,
    fit_masked_softmax,
    ilr_basis,
    predict_ilr,
    predict_masked_softmax,
)

# Below this many active examples, skip regression entirely (equal-share
# fallback at predict time) -- not enough data to fit or even cross-validate
# any model meaningfully.
_MIN_SAMPLES = 5


def _candidate_models(n_samples: int, n_features: int) -> dict[str, object]:
    # Leave slack below n_samples for the smaller folds cross_val_score will
    # actually fit on (worst case: Leave-One-Out, n_samples - 1 per fold).
    n_components = max(1, min(5, n_samples - 3, n_features))
    return {
        "mean": DummyRegressor(strategy="mean"),
        "ridge_1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "pls": make_pipeline(StandardScaler(), PLSRegression(n_components=n_components)),
        "gbr": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.08, max_depth=6, random_state=42
        ),
    }


def _select_model(X: pd.DataFrame, y: pd.Series) -> tuple[object | None, str | None, float]:
    """Cross-validated pick among a few model types; returns the winner refit
    on all of (X, y), plus its CV MAE (for comparison against the joint
    candidate families)."""
    n = len(y)
    cv = LeaveOneOut() if n < 15 else KFold(n_splits=5, shuffle=True, random_state=0)
    candidates = _candidate_models(n, X.shape[1])

    best_name, best_score, best_model = None, -np.inf, None
    for name, model in candidates.items():
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")
        except Exception:
            continue
        score = scores.mean()
        if score > best_score:
            best_name, best_score, best_model = name, score, model

    if best_model is None:
        return None, None, np.inf
    best_model.fit(X, y)
    return best_model, best_name, -best_score


def _tabicl_oof_mae(X: pd.DataFrame, y: pd.Series) -> tuple[object, float]:
    """Honest CV MAE for TabICLRegressor on one colorant's active rows, using
    the same CV protocol as _select_model, then refit on all rows for the
    deployable model."""
    n = len(y)
    cv = LeaveOneOut() if n < 15 else KFold(n_splits=5, shuffle=True, random_state=0)
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    abs_errors = []
    for train_idx, test_idx in cv.split(X_arr):
        model = TabICLRegressor(device="cuda")
        model.fit(X_arr[train_idx], y_arr[train_idx])
        pred = np.asarray(model.predict(X_arr[test_idx])).reshape(-1)
        abs_errors.extend(np.abs(pred - y_arr[test_idx]).tolist())
    mae = float(np.mean(abs_errors))

    final_model = TabICLRegressor(device="cuda")
    final_model.fit(X_arr, y_arr)
    return final_model, mae


def train_stage2(X: pd.DataFrame, df: pd.DataFrame) -> dict[str, object]:
    """Train Stage 2: per-colorant candidates + two joint (whole-composition)
    models + TabICL, picking whichever wins per colorant by cross-validated MAE."""
    colorant_cols = [f"colorant_{c}" for c in COLORANT_NAMES]
    Y_full = df[colorant_cols].to_numpy(dtype=float)
    active_mask_full = (Y_full > 0).astype(float)
    X_full = X.to_numpy(dtype=float)
    X_aug = np.hstack([X_full, active_mask_full])

    print("  Fitting joint masked-softmax MLP + ILR regressor (cross-validated)...")
    basis = ilr_basis(len(COLORANT_NAMES))

    # NOTE: masked-softmax gets plain X_full (not X_aug) -- its forward()
    # already concatenates the mask internally (see stage2_joint.py), so
    # passing the mask pre-concatenated here would feed it in twice, a
    # silent architecture bug that was making the deployed model bigger and
    # less well-behaved than anything validated in isolation. ILR has no
    # internal mask handling, so it still needs X_aug.
    softmax_oof = cv_masked_softmax(X_full, active_mask_full, Y_full)
    softmax_model = fit_masked_softmax(X_full, active_mask_full, Y_full)

    ilr_oof = cv_ilr(X_aug, Y_full, basis)
    ilr_model = fit_ilr(X_aug, Y_full, basis)

    per_colorant_models: dict[str, object] = {}
    tabicl_models: dict[str, object] = {}
    family: dict[str, str] = {}
    chosen: dict[str, str] = {}

    for j, cname in enumerate(COLORANT_NAMES):
        col = f"colorant_{cname}"
        mask = df[col].to_numpy(dtype=float) > 0
        if mask.sum() < _MIN_SAMPLES:
            per_colorant_models[cname] = None
            family[cname] = "per_colorant"
            continue

        model, name, per_colorant_mae = _select_model(X[mask], df.loc[mask, col])
        softmax_mae = np.abs(softmax_oof[mask, j] - Y_full[mask, j]).mean()
        ilr_mae = np.abs(ilr_oof[mask, j] - Y_full[mask, j]).mean()
        tabicl_model, tabicl_mae = _tabicl_oof_mae(X[mask], df.loc[mask, col])

        scores = {
            "per_colorant": per_colorant_mae,
            "joint_softmax": softmax_mae,
            "joint_ilr": ilr_mae,
            "tabicl": tabicl_mae,
        }
        winner = min(scores, key=scores.get)
        family[cname] = winner
        per_colorant_models[cname] = model if winner == "per_colorant" else None
        tabicl_models[cname] = tabicl_model if winner == "tabicl" else None
        chosen[cname] = (
            f"{winner} (per_colorant={per_colorant_mae:.4f}[{name}], "
            f"softmax={softmax_mae:.4f}, ilr={ilr_mae:.4f}, tabicl={tabicl_mae:.4f})"
        )

    print(f"  Stage 2 model choice per colorant: {chosen}")
    return {
        "per_colorant": per_colorant_models,
        "tabicl": tabicl_models,
        "family": family,
        "softmax_model": softmax_model,
        "ilr_model": ilr_model,
        "colorant_names": list(COLORANT_NAMES),
    }


def predict_fractions(
    models: dict[str, object],
    X: pd.DataFrame,
    active: np.ndarray,
) -> np.ndarray:
    """Predict normalized fractions (sum to 1) for the active colorants per row.

    active: boolean (n_samples, n_colorants) array from Stage 1.
    Returns: (n_samples, n_colorants) array, zero for inactive colorants.
    """
    colorant_names = models["colorant_names"]
    per_colorant_models = models["per_colorant"]
    tabicl_models = models.get("tabicl", {})
    family = models["family"]

    n_samples = len(X)
    n_colorants = len(colorant_names)
    raw = np.zeros((n_samples, n_colorants), dtype=float)

    X_full = None  # computed lazily, only if a joint family is actually needed
    X_aug = None
    softmax_pred = None
    ilr_pred = None

    for j, cname in enumerate(colorant_names):
        rows = np.where(active[:, j])[0]
        if len(rows) == 0:
            continue

        fam = family.get(cname, "per_colorant")
        if fam in ("joint_softmax", "joint_ilr"):
            if X_full is None:
                X_full = X.to_numpy(dtype=float)
            if fam == "joint_softmax" and softmax_pred is None:
                # plain X_full, not X_aug -- see the matching note in train_stage2
                softmax_pred = predict_masked_softmax(models["softmax_model"], X_full, active.astype(float))
            if fam == "joint_ilr" and ilr_pred is None:
                if X_aug is None:
                    X_aug = np.hstack([X_full, active.astype(float)])
                ilr_pred = predict_ilr(models["ilr_model"], X_aug)
            pred = softmax_pred if fam == "joint_softmax" else ilr_pred
            raw[rows, j] = np.clip(pred[rows, j], 0.0, None)
            continue

        if fam == "tabicl":
            model = tabicl_models.get(cname)
            if model is not None:
                X_rows = X.iloc[rows].to_numpy(dtype=float)
                preds = np.asarray(model.predict(X_rows)).reshape(-1)
                raw[rows, j] = np.clip(preds, 0.0, None)
            else:
                raw[rows, j] = 1.0
            continue

        model = per_colorant_models.get(cname)
        if model is None:
            raw[rows, j] = 1.0  # equal-share heuristic, normalized below
        else:
            preds = np.asarray(model.predict(X.iloc[rows])).reshape(-1)
            raw[rows, j] = np.clip(preds, 0.0, None)

    # Normalize each row over its active colorants to sum to 1.
    row_sums = raw.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    fractions = raw / safe_sums

    # Rows where every active prediction collapsed to zero: equal-split fallback.
    zero_rows = (row_sums.flatten() == 0)
    if zero_rows.any():
        for i in np.where(zero_rows)[0]:
            active_idx = np.where(active[i])[0]
            fractions[i, active_idx] = 1.0 / len(active_idx)

    return fractions


def save_stage2(models: dict, path: str) -> None:
    joblib.dump(models, path)


def load_stage2(path: str) -> dict:
    return joblib.load(path)
