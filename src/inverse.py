"""Inverse direction: a target colour -> the recipe that hits it.

The deployed pipeline runs FORWARD: you already mixed and measured something,
and it tells you what is in it ("what did I make?"). This module runs the other
way ("what should I make?"), which is the question a customer actually asks:
*here is the Pantone colour I need, what do I mix?*

That used to look blocked. Answering it appeared to need a forward model
(recipe -> colour) driven by an iterative search, and this project's
Kubelka-Munk physics layer was never trustworthy enough for that. The Pantone
linkage (src/pantone.py) removes the blocker entirely: because every UFO
formula name encodes the Pantone standard it was made to match, there are
~1800 direct (target spectrum -> known recipe) pairs sitting in the two CxF
files. No forward model, no search -- ordinary supervised learning.

Two structural differences from the forward pipeline:

  1. **Single backing.** A Pantone standard has one reflectance curve, not a
     light/dark pair, so this uses build_single_spectrum_features (142 features)
     instead of build_feature_matrix (285). The `contrast` group that encodes
     opacity simply does not exist here.
  2. **Leak-safe splitting.** Adjacent Pantone codes are often near-identical
     shades (2745 C vs 2746 C). A random split would put one in train and its
     twin in test and report a flattering number. `colour_groups` below merges
     any targets within GROUP_DE of each other into a single group, and splits
     whole groups -- so a test colour never has a near-twin in training.

Honest scope: for the ~2090 Pantone codes already formulated, the right answer
is a database lookup, not a prediction. This model exists for targets that were
never formulated -- newer Pantone ranges, or arbitrary customer colours -- so
the grouped evaluation below is the number that matters, not a random-split one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .colorimetry import delta_e_2000, reflectance_to_lab
from .cxf_parser import COLORANT_NAMES, WAVELENGTHS
from .features import build_single_spectrum_features
from .pantone import target_for_formula

# Targets closer than this are treated as the same colour for splitting
# purposes. 5.0 is deliberately generous -- well beyond a commercial match
# tolerance -- because the risk being defended against (a near-twin leaking
# across the split) is worse than the cost (slightly coarser groups).
GROUP_DE = 5.0

COLORANT_COLS = [f"colorant_{c}" for c in COLORANT_NAMES]


def build_pairs(df: pd.DataFrame, library: dict) -> pd.DataFrame:
    """Assemble the (target spectrum -> recipe) supervised set.

    Takes the parsed UFO dataset and the Pantone library, and returns one row
    per formula that has BOTH a resolvable Pantone target and a valid recipe.

    Note this deliberately does NOT require both backings: the input is the
    Pantone target's single curve, so light-only formulas are perfectly usable
    here even though train.py's forward pipeline discards them.
    """
    rows = []
    for _, r in df.iterrows():
        target = target_for_formula(r["formula"], library)
        if target is None:
            continue
        recipe = r[COLORANT_COLS].to_numpy(dtype=float)
        if abs(recipe.sum() - 1.0) >= 0.05:   # same corruption gate as train.py
            continue
        rows.append({
            "formula": r["formula"],
            "pantone_code": target.code,
            "pantone_name": target.name,
            "target_R": target.reflectance,
            **{c: recipe[i] for i, c in enumerate(COLORANT_COLS)},
        })

    if not rows:
        return pd.DataFrame(columns=["formula", "pantone_code", "pantone_name",
                                     "target_R"] + COLORANT_COLS)

    pairs = pd.DataFrame(rows)
    # One row per Pantone code: a code formulated more than once would
    # otherwise appear on both sides of a split with identical input features.
    return pairs.drop_duplicates(subset="pantone_code").reset_index(drop=True)


def colour_groups(target_R: np.ndarray, threshold: float = GROUP_DE) -> np.ndarray:
    """Group indices so that any two targets within `threshold` dE00 share a group.

    Connected components of the "closer than threshold" graph, found with a
    union-find over the pairwise CIEDE2000 matrix. Returns an integer group id
    per row.
    """
    lab = reflectance_to_lab(np.asarray(target_R, dtype=float))
    n = len(lab)

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Row-blocked so the pairwise dE00 never materialises as a full n x n array
    for i in range(n):
        d = delta_e_2000(np.repeat(lab[i:i + 1], n - i - 1, axis=0), lab[i + 1:])
        for j in np.nonzero(d < threshold)[0]:
            union(i, i + 1 + int(j))

    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


def purged_split(
    pairs: pd.DataFrame,
    test_size: float = 0.15,
    threshold: float = GROUP_DE,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Random split, then PURGE training rows that sit within `threshold` of any test row.

    Why not split by connected components of the "within threshold" graph
    (`colour_groups` below)? Measured: at dE00 5.0 the Pantone library collapses
    into one component of ~2000 of 2072 targets. Single-linkage chains through a
    dense colour library -- A is near B, B near C, and so on -- so whole-group
    splitting leaves a test set of ~12 rows, far too small to conclude anything
    from.

    Purging gives the same guarantee without the collapse: the test set is a
    normal random sample, and any training colour too close to a test colour is
    simply dropped. A test target therefore never has a near-twin in training,
    and the test set stays large enough to be meaningful. The cost is the purged
    training rows, which is reported.
    """
    rng = np.random.default_rng(random_state)
    n = len(pairs)
    idx = rng.permutation(n)
    n_test = max(1, int(round(n * test_size)))
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])

    lab = reflectance_to_lab(np.stack(pairs["target_R"].to_numpy()))
    te_lab, tr_lab = lab[test_idx], lab[train_idx]

    # Nearest test colour for each candidate training row
    nearest = np.full(len(tr_lab), np.inf)
    for i in range(len(te_lab)):
        d = delta_e_2000(np.repeat(te_lab[i:i + 1], len(tr_lab), axis=0), tr_lab)
        nearest = np.minimum(nearest, d)

    keep = nearest >= threshold
    kept_idx = train_idx[keep]

    info = {
        "n_test": int(len(test_idx)),
        "n_train_before_purge": int(len(train_idx)),
        "n_purged": int((~keep).sum()),
        "n_train": int(len(kept_idx)),
        "threshold": float(threshold),
        "min_train_test_de": float(nearest[keep].min()) if keep.any() else float("nan"),
    }
    return (pairs.iloc[kept_idx].reset_index(drop=True),
            pairs.iloc[test_idx].reset_index(drop=True),
            info)


def features_for(pairs: pd.DataFrame) -> pd.DataFrame:
    """Single-backing feature matrix for a set of (target -> recipe) pairs."""
    return build_single_spectrum_features(
        np.stack(pairs["target_R"].to_numpy()), index=pairs.index
    )


def recipes_for(pairs: pd.DataFrame) -> np.ndarray:
    """The (n_samples, 14) recipe matrix."""
    return pairs[COLORANT_COLS].to_numpy(dtype=float)


def lookup_baseline(
    train_pairs: pd.DataFrame, test_pairs: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Baseline: return the recipe of the colorimetrically nearest known target.

    This is what a formulator with a spreadsheet already does, and it is the bar
    the model has to clear. Returns (predicted recipes, dE00 to the neighbour
    that supplied each one).
    """
    tr_lab = reflectance_to_lab(np.stack(train_pairs["target_R"].to_numpy()))
    te_lab = reflectance_to_lab(np.stack(test_pairs["target_R"].to_numpy()))
    tr_rec = recipes_for(train_pairs)

    pred = np.empty((len(te_lab), len(COLORANT_COLS)))
    dists = np.empty(len(te_lab))
    for i in range(len(te_lab)):
        d = delta_e_2000(np.repeat(te_lab[i:i + 1], len(tr_lab), axis=0), tr_lab)
        j = int(np.argmin(d))
        pred[i] = tr_rec[j]
        dists[i] = d[j]
    return pred, dists


def score(pred: np.ndarray, true: np.ndarray, threshold: float = 0.0) -> dict:
    """Recipe-quality metrics, matching the forward pipeline's definitions.

    exact_match: the predicted *set* of active colorants equals the true set --
    the inverse-direction analogue of the Stage 1 metric.
    """
    active_true = true > threshold
    active_pred = pred > max(threshold, 1e-9)
    err = np.abs(pred - true)
    return {
        "exact_match": float((active_true == active_pred).all(axis=1).mean()),
        "mae_all": float(err.mean()),
        "mae_active": float(err[active_true].mean()) if active_true.any() else float("nan"),
        "n": int(len(true)),
    }


def predicted_recipe_dict(pred_row: np.ndarray, min_pct: float = 1e-4) -> dict[str, float]:
    """One prediction row -> {colorant: percentage}, matching predict.predict_recipe."""
    out = {
        COLORANT_NAMES[j]: round(float(v) * 100, 2)
        for j, v in enumerate(pred_row) if v > min_pct
    }
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
