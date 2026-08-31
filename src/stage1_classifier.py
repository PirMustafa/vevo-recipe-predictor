"""Stage 1: multi-label classification of which colorants are present.

Given the feature vector (raw R + K/S + contrast + derivative, light & dark),
predict which of the 14 colorants are actually used in the recipe.

Architecture: TabICLv2 (Soda team, Inria) -- a pretrained tabular foundation
model that does in-context learning, wrapped in a ClassifierChain for
multi-label support. Replaced the earlier 4-architecture GBM stack
(HistGradientBoosting/CatBoost/LightGBM/XGBoost + logistic meta-learner +
isotonic calibration, 82.5-82.8% exact-match) after validated testing found
TabICL ahead on every metric: 87.2% exact-match, 0.9498 macro-F1, and (the
standout result) Red 032's F1 jumping from 0.67 to 0.91 -- the largest
single improvement found anywhere in this project's history, on the
colorant every other technique (including the old stack) struggled with
most.

Why TabICL specifically: this session also tested SVM, ExtraTrees, PLS-DA,
k-NN, custom deep-learning MLP/CNN, extensive hyperparameter tuning, and
TabPFN (a closely related foundation model). TabPFN matched TabICL's Red 032
result but its model weights are restricted to non-commercial use; TabICL is
BSD-3-Clause (commercially usable) and equally fast once GPU-accelerated.
See repo memory for the full ablation history.

Hard requirement: a CUDA GPU. TabICL does inference-time computation over
the ENTIRE training set as context on every predict() call (that's what
"in-context learning" means -- there's no separate training phase that
bakes knowledge into fixed weights the way a GBM's tree splits do). On this
project's CPU-only environment, a single prediction took ~26 minutes; on a
GPU (RTX 5060 Laptop, sm_120), the same prediction takes ~87 seconds. Still
not instant -- callers (the web app) need to show a loading state, not
assume a fast response.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.multioutput import ClassifierChain
from sklearn.utils.validation import validate_data
from tabicl import TabICLClassifier

from .cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES

# ACTIVE_THRESHOLD lives in cxf_parser (the dataset-vocabulary module) so that
# CPU-only consumers -- train_inverse.py, src/evaluation.py -- can read it
# without importing this module and, through it, the GPU-only tabicl package.
# Re-exported here because this is where it is defined in the README and where
# callers reasonably look for it.
__all__ = ["ACTIVE_THRESHOLD", "build_labels", "fit_stage1", "load_stage1",
           "predict_active", "save_stage1"]


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Binary presence labels (fraction > ACTIVE_THRESHOLD) for each colorant."""
    colorant_cols = [f"colorant_{c}" for c in COLORANT_NAMES]
    return (df[colorant_cols] > ACTIVE_THRESHOLD).astype(int)


class TabICLStage1:
    """Wraps a fitted ClassifierChain(TabICLClassifier). Because TabICL is
    an in-context learner, the "fitted" chain retains the training data
    itself and re-processes it on every predict_proba() call -- that's why
    inference is slow regardless of how many rows are being predicted (one
    row costs almost as much as a few hundred)."""

    def __init__(self, chain: ClassifierChain):
        self.chain = chain
        # Per-colorant probability thresholds; None -> predict_active uses a
        # flat 0.5 (matches the old architecture's convention -- direct
        # threshold tuning against exact-match was tried and hurt
        # generalization, see README, so this stays unset by default).
        self.thresholds: np.ndarray | None = None

    def predict_proba_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Chain probabilities, decoded in ONE forward pass per link.

        sklearn's ClassifierChain._get_predictions calls each link's estimator
        TWICE on the same augmented input: once with chain_method="predict" to
        get the hard 0/1 it feeds forward, and once with predict_proba for the
        value it reports. TabICL's predict() is literally
        `argmax(self.predict_proba(X))` -- it recomputes the identical forward
        pass and discards the probabilities.

        So a 14-link chain runs 28 forward passes where 14 suffice, and each
        pass re-processes all 2051 training rows because TabICL is an
        in-context learner. Deriving the hard label from the probabilities we
        already have halves Stage 1 latency.

        Measured on the deployed light model: a single spectrum goes from
        62.2s to 29.9s (2.08x), and 297 spectra from 83.5s to 41.2s, with the
        output BIT-IDENTICAL (max abs difference 0.0, same predicted sets).

        Falls back to sklearn's own path if the chain does not look the way
        this decoding assumes, so an sklearn upgrade cannot silently change
        results -- it would only cost back the speed.
        """
        chain = self.chain
        ests = getattr(chain, "estimators_", None)
        order = getattr(chain, "order_", None)
        if not ests or order is None or not all(hasattr(e, "classes_") for e in ests):
            return chain.predict_proba(X)

        Xv = validate_data(chain, X, accept_sparse=True, reset=False)
        n, k = Xv.shape[0], len(ests)
        feature_chain = np.zeros((n, k))   # hard labels, in CHAIN order
        output_chain = np.zeros((n, k))    # probabilities, in CHAIN order
        for j, est in enumerate(ests):
            # Feature layout must match fit time exactly: X followed by the
            # previous links' HARD labels, in the chain's permuted order.
            # Appending them in original label order, or feeding soft
            # probabilities forward, both produce plausible-looking output
            # that is quietly wrong.
            X_aug = np.hstack((Xv, feature_chain[:, :j]))
            proba = est.predict_proba(X_aug)
            output_chain[:, j] = proba[:, 1]
            feature_chain[:, j] = est.classes_[np.argmax(proba, axis=1)]

        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(k)
        return output_chain[:, inv_order]   # back to original label order


def train_stage1(X: pd.DataFrame, Y: pd.DataFrame) -> TabICLStage1:
    print("  Fitting TabICL ClassifierChain (in-context learning, GPU)...")
    base = TabICLClassifier(device="cuda")
    chain = ClassifierChain(base, order="random", random_state=0)
    chain.fit(X, Y)
    return TabICLStage1(chain)


def predict_active(
    model: TabICLStage1, X: pd.DataFrame, threshold: float | np.ndarray | None = None
) -> np.ndarray:
    """Return a boolean (n_samples, n_colorants) array of predicted-active colorants.

    threshold: scalar, per-colorant array, or None (default) to use the
        model's calibrated thresholds if any were set, falling back to a
        flat 0.5 otherwise.

    Falls back to the single most-probable colorant if none clear the
    threshold, since every real recipe uses at least one colorant.
    """
    proba = model.predict_proba_matrix(X)  # (n_samples, n_colorants)
    if threshold is None:
        threshold = model.thresholds if getattr(model, "thresholds", None) is not None else 0.5
    thresh_arr = np.full(proba.shape[1], threshold) if np.isscalar(threshold) else np.asarray(threshold)
    active = proba >= thresh_arr[None, :]

    no_active = ~active.any(axis=1)
    if no_active.any():
        fallback_idx = proba[no_active].argmax(axis=1)
        rows = np.where(no_active)[0]
        active[rows, fallback_idx] = True

    return active


def save_stage1(model: TabICLStage1, path: str) -> None:
    joblib.dump(model, path)


def load_stage1(path: str) -> TabICLStage1:
    return joblib.load(path)
