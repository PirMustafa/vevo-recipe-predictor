"""Two "whole-composition" Stage 2 approaches, competing against the
per-colorant specialists in stage2_regressor.py: a masked-softmax neural net
and an ILR (isometric log-ratio) transform regressor. Both guarantee a valid
composition (every fraction >= 0, all summing to 1) *by construction*,
rather than via the clip-then-renormalize post-processing the rest of the
pipeline relies on.

Masked-softmax MLP: a small feed-forward net (PyTorch) takes the spectral
features plus the Stage-1 active-colorant mask, and outputs raw scores for
all 14 colorants. Before the final softmax, inactive colorants' scores are
forced to -inf, so the softmax can only place probability mass on the
active ones -- the output is a normalized composition over exactly the
active set with no extra masking step needed afterward.

ILR regressor: fractions are first transformed into ILR coordinates (a
(D-1)-dimensional, unconstrained real-valued representation of a D-part
composition -- the standard tool for regressing on "parts of a whole" data,
since ordinary regression on raw percentages doesn't respect the sum-to-one
constraint). A gradient-boosted regressor predicts the ILR coordinates from
the spectral features, and the inverse ILR transform (which is just a
softmax) maps back to a valid composition. Zeros (inactive colorants) are
handled via the standard "multiplicative replacement" trick (substitute a
small epsilon before taking logs) since ILR requires strictly positive
parts.

Both are trained ONCE on the full active-set-aware input (spectral features
+ Stage-1 mask) and evaluated via out-of-fold cross-validation predictions,
for a fair per-colorant comparison against the per-colorant specialists in
stage2_regressor.py -- see train_stage2() there for how the three families
compete.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Masked-softmax joint MLP
# ---------------------------------------------------------------------------


# Three hidden layers (128-128-64), up from the original two (128-64): a
# width/depth/lr/weight-decay sweep found this the only variant that
# improved held-out overall MAE (0.0133->0.0123 in a 3-seed-ensemble
# comparison) without also hurting Red 032 -- wider, higher-lr, and
# lower-weight-decay variants all won narrowly on overall MAE but noticeably
# overfit the data-starved colorants in exchange. See repo memory/README.
class _MaskedSoftmaxNet(nn.Module):
    def __init__(self, n_features: int, n_colorants: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features + n_colorants, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_colorants),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([x, mask], dim=1))
        logits = logits.masked_fill(mask < 0.5, -1e9)
        return torch.softmax(logits, dim=1)


_MAPE_EPS = 1e-3  # denominator floor -- without it, near-zero true values make
                  # relative error explode and swamp the gradient from everything else
_MAPE_WEIGHT = 0.1  # see README: swept 0.0-0.5, benefit plateaus by ~0.1 with
                    # dominant-colorant cost staying flat across the whole range


def _mape_like_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Relative error over active (masked) cells only -- inactive cells are
    already forced to exactly 0 by the softmax mask, so dividing by their
    (zero) true value would be meaningless noise, not signal."""
    denom = torch.clamp(target, min=_MAPE_EPS)
    rel_err = torch.abs(pred - target) / denom
    return (rel_err * mask).sum() / mask.sum().clamp(min=1.0)


def _train_masked_softmax(
    X: np.ndarray, mask: np.ndarray, Y: np.ndarray,
    n_epochs: int = 300, patience: int = 20, seed: int = 0,
) -> tuple[_MaskedSoftmaxNet, StandardScaler]:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    n = len(X)
    n_val = max(1, int(0.15 * n))
    perm = rng.permutation(n)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X)

    model = _MaskedSoftmaxNet(X.shape[1], Y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    l1_loss = nn.L1Loss()

    def training_loss(pred: torch.Tensor, target: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # MAE stays the primary term (dominant colorants keep driving most of
        # the gradient); the MAPE term is a smaller correction that gives
        # trace colorants enough weight to matter without letting them
        # dominate -- see README for the lambda sweep that picked 0.1.
        return l1_loss(pred, target) + _MAPE_WEIGHT * _mape_like_loss(pred, target, m)

    Xt = torch.tensor(Xs, dtype=torch.float32)
    Mt = torch.tensor(mask, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)

    best_val, best_state, bad_epochs = np.inf, None, 0
    batch_size = 64
    for _epoch in range(n_epochs):
        model.train()
        order = rng.permutation(len(train_idx))
        for start in range(0, len(order), batch_size):
            batch = train_idx[order[start : start + batch_size]]
            opt.zero_grad()
            pred = model(Xt[batch], Mt[batch])
            loss = training_loss(pred, Yt[batch], Mt[batch])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xt[val_idx], Mt[val_idx])
            # Validate on plain MAE (not the blended loss) so early stopping
            # optimizes the metric we actually report, not the training proxy.
            val_loss = l1_loss(val_pred, Yt[val_idx]).item()
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def _predict_masked_softmax(model: _MaskedSoftmaxNet, scaler: StandardScaler, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X)
    with torch.no_grad():
        pred = model(torch.tensor(Xs, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32))
    return pred.numpy()


# Validated: averaging N_ENSEMBLE_SEEDS independently-initialized models cut
# held-out overall MAE ~20% (0.0156->0.0126 at 5 seeds; diminishing returns
# beyond that) by reducing variance from random init/early-stopping -- a
# genuine accuracy gain for the well-supported colorants. It does NOT help
# (and mildly hurts) Red 032, which is expected: its problem is bias from
# too little distinguishing information, not variance, so averaging several
# models that share the same information gap doesn't cancel it out the way
# it does elsewhere. Accepted anyway -- Red 032 already gets routed to
# manual review regardless of its own R^2. See repo memory/README.
N_ENSEMBLE_SEEDS = 5


def fit_masked_softmax(X: np.ndarray, mask: np.ndarray, Y: np.ndarray) -> dict:
    members = [_train_masked_softmax(X, mask, Y, seed=s) for s in range(N_ENSEMBLE_SEEDS)]
    return {"members": members}


def predict_masked_softmax(fitted: dict, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preds = [_predict_masked_softmax(model, scaler, X, mask) for model, scaler in fitted["members"]]
    return np.mean(preds, axis=0)


def cv_masked_softmax(X: np.ndarray, mask: np.ndarray, Y: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions, for an honest per-colorant CV comparison
    against the other Stage 2 candidate families."""
    oof = np.zeros_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_i, (tr, va) in enumerate(kf.split(X)):
        members = [
            _train_masked_softmax(X[tr], mask[tr], Y[tr], seed=seed + fold_i * N_ENSEMBLE_SEEDS + s)
            for s in range(N_ENSEMBLE_SEEDS)
        ]
        preds = [_predict_masked_softmax(model, scaler, X[va], mask[va]) for model, scaler in members]
        oof[va] = np.mean(preds, axis=0)
    return oof


# ---------------------------------------------------------------------------
# ILR (isometric log-ratio) transform regressor
# ---------------------------------------------------------------------------

_ILR_EPS = 1e-6


def ilr_basis(d: int) -> np.ndarray:
    """Standard sequential (Helmert-style) ILR orthonormal basis, shape (d-1, d)."""
    v = np.zeros((d - 1, d))
    for k in range(1, d):
        v[k - 1, :k] = 1.0 / np.sqrt(k * (k + 1))
        v[k - 1, k] = -np.sqrt(k / (k + 1))
    return v


def _closure(x: np.ndarray, eps: float = _ILR_EPS) -> np.ndarray:
    """Multiplicative zero-replacement + renormalize onto the open simplex
    (every part > 0, sums to 1) -- ILR requires strictly positive parts, but
    most cells here are exact zeros (inactive colorants)."""
    x = np.clip(x, eps, None)
    return x / x.sum(axis=-1, keepdims=True)


def forward_ilr(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    x = _closure(x)
    logx = np.log(x)
    clr = logx - logx.mean(axis=-1, keepdims=True)
    return clr @ basis.T


def inverse_ilr(ilr_coords: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Inverse ILR is exactly a softmax over the reconstructed CLR vector --
    this is what guarantees the output is always a valid composition
    (positive, sums to 1) no matter what the regressor predicts."""
    clr = ilr_coords @ basis
    ex = np.exp(clr - clr.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)


def _make_ilr_regressor() -> MultiOutputRegressor:
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, max_depth=6, random_state=42)
    )


def fit_ilr(X: np.ndarray, Y: np.ndarray, basis: np.ndarray) -> dict:
    targets = forward_ilr(Y, basis)
    reg = _make_ilr_regressor()
    reg.fit(X, targets)
    return {"model": reg, "basis": basis}


def predict_ilr(fitted: dict, X: np.ndarray) -> np.ndarray:
    coords = fitted["model"].predict(X)
    return inverse_ilr(coords, fitted["basis"])


def cv_ilr(X: np.ndarray, Y: np.ndarray, basis: np.ndarray, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    oof = np.zeros_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        fitted = fit_ilr(X[tr], Y[tr], basis)
        oof[va] = predict_ilr(fitted, X[va])
    return oof
