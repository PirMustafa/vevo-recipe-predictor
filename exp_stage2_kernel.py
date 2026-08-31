"""Does a KERNEL method beat Stage 2's incumbent per-colorant ensemble?

Stage 2 predicts HOW MUCH of each colorant, given which are active. Its error
(test MAE 0.0090 on active cells, median 0.38 pp) is strongly tail-dominated
and costs roughly 2.98 dE against a client target of 1.0 dE, so it is the main
remaining lever on this project.

THE LEAD
--------
On the forward model built in exp_forward_kernel.py -- same project, same
feature construction, 2051 x 28 regression -- an MLP scored mean dE 1.667 and
kernel ridge (RBF, log-transformed target) scored 0.993. A 40% error reduction
purely from model class. Kernel/GP methods have never been tried for Stage 2,
and 2051 rows x 142 features is kernel territory, not neural-net territory.

THE COUNTER-EVIDENCE
--------------------
README.md:817 records SVM-RBF failing badly as STAGE 1's classifier on these
same 142 engineered features, attributed to distance degradation in high
dimensions. So PCA-reduced variants (20/40/80 components) are searched
alongside the plain kernel, with the PCA fitted INSIDE each CV fold so it
cannot leak. If plain KRR loses and PCA-KRR wins, that confirms the mechanism
rather than refuting the lead.

PROTOCOL (fixed before any result was seen)
-------------------------------------------
  * production_partition() -> train 1672 / test 297 / light_only 379.
  * Pool = training_frame_for_mode(MODE_LIGHT, train_df, light_only),
    2051 rows x 142 features -- exactly what production builds.
  * Per colorant, fit on ACTIVE rows only (df[col] > 0), matching
    src/stage2_regressor.py:167.
  * ALL model selection by CV on the 2051-row pool, using the SAME CV object
    as _select_model (src/stage2_regressor.py:95): LeaveOneOut if n < 15 else
    KFold(5, shuffle=True, random_state=0). Every colorant here has n >= 31,
    so in practice every fit uses KFold(5).
  * The INCUMBENT is re-measured under the identical protocol by calling its
    own functions (_select_model, _tabicl_oof_mae, cv_masked_softmax, cv_ilr),
    so the comparison is same-protocol rather than against a quoted number.
  * The 297 test rows are scored ONCE, at the very end.

PRE-COMMITTED THRESHOLDS
------------------------
  * Adopt the kernel PER COLORANT only where its CV MAE beats the best
    incumbent family by >= 5% relative.
  * Overall success: pooled CV MAE improves >= 8% relative, with no colorant
    of n_active >= 50 regressing.
  * If CV and the held-out test disagree, THE CV RESULT STANDS. The test
    number is reported as-is. This project has three documented failures from
    doing it the other way (README:810, :824).

Red 032 has 31 pool / 6 test examples. Any movement there is noise unless CV
moves with it.

This script writes NOTHING. It does not touch models/ and does not modify any
existing file.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.backing_modes import MODE_LIGHT, training_frame_for_mode
from src.cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES
from src.stage2_joint import (
    cv_ilr,
    cv_masked_softmax,
    fit_ilr,
    fit_masked_softmax,
    ilr_basis,
    predict_ilr,
    predict_masked_softmax,
)
from src.stage2_regressor import _candidate_models, _select_model, _tabicl_oof_mae
from train_backing_modes import production_partition

warnings.filterwarnings("ignore")

# --- pre-committed search grid -------------------------------------------
# gamma is quoted as a multiple of 1/n_features, sklearn's "scale" convention.
# For a PCA representation, n_features is that representation's own
# dimensionality (20/40/80), not 142 -- squared distances shrink with the
# number of retained components, so reusing 142 would search a systematically
# wrong band of length scales for the very variants the counter-evidence says
# are most likely to win. Stated here rather than discovered later.
ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0]
GAMMA_MULTS = [0.1, 0.3, 1.0, 3.0, 10.0]
PCA_COMPONENTS = [20, 40, 80]

# DECLARED DEVIATION, and why it is not goalpost-moving.
# A smoke test on three colorants put the CV optimum on the LOWER boundary of
# both grids (alpha=1e-3, gamma_mult=0.1) every time, and exp_forward_kernel.py
# -- the run that motivated this whole experiment -- settled at alpha=1e-4,
# which the pre-committed grid cannot even reach. An optimum pinned to a grid
# edge is the standard diagnostic that the grid is mis-centred, so the
# pre-committed result is a LOWER bound on what this model class can do.
# The extension is therefore searched too, but it is NEVER substituted for the
# pre-committed result: both verdicts are printed side by side, the
# pre-committed grid is the decision, and the extension is labelled
# exploratory. The union grid is scored in a single pass, so the pre-committed
# numbers are literally a subset of the same computation and cannot drift.
EXT_ALPHAS = [1e-6, 1e-5, 1e-4] + ALPHAS
EXT_GAMMA_MULTS = [0.01, 0.03] + GAMMA_MULTS

# Target treatments. All four share one Cholesky factorization per
# (fold, representation, gamma, alpha), so the centred and log variants are
# essentially free.
#   y        exactly sklearn's KernelRidge: solve (K + aI)d = y, no intercept.
#   y_ctr    same, on y - mean(y_train); the mean is added back. RBF-KRR
#            without an intercept decays toward ZERO away from the data, which
#            for a target whose mean is ~0.1-0.65 is a strong and unhelpful
#            prior.
#   log      log(y), inverted with exp. The tail-dominated error (mean 0.90 pp
#            vs median 0.38 pp) is exactly what a log target attacks, and this
#            was the decisive win on the forward-model task.
#   log_ctr  log(y) centred, same reasoning as y_ctr but far more important
#            here: mean log(y) is around -2, so an uncentred log-KRR decays
#            toward log(y)=0, i.e. a predicted fraction of 1.0.
TARGETS = ["y", "y_ctr", "log", "log_ctr"]

# Numerical guard only. exp() of an unconstrained kernel prediction can
# overflow; a colorant fraction cannot exceed 1.0, so capping the exponent at
# log(2) is well outside anything physical and cannot flatter the log variants
# in the region where they are actually judged. How often it binds is reported.
LOG_PRED_CAP = np.log(2.0)

MIN_N_FOR_NO_REGRESSION = 50   # colorants the "no regression" rule applies to
ADOPT_REL = 0.05               # per-colorant adoption threshold
POOLED_REL = 0.08              # overall success threshold
GPR_MAX_N = 150                # GP is run only for colorants below this


def rule(ch: str = "-") -> None:
    print(ch * 100)


def cv_for(n: int):
    """The SAME CV object as src/stage2_regressor.py:95."""
    return LeaveOneOut() if n < 15 else KFold(n_splits=5, shuffle=True, random_state=0)


# --------------------------------------------------------------------------
# Fast RBF kernel ridge: one Cholesky serves all four target treatments.
# --------------------------------------------------------------------------

def _sqdist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)


def _target_matrix(y_tr: np.ndarray) -> tuple[np.ndarray, dict]:
    """(n_tr, 4) right-hand side plus what each column needs to be inverted."""
    log_y = np.log(y_tr)
    meta = {"y_mean": float(y_tr.mean()), "log_mean": float(log_y.mean())}
    B = np.column_stack([y_tr, y_tr - meta["y_mean"], log_y, log_y - meta["log_mean"]])
    return B, meta


def _invert(col: np.ndarray, target: str, meta: dict) -> tuple[np.ndarray, int]:
    """Map a raw dual prediction back to a fraction. Returns (pred, n_capped)."""
    if target == "y":
        return col, 0
    if target == "y_ctr":
        return col + meta["y_mean"], 0
    z = col if target == "log" else col + meta["log_mean"]
    capped = int((z > LOG_PRED_CAP).sum())
    return np.exp(np.minimum(z, LOG_PRED_CAP)), capped


def _verify_against_sklearn(Z_tr, y_tr, Z_te) -> float:
    """The 'y' treatment must reproduce sklearn's KernelRidge exactly."""
    gamma, alpha = 1.0 / Z_tr.shape[1], 1e-2
    ref = KernelRidge(kernel="rbf", gamma=gamma, alpha=alpha).fit(Z_tr, y_tr)
    K = np.exp(-gamma * _sqdist(Z_tr, Z_tr))
    K[np.diag_indices_from(K)] += alpha
    dual = cho_solve(cho_factor(K, lower=True), y_tr)
    mine = np.exp(-gamma * _sqdist(Z_te, Z_tr)) @ dual
    return float(np.max(np.abs(mine - ref.predict(Z_te))))


def in_precommitted(key) -> bool:
    return key[1] in GAMMA_MULTS and key[2] in ALPHAS


def kernel_cv_scores(X: np.ndarray, y: np.ndarray, cv, verify: bool = False):
    """CV MAE for every (representation, gamma, alpha, target) combination.

    Scores the UNION of the pre-committed and extended grids in one pass;
    in_precommitted() recovers the pre-committed subset exactly, so the
    decision-bearing numbers are a filter over the same computation rather
    than a separate re-run that could drift.

    PCA and StandardScaler are fitted inside each fold, so nothing leaks.
    Returns (scores, n_capped, verify_err) where scores maps config-key ->
    (pooled_mae, mean_of_fold_means).
    """
    reps = ["raw"] + [f"pca{k}" for k in PCA_COMPONENTS]
    abs_err = {}    # key -> list of per-fold arrays of |error|
    capped = {}
    verify_err = np.nan

    for fold_i, (tr, te) in enumerate(cv.split(X)):
        sc = StandardScaler().fit(X[tr])
        Z_tr_raw, Z_te_raw = sc.transform(X[tr]), sc.transform(X[te])
        y_tr, y_te = y[tr], y[te]
        B, meta = _target_matrix(y_tr)

        if verify and fold_i == 0:
            verify_err = _verify_against_sklearn(Z_tr_raw, y_tr, Z_te_raw)

        for rep in reps:
            if rep == "raw":
                Z_tr, Z_te, dim = Z_tr_raw, Z_te_raw, Z_tr_raw.shape[1]
            else:
                k = int(rep[3:])
                if k >= min(len(tr), Z_tr_raw.shape[1]):
                    continue
                p = PCA(n_components=k, random_state=0).fit(Z_tr_raw)
                Z_tr, Z_te, dim = p.transform(Z_tr_raw), p.transform(Z_te_raw), k

            D_tr, D_te = _sqdist(Z_tr, Z_tr), _sqdist(Z_te, Z_tr)
            for gm in EXT_GAMMA_MULTS:
                gamma = gm / dim
                K_tr = np.exp(-gamma * D_tr)
                K_te = np.exp(-gamma * D_te)
                for alpha in EXT_ALPHAS:
                    Ka = K_tr.copy()
                    Ka[np.diag_indices_from(Ka)] += alpha
                    try:
                        dual = cho_solve(cho_factor(Ka, lower=True), B)
                    except Exception:
                        continue
                    raw_pred = K_te @ dual
                    for ci, target in enumerate(TARGETS):
                        pred, nc = _invert(raw_pred[:, ci], target, meta)
                        key = (rep, gm, alpha, target)
                        abs_err.setdefault(key, []).append(np.abs(pred - y_te))
                        capped[key] = capped.get(key, 0) + nc

    scores = {}
    for key, folds in abs_err.items():
        allerr = np.concatenate(folds)
        scores[key] = (float(allerr.mean()),
                       float(np.mean([f.mean() for f in folds])))
    return scores, capped, verify_err


def fit_kernel_final(X: np.ndarray, y: np.ndarray, key):
    """Refit one config on all active pool rows; returns a predict callable."""
    rep, gm, alpha, target = key
    sc = StandardScaler().fit(X)
    Z = sc.transform(X)
    pca = None
    if rep != "raw":
        pca = PCA(n_components=int(rep[3:]), random_state=0).fit(Z)
        Z = pca.transform(Z)
    gamma = gm / Z.shape[1]
    B, meta = _target_matrix(y)
    K = np.exp(-gamma * _sqdist(Z, Z))
    K[np.diag_indices_from(K)] += alpha
    dual = cho_solve(cho_factor(K, lower=True), B)
    col = TARGETS.index(target)

    def predict(Xq: np.ndarray) -> np.ndarray:
        Zq = sc.transform(Xq)
        if pca is not None:
            Zq = pca.transform(Zq)
        raw = np.exp(-gamma * _sqdist(Zq, Z)) @ dual[:, col]
        pred, _ = _invert(raw, target, meta)
        return pred

    return predict


def gpr_cv(X: np.ndarray, y: np.ndarray, cv):
    """GaussianProcessRegressor CV MAE plus its mean predictive std.

    The calibrated variance is independently useful for routing uncertain
    predictions to manual review, so it is reported even when the MAE loses.
    """
    errs, stds = [], []
    for tr, te in cv.split(X):
        sc = StandardScaler().fit(X[tr])
        Z_tr, Z_te = sc.transform(X[tr]), sc.transform(X[te])
        ls0 = float(np.sqrt(np.median(_sqdist(Z_tr, Z_tr)))) or 1.0
        k = (ConstantKernel(1.0, (1e-3, 1e3))
             * RBF(length_scale=ls0, length_scale_bounds=(1e-2 * ls0, 1e2 * ls0))
             + WhiteKernel(1e-3, (1e-8, 1e1)))
        g = GaussianProcessRegressor(kernel=k, normalize_y=True,
                                     n_restarts_optimizer=1, random_state=0)
        g.fit(Z_tr, y[tr])
        mu, sd = g.predict(Z_te, return_std=True)
        errs.append(np.abs(mu - y[te]))
        stds.append(sd)
    return float(np.concatenate(errs).mean()), float(np.concatenate(stds).mean())


# --------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    rule("=")
    print("exp_stage2_kernel -- kernel/GP methods vs Stage 2's incumbent ensemble")
    print(f"  ACTIVE_THRESHOLD {ACTIVE_THRESHOLD:g}   (nothing is written; models/ untouched)")
    rule("=")

    print("\nLoading data and reproducing train.py's partition...")
    train_df, test_df, light_only = production_partition()
    print(f"  train {len(train_df)}   test {len(test_df)}   light-only {len(light_only)}")

    X_pool_df, frame = training_frame_for_mode(MODE_LIGHT, train_df, light_only)
    X_test_df, _ = training_frame_for_mode(MODE_LIGHT, test_df, None)
    X_pool = X_pool_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    print(f"  pool {X_pool.shape}   test {X_test.shape}")

    cols = [f"colorant_{c}" for c in COLORANT_NAMES]
    Y_pool = frame[cols].to_numpy(dtype=float)
    Y_test = test_df[cols].to_numpy(dtype=float)
    active_pool = (Y_pool > 0).astype(float)
    active_test = Y_test > 0

    # ---------------- incumbent, re-measured under this exact protocol ----
    rule("=")
    print("INCUMBENT re-measured under the identical protocol")
    rule("=")

    print("  joint masked-softmax MLP (CV over the full pool)...")
    t0 = time.time()
    softmax_oof = cv_masked_softmax(X_pool, active_pool, Y_pool)
    print(f"    {time.time() - t0:.0f}s")

    print("  joint ILR regressor (CV over the full pool)...")
    t0 = time.time()
    basis = ilr_basis(len(COLORANT_NAMES))
    X_aug = np.hstack([X_pool, active_pool])
    ilr_oof = cv_ilr(X_aug, Y_pool, basis)
    print(f"    {time.time() - t0:.0f}s")

    print("  fitting the joint models on the full pool (for the single test scoring)...")
    t0 = time.time()
    softmax_model = fit_masked_softmax(X_pool, active_pool, Y_pool)
    ilr_model = fit_ilr(X_aug, Y_pool, basis)
    print(f"    {time.time() - t0:.0f}s")

    inc: dict[str, dict] = {}
    for j, cname in enumerate(COLORANT_NAMES):
        col = cols[j]
        mask = frame[col].to_numpy(dtype=float) > 0
        n = int(mask.sum())
        Xa_df, ya_s = X_pool_df[mask], frame.loc[mask, col]
        cv = cv_for(n)

        t0 = time.time()
        pc_model, pc_name, pc_mae = _select_model(Xa_df, ya_s)
        # Same splits, pooled aggregation, so the incumbent has BOTH
        # aggregations available and the kernel can be compared like-for-like.
        pc_pooled = np.nan
        if pc_name is not None:
            cand = _candidate_models(n, Xa_df.shape[1])[pc_name]
            oof = cross_val_predict(cand, Xa_df, ya_s, cv=cv)
            pc_pooled = float(np.abs(np.asarray(oof).reshape(-1) - ya_s.to_numpy()).mean())
        t_pc = time.time() - t0

        t0 = time.time()
        tab_model, tab_mae = _tabicl_oof_mae(Xa_df, ya_s)
        t_tab = time.time() - t0

        sm_mae = float(np.abs(softmax_oof[mask, j] - Y_pool[mask, j]).mean())
        il_mae = float(np.abs(ilr_oof[mask, j] - Y_pool[mask, j]).mean())

        fam = {"per_colorant": pc_mae, "joint_softmax": sm_mae,
               "joint_ilr": il_mae, "tabicl": tab_mae}
        winner = min(fam, key=fam.get)
        inc[cname] = {"n": n, "fam": fam, "winner": winner, "pc_name": pc_name,
                      "pc_model": pc_model, "pc_pooled": pc_pooled,
                      "tab_model": tab_model, "best": fam[winner], "j": j}
        print(f"  {cname:<28} n={n:5d}  per_colorant={pc_mae:.5f}[{pc_name}]  "
              f"softmax={sm_mae:.5f}  ilr={il_mae:.5f}  tabicl={tab_mae:.5f}  "
              f"-> {winner}   ({t_pc:.0f}s+{t_tab:.0f}s)")

    # ---------------- kernel search --------------------------------------
    rule("=")
    print("KERNEL / GP search  (PCA and scaling fitted inside every fold)")
    n_cfg = (1 + len(PCA_COMPONENTS)) * len(GAMMA_MULTS) * len(ALPHAS) * len(TARGETS)
    print(f"  {n_cfg} configs per colorant: "
          f"{['raw'] + [f'pca{k}' for k in PCA_COMPONENTS]} x "
          f"gamma{GAMMA_MULTS}/dim x alpha{ALPHAS} x {TARGETS}")
    rule("=")

    ker: dict[str, dict] = {}
    for j, cname in enumerate(COLORANT_NAMES):
        col = cols[j]
        mask = frame[col].to_numpy(dtype=float) > 0
        n = int(mask.sum())
        Xa, ya = X_pool[mask], Y_pool[mask, j]
        cv = cv_for(n)

        t0 = time.time()
        scores, capped, verr = kernel_cv_scores(Xa, ya, cv, verify=(j == 0))
        if j == 0:
            print(f"  [self-check] max |mine - sklearn.KernelRidge| = {verr:.3e}")

        pre = {k: v for k, v in scores.items() if in_precommitted(k)}
        best_key = min(pre, key=lambda k: pre[k][0])
        best_pooled, best_foldmean = pre[best_key]
        ext_key = min(scores, key=lambda k: scores[k][0])
        ext_pooled = scores[ext_key][0]

        # Best within each family, for the mechanism question (pre-committed
        # grid, so the mechanism table matches the decision table).
        by_rep, by_tgt = {}, {}
        for k, v in pre.items():
            by_rep[k[0]] = min(by_rep.get(k[0], np.inf), v[0])
            by_tgt[k[3]] = min(by_tgt.get(k[3], np.inf), v[0])

        gp_mae = gp_std = np.nan
        if n < GPR_MAX_N:
            gp_mae, gp_std = gpr_cv(Xa, ya, cv)

        ker[cname] = {"key": best_key, "pooled": best_pooled,
                      "foldmean": best_foldmean, "by_rep": by_rep, "by_tgt": by_tgt,
                      "ext_key": ext_key, "ext_pooled": ext_pooled,
                      "gp_mae": gp_mae, "gp_std": gp_std,
                      "capped": capped.get(best_key, 0)}
        rep_s = "  ".join(f"{r}={by_rep[r]:.5f}" for r in sorted(by_rep))
        print(f"  {cname:<28} n={n:5d}  best={best_pooled:.5f} "
              f"{best_key}   [{rep_s}]"
              + (f"  ext={ext_pooled:.5f} {ext_key}")
              + (f"  gp={gp_mae:.5f}(sd {gp_std:.4f})" if np.isfinite(gp_mae) else "")
              + f"   ({time.time() - t0:.0f}s)")

    # ---------------- decision table -------------------------------------
    rule("=")
    print("PER-COLORANT CV COMPARISON   (decision metric: pooled OOF MAE on active rows)")
    rule("=")
    hdr = (f"{'colorant':<28} {'n':>5} {'incumbent':>10} {'family':>14} "
           f"{'kernel':>10} {'variant':>26} {'rel':>8}  adopt")
    print(hdr)
    rule()

    base_v, adopted, adopted_ext, regressed, regressed_ext = [], [], [], [], []
    for cname in COLORANT_NAMES:
        I, K = inc[cname], ker[cname]
        # The per_colorant family's native score is a mean of fold means;
        # compare the kernel to whichever aggregation the winning family uses.
        if I["winner"] == "per_colorant" and np.isfinite(I["pc_pooled"]):
            base = I["pc_pooled"]
        else:
            base = I["best"]
        base_v.append(base)
        rel = (K["pooled"] - base) / base
        rel_e = (K["ext_pooled"] - base) / base
        ad, ad_e = rel <= -ADOPT_REL, rel_e <= -ADOPT_REL
        if ad:
            adopted.append(cname)
        if ad_e:
            adopted_ext.append(cname)
        if rel > 0 and I["n"] >= MIN_N_FOR_NO_REGRESSION:
            regressed.append((cname, rel))
        if rel_e > 0 and I["n"] >= MIN_N_FOR_NO_REGRESSION:
            regressed_ext.append((cname, rel_e))
        K["base"], K["rel"], K["rel_ext"], K["adopt"] = base, rel, rel_e, ad
        print(f"{cname:<28} {I['n']:>5} {base:>10.5f} {I['winner']:>14} "
              f"{K['pooled']:>10.5f} {str(K['key']):>26} {rel:>+7.1%}  "
              f"{'YES' if ad else '-'}")

    # ---------------- pooled ---------------------------------------------
    w = np.array([inc[c]["n"] for c in COLORANT_NAMES], dtype=float)
    inc_v = np.array(base_v)
    ker_v = np.array([ker[c]["pooled"] for c in COLORANT_NAMES])
    ext_v = np.array([ker[c]["ext_pooled"] for c in COLORANT_NAMES])
    ad_m = np.array([ker[c]["adopt"] for c in COLORANT_NAMES])
    ad_e_m = np.array([c in adopted_ext for c in COLORANT_NAMES])

    def pooled(v):
        return float((w * v).sum() / w.sum())

    pooled_inc = pooled(inc_v)
    pooled_ker = pooled(ker_v)
    pooled_ens = pooled(np.where(ad_m, ker_v, inc_v))
    pooled_ens_e = pooled(np.where(ad_e_m, ext_v, inc_v))
    rel_ens = (pooled_ens - pooled_inc) / pooled_inc
    rel_ens_e = (pooled_ens_e - pooled_inc) / pooled_inc
    rel_ker = (pooled_ker - pooled_inc) / pooled_inc

    rule("=")
    print("POOLED CV  (active-cell-weighted mean over all 14 colorants)")
    rule("=")
    print(f"  incumbent (best family per colorant)     {pooled_inc:.5f}")
    print(f"  kernel everywhere                        {pooled_ker:.5f}   ({rel_ker:+.1%})")
    print(f"  kernel adopted only where it wins >=5%   {pooled_ens:.5f}   ({rel_ens:+.1%})")
    print(f"  [exploratory] extended-grid, same rule   {pooled_ens_e:.5f}   ({rel_ens_e:+.1%})")

    rule("=")
    print("VERDICT against the pre-committed thresholds  (PRE-COMMITTED GRID = THE DECISION)")
    rule("=")
    ok_pooled = rel_ens <= -POOLED_REL
    ok_reg = len(regressed) == 0
    print(f"  [{'PASS' if ok_pooled else 'FAIL'}] pooled CV MAE improves >= {POOLED_REL:.0%} "
          f"relative: {pooled_inc:.5f} -> {pooled_ens:.5f} ({rel_ens:+.1%})")
    print(f"  [{'PASS' if ok_reg else 'FAIL'}] no colorant with n_active >= "
          f"{MIN_N_FOR_NO_REGRESSION} regresses"
          + ("" if ok_reg else ": " + ", ".join(f"{c} {r:+.1%}" for c, r in regressed)))
    print(f"  adopted ({len(adopted)}/14): {', '.join(adopted) if adopted else 'none'}")
    print(f"\n  OVERALL: {'SUCCESS' if (ok_pooled and ok_reg) else 'DOES NOT MEET THE BAR'}")

    rule()
    print("  Same thresholds applied to the EXPLORATORY extended grid (NOT the decision;")
    print("  its winners were selected over a 4x larger space, so it is optimistic):")
    oke_p, oke_r = rel_ens_e <= -POOLED_REL, len(regressed_ext) == 0
    print(f"    [{'PASS' if oke_p else 'FAIL'}] pooled {pooled_inc:.5f} -> "
          f"{pooled_ens_e:.5f} ({rel_ens_e:+.1%})")
    print(f"    [{'PASS' if oke_r else 'FAIL'}] no n>={MIN_N_FOR_NO_REGRESSION} regression"
          + ("" if oke_r else ": " + ", ".join(f"{c} {r:+.1%}" for c, r in regressed_ext)))
    print(f"    adopted ({len(adopted_ext)}/14): "
          f"{', '.join(adopted_ext) if adopted_ext else 'none'}")
    n_edge = sum(1 for c in COLORANT_NAMES
                 if ker[c]["key"][1] == min(GAMMA_MULTS) or ker[c]["key"][2] == min(ALPHAS))
    print(f"    pre-committed winners pinned to a grid edge: {n_edge}/14 "
          f"(this is why the extension was searched)")

    # ---------------- mechanism ------------------------------------------
    rule("=")
    print("MECHANISM: best CV MAE by representation and by target treatment")
    rule("=")
    reps = ["raw"] + [f"pca{k}" for k in PCA_COMPONENTS]
    print(f"{'colorant':<28} {'n':>5} " + " ".join(f"{r:>9}" for r in reps)
          + "  |" + " ".join(f"{t:>9}" for t in TARGETS))
    for cname in COLORANT_NAMES:
        K = ker[cname]
        print(f"{cname:<28} {inc[cname]['n']:>5} "
              + " ".join(f"{K['by_rep'].get(r, np.nan):>9.5f}" for r in reps)
              + "  |" + " ".join(f"{K['by_tgt'].get(t, np.nan):>9.5f}" for t in TARGETS))
    win_rep = pd.Series([ker[c]["key"][0] for c in COLORANT_NAMES]).value_counts()
    win_tgt = pd.Series([ker[c]["key"][3] for c in COLORANT_NAMES]).value_counts()
    print(f"\n  winning representation: {dict(win_rep)}")
    print(f"  winning target:         {dict(win_tgt)}")
    gp_rows = [(c, ker[c]["gp_mae"], ker[c]["gp_std"], ker[c]["pooled"], inc[c]["best"])
               for c in COLORANT_NAMES if np.isfinite(ker[c]["gp_mae"])]
    if gp_rows:
        print("\n  GaussianProcessRegressor (n_active < %d):" % GPR_MAX_N)
        for c, gm, gs, km, im in gp_rows:
            print(f"    {c:<28} gp={gm:.5f}  krr={km:.5f}  incumbent={im:.5f}  "
                  f"mean predictive sd={gs:.5f}")

    # ---------------- SINGLE test scoring --------------------------------
    rule("=")
    print("HELD-OUT TEST (297 rows) -- scored ONCE. CV above is the decision.")
    rule("=")

    sm_test = predict_masked_softmax(softmax_model, X_test, active_test.astype(float))
    il_test = predict_ilr(ilr_model, np.hstack([X_test, active_test.astype(float)]))

    raw_inc = np.zeros_like(Y_test)
    raw_ens = np.zeros_like(Y_test)
    raw_ens_e = np.zeros_like(Y_test)
    print(f"{'colorant':<28} {'n_test':>7} {'incumbent':>10} {'kernel':>10} {'rel':>8}"
          f" {'ext':>10}  used")
    rule()
    t_inc, t_ker, t_ext, t_w = [], [], [], []
    for j, cname in enumerate(COLORANT_NAMES):
        I, K = inc[cname], ker[cname]
        rows_j = np.where(active_test[:, j])[0]

        if I["winner"] == "per_colorant":
            p_inc = np.asarray(I["pc_model"].predict(X_test_df)).reshape(-1)
        elif I["winner"] == "tabicl":
            p_inc = np.asarray(I["tab_model"].predict(X_test)).reshape(-1)
        elif I["winner"] == "joint_softmax":
            p_inc = sm_test[:, j]
        else:
            p_inc = il_test[:, j]
        p_inc = np.clip(p_inc, 0.0, None)

        mask = frame[cols[j]].to_numpy(dtype=float) > 0
        Xa, ya = X_pool[mask], Y_pool[mask, j]
        p_ker = np.clip(fit_kernel_final(Xa, ya, K["key"])(X_test), 0.0, None)
        p_ext = np.clip(fit_kernel_final(Xa, ya, K["ext_key"])(X_test), 0.0, None)

        raw_inc[rows_j, j] = p_inc[rows_j]
        use_ker = cname in adopted
        raw_ens[rows_j, j] = (p_ker if use_ker else p_inc)[rows_j]
        raw_ens_e[rows_j, j] = (p_ext if cname in adopted_ext else p_inc)[rows_j]

        if len(rows_j):
            mi = float(np.abs(p_inc[rows_j] - Y_test[rows_j, j]).mean())
            mk = float(np.abs(p_ker[rows_j] - Y_test[rows_j, j]).mean())
            me = float(np.abs(p_ext[rows_j] - Y_test[rows_j, j]).mean())
            t_inc.append(mi), t_ker.append(mk), t_ext.append(me), t_w.append(len(rows_j))
            print(f"{cname:<28} {len(rows_j):>7} {mi:>10.5f} {mk:>10.5f} "
                  f"{(mk - mi) / mi:>+7.1%} {me:>10.5f}  "
                  f"{'kernel' if use_ker else I['winner']}")

    tw = np.array(t_w, float)
    print(f"\n  test MAE, un-normalized, active cells:")
    print(f"    incumbent          {float((tw * np.array(t_inc)).sum() / tw.sum()):.5f}")
    print(f"    kernel every col   {float((tw * np.array(t_ker)).sum() / tw.sum()):.5f}")
    print(f"    ext kernel ev. col {float((tw * np.array(t_ext)).sum() / tw.sum()):.5f}")

    # Normalized, i.e. directly comparable to train_backing_modes' 0.0090.
    def normalized_mae(raw: np.ndarray) -> float:
        s = raw.sum(axis=1, keepdims=True)
        f = raw / np.where(s > 0, s, 1.0)
        zero = (s.ravel() == 0)
        for i in np.where(zero)[0]:
            idx = np.where(active_test[i])[0]
            f[i, idx] = 1.0 / len(idx)
        return float(np.abs(f - Y_test)[active_test].mean())

    m_inc, m_ens = normalized_mae(raw_inc), normalized_mae(raw_ens)
    m_ens_e = normalized_mae(raw_ens_e)
    print(f"\n  test MAE, row-normalized (the 0.0090 headline's definition):")
    print(f"    incumbent ensemble            {m_inc:.5f}")
    print(f"    kernel adopted where CV won   {m_ens:.5f}   ({(m_ens - m_inc) / m_inc:+.1%})")
    print(f"    [exploratory] extended grid   {m_ens_e:.5f}   "
          f"({(m_ens_e - m_inc) / m_inc:+.1%})")
    print(f"\n  (CV stands as the decision; the test figure is reported as-is.)")
    print(f"\nTotal runtime {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
