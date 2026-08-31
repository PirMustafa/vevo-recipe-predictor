"""Can a LEARNED forward model (ink recipe -> light-backing spectrum) score dE without mixing ink?

    python forward_model_test.py                  # full run, ~12 min on CPU (measured 701s)
    python forward_model_test.py --seeds 5        # better ensemble, ~18 min
    python forward_model_test.py --no-sweep       # skip config selection, ~10 min

Runtime note: one network costs 12-18s on this CPU and the two 5-fold splits
need seeds x 5 of them each, so the ensemble defaults to 3 seeds rather than 5
and the selection sweep runs on 3 folds rather than 5. Both are budget
decisions, not measurement ones -- every reported figure still comes from a
full 5-fold split. --seeds 5 is the higher-fidelity run.

THE QUESTION. Stage 2 predicts a recipe. Nobody can say how good that recipe is
without physically mixing the ink and measuring it. A forward model would close
that loop in software: recipe -> predicted spectrum -> CIEDE2000 against the
target. A physics (Kubelka-Munk) attempt was removed from this repo for scoring
a mean dE of ~14, which is not a usable answer to anything.

THE BAR IS 2.35, NOT 14. A closed-form least squares of `log R ~ F + all 91
pairwise F_i*F_j` (105 terms, no ML at all) reaches mean dE 2.35 on the test
set. Any learned model must be read against THAT, not against the physics
model. Beating 14 proves nothing and is not reported here as success.

TARGET SPACE IS log R -- settled by measurement, do not revisit with K/S:
    space    linear   + nonlinear terms
    log R      5.47      2.35
    raw R     13.49     10.22
    K/S       12.15     12.34   (WORSE given more capacity)
K/S spans 0..85 on this data because the minimum reflectance is 0.00577, so a
least-squares fit is dominated by a handful of dark bands. Reflectance is
clipped to 1e-4 before the log, the model predicts log R, and every dE is
computed after exponentiating back into R space.

METHOD. 14 raw colorant fractions -> 36 log-reflectance values, via a small MLP
(fractions concatenated with their square roots, 28 -> 256 -> ... -> 36, GELU,
dropout, Adam + cosine decay, early stopping, seeds averaged in log R space).
Fractions are fed RAW: ACTIVE_THRESHOLD is a Stage-1 labelling device, and
applying it here would train the model on recipes that were never mixed.

TWO SPLITS, BOTH REPORTED. A random 5-fold on the pool selects the
configuration. A combination-held-out 5-fold -- grouped by which colorants are
non-zero, 174 distinct support patterns, so a fold never sees the pattern it is
scored on -- is the memorisation control and the honest read. The smooth
closed-form reference already loses ~63% accuracy under that split, so some
degradation is expected even without memorisation.

CAVEATS THAT TRAVEL WITH THE NUMBERS (reprinted at the end of every run):
  - The noise floor is ~0.8 mean dE. No replicates exist in this dataset, so
    that figure comes from leave-one-out inside the four densest colorant
    groups (LOO mean dE 0.74-0.99). A model scoring below 0.8 has leaked.
  - production_partition() drops formulas more than 8 dE from their Pantone
    target from the TEST side too, so this test set is pre-sweetened relative
    to production traffic.
  - R[380] = R[390] = R[400] in 99.95% of rows (instrument padding) and those
    bands carry 0.0023% of CIE luminance weight. Spectral MAE is flattered by
    them and is never quoted here without dE beside it.

NOTHING HERE IS WRITTEN TO DISK. No file under models/ is read or created; the
production pipeline is untouched. This script only prints.
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, KFold

from src.colorimetry import delta_e_2000, delta_e_2000_spectra, reflectance_to_lab
from src.pantone import load_pantone_library, target_for_formula
from train_backing_modes import CC, LIGHT_COLS, PANTONE_PATH, production_partition

# ---------------------------------------------------------------------------
# Fixed choices. These are settled by the measurements in the docstring; they
# are constants rather than flags so a run cannot quietly change target space.
# ---------------------------------------------------------------------------
R_FLOOR = 1e-4          # clip before log; min observed reflectance is 0.00577
R_CEIL = 1.0            # reflectance is a fraction; exp() must be pulled back
N_FOLDS = 5
NOISE_FLOOR_DE = 0.8    # below this is leakage, not skill -- see docstring
SEED = 0

# CPU is forced deliberately. At 2051 rows x 28 inputs the host-device transfer
# and kernel-launch overhead dominate the arithmetic, so a GPU is slower here.
DEVICE = torch.device("cpu")

CONFIGS = {
    #  name          width depth dropout   lr    epochs
    "mlp-256x2": dict(width=256, depth=2, dropout=0.10, lr=3e-3, epochs=140),
    "mlp-256x3": dict(width=256, depth=3, dropout=0.05, lr=3e-3, epochs=140),
}
DEFAULT_CONFIG = "mlp-256x2"   # used when --no-sweep

# Selection runs on a 3-fold subset of the random split rather than all five.
# One net costs 12-18s on this CPU, so a full 5-fold sweep at 5 seeds would put
# the script past 30 minutes. Selection only has to rank two configurations,
# which 3 folds over ~1230 scored rows does adequately; every REPORTED number
# still comes from the full 5-fold splits below.
SWEEP_FOLDS = 3


# ---------------------------------------------------------------------------
# Target space
# ---------------------------------------------------------------------------
def to_log(R: np.ndarray) -> np.ndarray:
    """Reflectance -> log reflectance, with the floor that keeps K/S-style blowup out."""
    return np.log(np.clip(np.asarray(R, float), R_FLOOR, None))


def from_log(L: np.ndarray) -> np.ndarray:
    """log reflectance -> reflectance in (0, 1]. Every dE is scored after this."""
    return np.clip(np.exp(np.asarray(L, float)), 1e-9, R_CEIL)


def features(F: np.ndarray) -> np.ndarray:
    """14 raw fractions -> 28 inputs: the fractions and their square roots.

    sqrt(f) gives the network a cheap way to express the strongly sub-linear
    response of a colorant near zero dose without having to learn that
    curvature from scratch. No thresholding -- see the module docstring.
    """
    F = np.asarray(F, float)
    return np.hstack([F, np.sqrt(np.clip(F, 0.0, None))]).astype(np.float32)


# ---------------------------------------------------------------------------
# Scoring. Never a single number -- the dE distribution here is heavy-tailed,
# so a mean on its own hides exactly the failures that matter.
# ---------------------------------------------------------------------------
METRIC_HEADER = (f"  {'model / baseline':<30}{'n':>5}{'mean':>8}{'med':>7}{'p75':>7}"
                 f"{'p90':>7}{'p95':>7}{'max':>8}{'<1':>7}{'<2':>7}{'<3':>7}{'R-MAE':>9}")


def metrics(pred_R: np.ndarray, true_R: np.ndarray) -> dict:
    pred_R = np.clip(np.asarray(pred_R, float), 1e-9, R_CEIL)
    true_R = np.asarray(true_R, float)
    d = np.asarray(delta_e_2000_spectra(pred_R, true_R), float).ravel()
    return {
        "n": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
        "p75": float(np.percentile(d, 75)), "p90": float(np.percentile(d, 90)),
        "p95": float(np.percentile(d, 95)), "max": float(d.max()),
        "u1": float((d < 1.0).mean()), "u2": float((d < 2.0).mean()),
        "u3": float((d < 3.0).mean()),
        "mae": float(np.abs(pred_R - true_R).mean()), "dE": d,
    }


def show(name: str, m: dict, note: str = "") -> None:
    print(f"  {name:<30}{m['n']:>5}{m['mean']:>8.3f}{m['median']:>7.3f}{m['p75']:>7.3f}"
          f"{m['p90']:>7.3f}{m['p95']:>7.3f}{m['max']:>8.2f}"
          f"{m['u1']:>6.1%}{m['u2']:>7.1%}{m['u3']:>7.1%}{m['mae']:>9.5f}  {note}")


# ---------------------------------------------------------------------------
# Baselines. Printed first as a harness correctness check: if these do not
# reproduce the measured references, nothing below them means anything.
# ---------------------------------------------------------------------------
PAIRS = list(itertools.combinations(range(14), 2))


def quad_design(F: np.ndarray) -> np.ndarray:
    """105 terms: 14 fractions + 91 pairwise products.

    Squares are deliberately absent. Fractions sum to 1, so
    f_i = f_i * sum_j f_j = f_i^2 + sum_{j!=i} f_i f_j -- the squares are
    already spanned by the linear and off-diagonal terms and would add only
    collinearity.
    """
    F = np.asarray(F, float)
    return np.hstack([F, np.column_stack([F[:, i] * F[:, j] for i, j in PAIRS])])


def fit_linear_log(Xtr: np.ndarray, Ltr: np.ndarray, Xte: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(Xtr, Ltr, rcond=None)
    return from_log(Xte @ coef)


def baselines(Ftr, Rtr, Fte):
    """Every reference prediction for one (train pool, query set) pair."""
    Ltr = to_log(Rtr)
    out = {"train-mean": np.tile(Rtr.mean(axis=0), (len(Fte), 1))}

    # L1 in recipe space: the natural "have I seen this mixture before" metric.
    D = np.abs(Fte[:, None, :] - Ftr[None, :, :]).sum(axis=-1)
    out["1-NN (L1 recipe)"] = Rtr[D.argmin(axis=1)]

    idx = np.argsort(D, axis=1)[:, :3]
    w = 1.0 / np.maximum(np.take_along_axis(D, idx, 1), 1e-9)
    w /= w.sum(axis=1, keepdims=True)
    # Averaged in log R for consistency with the model's target space. Doing it
    # in R space instead gives 3.878 rather than 3.840 -- both reproduce the
    # 3.88 reference within tolerance.
    out["3-NN inv-dist"] = from_log((Ltr[idx] * w[:, :, None]).sum(axis=1))

    out["linear log R (14)"] = fit_linear_log(Ftr, Ltr, Fte)
    out["pairwise log R (105)"] = fit_linear_log(quad_design(Ftr), Ltr, quad_design(Fte))
    return out, D


EXPECTED = {"train-mean": 27.93, "1-NN (L1 recipe)": 4.65, "3-NN inv-dist": 3.88,
            "linear log R (14)": 5.47, "pairwise log R (105)": 2.35}


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
def build_mlp(cfg: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    layers, d = [], 28
    for _ in range(cfg["depth"]):
        layers += [nn.Linear(d, cfg["width"]), nn.GELU(), nn.Dropout(cfg["dropout"])]
        d = cfg["width"]
    layers += [nn.Linear(d, 36)]
    return nn.Sequential(*layers).to(DEVICE)


def fit_mlp(Xfit, Yfit, Xes, Yes, cfg, seed, batch=128, check_every=5, patience=12):
    """One network. Early stopping watches a held-out slice of the FIT rows only.

    Yfit / Yes are standardised log R. The early-stopping slice never contains a
    row -- or, under the group split, a support pattern -- that will be scored.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_mlp(cfg, seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    Xf = torch.as_tensor(Xfit, device=DEVICE)
    Yf = torch.as_tensor(Yfit, device=DEVICE)
    Xv = torch.as_tensor(Xes, device=DEVICE)
    Yv = torch.as_tensor(Yes, device=DEVICE)

    n, best, best_state, stale = len(Xf), float("inf"), None, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            nn.functional.mse_loss(model(Xf[b]), Yf[b]).backward()
            opt.step()
        sched.step()
        if epoch % check_every == check_every - 1:
            model.eval()
            with torch.no_grad():
                v = float(nn.functional.mse_loss(model(Xv), Yv))
            if v < best - 1e-6:
                best, stale = v, 0
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_logR(models, X, mu, sd) -> np.ndarray:
    """Ensemble average taken in standardised log R, then un-standardised."""
    with torch.no_grad():
        p = np.mean([m(torch.as_tensor(X, device=DEVICE)).cpu().numpy()
                     for m in models], axis=0)
    return p * sd + mu


def train_predict(Ffit, Rfit, Fes, Res, Fquery, cfg, seeds):
    """Standardise on the fit rows only, train `seeds` nets, return predicted R."""
    Lfit = to_log(Rfit)
    mu, sd = Lfit.mean(axis=0), Lfit.std(axis=0) + 1e-8
    Yfit = ((Lfit - mu) / sd).astype(np.float32)
    Yes = ((to_log(Res) - mu) / sd).astype(np.float32)
    Xfit, Xes, Xq = features(Ffit), features(Fes), features(Fquery)
    models = [fit_mlp(Xfit, Yfit, Xes, Yes, cfg, seed) for seed in range(seeds)]
    return from_log(predict_logR(models, Xq, mu, sd))


# ---------------------------------------------------------------------------
# Folds. Each fold is (fit rows, early-stopping rows, scored rows).
# ---------------------------------------------------------------------------
def random_folds(n: int, seed: int = SEED):
    rng = np.random.RandomState(seed)
    splitter = KFold(N_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, out_idx in splitter.split(np.arange(n)):
        tr_idx = rng.permutation(tr_idx)
        cut = max(1, int(round(0.10 * len(tr_idx))))
        yield tr_idx[cut:], tr_idx[:cut], out_idx


def group_folds(groups: np.ndarray, seed: int = SEED):
    """Split by colorant SUPPORT PATTERN, so a fold never sees the pattern it scores.

    The early-stopping slice is carved out by whole groups as well. Taking it at
    random instead would let a pattern that is about to be scored leak in
    through the stopping criterion, which is precisely the memorisation this
    split exists to detect.
    """
    rng = np.random.RandomState(seed)
    for tr_idx, out_idx in GroupKFold(N_FOLDS).split(np.arange(len(groups)),
                                                     groups=groups):
        uniq = np.unique(groups[tr_idx])
        n_held = max(1, int(round(0.10 * len(uniq))))
        held = set(rng.permutation(uniq)[:n_held].tolist())
        mask = np.array([g in held for g in groups[tr_idx]])
        if mask.all() or not mask.any():
            # Degenerate carve-out on a fold with very few groups; fall back to
            # a row-wise slice rather than emitting an empty tensor.
            cut = max(1, len(tr_idx) // 10)
            yield tr_idx[cut:], tr_idx[:cut], out_idx
        else:
            yield tr_idx[~mask], tr_idx[mask], out_idx


def run_cv(folds, F, R, cfg, seeds, label):
    """Out-of-fold predicted reflectance, plus the indices actually scored.

    The caller must score only the returned indices: a partial fold list (the
    selection sweep) leaves the rest of the pool unpredicted, and scoring those
    zero rows would silently invent a huge error.
    """
    pred = np.zeros_like(R)
    covered, t0 = [], time.time()
    for k, (fit_i, es_i, out_i) in enumerate(folds, 1):
        pred[out_i] = train_predict(F[fit_i], R[fit_i], F[es_i], R[es_i],
                                    F[out_i], cfg, seeds)
        covered.append(out_i)
        print(f"      {label} fold {k}/{len(folds)}  fit {len(fit_i):>5}  "
              f"stop {len(es_i):>4}  scored {len(out_i):>5}   "
              f"[{time.time() - t0:5.1f}s]", flush=True)
    return pred, np.concatenate(covered)


def run_cv_closed_form(folds, F, R) -> np.ndarray:
    """The 105-term reference under the same folds.

    A smooth least-squares fit has no recipe memory, so whatever it loses under
    the group split is the cost of the SPLIT rather than evidence of
    memorisation. That is the yardstick the MLP's degradation is read against.
    """
    pred = np.zeros_like(R)
    for fit_i, es_i, out_i in folds:
        keep = np.concatenate([fit_i, es_i])       # no early stopping to protect
        pred[out_i] = fit_linear_log(quad_design(F[keep]), to_log(R[keep]),
                                     quad_design(F[out_i]))
    return pred


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=3,
                    help="networks averaged per ensemble (default 3, ~12 min total; "
                         "5 is better but takes ~18 min)")
    ap.add_argument("--no-sweep", action="store_true",
                    help=f"skip config selection and use {DEFAULT_CONFIG}")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads; more is not faster at this size")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    started = time.time()
    print("=" * 118)
    print("FORWARD MODEL TEST -- recipe -> light-backing spectrum, scored in CIEDE2000")
    print(f"  target space : log R (clip {R_FLOOR:g})     device: CPU (forced)     "
          f"seeds/ensemble: {args.seeds}")
    print("  the number to beat is the 105-term closed form at mean dE 2.35, NOT the")
    print("  removed Kubelka-Munk model at ~14.")
    print("=" * 118)

    print("\nLoading data (reproducing train.py's partition via production_partition)...",
          flush=True)
    train_df, test_df, light_only = production_partition()
    pool = pd.concat([train_df, light_only], ignore_index=True)
    F = pool[CC].to_numpy(float)
    R = pool[LIGHT_COLS].to_numpy(float)
    Fte = test_df[CC].to_numpy(float)
    Rte = test_df[LIGHT_COLS].to_numpy(float)

    patterns = np.array(["".join("1" if b else "0" for b in row) for row in (F > 0)])
    print(f"  train {len(train_df)}   test {len(test_df)}   light-only {len(light_only)}"
          f"   ->  POOL {len(pool)}")
    print(f"  colorants {F.shape[1]}   bands {R.shape[1]}   "
          f"distinct support patterns in pool {len(np.unique(patterns))}")
    print(f"  fractions fed RAW (no ACTIVE_THRESHOLD); row sums "
          f"{F.sum(1).min():.4f}-{F.sum(1).max():.4f}")
    print(f"  measured reflectance range {R.min():.5f}-{R.max():.5f}")

    # -- 1. Baselines ------------------------------------------------------
    print("\n" + "-" * 118)
    print("1. BASELINES on the 297 held-out test rows (harness check: each should")
    print("   land within +-0.05 of its measured reference)")
    print("-" * 118)
    print(METRIC_HEADER)
    base_preds, D_test = baselines(F, R, Fte)
    base_metrics, harness_ok = {}, True
    for name, pred in base_preds.items():
        m = metrics(pred, Rte)
        base_metrics[name] = m
        delta = m["mean"] - EXPECTED[name]
        ok = abs(delta) <= 0.05
        harness_ok = harness_ok and ok
        show(name, m, f"ref {EXPECTED[name]:5.2f} ({delta:+.3f})"
                      f"{'' if ok else '  <-- MISMATCH'}")
    print("\n  harness: " + ("OK -- all references reproduced" if harness_ok
                             else "FAILED -- do not trust anything below"))

    # -- 2. Config selection on the random split ---------------------------
    if args.no_sweep:
        chosen = DEFAULT_CONFIG
        print(f"\n2. CONFIG SELECTION skipped (--no-sweep); using {chosen}")
    else:
        print("\n" + "-" * 118)
        print(f"2. CONFIG SELECTION -- random split, first {SWEEP_FOLDS} of {N_FOLDS} folds, "
              f"1 seed per fold.")
        print("   A budget subset (see SWEEP_FOLDS); it only has to RANK the configs, and")
        print("   selection never touches the 297 test rows or the group split.")
        print("-" * 118)
        sweep = {}
        for name, cfg in CONFIGS.items():
            print(f"    {name}: {cfg}", flush=True)
            p, cov = run_cv(list(random_folds(len(F)))[:SWEEP_FOLDS],
                            F, R, cfg, 1, name)
            sweep[name] = metrics(p[cov], R[cov])
        print("\n" + METRIC_HEADER)
        for name, m in sweep.items():
            show(name + " (1 seed, CV)", m)
        chosen = min(sweep, key=lambda k: sweep[k]["mean"])
        print(f"\n  selected: {chosen}  (lowest random-CV mean dE)")
    cfg = CONFIGS[chosen]

    # -- 3. The two splits -------------------------------------------------
    print("\n" + "-" * 118)
    print(f"3. THE TWO SPLITS, {args.seeds}-seed ensemble, config {chosen}")
    print("   random 5-fold = optimistic; the pool contains near-twins of most rows")
    print("   group 5-fold  = held out by colorant support pattern; the honest read")
    print("-" * 118)

    print("  random 5-fold:", flush=True)
    pred_random, _ = run_cv(list(random_folds(len(F))), F, R, cfg, args.seeds, "random")
    m_random = metrics(pred_random, R)

    print("  group 5-fold (support-pattern held out):", flush=True)
    pred_group, _ = run_cv(list(group_folds(patterns)), F, R, cfg, args.seeds, "group ")
    m_group = metrics(pred_group, R)

    cf_random = metrics(run_cv_closed_form(list(random_folds(len(F))), F, R), R)
    cf_group = metrics(run_cv_closed_form(list(group_folds(patterns)), F, R), R)

    print("\n" + METRIC_HEADER)
    show("MLP  random 5-fold", m_random)
    show("MLP  group 5-fold", m_group)
    show("105-term  random 5-fold", cf_random)
    show("105-term  group 5-fold", cf_group)

    degr = (m_group["mean"] - m_random["mean"]) / m_random["mean"]
    cf_degr = (cf_group["mean"] - cf_random["mean"]) / cf_random["mean"]

    # -- 4. Test set, scored once ------------------------------------------
    print("\n" + "-" * 118)
    print("4. HELD-OUT TEST SET -- 297 rows, scored ONCE with the selected config")
    print("-" * 118, flush=True)
    rng = np.random.RandomState(SEED)
    order = rng.permutation(len(F))
    cut = int(round(0.10 * len(F)))
    es_i, fit_i = order[:cut], order[cut:]
    pred_test = train_predict(F[fit_i], R[fit_i], F[es_i], R[es_i], Fte, cfg, args.seeds)
    m_test = metrics(pred_test, Rte)
    print(METRIC_HEADER)
    show(f"MLP ({chosen})", m_test, "<-- vs measured")
    show("pairwise log R (105)", base_metrics["pairwise log R (105)"], "the bar")
    show("3-NN inv-dist", base_metrics["3-NN inv-dist"])

    # -- 5. Diagnostics ----------------------------------------------------
    print("\n" + "-" * 118)
    print("5. DIAGNOSTICS")
    print("-" * 118)

    nearest = D_test.min(axis=1)
    dE_test = m_test["dE"]
    r = float(np.corrcoef(nearest, dE_test)[0, 1])
    print("\n  (a) test dE by distance to the NEAREST pooled recipe (L1 over 14 fractions)")
    print(f"      {'quintile':<12}{'n':>5}{'L1 range':>22}{'mean dE':>10}{'median dE':>11}")
    edges = np.quantile(nearest, np.linspace(0, 1, 6))
    for q in range(5):
        lo, hi = edges[q], edges[q + 1]
        sel = ((nearest >= lo) & (nearest <= hi)) if q == 4 else ((nearest >= lo) & (nearest < hi))
        if sel.sum():
            print(f"      Q{q + 1:<11}{int(sel.sum()):>5}{f'{lo:.4f} - {hi:.4f}':>22}"
                  f"{dE_test[sel].mean():>10.3f}{np.median(dE_test[sel]):>11.3f}")
    print(f"      Pearson r(nearest-recipe distance, dE) = {r:+.3f}")
    spread = dE_test[nearest >= edges[4]].mean() - dE_test[nearest < edges[1]].mean()
    print(f"      Q5-Q1 gap = {spread:+.3f} dE")
    if r >= 0.15 and spread >= 0.5:
        print("      SLOPED UPWARD: error genuinely grows with distance from known recipes,")
        print("      which is the extrapolation signature. Treat the far quintile as the")
        print("      realistic figure for a novel recipe, not the overall mean.")
    else:
        # Either flat, or -- as measured here -- tilted the WRONG way. Both mean
        # the same thing, and neither is a safety result.
        if r <= -0.15 or spread <= -0.5:
            print("      INVERTED: dE is LOWER on the most distant rows. Distance to the")
            print("      nearest known recipe is not tracking difficulty at all here; the")
            print("      far quintile is simply dominated by easier (often lighter, less")
            print("      saturated) colours. This is not evidence of extrapolation safety.")
        else:
            print("      FLAT. The smooth 105-term reference is flat here too (r=0.047).")
        print("      Either way THE TEST SET CONTAINS ALMOST NO EXTRAPOLATION: read this as")
        print("      'this test set cannot detect the failure mode', NOT as 'the model is")
        print("      safe on novel recipes'. The group split is the evidence that can.")

    print("\n  (b) test rows whose colorant COMBINATION never appears in the pool")
    pool_patterns = set(np.unique(patterns).tolist())
    te_patterns = np.array(["".join("1" if b else "0" for b in row) for row in (Fte > 0)])
    unseen = np.where(~np.array([p in pool_patterns for p in te_patterns]))[0]
    print(f"      {len(unseen)} of {len(test_df)} test rows. Individually:")
    if len(unseen):
        print(f"      {'row':>5}  {'formula':<16}{'n_colorants':>12}{'dE':>9}{'nearest L1':>12}")
        for i in unseen:
            print(f"      {i:>5}  {str(test_df['formula'].iloc[i]):<16}"
                  f"{int((Fte[i] > 0).sum()):>12}{dE_test[i]:>9.3f}{nearest[i]:>12.4f}")
        print(f"      mean dE on these {len(unseen)}: {dE_test[unseen].mean():.3f}   "
              f"(rest of test set: {np.delete(dE_test, unseen).mean():.3f})")
        print("      n is far too small to conclude anything on its own -- it is here so the")
        print("      group-split number has something concrete behind it.")

    print("\n  (c) MEMORISATION SIGNATURE -- random CV vs group CV, same config, same seeds")
    print(f"      {'split':<26}{'mean dE':>10}{'median':>9}{'<1':>8}{'<2':>8}")
    print(f"      {'MLP  random 5-fold':<26}{m_random['mean']:>10.3f}"
          f"{m_random['median']:>9.3f}{m_random['u1']:>8.1%}{m_random['u2']:>8.1%}")
    print(f"      {'MLP  group 5-fold':<26}{m_group['mean']:>10.3f}"
          f"{m_group['median']:>9.3f}{m_group['u1']:>8.1%}{m_group['u2']:>8.1%}")
    print(f"      MLP degradation      {degr:+.1%}")
    print(f"      105-term degradation {cf_degr:+.1%}   <- cost of the split for a model")
    print("                                 with no recipe memory at all")
    print(f"      excess attributable to memorisation: {degr - cf_degr:+.1%}")

    print("\n  (d) SECONDARY: dE(predicted spectrum, PANTONE target)")
    library = load_pantone_library(PANTONE_PATH)
    rows, labs = [], []
    for i, formula in enumerate(test_df["formula"]):
        t = target_for_formula(str(formula), library)
        if t is not None:
            rows.append(i)
            labs.append(t.lab)
    if rows:
        rows = np.array(rows)
        labs = np.array(labs)
        d_pan = np.asarray(delta_e_2000(reflectance_to_lab(pred_test[rows]), labs),
                           float).ravel()
        d_meas = np.asarray(delta_e_2000(reflectance_to_lab(Rte[rows]), labs),
                            float).ravel()
        print(f"      DENOMINATOR IS {len(rows)} of {len(test_df)} test rows -- only these have")
        print("      a Pantone match. Do NOT mix this sample with the 297 above.")
        print(f"      {'comparison':<34}{'n':>5}{'mean':>9}{'median':>9}{'p95':>9}")
        print(f"      {'predicted spectrum vs Pantone':<34}{len(rows):>5}"
              f"{d_pan.mean():>9.3f}{np.median(d_pan):>9.3f}"
              f"{np.percentile(d_pan, 95):>9.3f}")
        print(f"      {'MEASURED spectrum vs Pantone':<34}{len(rows):>5}"
              f"{d_meas.mean():>9.3f}{np.median(d_meas):>9.3f}"
              f"{np.percentile(d_meas, 95):>9.3f}")
        print("      The measured row is the floor: it is how far the real ink sits from")
        print("      target, and no forward model can beat it.")
    else:
        print("      No Pantone library available -- skipped.")

    # -- 6. Caveats, printed so the numbers cannot travel without them -----
    print("\n" + "-" * 118)
    print("6. CAVEATS -- these belong beside every number above")
    print("-" * 118)
    print("  * PRE-SWEETENED TEST SET. production_partition() drops formulas more than")
    print("    8 dE from their Pantone target from the TEST side as well as the train")
    print("    side. This test set is therefore easier than production traffic, and")
    print("    every figure here is optimistic by an unmeasured margin.")
    pad = float(((R[:, 0] == R[:, 1]) & (R[:, 1] == R[:, 2])).mean())
    print(f"  * PADDED BANDS. R[380]=R[390]=R[400] in {pad:.2%} of pooled rows -- that is")
    print("    instrument padding, not measurement, and those bands carry 0.0023% of CIE")
    print("    luminance weight. Spectral MAE is flattered by them. NEVER headline R-MAE")
    print("    without dE beside it.")
    print(f"  * NOISE FLOOR ~{NOISE_FLOOR_DE} MEAN dE. The dataset has zero duplicate recipes and")
    print("    zero repeat measurements, so this comes from leave-one-out inside the four")
    print("    densest colorant groups (LOO mean dE 0.74-0.99, 68-81% under dE 1.0). A")
    print("    realistic ceiling on the under-dE-1.0 rate is ~70-80% in dense regions and")
    print("    ~45-55% globally. Scoring below the floor means leakage, not skill.")
    print("  * NO REPLICATES means none of this has a proper error bar.")

    # -- 7. Verdict, against thresholds committed before the run -----------
    mean_r, med_r = m_random["mean"], m_random["median"]
    print("\n" + "=" * 118)
    print("7. VERDICT -- thresholds below were committed IN ADVANCE of this run")
    print("=" * 118)
    print("  judged on the RANDOM 5-fold CV, predicted vs MEASURED reflectance:")
    print(f"     mean dE {mean_r:.3f}   median {med_r:.3f}   "
          f"under dE 1.0 {m_random['u1']:.1%}   under dE 2.0 {m_random['u2']:.1%}   "
          f"group degradation {degr:+.1%}")
    print(f"  reference points: closed form 2.35 | 3-NN 3.88 | noise floor "
          f"~{NOISE_FLOOR_DE} | removed physics model ~14")

    missed = []
    if med_r > 1.0:
        missed.append(f"median {med_r:.3f} > 1.0")
    if m_random["u1"] < 0.45:
        missed.append(f"under-dE-1 {m_random['u1']:.1%} < 45%")
    if m_random["u2"] < 0.85:
        missed.append(f"under-dE-2 {m_random['u2']:.1%} < 85%")
    if degr >= 0.50:
        missed.append(f"group degradation {degr:.1%} >= 50%")

    if mean_r < NOISE_FLOOR_DE:
        label = "SUSPICIOUS -- INVESTIGATE LEAKAGE"
        body = (f"mean dE {mean_r:.3f} is below the ~{NOISE_FLOOR_DE} noise floor. No model can\n"
                "     predict a measurement more precisely than the measurement repeats.\n"
                "     Find the leak before reporting this to anyone.")
    elif mean_r > 2.5 or med_r > 2.0 or m_random["u2"] < 0.30:
        hard = []
        if mean_r > 2.5:
            hard.append(f"mean {mean_r:.3f} > 2.5")
        if med_r > 2.0:
            hard.append(f"median {med_r:.3f} > 2.0")
        if m_random["u2"] < 0.30:
            hard.append(f"under-dE-2 {m_random['u2']:.1%} < 30%")
        label = "DEAD END"
        body = ("; ".join(hard) +
                ".\n     A learned forward model does not replace mixing the ink.")
    elif mean_r <= 1.3 and not missed:
        label = "VIABLE"
        body = ("every committed threshold met, on both splits.\n"
                "     Still read section 6 before acting: the test set is pre-sweetened,\n"
                "     there are no replicates, and the noise floor is ~0.8.")
    else:
        label = "BORDERLINE"
        band = (f"mean dE {mean_r:.3f} sits in the 1.3-2.5 band" if mean_r >= 1.3
                else f"mean dE {mean_r:.3f} clears 1.3 but a committed threshold did not")
        body = (band + (". Missed: " + "; ".join(missed) if missed else "") +
                ".\n     Usable for RANKING candidate recipes, not for quoting a dE to a"
                " customer.")
    print(f"\n  >>> {label}")
    print(f"     {body}")
    if not harness_ok:
        print("\n  NOTE: the baseline harness check FAILED. Treat the verdict as void.")
    print(f"\n  wall clock {time.time() - started:.1f}s")
    print("=" * 118)


if __name__ == "__main__":
    main()
