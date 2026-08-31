"""Stage 2 error anatomy: WHERE does the 0.0090 MAE actually come from?

DIAGNOSIS ONLY. Trains nothing, writes nothing, modifies nothing. Loads the
deployed light-mode Stage 2 artifact read-only and dissects its test-set error.
No model is proposed here -- that is deliberately somebody else's job.

Five questions, in the order effort should be spent on the answers:

  1. TAIL SHAPE      is the error a few bad formulas or a broad spread?
  2. UNDERDETERMINED does error rise where colour carries less information?
                     (white loading, colorant count, spectral crowding)
  3. COLOUR COST     which concentration errors actually cost dE? Measured
                     MODEL-FREE: for a row's signed error vector e, find real
                     pool formula pairs (a,b) sharing the row's active set
                     whose difference F_a - F_b points the same way as e, take
                     their MEASURED dE00, and scale it to |e|_1. No forward
                     model, no Kubelka-Munk, no assumption beyond local
                     linearity of dE in a concentration displacement.
  4. BIAS            shrinkage, per-colorant over/under-prediction, and the
                     compositional constraint (fractions sum to 1, so every
                     over-prediction is paid for by an under-prediction --
                     which ink pays?)
  5. CEILING         given the measured dE noise floor and the irreducible
                     portion from (2)/(3), what is the best plausible MAE?

Stage 2 is isolated from Stage 1 throughout by predicting under the TRUE
active mask (Ft > 0), so nothing here is contaminated by set-selection error.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.backing_modes import MODE_LIGHT, training_frame_for_mode
from src.colorimetry import delta_e_2000, reflectance_to_lab
from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS
from src.stage2_regressor import load_stage2, predict_fractions
from train_backing_modes import production_partition

ROOT = pathlib.Path(__file__).parent
STAGE2_PATH = ROOT / "models" / "wolves_v1" / "backing_modes" / "light" / "stage2.joblib"

CC = [f"colorant_{c}" for c in COLORANT_NAMES]
LIGHT_COLS = [f"R_light_{w}" for w in WAVELENGTHS]
WHITE = "UFO00061 transp. white"
WHITE_J = COLORANT_NAMES.index(WHITE)
SHORT = [c.split(" ", 1)[1] for c in COLORANT_NAMES]

# dE00 below which two measurements are treated as the same colour. The
# client target is 1.0; the measured repeatability floor is printed in
# section 5 and the ceiling is quoted over a band, not a point.
DE_FLOOR_BAND = (0.8, 1.0, 1.1)

RNG = np.random.default_rng(0)


def hdr(title, ch="="):
    print("\n" + ch * 78)
    print(title)
    print(ch * 78)


def pct(x):
    return f"{100.0 * x:.1f}%"


def corr_line(name, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        print(f"  {name:<36} (too few finite values)")
        return
    r, _ = pearsonr(x[m], y[m])
    rho, p = spearmanr(x[m], y[m])
    flag = "  <-- strong" if abs(rho) >= 0.30 else ("  <-- weak" if abs(rho) >= 0.15 else "")
    print(f"  {name:<36} pearson {r:+.3f}   spearman {rho:+.3f} "
          f"(p={p:.1e}, n={int(m.sum())}){flag}")


def quintile_table(name, x, err, n_bins=5, value_label="mean L1 err (pp)"):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    chunks = np.array_split(order, n_bins)
    print(f"  {name:<22}{'n':>5}{'range':>22}{value_label:>19}{'median':>10}")
    for c in chunks:
        lo, hi = x[c].min(), x[c].max()
        rng_s = f"{lo:.4g} .. {hi:.4g}"
        print(f"  {'':<22}{len(c):>5}{rng_s:>22}"
              f"{100 * err[c].mean():>19.2f}{100 * np.median(err[c]):>10.2f}")


# ---------------------------------------------------------------------------
# The model-free colour-cost estimator.
# ---------------------------------------------------------------------------
TIERS_DEFAULT = ((0.80, 0.25, 4.0), (0.60, 0.10, 10.0), (0.40, 0.02, 50.0))


def direction_matched_de(e, cand_idx, F, lab, max_rows=120, max_pairs=8000,
                         tiers=TIERS_DEFAULT, agg="median", seed=0):
    """dE00 cost of a signed concentration displacement `e`, without a model.

    Among candidate pool rows, form real formula PAIRS, keep those whose recipe
    difference points the same way as `e` (cosine) and is of comparable
    magnitude (so we interpolate rather than extrapolate), take their MEASURED
    dE00 and scale it linearly to |e|_1.

    Tries progressively looser cosine/magnitude gates and returns which tier
    produced the answer, so a loosely-matched row is never mistaken for a
    tightly-matched one.
    """
    e = np.asarray(e, dtype=float)
    l1e = float(np.abs(e).sum())
    en = float(np.linalg.norm(e))
    if l1e <= 0 or en <= 0 or len(cand_idx) < 2:
        return np.nan, 0, -1

    # Seeded per call, not from a shared stream: the candidate subsample must not
    # depend on how many rows were scored before this one, or the same settings
    # give a different answer on a re-run and the number stops being checkable.
    rng = np.random.default_rng(seed)
    idx = np.asarray(cand_idx)
    if len(idx) > max_rows:
        idx = rng.choice(idx, max_rows, replace=False)
    ia, ib = np.triu_indices(len(idx), k=1)
    if len(ia) > max_pairs:
        sel = rng.choice(len(ia), max_pairs, replace=False)
        ia, ib = ia[sel], ib[sel]
    ia, ib = idx[ia], idx[ib]

    d = F[ia] - F[ib]
    dn = np.linalg.norm(d, axis=1)
    l1d = np.abs(d).sum(axis=1)
    ok = (dn > 1e-9) & (l1d > 1e-9)
    if not ok.any():
        return np.nan, 0, -1
    cos = np.zeros(len(d))
    cos[ok] = (d[ok] @ e) / (dn[ok] * en)
    ratio = np.full(len(d), np.inf)
    ratio[ok] = l1e / l1d[ok]

    # (min_cos, ratio_lo, ratio_hi) -- tier 0 is the tight, trustworthy match.
    for tier, (min_cos, r_lo, r_hi) in enumerate(tiers):
        keep = ok & (cos >= min_cos) & (ratio >= r_lo) & (ratio <= r_hi)
        if keep.sum() >= 3:
            de = delta_e_2000(lab[ia[keep]], lab[ib[keep]]).reshape(-1)
            scaled = de * ratio[keep]
            val = np.mean(scaled) if agg == "mean" else np.median(scaled)
            return float(val), int(keep.sum()), tier
    return np.nan, 0, -1


def colour_cost_all(E, active, F_pool, supp_pool, pat_pool, lab_pool, **kw):
    """Per-row model-free dE for every test row's signed error vector."""
    n = len(E)
    de_row = np.full(n, np.nan)
    npairs = np.zeros(n, dtype=int)
    tier_row = np.full(n, -1)
    for i in range(n):
        p = active[i]
        cand = pat_pool.get(p.tobytes(), [])
        bump = 0
        if len(cand) < 4:            # same-active-set pool too thin -> subsets of it
            cand = np.where(~(supp_pool & ~p).any(axis=1))[0]
            bump = 10
        if len(cand) < 4:            # last resort: whole pool, cosine still gates
            cand = np.arange(len(F_pool))
            bump = 20
        d, k, t = direction_matched_de(E[i], cand, F_pool, lab_pool, **kw)
        de_row[i], npairs[i] = d, k
        tier_row[i] = -1 if t < 0 else t + bump
    return de_row, npairs, tier_row


def main():
    t0 = time.time()
    hdr("STAGE 2 ERROR ANATOMY  --  read-only diagnosis")
    print("run " + time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"artifact: {STAGE2_PATH}")

    train_df, test_df, light_only = production_partition()
    print(f"partition: train {len(train_df)}  test {len(test_df)}  "
          f"light_only {len(light_only)}")

    X_test, _ = training_frame_for_mode(MODE_LIGHT, test_df, None)
    true_f = test_df[CC].to_numpy(dtype=float)
    active = true_f > 0

    s2 = load_stage2(str(STAGE2_PATH))
    print("predicting under the TRUE active mask (Stage 1 excluded)...")
    pred = predict_fractions(s2, X_test, active)

    E = pred - true_f                      # signed error
    A = np.abs(E)
    n_act = active.sum(axis=1)
    row_l1 = A.sum(axis=1)                 # per-formula L1 error (fraction units)
    row_mae = row_l1 / n_act
    mae = float(A[active].mean())
    n = len(test_df)

    print(f"\nheadline: MAE over active cells {mae:.4f}  ({100 * mae:.2f} pp)   "
          f"active cells {int(active.sum())}  rows {n}")
    print(f"          per-formula L1 error: mean {100 * row_l1.mean():.2f} pp   "
          f"median {100 * np.median(row_l1):.2f} pp   max {100 * row_l1.max():.2f} pp")

    # -----------------------------------------------------------------
    # POOL (what the light model actually trained on) + colour of everything
    # -----------------------------------------------------------------
    pool = pd.concat([train_df, light_only], ignore_index=True)
    F_pool = pool[CC].to_numpy(dtype=float)
    lab_pool = reflectance_to_lab(pool[LIGHT_COLS].to_numpy(dtype=float))
    lab_test = reflectance_to_lab(test_df[LIGHT_COLS].to_numpy(dtype=float))
    supp_pool = F_pool > 0
    n_patterns = len({r.tobytes() for r in supp_pool})
    print(f"pool (train + light_only) {len(pool)} formulas; "
          f"{n_patterns} distinct active-set patterns")

    # =================================================================
    hdr("1. TAIL SHAPE -- a few bad formulas, or a broad spread?")
    # =================================================================
    total_mass = row_l1.sum()
    order = np.argsort(-row_l1)
    print(f"  total error mass (sum of |err| over all cells) = {100 * total_mass:.1f} pp")
    print(f"\n  {'worst k rows':<16}{'k':>5}{'share of error mass':>22}"
          f"{'MAE of the REST':>22}")
    for frac in (0.01, 0.05, 0.10, 0.25, 0.50):
        k = max(1, int(round(frac * n)))
        top, rest = order[:k], order[k:]
        rest_mae = A[rest][active[rest]].mean()
        share = pct(row_l1[top].sum() / total_mass)
        s = f"{rest_mae:.4f} ({100 * rest_mae:.2f} pp)"
        print(f"  {'top ' + pct(frac):<16}{k:>5}{share:>22}{s:>22}")

    for k in (5, 15, 30):
        rest = order[k:]
        rm = A[rest][active[rest]].mean()
        print(f"  drop the worst {k:>2} rows -> MAE {rm:.4f} ({100 * rm:.2f} pp), "
              f"{pct(1 - rm / mae)} lower than {mae:.4f}")

    # cell level
    cell_err = A[active]
    cs = np.sort(cell_err)[::-1]
    print(f"\n  cell level ({len(cell_err)} active cells): worst 1% of cells hold "
          f"{pct(cs[:max(1, len(cs) // 100)].sum() / cs.sum())} of the mass, "
          f"worst 5% hold {pct(cs[:len(cs) // 20].sum() / cs.sum())}")
    print(f"  cells inside the lab 0.1 pp dosing step: {pct((cell_err <= 0.001).mean())}")

    # characterise the worst rows
    white = true_f[:, WHITE_J]
    min_conc = np.nanmin(np.where(active, true_f, np.nan), axis=1)
    sum_dev = np.abs(true_f.sum(axis=1) - 1.0)
    n_trace = (active & (true_f < 0.001)).sum(axis=1)
    driver = np.argmax(A, axis=1)

    hdr("   the worst 15 formulas", "-")
    print(f"  {'formula':<16}{'L1 pp':>8}{'nact':>5}{'white':>8}{'minconc':>9}"
          f"{'|sum-1|':>9}{'trace':>6}  {'biggest-error ink':<16}{'true->pred (pp)':>18}")
    for i in order[:15]:
        j = driver[i]
        print(f"  {str(test_df['formula'][i])[:15]:<16}{100 * row_l1[i]:>8.2f}{n_act[i]:>5}"
              f"{white[i]:>8.3f}{min_conc[i]:>9.4f}{sum_dev[i]:>9.4f}{n_trace[i]:>6}  "
              f"{SHORT[j][:15]:<16}{100 * true_f[i, j]:>8.2f} ->{100 * pred[i, j]:>7.2f}")

    worst, rest = order[:15], order[15:]
    print(f"\n  {'attribute':<28}{'worst 15':>12}{'other ' + str(n - 15):>12}")
    for label, v in (("white fraction", white),
                     ("n active colorants", n_act.astype(float)),
                     ("min active conc", min_conc),
                     ("|sum(true) - 1|", sum_dev),
                     ("trace cells (<0.1 pp)", n_trace.astype(float))):
        print(f"  {label:<28}{v[worst].mean():>12.4f}{v[rest].mean():>12.4f}")
    rare_names = ("red 032", "rhodamine red", "orange 021", "warm red")
    rare = [j for j, s in enumerate(SHORT) if s in rare_names]
    has_rare = active[:, rare].any(axis=1)
    print(f"  {'contains a rare ink':<28}{pct(has_rare[worst].mean()):>12}"
          f"{pct(has_rare[rest].mean()):>12}")
    print(f"  {'near corrupt bound (>0.03)':<28}{pct((sum_dev[worst] > 0.03).mean()):>12}"
          f"{pct((sum_dev[rest] > 0.03).mean()):>12}")

    # =================================================================
    hdr("2. IS ERROR CONCENTRATED WHERE COLOUR IS UNDERDETERMINED?")
    # =================================================================
    print("  computing test x pool dE00 (light backing)...")
    D = delta_e_2000(lab_test[:, None, :], lab_pool[None, :, :])   # (n_test, n_pool)
    nn_de = D.min(axis=1)
    nn_idx = D.argmin(axis=1)
    nn_recipe_l1 = np.abs(F_pool[nn_idx] - true_f).sum(axis=1)

    amb1 = (D < 1.0).sum(axis=1)
    spread1 = np.full(n, np.nan)
    for i in range(n):
        near = np.where(D[i] < 1.0)[0]
        if len(near):
            spread1[i] = np.abs(F_pool[near] - true_f[i]).sum(axis=1).mean()
    spread_filled = np.where(np.isfinite(spread1), spread1, nn_recipe_l1)

    print(f"\n  spectral crowding: median nearest-pool dE {np.median(nn_de):.2f}, "
          f"{pct((nn_de < 1.0).mean())} of test rows have a pool formula within dE 1.0")
    print(f"  among those, mean recipe L1 difference {100 * np.nanmean(spread1):.2f} pp "
          f"-- recipes that measure the SAME but ARE different")
    print(f"  1-NN-in-colour recipe lookup would give L1 {100 * np.mean(nn_recipe_l1):.2f} pp "
          f"(model gives {100 * row_l1.mean():.2f} pp)")

    hdr("   per-formula L1 error vs. each candidate driver", "-")
    corr_line("white loading", white, row_l1)
    corr_line("n active colorants", n_act.astype(float), row_l1)
    corr_line("dE to nearest pool formula", nn_de, row_l1)
    corr_line("recipe spread among dE<1 twins", spread_filled, row_l1)
    corr_line("n pool formulas within dE 1.0", amb1.astype(float), row_l1)
    corr_line("min active concentration", min_conc, row_l1)
    corr_line("|sum(true) - 1|", sum_dev, row_l1)
    corr_line("1-NN-in-colour recipe L1", nn_recipe_l1, row_l1)
    print("\n  same, against per-CELL error (L1 / n_active) so colorant count cannot")
    print("  mechanically inflate it:")
    corr_line("white loading", white, row_mae)
    corr_line("n active colorants", n_act.astype(float), row_mae)
    corr_line("recipe spread among dE<1 twins", spread_filled, row_mae)

    hdr("   binned", "-")
    quintile_table("white loading", white, row_l1)
    print()
    for k in sorted(set(n_act.tolist())):
        m = n_act == k
        if m.sum() >= 3:
            print(f"  n_active = {k:<2}  n={int(m.sum()):>4}   "
                  f"mean L1 {100 * row_l1[m].mean():>6.2f} pp   "
                  f"mean per-cell {100 * row_mae[m].mean():>5.2f} pp")
    print()
    quintile_table("recipe spread(dE<1)", spread_filled, row_l1)

    hdr("   is the white correlation a MODEL failure, or just SCALE?", "-")
    print("  A recipe that is 95% white has only 5 pp of coloured mass to get wrong, so a")
    print("  low ABSOLUTE error there is arithmetic, not skill. Dividing the row error by")
    print("  the coloured mass (1 - white) removes that effect. If the correlation")
    print("  survives, low-white formulas really are harder; if it flips, the error is")
    print("  simply proportional to how much colorant is in the can.")
    coloured = np.maximum(1.0 - white, 1e-6)
    rel = row_l1 / coloured
    corr_line("white  vs  ABSOLUTE row L1", white, row_l1)
    corr_line("white  vs  L1 / coloured mass", white, rel)
    print()
    quintile_table("white loading", white, rel, value_label="L1 / coloured (%)")
    for label, m in (("white ABSENT from recipe", ~active[:, WHITE_J]),
                     ("white present", active[:, WHITE_J])):
        if m.any():
            print(f"  {label:<26} n={int(m.sum()):>4}  mean L1 {100*row_l1[m].mean():>6.2f} pp"
                  f"   mean L1/coloured {100*rel[m].mean():>6.2f}%"
                  f"   share of error mass {pct(row_l1[m].sum()/total_mass)}")

    # =================================================================
    hdr("3. WHICH ERRORS ACTUALLY COST COLOUR?  (model-free dE)")
    # =================================================================
    pat_pool = {}
    for r in range(len(F_pool)):
        pat_pool.setdefault(supp_pool[r].tobytes(), []).append(r)

    de_row, npairs, tier_row = colour_cost_all(
        E, active, F_pool, supp_pool, pat_pool, lab_pool)

    ok = np.isfinite(de_row)
    print(f"  matched {int(ok.sum())}/{n} rows   median {int(np.median(npairs[ok]))} pairs/row")
    print(f"  match quality: same-active-set "
          f"{int(((tier_row >= 0) & (tier_row < 10)).sum())}, subset-of-active-set "
          f"{int(((tier_row >= 10) & (tier_row < 20)).sum())}, whole-pool "
          f"{int((tier_row >= 20).sum())}, unmatched {int((tier_row < 0).sum())}")
    print(f"\n  COLOUR COST OF STAGE 2 ERROR: mean {np.nanmean(de_row):.2f} dE00   "
          f"median {np.nanmedian(de_row):.2f}   p90 {np.nanpercentile(de_row, 90):.2f}   "
          f"max {np.nanmax(de_row):.2f}")
    print(f"  rows already under the client dE 1.0 target: {pct(np.nanmean(de_row[ok] < 1.0))}")

    sens = de_row / row_l1                       # dE per unit L1 of concentration error
    print(f"\n  sensitivity, dE per 1 pp of L1 error: median {np.nanmedian(sens) / 100:.3f}   "
          f"p10 {np.nanpercentile(sens, 10) / 100:.3f}   "
          f"p90 {np.nanpercentile(sens, 90) / 100:.3f}   "
          f"ratio p90/p10 {np.nanpercentile(sens, 90) / np.nanpercentile(sens, 10):.1f}x")
    print("  (a wide ratio means equal pp errors do NOT cost equal colour)")

    hdr("   how much does that number depend on the estimator's settings?", "-")
    print("  The scaling assumes dE is locally linear in the displacement, so a looser")
    print("  cosine gate or a mean-instead-of-median moves it. Reported so the headline")
    print("  is read as a band, not a point.")
    print(f"  {'settings':<40}{'matched':>9}{'mean dE':>10}{'median dE':>11}")
    variants = [
        ("median, cos>=0.80 (headline)", dict(agg="median")),
        ("mean of matched pairs", dict(agg="mean")),
        ("median, cos>=0.90 tight only", dict(agg="median", tiers=((0.90, 0.5, 2.0),))),
        ("median, cos>=0.50 loose only", dict(agg="median", tiers=((0.50, 0.05, 20.0),))),
        ("headline, different subsample seed", dict(agg="median", seed=7)),
        ("headline, different subsample seed", dict(agg="median", seed=13)),
    ]
    for label, kw in variants:
        dv, _, _ = colour_cost_all(E, active, F_pool, supp_pool, pat_pool, lab_pool, **kw)
        okv = np.isfinite(dv)
        print(f"  {label:<40}{int(okv.sum()):>9}{np.nanmean(dv):>10.2f}"
              f"{np.nanmedian(dv):>11.2f}")

    hdr("   is MAE ranking the same rows dE does?", "-")
    corr_line("row L1 (pp)  vs  row dE", row_l1[ok], de_row[ok])
    corr_line("row per-cell MAE vs row dE", row_mae[ok], de_row[ok])
    de_mass = np.nansum(de_row)
    o_de = np.argsort(-np.nan_to_num(de_row))
    for frac in (0.05, 0.10):
        k = max(1, int(round(frac * n)))
        overlap = len(set(o_de[:k].tolist()) & set(order[:k].tolist()))
        print(f"  top {pct(frac)} by dE holds "
              f"{pct(np.nansum(de_row[o_de[:k]]) / de_mass)} of dE mass; "
              f"overlap with top {pct(frac)} by pp = {overlap}/{k} rows")
    print(f"\n  rows in the worst 30 by pp:  mean {np.nanmean(de_row[order[:30]]):.2f} dE")
    print(f"  rows in the worst 30 by dE:  mean {100 * row_l1[o_de[:30]].mean():.2f} pp L1")
    cheap = ok & (row_l1 > np.percentile(row_l1, 90)) & (de_row < 1.0)
    dear = ok & (row_l1 < np.percentile(row_l1, 50)) & (de_row > 3.0)
    print(f"  BIG pp but dE < 1.0 (harmless):    {int(cheap.sum())} rows")
    print(f"  SMALL pp but dE > 3.0 (expensive): {int(dear.sum())} rows")

    hdr("   the worst 15 by pp, priced in dE and placed in the pool", "-")
    print(f"  {'formula':<16}{'L1 pp':>8}{'dE cost':>9}{'nearest pool dE':>17}"
          f"{'twin recipe L1 pp':>19}")
    for i in order[:15]:
        print(f"  {str(test_df['formula'][i])[:15]:<16}{100 * row_l1[i]:>8.2f}"
              f"{de_row[i]:>9.2f}{nn_de[i]:>17.2f}{100 * spread_filled[i]:>19.2f}")

    hdr("   what one percentage point of each ink costs, in dE", "-")
    print("  (model-free: +1 pp of the ink against -1 pp of transparent white,")
    print("   matched to real pool pairs containing both)")
    print(f"  {'ink':<18}{'dE per 1 pp':>13}{'mean |err| pp':>15}"
          f"{'dE at that error':>19}{'pairs':>7}")
    unit_rows = []
    for j, name in enumerate(COLORANT_NAMES):
        if j == WHITE_J:
            continue
        e = np.zeros(len(COLORANT_NAMES))
        e[j], e[WHITE_J] = 0.01, -0.01
        cand = np.where(supp_pool[:, j] & supp_pool[:, WHITE_J])[0]
        # Near-exhaustive here: qualifying pairs (a small, real dose change of
        # this one ink) are rare, so a 120-row subsample finds only a handful
        # and the estimate swings with the draw.
        d, k, _ = direction_matched_de(e, cand, F_pool, lab_pool,
                                       max_rows=600, max_pairs=400_000)
        m = active[:, j]
        mean_err = A[m, j].mean() if m.any() else np.nan
        cost = d * mean_err / 0.01 if np.isfinite(d) else np.nan
        unit_rows.append((SHORT[j], d, mean_err, cost, k))
    for nm, d, me, cost, k in sorted(
            unit_rows, key=lambda r: -(r[3] if np.isfinite(r[3]) else -1)):
        flag = "  (few pairs -- imprecise)" if k < 10 else ""
        print(f"  {nm:<18}{d:>13.2f}{100 * me:>15.2f}{cost:>19.2f}{k:>7}{flag}")

    # =================================================================
    hdr("4. SYSTEMATIC BIAS")
    # =================================================================
    row_signed = E.sum(axis=1)
    print(f"  predicted rows sum to 1 by construction: max |sum(pred)-1| = "
          f"{np.abs(pred.sum(axis=1) - 1).max():.2e}")
    print(f"  true rows sum to 1 +/- {sum_dev.mean():.4f} (mean), max {sum_dev.max():.4f}")
    print(f"  => signed errors within a row sum to {row_signed.mean():+.4f} (mean), "
          f"|.| max {np.abs(row_signed).max():.4f}: an over-prediction IS an "
          f"under-prediction elsewhere")

    hdr("   per-colorant: shrinkage and direction", "-")
    print(f"  {'ink':<18}{'n':>5}{'MAE pp':>9}{'bias pp':>9}{'over%':>7}"
          f"{'slope':>8}{'sd(pred)/sd(true)':>19}{'r':>7}")
    for j, name in enumerate(COLORANT_NAMES):
        m = active[:, j]
        if m.sum() < 3:
            continue
        t, p_ = true_f[m, j], pred[m, j]
        bias = (p_ - t).mean()
        over = (p_ > t).mean()
        if t.std() > 1e-9:
            slope = np.polyfit(t, p_, 1)[0]
            r = np.corrcoef(t, p_)[0, 1]
            sdr = p_.std() / t.std()
        else:
            slope = r = sdr = np.nan
        print(f"  {SHORT[j]:<18}{int(m.sum()):>5}{100 * np.abs(p_ - t).mean():>9.2f}"
              f"{100 * bias:>9.2f}{100 * over:>6.0f}%{slope:>8.2f}{sdr:>19.2f}{r:>7.2f}")
    print("  slope = OLS of pred on true over active cells; < 1.0 means shrinkage "
          "toward the mean")

    hdr("   who absorbs the compensating error?", "-")
    absorb = np.zeros(len(COLORANT_NAMES), dtype=int)
    drive = np.zeros(len(COLORANT_NAMES), dtype=int)
    for i in range(n):
        j = int(np.argmax(A[i]))
        drive[j] += 1
        opp = np.where(np.sign(E[i]) == -np.sign(E[i, j]))[0]
        if len(opp):
            absorb[int(opp[np.argmax(A[i, opp])])] += 1
    signed_mass = E.sum(axis=0)
    print(f"  {'ink':<18}{'largest error in row':>22}{'largest opposing error':>24}"
          f"{'net signed pp':>15}")
    for j in np.argsort(-(drive + absorb)):
        if drive[j] + absorb[j] == 0:
            continue
        print(f"  {SHORT[j]:<18}{drive[j]:>16} rows {absorb[j]:>18} rows"
              f"{100 * signed_mass[j]:>15.1f}")
    col = np.ones(len(COLORANT_NAMES), dtype=bool)
    col[WHITE_J] = False
    r_wc = np.corrcoef(E[:, WHITE_J], E[:, col].sum(axis=1))[0, 1]
    print(f"\n  corr(white signed error, summed coloured signed error) = {r_wc:+.3f} "
          f"(-1 = white absorbs everything)")
    print(f"  share of total |error| mass carried by transparent white: "
          f"{pct(A[:, WHITE_J].sum() / total_mass)}")

    # =================================================================
    hdr("5. CEILING -- what is the best plausible Stage 2 MAE?")
    # =================================================================
    allf = pd.concat([pool, test_df], ignore_index=True)
    F_all = allf[CC].to_numpy(dtype=float)
    lab_all = reflectance_to_lab(allf[LIGHT_COLS].to_numpy(dtype=float))
    groups = {}
    for r in range(len(F_all)):
        groups.setdefault(np.round(F_all[r], 6).tobytes(), []).append(r)
    dup = [g for g in groups.values() if len(g) > 1]
    dup_de = []
    for g in dup:
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                dup_de.append(float(
                    delta_e_2000(lab_all[g[a]], lab_all[g[b]]).reshape(-1)[0]))
    if dup_de:
        dup_de = np.array(dup_de)
        print(f"  REPEATABILITY: {len(dup)} recipes appear more than once in the file "
              f"({len(dup_de)} pairs).")
        print(f"    dE00 between measurements of the SAME recipe: mean {dup_de.mean():.2f}  "
              f"median {np.median(dup_de):.2f}  p90 {np.percentile(dup_de, 90):.2f}")
        print("    This is a floor: identical recipes do not measure identically.")
    else:
        print("  REPEATABILITY: no exactly-duplicated recipes in the file.")

    # No exact duplicates means the floor has to come from NEAR-duplicates: pairs
    # of recipes so close that no dosing difference could explain a large dE.
    print("\n  NEAR-DUPLICATE FLOOR: pairs of recipes differing by at most a few tenths")
    print("  of a pp in total. Whatever dE separates them is measurement + process")
    print("  noise, not formulation -- the hard floor on any model.")
    print(f"  {'recipe L1 <=':>14}{'pairs':>8}{'mean dE':>10}{'median dE':>11}{'p90 dE':>9}")
    for tol in (0.002, 0.005, 0.01):
        aa, bb = [], []
        for start in range(0, len(F_all), 200):
            blk = F_all[start:start + 200]
            l1 = np.abs(blk[:, None, :] - F_all[None, :, :]).sum(axis=2)
            ii, jj = np.where(l1 <= tol)
            keep = (ii + start) < jj
            aa.extend((ii[keep] + start).tolist())
            bb.extend(jj[keep].tolist())
        if len(aa) >= 3:
            de = delta_e_2000(lab_all[np.array(aa)], lab_all[np.array(bb)]).reshape(-1)
            print(f"  {100 * tol:>11.1f} pp{len(de):>8}{de.mean():>10.2f}"
                  f"{np.median(de):>11.2f}{np.percentile(de, 90):>9.2f}")
        else:
            print(f"  {100 * tol:>11.1f} pp{len(aa):>8}   (too few pairs)")

    amb_pairs = []
    for i in range(n):
        near = np.where(D[i] < 1.0)[0]
        for j_ in near:
            l1 = float(np.abs(F_pool[j_] - true_f[i]).sum())
            if l1 > 0.005:
                amb_pairs.append(l1)
    if amb_pairs:
        amb_pairs = np.array(amb_pairs)
        print(f"\n  AMBIGUITY: {len(amb_pairs)} test/pool pairs measure within dE 1.0 of "
              f"each other yet differ in recipe.")
        print(f"    recipe L1 between them: median {100 * np.median(amb_pairs):.2f} pp  "
              f"mean {100 * amb_pairs.mean():.2f} pp")
        print(f"    A model seeing only the light spectrum can be handed either recipe; "
              f"half that L1")
        print(f"    ({100 * np.median(amb_pairs) / 2:.2f} pp) is the expected error from "
              f"picking the wrong twin.")

    print("\n  IRREDUCIBLE SHARE, per row: a row error is unfixable once its colour cost")
    print("  falls under the measurement floor -- nothing in the spectrum distinguishes it.")
    print(f"  {'dE floor':>9}{'rows already under it':>24}"
          f"{'their share of error mass':>28}{'ceiling MAE':>22}")
    for floor in DE_FLOOR_BAND:
        l1_irr = floor / sens                     # L1 that costs exactly `floor` dE
        under = ok & (de_row <= floor)
        best_l1 = np.where(ok, np.minimum(row_l1, l1_irr), row_l1)
        ceiling = best_l1.sum() / n_act.sum()
        s = f"{ceiling:.4f} ({100 * ceiling:.2f} pp)"
        print(f"  {floor:>9.1f}{pct(under.mean()):>24}"
              f"{pct(row_l1[under].sum() / total_mass):>28}{s:>22}")
    print("\n  Read: 'ceiling MAE' is what a PERFECT model would still score -- one that")
    print("  fixed every row down to the point where the residual is invisible in colour.")
    print(f"  Current MAE {mae:.4f}. The gap between the two is the only part worth chasing.")

    print(f"\n[done in {time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
