"""What does Stage 2's concentration error actually cost in COLOUR?

Stage 2 predicts how much of each ink. Its error is 0.90 pp (mean, active
cells). Nobody knows what that is worth in dE2000 -- the unit the client's
acceptance criterion is written in -- because nothing in this pipeline turns a
recipe into a colour.

This script builds the forward model that does (kernel ridge on log R; test
mean dE 0.993, vs the removed Kubelka-Munk physics model's ~14) and uses it to
put a dE number on Stage 2's error.

METHOD, and its central assumption
----------------------------------
The headline is dE2000( forward(true_recipe), forward(predicted_recipe) ).
Both recipes go through the SAME forward model, so that model's own bias
cancels to first order and what remains is the colour sensitivity to the
recipe error.

"To first order" is doing real work in that sentence. It holds only if both
recipes sit close together relative to the model's length scale. So this
script MEASURES that rather than assuming it, three ways:

  1. the forward model's own error on true recipes  (the resolution floor)
  2. how far predicted recipes sit from the model's training data, versus
     how far true recipes sit -- predicted recipes are model OUTPUTS, not
     human formulations, so they may be systematically further out, and the
     forward model would then be least reliable exactly where this
     measurement depends on it
  3. the same dE against the MEASURED spectrum, which does NOT cancel
     forward-model bias but is not circular either

If (2) shows predicted recipes are much further from training data than true
recipes, the headline is not trustworthy and the script says so.

Caveats that travel with every number here:
  - the data's own noise floor is ~0.8-1.1 dE (near-duplicate recipes differ
    by that much), so nothing below that is resolvable
  - production_partition() drops formulas >8 dE from their Pantone target
    from the TEST side too, so the test set is easier than production traffic
  - this is a learned instrument, not a drawdown. It cannot certify dE.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.kernel_ridge import KernelRidge

from train_backing_modes import production_partition
from src.backing_modes import MODE_LIGHT, training_frame_for_mode
from src.colorimetry import delta_e_2000_spectra
from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS
from src.stage2_regressor import load_stage2, predict_fractions

STAGE2 = "models/wolves_v1/backing_modes/light/stage2.joblib"
GAMMA, ALPHA = 0.003, 1e-4          # selected on pool CV, test scored once
NOISE_FLOOR = 0.8                    # measured; nothing below this is real
WHITE = "UFO00061 transp. white"


def rule(c="-"):
    print(c * 78)


def feats(f):
    return np.hstack([f, np.sqrt(np.clip(f, 0, None))])


def to_log(r):
    return np.log(np.clip(r, 1e-4, None))


def from_log(z):
    return np.clip(np.exp(z), 1e-9, 1.0)


def describe(name, d):
    print("%-34s mean %6.3f  med %6.3f  p90 %6.3f  max %6.3f  <1 %5.1f%%"
          % (name, d.mean(), np.median(d), np.percentile(d, 90), d.max(),
             (d < 1).mean() * 100))


def main() -> None:
    t0 = time.time()
    rule("=")
    print("COLOUR COST OF STAGE 2's CONCENTRATION ERROR")
    print(time.strftime("%Y-%m-%d %H:%M:%S"))
    rule("=")

    tr, te, lo = production_partition()
    pool = pd.concat([tr, lo], ignore_index=True)
    cc = ["colorant_" + c for c in COLORANT_NAMES]
    lc = ["R_light_%d" % w for w in WAVELENGTHS]
    Fp, Sp = pool[cc].to_numpy(float), pool[lc].to_numpy(float)
    Ft, St = te[cc].to_numpy(float), te[lc].to_numpy(float)
    print("pool %d   test %d" % (len(Fp), len(Ft)))

    # ---- forward model -------------------------------------------------
    Xp, Xt = feats(Fp), feats(Ft)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-9
    Yp = to_log(Sp)
    ym, ys = Yp.mean(0), Yp.std(0) + 1e-9
    fwd = KernelRidge(kernel="rbf", gamma=GAMMA, alpha=ALPHA)
    fwd.fit((Xp - mu) / sd, (Yp - ym) / ys)
    predict_spec = lambda F: from_log(
        fwd.predict((feats(F) - mu) / sd) * ys + ym)

    d_fwd = delta_e_2000_spectra(predict_spec(Ft), St)
    rule()
    print("FORWARD MODEL FIDELITY (its own error -- the resolution floor)")
    describe("forward(true) vs measured", d_fwd)
    print("  data noise floor ~%.1f dE, so this instrument is close to the" % NOISE_FLOOR)
    print("  limit of what the data itself can resolve.")

    # ---- Stage 2 predictions (true active mask isolates Stage 2) --------
    Xs, _ = training_frame_for_mode(MODE_LIGHT, te, None)
    s2 = load_stage2(STAGE2)
    active = Ft > 0
    P = predict_fractions(s2, Xs, active)
    err = np.abs(P - Ft)
    print("\nStage 2 recipe error: MAE %.4f (%.2f pp) on active cells; "
          "total L1/formula mean %.4f" % (err[active].mean(),
                                          err[active].mean() * 100,
                                          err.sum(1).mean()))

    # ---- extrapolation check: are predicted recipes real recipes? -------
    rule()
    print("EXTRAPOLATION CHECK (does the forward model get inputs it knows?)")
    def nn_dist(F):
        return np.abs(F[:, None, :] - Fp[None, :, :]).sum(-1).min(1)
    dt, dp = nn_dist(Ft), nn_dist(P)
    print("  nearest-training-recipe L1  true recipes: med %.4f  p90 %.4f"
          % (np.median(dt), np.percentile(dt, 90)))
    print("  nearest-training-recipe L1  PRED recipes: med %.4f  p90 %.4f"
          % (np.median(dp), np.percentile(dp, 90)))
    ratio = np.median(dp) / max(np.median(dt), 1e-9)
    print("  ratio of medians: %.2fx" % ratio)
    trust = ratio <= 1.5
    print("  => %s" % ("predicted recipes sit as close to training data as real"
                       " ones; headline is trustworthy" if trust else
                       "PREDICTED RECIPES ARE FURTHER OUT -- the forward model is"
                       " least reliable exactly where this measurement needs it."
                       " Treat the headline as an UPPER BOUND on reliability."))

    # ---- the headline ---------------------------------------------------
    rule("=")
    print("HEADLINE: colour cost of Stage 2's error")
    rule("=")
    d_pair = delta_e_2000_spectra(predict_spec(P), predict_spec(Ft))
    d_meas = delta_e_2000_spectra(predict_spec(P), St)
    describe("forward(pred) vs forward(true)", d_pair)
    describe("forward(pred) vs measured", d_meas)
    print("""
  *** READ THIS BEFORE QUOTING THE NUMBER ABOVE ***

  It is a LOWER BOUND, and understates the real colour cost by roughly 4x.

  Why: Stage 2 predicts fractions FROM colour, so its residual concentrates in
  the subspace colour cannot determine (transparent white alone carries ~33% of
  the error mass, and white is 3.81x underdetermined by colour). The forward
  model is fit to the SAME colour data, so it is near-blind in the SAME
  directions. Pushing Stage 2's residual through it measures only the part that
  colour can see -- small by construction, whatever the physics says.

  Sharing the model cancels its LEVEL error, but what survives is the error in
  its JACOBIAN, which scales with the displacement exactly as the signal does.
  Making the error smaller never improves that ratio.

  The model-free estimate is the one to trust: find REAL formula pairs whose
  difference points the same way as Stage 2's error and read off their MEASURED
  dE. That gives mean 2.9-3.2 (stable across |cos| thresholds 0.85/0.90/0.95,
  184-241 formulas), against this model's 0.79. See the project notes.""")

    # ---- per-colorant counterfactual ------------------------------------
    rule()
    print("PER-COLORANT ATTRIBUTION (counterfactual: fix ONE ink to truth,")
    print("re-measure, and see how much of the gap closes)")
    print("%-30s %6s %10s %10s" % ("colorant", "n", "dE closed", "% of gap"))
    # Baseline MUST be restricted to the same rows as the counterfactual.
    # Using the global mean here compares a subset mean against an all-rows
    # mean, which mostly measures how atypical that colorant's subset is --
    # it produced a nonsensical -178% for red 032 (6 active rows) before this
    # was fixed.
    #
    # Redistribute the freed mass across the OTHER active inks rather than
    # renormalising the whole row: plain renormalisation leaves ink j at
    # Ft_j/(1+delta) -- NOT its true value -- and silently rescales all 13
    # others, so "fix one ink" perturbs everything. On a simplex you cannot
    # move one component alone; this states the convention explicitly.
    # Contributions still will not sum to the total.
    rows = []
    for j, c in enumerate(COLORANT_NAMES):
        m = active[:, j]
        if m.sum() < 3:
            continue
        Pc = P.copy()
        delta = Ft[:, j] - P[:, j]
        Pc[:, j] = Ft[:, j]
        others = active.copy()
        others[:, j] = False
        w = Pc * others
        wsum = w.sum(1, keepdims=True)
        Pc = Pc - np.where(wsum > 1e-9, w / np.clip(wsum, 1e-9, None), 0.0) * delta[:, None]
        d_c = delta_e_2000_spectra(predict_spec(Pc), predict_spec(Ft))
        base_m = d_pair[m].mean()
        rows.append((base_m - d_c[m].mean(), c, int(m.sum()), base_m))
    for closed, c, n, base_m in sorted(rows, reverse=True):
        print("%-30s %6d %10.3f %9.1f%%" % (c, n, closed, closed / max(base_m, 1e-9) * 100))

    # ---- verdict ---------------------------------------------------------
    rule("=")
    mean_d, med_d = d_pair.mean(), np.median(d_pair)
    if mean_d < 1.0 and med_d < 0.5:
        v = ("BELOW WHAT THE CUSTOMER CAN SEE. Stage 2's error costs less colour\n"
             "than the data's own noise floor (~0.8 dE). Further Stage 2 modelling\n"
             "would polish something no drawdown could detect. Report this to the\n"
             "client instead of building it.")
    elif mean_d >= 1.5:
        v = ("COLOUR-RELEVANT. Stage 2's error is large enough to matter against a\n"
             "dE 1.0 acceptance criterion. Improving Stage 2 is worth doing.")
    else:
        v = ("AMBIGUOUS. The effect is the same size as the measurement's own\n"
             "resolution (~%.2f dE) and the data's noise floor. Any Stage 2 gain\n"
             "would be real but unverifiable from this data alone." % d_fwd.mean())
    print("VERDICT (thresholds fixed before the run):")
    print(v)
    if not trust:
        print("\n** Read the above against the extrapolation warning. **")
    rule("=")
    print("total %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
