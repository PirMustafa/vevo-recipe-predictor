"""Wolves v1 evaluation: does lowering the presence threshold to the client's
real working minimum (0.1%) produce a better model than the 2% one?

Read-only. Loads existing artifacts, trains nothing, writes no models.

Why this script computes labels itself instead of using build_labels():
ACTIVE_THRESHOLD is bound at import time, so a single process can only ever
see one value of it. The whole point here is to score the SAME predictions
against TWO label definitions, so labels are derived directly from the true
fractions at explicit thresholds. That also makes the numbers below immune to
whether VEVO_ACTIVE_THRESHOLD happened to be set when this was run.

Stage 1 predictions do not depend on ACTIVE_THRESHOLD at all: predict_active()
thresholds PROBABILITIES (default 0.5), not concentrations.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from train_backing_modes import production_partition
from src.backing_modes import MODE_LIGHT, training_frame_for_mode
from src.cxf_parser import COLORANT_NAMES
from src.stage1_classifier import load_stage1, predict_active
from src.stage2_regressor import load_stage2, predict_fractions

ROOT = pathlib.Path(__file__).parent
OLD = ROOT / "models" / "backing_modes" / "light"
NEW = ROOT / "models" / "wolves_v1" / "backing_modes" / "light"
T_OLD, T_NEW = 0.02, 0.001
WHITE = "UFO00061 transp. white"

OLD_TAG = "old (2%-trained)"
NEW_TAG = "new (0.1%-trained)"


def rule(ch="-"):
    print(ch * 78)


def exact(pred, lab):
    return float((pred == lab).all(axis=1).mean())


def main() -> None:
    t0 = time.time()
    rule("=")
    print("WOLVES v1 - presence-threshold evaluation")
    print("run at " + time.strftime("%Y-%m-%d %H:%M:%S"))
    rule("=")

    train_df, test_df, light_only = production_partition()
    X_test, _ = training_frame_for_mode(MODE_LIGHT, test_df, None)
    cc = ["colorant_" + c for c in COLORANT_NAMES]
    true_f = test_df[cc].to_numpy(dtype=float)
    n, k = true_f.shape

    lab_old = true_f > T_OLD
    lab_new = true_f > T_NEW
    same = int((lab_old == lab_new).all(axis=1).sum())
    ceiling = same / n
    gain = (lab_new.sum() - lab_old.sum()) / lab_old.sum() * 100

    print("\ntest rows %d - %d binary decisions" % (n, n * k))
    print("active cells   @%.3f: %d   @%.3f: %d  (+%.1f%%)"
          % (T_OLD, lab_old.sum(), T_NEW, lab_new.sum(), gain))
    print("mean actives/row @%.3f: %.3f   @%.3f: %.3f"
          % (T_OLD, lab_old.sum(1).mean(), T_NEW, lab_new.sum(1).mean()))
    print("\nrows whose active set is IDENTICAL under both: %d/%d" % (same, n))
    print("==> CEILING for a 2%%-trained model scored on 0.1%% labels: %.2f%%"
          % (ceiling * 100))
    print("    (a 2% model that were PERFECT at its own task could not beat this)")

    available = {}
    for tag, d in ((OLD_TAG, OLD), (NEW_TAG, NEW)):
        s1, s2 = d / "stage1.joblib", d / "stage2.joblib"
        if s1.exists() and s2.exists():
            available[tag] = (s1, s2)
        else:
            print("\n[skip] %s: artifacts not found at %s" % (tag, d))

    results = {}
    for tag, (s1p, s2p) in available.items():
        rule()
        print("MODEL: %s   <- %s" % (tag, s1p.parent))
        t = time.time()
        m1 = load_stage1(str(s1p))
        pred = predict_active(m1, X_test)
        print("  Stage 1 predict: %.1fs" % (time.time() - t))
        m2 = load_stage2(str(s2p))
        mask_true = true_f > 0
        frac_anchor = predict_fractions(m2, X_test, mask_true)
        frac_e2e = predict_fractions(m2, X_test, pred)

        union = mask_true | pred.astype(bool)
        nonwhite = [i for i, c in enumerate(COLORANT_NAMES) if c != WHITE]
        r = dict(
            e_old=exact(pred, lab_old),
            e_new=exact(pred, lab_new),
            mae_anchor=float(np.abs(frac_anchor - true_f)[mask_true].mean()),
            mae_e2e=float(np.abs(frac_e2e - true_f)[union].mean()),
            per_dec=float((pred == lab_new).mean()),
            over3=int((pred[:, nonwhite].sum(axis=1) > 3).sum()),
            pred=pred,
            frac_e2e=frac_e2e,
        )
        results[tag] = r
        print("  exact-match vs 2%%   labels: %.2f%%" % (r["e_old"] * 100))
        print("  exact-match vs 0.1%% labels: %.2f%%" % (r["e_new"] * 100))
        print("  Stage 2 MAE (true-active mask, THRESHOLD-INVARIANT): %.4f" % r["mae_anchor"])
        print("  end-to-end MAE (OWN union mask - NOT comparable across models): %.4f"
              % r["mae_e2e"])
        print("  per-decision accuracy vs 0.1%%: %.4f%%" % (r["per_dec"] * 100))
        print("  predictions with >3 non-white colorants: %d/%d" % (r["over3"], n))

    if OLD_TAG in results and NEW_TAG in results:
        rule("=")
        print("THE 2x2 - same %d rows, two label definitions" % n)
        rule("=")
        print("%-26s %16s %16s" % ("", "vs 2% labels", "vs 0.1% labels"))
        for tag in (OLD_TAG, NEW_TAG):
            r = results[tag]
            print("%-26s %15.2f%% %15.2f%%" % (tag, r["e_old"] * 100, r["e_new"] * 100))
        d = (results[NEW_TAG]["e_new"] - results[OLD_TAG]["e_new"]) * 100
        print("\nceiling for the old model on 0.1%% labels: %.2f%%" % (ceiling * 100))
        print("HEADLINE: on the client's actual question (0.1%%), new scores "
              "%.2f%% vs old %.2f%%  (%+.2f pts)"
              % (results[NEW_TAG]["e_new"] * 100, results[OLD_TAG]["e_new"] * 100, d))
        print("\nTHRESHOLD-INVARIANT ANCHORS (the fair did-we-regress test)")
        # End-to-end MAE must be scored over ONE fixed cell population or the
        # two numbers are means over different denominators: each model's own
        # union mask includes its own predictions, and the 0.001 model flags
        # ~13% more actives. Fixed mask = true actives OR either model's
        # prediction, so both are scored on identical cells.
        shared = (true_f > 0) | results[OLD_TAG]["pred"].astype(bool) \
                              | results[NEW_TAG]["pred"].astype(bool)
        print("  (end-to-end MAE below uses ONE shared mask of %d cells,"
              " so the two are directly comparable)" % int(shared.sum()))
        for tag in (OLD_TAG, NEW_TAG):
            r = results[tag]
            e2e_shared = float(np.abs(r["frac_e2e"] - true_f)[shared].mean())
            print("  %-26s Stage2 MAE %.4f | e2e MAE (shared mask) %.4f"
                  % (tag, r["mae_anchor"], e2e_shared))
        print("\n  NOTE: Stage 2 never reads ACTIVE_THRESHOLD (it masks on >0),")
        print("  so the two Stage 2 models solve an IDENTICAL problem. Any gap")
        print("  in Stage2 MAE is retraining noise, not a threshold effect.")
        print("  Only Stage 1 can genuinely move.")

    if results:
        tag = NEW_TAG if NEW_TAG in results else list(results)[0]
        pred = results[tag]["pred"]
        rule()
        print("PER-COLORANT (model: %s, vs 0.1%% labels)" % tag)
        print("%-32s %8s %7s %7s %7s" % ("colorant", "support", "prec", "rec", "F1"))
        for j, c in enumerate(COLORANT_NAMES):
            print("%-32s %8d %7.3f %7.3f %7.3f" % (
                c, int(lab_new[:, j].sum()),
                precision_score(lab_new[:, j], pred[:, j], zero_division=0),
                recall_score(lab_new[:, j], pred[:, j], zero_division=0),
                f1_score(lab_new[:, j], pred[:, j], zero_division=0)))
        print("%-32s %8s %7s %7s %7.3f" % (
            "macro-F1", "", "", "",
            f1_score(lab_new, pred, average="macro", zero_division=0)))

    rule("=")
    print("COLOUR ACCURACY (dE2000) - NOT REPORTED, AND WHY")
    rule("=")
    print(
        "The client's acceptance criterion is dE2000 1:1:1 below 1.0. No number\n"
        "resembling that appears above, and none can. dE2000 needs a SPECTRUM.\n"
        "This pipeline maps spectrum -> recipe and has no forward model\n"
        "(recipe -> spectrum); the Kubelka-Munk stage that once did was removed\n"
        "for being too inaccurate to use. No predicted recipe has ever been\n"
        "mixed and drawn down. Everything above is RECIPE-space accuracy: does\n"
        "the predicted recipe match the recorded one. It is not colour accuracy\n"
        "and must not be presented as such. The client's planned validations on\n"
        "these samples are what would close this gap."
    )
    print("\ntotal %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
