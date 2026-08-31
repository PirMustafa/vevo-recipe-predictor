"""Recipe-quality metrics that account for how much a disagreement actually matters.

Why this exists. Stage 1's headline metric is *exact colorant-set match*: a row
counts only if all 14 presence decisions are simultaneously right. Presence is
defined by `ACTIVE_THRESHOLD`, which is CONFIGURABLE (it defaults to 0.02 but
is overridable via the VEVO_ACTIVE_THRESHOLD environment variable -- see
src/cxf_parser.py). The numbers quoted below were measured at the 0.02 default,
where a formula containing 1.96% of a colorant is labelled ABSENT and one
containing 2.06% is labelled PRESENT. At a different threshold they no longer
apply and must be re-measured.

Measured on the production test set by re-running the deployed classifier
(288 of 297 rows, 43 wrong decisions; that subset reproduces the headline at
87.153% vs 87.205%, so it is representative):

    21 of 43 wrong decisions (49%) involve a trace amount straddling that
    0.02 line -- e.g. rubine red at 0.0196 vs 0.0224, black at 0.0199 vs
    0.0206, process blue at 0.0193 vs 0.0202.

CORRECTION 2026-08-20: this said "25 of 44 (57%)" until classify_errors was
run against the deployed model's own predictions rather than quoted from an
earlier note. The real split is very close to half, not 57%, which leaves
~22 genuinely wrong decisions out of 4158 -- the figure the rest of the
project's analysis already used.

The model is being scored wrong for failing to resolve one part in a thousand
of ink, a distinction no formulator can act on. That understates real
performance and, worse, misdirects effort: it makes trace-colorant behaviour
look like the dominant problem when it is mostly a labelling artifact.

So this module scores the *recipe*, not the presence flags: two recipes agree
when no colorant differs by more than a tolerance. Report it ALONGSIDE
exact-match, never instead of it -- exact-match remains the strict,
comparable-to-history number, and every figure in README.md is quoted in those
terms.

This is NOT the per-colorant threshold calibration that failed three times
(see README "What didn't work"). Those attempts tuned the classifier's
*decision* thresholds to score better against an unchanged metric. This changes
what counts as an error in the first place, and tunes nothing.
"""
from __future__ import annotations

import numpy as np

from .cxf_parser import ACTIVE_THRESHOLD

# Half-width of the band around ACTIVE_THRESHOLD inside which a presence
# disagreement is not counted. 0.01 around a 0.02 threshold forgives true
# fractions in [0.01, 0.03] -- the region where "present" vs "absent" is a
# labelling convention rather than a fact about the ink.
#
# IT MUST SCALE WITH THE THRESHOLD, and a fixed 0.01 does not. Any deadband >=
# ACTIVE_THRESHOLD forgives EVERY true zero (|0 - T| = T <= deadband), which
# silently turns deadband_match into "did you get the non-zeros right" and
# inflates it towards 100%. Concretely, at ACTIVE_THRESHOLD = 0.001 a fixed
# 0.01 would forgive all 3068 true-zero cells of the 4158-cell test set. Half
# the threshold keeps the band strictly inside it at every setting, and is
# exactly 0.01 at the 0.02 default, so nothing about the historical figures
# changes.
DEFAULT_DEADBAND = min(0.01, 0.5 * ACTIVE_THRESHOLD)

# Deadband sweep for summarise(), as fractions of the threshold rather than
# absolute numbers, for the same reason. The middle point is 0.5*T, i.e.
# DEFAULT_DEADBAND whenever the threshold is at or below 0.02.
DEFAULT_DEADBANDS = (0.2 * ACTIVE_THRESHOLD, 0.5 * ACTIVE_THRESHOLD, 0.8 * ACTIVE_THRESHOLD)

# Absolute fraction tolerance for whole-recipe agreement. NOTE this is a
# STRICTER test than exact-match, not a more forgiving one: it requires the
# quantities to be right too, where exact-match only checks presence.
DEFAULT_TOLERANCE = 0.03


def _true_active(true_fractions: np.ndarray) -> np.ndarray:
    """Presence as the pipeline defines it -- must match stage1's build_labels."""
    return np.asarray(true_fractions, dtype=float) > ACTIVE_THRESHOLD


def exact_match(pred_active: np.ndarray, true_fractions: np.ndarray) -> float:
    """The strict historical metric: all 14 presence decisions right at once."""
    return float((np.asarray(pred_active).astype(bool)
                  == _true_active(true_fractions)).all(axis=1).mean())


def deadband_match(
    pred_active: np.ndarray, true_fractions: np.ndarray, deadband: float = DEFAULT_DEADBAND
) -> np.ndarray:
    """Exact-match, but disagreements inside the threshold's own noise band don't count.

    This is the direct fix for being scored wrong on 1.96% vs 2.06%. A
    disagreement is forgiven when the true fraction lies within `deadband` of
    ACTIVE_THRESHOLD, in either direction -- so it forgives both a missed
    colorant just above the line and a hallucinated one just below it.

    Still a presence metric, so it stays comparable in kind to exact-match.
    """
    if not deadband < ACTIVE_THRESHOLD:
        raise ValueError(
            f"deadband {deadband} must be strictly less than ACTIVE_THRESHOLD "
            f"{ACTIVE_THRESHOLD}. A deadband at or above the threshold forgives "
            f"every true zero (|0 - T| = T <= deadband), so the metric stops "
            f"measuring false positives at all and is not comparable to "
            f"exact-match. Scale the deadband with the threshold instead "
            f"(see DEFAULT_DEADBAND)."
        )

    pred_active = np.asarray(pred_active).astype(bool)
    true_fractions = np.asarray(true_fractions, dtype=float)

    disagree = pred_active != _true_active(true_fractions)
    forgiven = np.abs(true_fractions - ACTIVE_THRESHOLD) <= deadband
    return ~(disagree & ~forgiven).any(axis=1)


def tolerant_match(pred: np.ndarray, true: np.ndarray, tolerance: float = DEFAULT_TOLERANCE) -> np.ndarray:
    """Per-row: do the predicted and true recipes agree within `tolerance` everywhere?

    Quantitative, so it subsumes presence AND amount -- a harder bar than
    exact-match. Both arrays are (n_samples, n_colorants) of fractions.
    """
    return (np.abs(np.asarray(pred, dtype=float) - np.asarray(true, dtype=float))
            <= tolerance).all(axis=1)


def classify_errors(
    pred_active: np.ndarray, true_fractions: np.ndarray, deadband: float = DEFAULT_DEADBAND
) -> dict:
    """Split presence mistakes into categories that mean different things.

    Measured against the pipeline's own presence definition (> ACTIVE_THRESHOLD):
      - threshold_artifact: true fraction within `deadband` of the threshold,
        so which side it falls on is a convention, not a fact
      - false_positive: said present, true fraction is exactly zero
      - substantial_miss: said absent, true fraction well above the threshold
      - other: everything else (e.g. a false positive on a real trace amount)
    """
    pred_active = np.asarray(pred_active).astype(bool)
    true_fractions = np.asarray(true_fractions, dtype=float)
    true_act = _true_active(true_fractions)

    disagree = pred_active != true_act
    near = np.abs(true_fractions - ACTIVE_THRESHOLD) <= deadband

    artifact = disagree & near
    fp_zero = disagree & ~near & pred_active & (true_fractions == 0.0)
    sub_miss = disagree & ~near & ~pred_active & (true_fractions > ACTIVE_THRESHOLD + deadband)
    other = disagree & ~artifact & ~fp_zero & ~sub_miss

    total = int(disagree.sum())
    return {
        "threshold_artifact": int(artifact.sum()),
        "false_positive": int(fp_zero.sum()),
        "substantial_miss": int(sub_miss.sum()),
        "other": int(other.sum()),
        "total": total,
        "artifact_share": (float(artifact.sum()) / total) if total else 0.0,
    }


def summarise(
    pred_fractions: np.ndarray,
    true_fractions: np.ndarray,
    pred_active: np.ndarray | None = None,
    tolerances=(0.01, 0.02, 0.03, 0.05),
    deadbands=DEFAULT_DEADBANDS,
) -> dict:
    """Everything at once, for reporting after a training run."""
    pred_fractions = np.asarray(pred_fractions, dtype=float)
    true_fractions = np.asarray(true_fractions, dtype=float)

    out: dict = {"tolerance_curve": {}, "deadband_curve": {}}
    for t in tolerances:
        out["tolerance_curve"][t] = float(tolerant_match(pred_fractions, true_fractions, t).mean())

    if pred_active is not None:
        out["exact_match"] = exact_match(pred_active, true_fractions)
        for d in deadbands:
            out["deadband_curve"][d] = float(deadband_match(pred_active, true_fractions, d).mean())
        out["error_breakdown"] = classify_errors(pred_active, true_fractions)
    return out


def format_summary(summary: dict) -> str:
    """Human-readable block for train.py's stdout."""
    lines = []
    if "exact_match" in summary:
        lines.append(f"  Exact colorant-set match (strict, historical): {summary['exact_match']:.3%}")
    if summary.get("deadband_curve"):
        lines.append(f"  Same metric, ignoring disagreements within +/-d of the "
                     f"{ACTIVE_THRESHOLD} presence threshold:")
        for d, v in summary["deadband_curve"].items():
            note = "  <- recommended companion figure" if d == DEFAULT_DEADBAND else ""
            # %.3f collapses every sub-0.001 deadband to "0.000"; %g keeps the
            # small values legible once the threshold drops.
            lines.append(f"    deadband {d:<8g}: {v:.3%}{note}")
    lines.append("  Whole-recipe agreement within an absolute tolerance "
                 "(HARDER -- checks amounts too):")
    for t, v in summary["tolerance_curve"].items():
        lines.append(f"    within {t:.2f}: {v:.3%}")
    if "error_breakdown" in summary:
        b = summary["error_breakdown"]
        lines.append(f"  Presence mistakes: {b['total']} total")
        lines.append(f"    threshold artifact (near the line):  {b['threshold_artifact']}")
        lines.append(f"    false positive on a true zero:       {b['false_positive']}")
        lines.append(f"    substantial miss:                    {b['substantial_miss']}")
        lines.append(f"    other:                               {b['other']}")
        lines.append(f"    -> {b['artifact_share']:.0%} of presence mistakes are threshold artifacts")
    return "\n".join(lines)
