"""Client formulation rules applied on top of the model's raw prediction.

Two rules, taken from how the client actually mixes ink:

  1. AT MOST THREE non-white colorants, PLUS transparent white if it is needed.
     White is exempt because it is the extender/tint base, not one of the
     chromatic inks a press operator weighs out.
  2. NOTHING BELOW 0.1% (0.001 as a fraction). Below that the dosing equipment
     cannot place it and the formula sheet would not carry it, so a predicted
     0.03% is not a small answer -- it is a wrong one.

ORDER MATTERS, and it is deliberate: the two rules sit on OPPOSITE SIDES of
Stage 2.

    Stage 1 (presence)
        -> enforce_max_colorants          <-- BEFORE Stage 2
            -> Stage 2 (amounts)
                -> apply_min_concentration    <-- AFTER Stage 2

`enforce_max_colorants` runs BETWEEN the stages because Stage 2 renormalises
the colorants it is given so they sum to 1.0 (a masked softmax over the active
set). Removing a colorant before that pass means the remaining amounts are
re-solved once, in a single consistent step. Removing it afterwards would leave
a recipe summing to less than 1 and force a second, cruder renormalisation on
top of predictions that were fitted assuming the dropped colorant was present
-- the survivors would all be biased in a way rescaling cannot undo.

`apply_min_concentration` must run AFTER Stage 2 for the mirror reason: before
Stage 2 there are no amounts to apply a floor to.

Stage 1 probabilities come from `TabICLStage1.predict_proba_matrix(X)`
(src/stage1_classifier.py:73). Its columns are in the ORIGINAL label order --
the same order as COLORANT_NAMES, not the internal random chain order. Do not
reorder them.

Nothing here is applied by default. predict.py exposes it behind
`enforce_constraints=True`, so the deployed pipeline stays byte-identical until
someone asks for these rules.
"""
from __future__ import annotations

import numpy as np

# The one colorant exempt from the max-colorant count: it is the tint base.
WHITE = "UFO00061 transp. white"

# Client rule: at most 3 chromatic colorants, plus white on top if needed.
MAX_NON_WHITE = 3

# Client rule: 0.1% is the working minimum (they occasionally go to 0.01%, but
# 0.1% is what the standard process supports).
MIN_CONCENTRATION = 0.001


def enforce_max_colorants(
    active: np.ndarray,
    proba: np.ndarray,
    colorant_names,
    max_non_white: int = MAX_NON_WHITE,
    white: str = WHITE,
) -> np.ndarray:
    """Trim each row's active set to <= `max_non_white` non-white colorants (+ white).

    active : (n, n_colorants) bool  -- Stage 1's presence decisions
    proba  : (n, n_colorants) float -- Stage 1's probabilities, ORIGINAL label
             order, from TabICLStage1.predict_proba_matrix(X)
    colorant_names : the label order of both arrays' columns

    Keeps white if Stage 1 flagged it, then keeps the `max_non_white` most
    probable non-white colorants. Rows already inside the limit come back
    unchanged.

    A row is never left with only white, or with nothing: white on its own is
    not an ink, so if no non-white colorant is active the single most probable
    one is promoted. Run this BETWEEN Stage 1 and Stage 2 -- see the module
    docstring for why the order is not interchangeable.
    """
    active = np.asarray(active).astype(bool)
    proba = np.asarray(proba, dtype=float)
    names = list(colorant_names)

    if active.ndim != 2 or proba.ndim != 2:
        raise ValueError(f"active and proba must be 2-D, got {active.shape} and {proba.shape}")
    if active.shape != proba.shape:
        raise ValueError(f"active {active.shape} and proba {proba.shape} must match")
    if active.shape[1] != len(names):
        raise ValueError(
            f"active has {active.shape[1]} columns but {len(names)} colorant names were given"
        )
    if max_non_white < 1:
        raise ValueError(f"max_non_white must be >= 1, got {max_non_white}")

    white_idx = names.index(white) if white in names else None
    non_white = np.array([j for j in range(len(names)) if j != white_idx], dtype=int)

    out = np.zeros_like(active)
    for i in range(active.shape[0]):
        if white_idx is not None and active[i, white_idx]:
            out[i, white_idx] = True

        cand = non_white[active[i, non_white]]
        if cand.size == 0:
            # White-only or empty row: promote the most probable non-white.
            cand = non_white[np.argsort(-proba[i, non_white], kind="stable")[:1]]
        elif cand.size > max_non_white:
            cand = cand[np.argsort(-proba[i, cand], kind="stable")[:max_non_white]]
        out[i, cand] = True

    return out


def apply_min_concentration(
    fractions: np.ndarray, min_conc: float = MIN_CONCENTRATION
) -> np.ndarray:
    """Zero anything below `min_conc` and renormalise each row back to 1.0.

    fractions : (n, n_colorants) float, rows summing to ~1.0 (Stage 2 output)

    A row whose every non-zero entry sits below the floor is left EXACTLY as it
    was: zeroing it would produce an empty recipe, which is worse than an
    out-of-spec one, and it is a signal worth surfacing rather than erasing.

    Renormalisation cannot push a kept value back under the floor: the row sum
    after zeroing is <= 1, so dividing by it only scales values up. One pass is
    therefore enough. Run this AFTER Stage 2 -- see the module docstring.
    """
    frac = np.array(fractions, dtype=float, copy=True)
    if frac.ndim != 2:
        raise ValueError(f"fractions must be 2-D, got {frac.shape}")

    for i in range(frac.shape[0]):
        row = frac[i]
        keep = row >= min_conc
        if not keep.any():
            continue  # flooring would empty the row -- leave it untouched
        row = np.where(keep, row, 0.0)
        total = row.sum()
        frac[i] = row / total if total > 0 else row

    return frac


if __name__ == "__main__":
    # Tests by construction: small hand-checked arrays, edge cases included.
    names = [WHITE, "yellow", "warm red", "reflex blue", "black"]

    def check(label, cond):
        print(f"  {'PASS' if cond else 'FAIL'}  {label}")
        assert cond, label

    print("enforce_max_colorants")

    # 1. Already inside the limit -> unchanged (white + 2 non-white).
    a = np.array([[True, True, True, False, False]])
    p = np.array([[0.9, 0.8, 0.7, 0.1, 0.05]])
    out = enforce_max_colorants(a, p, names)
    check("row inside the limit is untouched", bool((out == a).all()))

    # 2. Four non-white active -> the least probable is dropped, white kept.
    a = np.array([[True, True, True, True, True]])
    p = np.array([[0.99, 0.90, 0.80, 0.70, 0.10]])
    out = enforce_max_colorants(a, p, names)
    check("white survives", bool(out[0, 0]))
    check("exactly 3 non-white survive", int(out[0, 1:].sum()) == 3)
    check("weakest non-white (black, p=0.10) dropped", not bool(out[0, 4]))

    # 3. EDGE: five non-white active, white absent -> top 3 only, white not added.
    names5 = [WHITE, "c1", "c2", "c3", "c4", "c5"]
    a = np.array([[False, True, True, True, True, True]])
    p = np.array([[0.02, 0.30, 0.95, 0.10, 0.88, 0.60]])
    out = enforce_max_colorants(a, p, names5)
    check("white not resurrected", not bool(out[0, 0]))
    check("exactly 3 non-white survive", int(out[0, 1:].sum()) == 3)
    check("kept the top 3 by probability (c2, c4, c5)",
          list(np.flatnonzero(out[0])) == [2, 4, 5])

    # 4. EDGE: all-white row -> most probable non-white is promoted.
    a = np.array([[True, False, False, False, False]])
    p = np.array([[0.95, 0.20, 0.44, 0.31, 0.02]])
    out = enforce_max_colorants(a, p, names)
    check("white kept", bool(out[0, 0]))
    check("one non-white promoted", int(out[0, 1:].sum()) == 1)
    check("promoted the most probable non-white (warm red, p=0.44)", bool(out[0, 2]))

    # 5. EDGE: fully empty row -> still ends up with one non-white.
    a = np.zeros((1, 5), dtype=bool)
    p = np.array([[0.01, 0.02, 0.03, 0.90, 0.04]])
    out = enforce_max_colorants(a, p, names)
    check("empty row gets exactly one colorant", int(out[0].sum()) == 1)
    check("and it is the most probable non-white", bool(out[0, 3]))

    # 6. Multi-row: rows are handled independently.
    a = np.array([[True, True, True, True, True],
                  [False, True, True, False, False]])
    p = np.array([[0.9, 0.8, 0.7, 0.6, 0.5],
                  [0.1, 0.7, 0.6, 0.2, 0.1]])
    out = enforce_max_colorants(a, p, names)
    check("row 0 trimmed to 3 non-white", int(out[0, 1:].sum()) == 3)
    check("row 1 untouched", bool((out[1] == a[1]).all()))

    print("apply_min_concentration")

    # 7. Nothing below the floor -> unchanged.
    f = np.array([[0.50, 0.30, 0.20, 0.0, 0.0]])
    out = apply_min_concentration(f)
    check("row above the floor is unchanged", bool(np.allclose(out, f)))

    # 8. A sub-floor trace is zeroed and the rest renormalised to 1.0.
    f = np.array([[0.6000, 0.3995, 0.0005, 0.0, 0.0]])
    out = apply_min_concentration(f)
    check("trace zeroed", out[0, 2] == 0.0)
    check("row renormalised to 1.0", abs(out[0].sum() - 1.0) < 1e-12)
    check("survivors keep their ratio",
          abs(out[0, 0] / out[0, 1] - 0.6000 / 0.3995) < 1e-9)

    # 9. Exactly at the floor -> kept (the rule is "below min_conc").
    f = np.array([[0.5, 0.499, 0.001, 0.0, 0.0]])
    out = apply_min_concentration(f)
    check("value exactly at the floor survives", out[0, 2] > 0)

    # 10. EDGE: flooring would empty the row -> left exactly as it was.
    f = np.array([[0.0004, 0.0003, 0.0002, 0.0, 0.0]])
    out = apply_min_concentration(f)
    check("row that would be emptied is untouched", bool(np.array_equal(out, f)))

    # 11. Multi-row, mixed: one normal, one that would be emptied.
    f = np.array([[0.6000, 0.3995, 0.0005, 0.0, 0.0],
                  [0.0004, 0.0003, 0.0002, 0.0, 0.0]])
    out = apply_min_concentration(f)
    check("normal row renormalised", abs(out[0].sum() - 1.0) < 1e-12)
    check("emptiable row preserved", bool(np.array_equal(out[1], f[1])))

    # 12. The input array is not mutated in place.
    f = np.array([[0.6000, 0.3995, 0.0005, 0.0, 0.0]])
    before = f.copy()
    apply_min_concentration(f)
    check("input array not mutated", bool(np.array_equal(f, before)))

    print("\nAll constraint checks passed.")
