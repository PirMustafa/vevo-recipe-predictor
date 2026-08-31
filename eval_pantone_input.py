"""What happens if the INPUT is a Pantone standard's spectrum, not a measurement?

    python eval_pantone_input.py [--n 60]

The client's question: the tool takes a measured reflectance spectrum -- can it
take a Pantone standard's spectrum instead? ("Here is the colour I want, tell me
the recipe.")

It is a fair question because the two look interchangeable: both are 36-point
reflectance curves over 380-730nm, and a UFO formula's light-backing measurement
sits a mean dE00 0.75 from the Pantone standard it was made to match. But they
are different physical objects -- a Pantone Coated swatch is opaque ink on
coated stock, a UFO reading is a translucent film over a light substrate -- and
the model was trained on one of them.

This script measures the difference instead of arguing about it. On the held-out
test formulas only (never trained on), it runs the deployed light-only model
twice:

    A. the UFO light-backing measurement   <- what the tool is built for
    B. the Pantone standard's spectrum     <- what the client is proposing

Both are scored against the same known recipe, so the gap between A and B is
exactly the cost of substituting the standard for the measurement.

It also reports how often the question is moot: most Pantone codes in this
library have ALREADY been formulated, so the correct answer is a lookup of the
recipe that was actually mixed, not a prediction at all.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from predict import load_mode_artifacts
from src.backing_modes import MODE_LIGHT
from src.colorimetry import delta_e_2000, reflectance_to_lab
from src.cxf_parser import ACTIVE_THRESHOLD, COLORANT_NAMES, WAVELENGTHS, load_dataset
from src.features import build_single_spectrum_features
from src.pantone import formula_to_pantone_code, load_pantone_library, target_deltas
from src.stage1_classifier import predict_active
from src.stage2_regressor import predict_fractions

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"
TARGET_DELTA_MAX_DE = 8.0

CC = [f"colorant_{c}" for c in COLORANT_NAMES]
LIGHT = [f"R_light_{w}" for w in WAVELENGTHS]


def production_partition(df, library):
    """Exactly train.py's sequence: split on dirty data, then clean each side."""
    both = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)
    tr, te = train_test_split(both, test_size=0.15, random_state=42)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

    def drop_corrupt(d):
        return d[(d[CC].sum(axis=1) - 1.0).abs() < 0.05].reset_index(drop=True)

    def drop_off_target(d):
        dd = target_deltas(d["formula"], d[LIGHT].to_numpy(float), library)
        return d[~(~np.isnan(dd) & (dd > TARGET_DELTA_MAX_DE))].reset_index(drop=True)

    tr, te = drop_corrupt(tr), drop_corrupt(te)
    if library:
        tr, te = drop_off_target(tr), drop_off_target(te)
    return tr, te


def run(X, artifacts):
    stage1, stage2 = artifacts
    active = predict_active(stage1, X)
    return active, predict_fractions(stage2, X, active)


def score(active, frac, true_frac):
    true_active = true_frac > ACTIVE_THRESHOLD
    exact = float((active.astype(bool) == true_active).all(axis=1).mean())
    per_dec = float((active.astype(bool) == true_active).mean())
    # MAE over cells the recipe actually uses -- the number a formulator feels.
    mask = true_active
    mae = float(np.abs(frac - true_frac)[mask].mean()) if mask.any() else float("nan")
    return exact, per_dec, mae


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="how many test formulas to use")
    args = ap.parse_args()

    library = load_pantone_library(PANTONE_PATH)
    df = load_dataset(str(CXF_PATH))
    tr, te = production_partition(df, library)

    # ---- how often is a prediction even the right tool? -------------------
    formulated = {c for f in df["formula"]
                  if (c := formula_to_pantone_code(f)) and c in library}
    print("PANTONE LIBRARY COVERAGE")
    print(f"  standards in the library          {len(library)}")
    print(f"  ...already formulated as a UFO ink {len(formulated)} "
          f"({len(formulated) / len(library):.1%})")
    print("  For those, the exact recipe that was mixed to match the standard is")
    print("  already on record -- looking it up beats predicting it.\n")

    # ---- the substitution test --------------------------------------------
    codes = [formula_to_pantone_code(f) for f in te["formula"]]
    keep = [i for i, c in enumerate(codes) if c and c in library][: args.n]
    sub = te.iloc[keep].reset_index(drop=True)
    targets = np.array([library[codes[i]].reflectance for i in keep], dtype=float)
    ufo_light = sub[LIGHT].to_numpy(float)
    true_frac = sub[CC].to_numpy(float)

    # How far apart are the two inputs, colorimetrically?
    d = np.array([delta_e_2000(reflectance_to_lab(a), reflectance_to_lab(b))
                  for a, b in zip(ufo_light, targets)])
    print(f"INPUT GAP  (n={len(sub)} held-out formulas)")
    print(f"  dE00 between the UFO measurement and its Pantone standard:")
    print(f"    mean {d.mean():.2f}   median {np.median(d):.2f}   "
          f"90th pct {np.percentile(d, 90):.2f}   max {d.max():.2f}\n")

    artifacts = load_mode_artifacts(MODE_LIGHT)

    print("SAME MODEL, TWO INPUTS, SAME KNOWN RECIPES")
    rows = []
    for label, X_raw in (("A  UFO light measurement (designed input)", ufo_light),
                         ("B  Pantone standard spectrum (proposed)", targets)):
        active, frac = run(build_single_spectrum_features(X_raw), artifacts)
        e, pd_, mae = score(active, frac, true_frac)
        rows.append((label, e, pd_, mae))
        print(f"  {label:<42} exact {e:7.2%}   per-decision {pd_:7.3%}   "
              f"active-cell MAE {mae:.4f}")

    (_, ea, pa, ma), (_, eb, pb, mb) = rows
    print(f"\n  COST OF THE SUBSTITUTION: exact-match {eb - ea:+.2%}, "
          f"per-decision {pb - pa:+.3%}, MAE {mb - ma:+.4f}")


if __name__ == "__main__":
    main()
