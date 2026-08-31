"""What does each backing mode cost, measured on the PRODUCTION architecture?

    python eval_backing_modes.py

Client requirement: accept a light-backing spectrum alone, a dark-backing
spectrum alone, or both. Today the pipeline hard-requires both (285 features)
and train.py discards any formula missing either.

Every number here uses the deployed Stage 1 architecture (TabICL in a
ClassifierChain) and reproduces train.py's partition exactly -- split FIRST on
the still-dirty data, then drop corrupted rows per side. A CPU proxy run of
this comparison scored the both-backing mode at 82.77%, close to the project's
historical GBM figure but ~4.5 points below the deployed TabICL number, which
is why it could not be quoted against 87.2%. These can be.

Note on data: the dataset contains 1988 both-backing and 384 light-only
formulas, and ZERO dark-only. A dark-only model is still trainable from the
dark half of paired formulas, but no real dark-only submission has ever been
recorded -- worth confirming that mode is a genuine requirement.
"""
from __future__ import annotations

import pathlib

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src.cxf_parser import COLORANT_NAMES, WAVELENGTHS, load_dataset
from src.features import build_feature_matrix, build_single_spectrum_features
from src.pantone import load_pantone_library, target_deltas
from src.stage1_classifier import ACTIVE_THRESHOLD, predict_active, train_stage1

ROOT = pathlib.Path(__file__).parent
CXF_PATH = ROOT / "NPXXXXCPWNUF UFO White PE.cxf"
PANTONE_PATH = ROOT / "Pantone_Coated_V_4.cxf"
TARGET_DELTA_MAX_DE = 8.0

CC = [f"colorant_{c}" for c in COLORANT_NAMES]
LIGHT = [f"R_light_{w}" for w in WAVELENGTHS]
DARK = [f"R_dark_{w}" for w in WAVELENGTHS]


def _labels(df):
    return (df[CC] > ACTIVE_THRESHOLD).astype(int)


def production_partition(df):
    """Exactly train.py's sequence: split on dirty data, then clean each side."""
    both = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)
    tr, te = train_test_split(both, test_size=0.15, random_state=42)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)

    def drop_corrupt(d):
        t = d[CC].sum(axis=1)
        return d[(t - 1.0).abs() < 0.05].reset_index(drop=True)

    tr, te = drop_corrupt(tr), drop_corrupt(te)

    library = load_pantone_library(PANTONE_PATH)
    if library:
        def drop_off_target(d):
            dd = target_deltas(d["formula"], d[LIGHT].to_numpy(float), library)
            return d[~(~np.isnan(dd) & (dd > TARGET_DELTA_MAX_DE))].reset_index(drop=True)
        tr, te = drop_off_target(tr), drop_off_target(te)
    return tr, te


def evaluate(name, Xtr, Ytr, Xte, Yte):
    model = train_stage1(Xtr, Ytr)
    pred = predict_active(model, Xte)
    yt = Yte.to_numpy() if hasattr(Yte, "to_numpy") else Yte
    exact = float((pred == yt).all(axis=1).mean())
    macro = float(f1_score(yt, pred, average="macro", zero_division=0))
    per_dec = float((pred == yt).mean())
    print(f"  {name:<34} train {len(Xtr):>5}  feat {Xtr.shape[1]:>4}   "
          f"exact {exact:7.2%}   macro-F1 {macro:.4f}   per-decision {per_dec:.3%}")
    return exact, macro


def main() -> None:
    df = load_dataset(str(CXF_PATH))
    tr, te = production_partition(df)
    Ytr, Yte = _labels(tr), _labels(te)

    light_only = df[df["has_light"] & ~df["has_dark"]].reset_index(drop=True)
    lo_ok = light_only[(light_only[CC].sum(axis=1) - 1.0).abs() < 0.05].reset_index(drop=True)

    print("DATA AVAILABILITY")
    print(f"  both backings: {int((df['has_light'] & df['has_dark']).sum())}   "
          f"light only: {len(light_only)}   "
          f"dark only: {int((~df['has_light'] & df['has_dark']).sum())}")
    print(f"\nPRODUCTION PARTITION (matches train.py): train {len(tr)}  test {len(te)}\n")
    print("EACH INPUT MODE, DEPLOYED ARCHITECTURE (TabICL ClassifierChain)")

    base, _ = evaluate("both backings (current)",
                       build_feature_matrix(tr), Ytr, build_feature_matrix(te), Yte)

    Xl_te = build_single_spectrum_features(te[LIGHT].to_numpy(float))
    e_light, _ = evaluate("light backing only",
                          build_single_spectrum_features(tr[LIGHT].to_numpy(float)),
                          Ytr, Xl_te, Yte)

    e_dark, _ = evaluate("dark backing only",
                         build_single_spectrum_features(tr[DARK].to_numpy(float)),
                         Ytr,
                         build_single_spectrum_features(te[DARK].to_numpy(float)), Yte)

    # Light-only mode can train on the formulas train.py discards
    import pandas as pd
    aug_R = np.vstack([tr[LIGHT].to_numpy(float), lo_ok[LIGHT].to_numpy(float)])
    aug_Y = pd.concat([Ytr, _labels(lo_ok)], ignore_index=True)
    e_light_aug, _ = evaluate("light only + recovered formulas",
                              build_single_spectrum_features(aug_R), aug_Y, Xl_te, Yte)

    print("\nCOST RELATIVE TO THE CURRENT BOTH-BACKING MODEL")
    for nm, v in [("light only", e_light), ("dark only", e_dark),
                  ("light only + recovered", e_light_aug)]:
        print(f"  {nm:<26} {v - base:+.2%}")
    print(f"\n  value of the {len(lo_ok)} recovered light-only formulas: "
          f"{e_light_aug - e_light:+.2%}")


if __name__ == "__main__":
    main()
