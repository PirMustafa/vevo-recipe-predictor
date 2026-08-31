"""Why does a dE00 0.75 match still break the model?

    python eval_pantone_gap.py

The apparent contradiction: a Pantone standard sits a median CIEDE2000 of 0.45
from its own UFO light-backing measurement -- inside instrument noise, the same
colour to any human eye -- yet substituting it as model input drops exact-match
from 85% to 22% (eval_pantone_input.py).

The resolution is that CIEDE2000 and the model are not looking at the same
thing. dE00 is computed from Lab, which is 36 reflectance numbers integrated
through 3 CIE colour-matching functions: a 12:1 lossy compression, deliberately
designed to discard everything the eye cannot see. The model reads 142 features
built from all 36 points -- raw R, the Kubelka-Munk K/S transform, and their
derivatives -- precisely BECAUSE colour alone does not determine a recipe. Two
inks of identical colour and different composition is metamerism, the everyday
problem this whole project exists to solve.

So "same colour" cannot certify "same input". This script quantifies the gap the
model actually sees, in units of the training data's own spread: a shift of 1.0
means the Pantone input is displaced by one training standard deviation on that
feature, which is a large move for a model fitted on that distribution.

CPU only.
"""
from __future__ import annotations

import pathlib

import numpy as np

from src.colorimetry import delta_e_2000, reflectance_to_lab
from src.cxf_parser import WAVELENGTHS, load_dataset
from src.features import build_single_spectrum_features
from src.pantone import formula_to_pantone_code, load_pantone_library

ROOT = pathlib.Path(__file__).parent
LIGHT = [f"R_light_{w}" for w in WAVELENGTHS]


def main() -> None:
    library = load_pantone_library(ROOT / "Pantone_Coated_V_4.cxf")
    df = load_dataset(str(ROOT / "NPXXXXCPWNUF UFO White PE.cxf"))
    both = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)

    codes = [formula_to_pantone_code(f) for f in both["formula"]]
    keep = [i for i, c in enumerate(codes) if c and c in library]
    sub = both.iloc[keep].reset_index(drop=True)
    tgt = np.array([library[codes[i]].reflectance for i in keep], dtype=float)
    Rl = sub[LIGHT].to_numpy(float)

    dE = np.asarray([delta_e_2000(reflectance_to_lab(a), reflectance_to_lab(b))
                     for a, b in zip(Rl, tgt)], dtype=float).ravel()

    Xu = build_single_spectrum_features(Rl).to_numpy(float)
    Xp = build_single_spectrum_features(tgt).to_numpy(float)
    sd = Xu.std(axis=0)

    n = len(sub)
    print(f"MATCHED FORMULAS: {n}\n")

    # ---- the structural mismatch, found while building this script ---------
    # The UFO instrument measures 400-700nm; the CxF pads out to the 380-730
    # grid by REPEATING the edge values. Pantone's file carries real data across
    # the whole range. So the derivative features at those padded positions are
    # exactly constant in every training row -- the model has never seen them
    # vary -- while a Pantone input makes them non-zero.
    dead = sd == 0
    print("STRUCTURAL MISMATCH: EDGE PADDING")
    lead_u = (np.diff(Rl[:, :4], axis=1) == 0).sum(axis=1)
    lead_p = (np.diff(tgt[:, :4], axis=1) == 0).sum(axis=1)
    print(f"  UFO rows whose 380/390/400nm values are identical (padded): "
          f"{(lead_u >= 2).mean():6.1%}")
    print(f"  Pantone rows likewise                                     : "
          f"{(lead_p >= 2).mean():6.1%}")
    print(f"  features with ZERO variance across all training rows      : {int(dead.sum())} of 142")
    if dead.any():
        moved = (np.abs(Xp - Xu)[:, dead] > 1e-9).any(axis=1).mean()
        print(f"  ...and a Pantone input makes them non-zero on           : {moved:6.1%} of formulas")
        print("  The model cannot weigh a feature it has only ever seen as a constant.\n")

    live = ~dead
    z = np.abs(Xp - Xu)[:, live] / sd[live]

    print("WHAT THE EYE SEES vs WHAT THE MODEL SEES  (the 132 live features)")
    print(f"  CIEDE2000 (Lab, 3 numbers)      median {np.median(dE):.2f}  "
          f"-- inside the dE00 2.0 'commercial match' band")
    print(f"  Feature displacement            median {np.median(z.mean(axis=1)):.2f} "
          f"training SD per feature, worst feature {np.median(z.max(axis=1)):.2f} SD\n")

    # Which feature families move most? build_single_spectrum_features packs
    # R(36) | KS(36) | dR(35) | dKS(35).
    groups = [("raw reflectance  R", 0, 36),
              ("Kubelka-Munk  K/S", 36, 72),
              ("dR   (curve slope)", 72, 107),
              ("dK/S (slope of K/S)", 107, 142)]
    print("DISPLACEMENT BY FEATURE FAMILY (median over formulas, in training SD)")
    idx = np.arange(142)[live]          # map live columns back to family ranges
    for name, a, b in groups:
        g = z[:, (idx >= a) & (idx < b)]
        if not g.size:
            continue
        print(f"  {name:<22} mean {np.median(g.mean(axis=1)):5.2f} SD    "
              f"worst feature {np.median(g.max(axis=1)):6.2f} SD")

    # K/S is the amplifier: it is (1-R)^2/2R, which blows up as R -> 0, so a
    # difference invisible in reflectance becomes enormous in the model's input.
    print("\nWHY K/S AMPLIFIES AN INVISIBLE DIFFERENCE")
    for r_a, r_b in ((0.010, 0.015), (0.050, 0.055), (0.400, 0.405)):
        ks = lambda r: (1 - r) ** 2 / (2 * r)
        print(f"  R {r_a:.3f} -> {r_b:.3f}  (delta {r_b - r_a:.3f}, invisible)   "
              f"K/S {ks(r_a):8.2f} -> {ks(r_b):8.2f}   "
              f"= {abs(ks(r_b) - ks(r_a)) / ks(r_a):6.1%} change")

    # Is the shift systematic (a domain offset the model cannot absorb) or just
    # noise? A signed mean far from zero means systematic.
    signed = ((Xp - Xu)[:, live] / sd[live]).mean(axis=0)
    print(f"\nIS THE SHIFT SYSTEMATIC OR RANDOM?")
    print(f"  |mean signed shift| averaged over features : {np.abs(signed).mean():.3f} SD")
    print(f"  mean |unsigned| shift                      : {z.mean():.3f} SD")
    ratio = np.abs(signed).mean() / z.mean()
    print(f"  systematic fraction                        : {ratio:.1%}")
    print("  (high = a consistent domain offset, not random measurement noise)")

    # The punchline: colour agreement does not predict feature agreement.
    close = dE < 1.0
    print(f"\nEVEN THE PERFECT-COLOUR CASES ARE DISPLACED")
    print(f"  formulas with dE00 < 1.0 (visually identical): {int(close.sum())} of {n}")
    print(f"  their median feature displacement            : "
          f"{np.median(z[close].mean(axis=1)):.2f} SD")
    print(f"  correlation between dE00 and displacement    : "
          f"{np.corrcoef(dE, z.mean(axis=1))[0, 1]:.3f}")


if __name__ == "__main__":
    main()
