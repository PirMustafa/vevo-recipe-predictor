"""Which UFO backing does a Pantone standard actually resemble -- light or dark?

    python eval_pantone_backing.py

src/pantone.py compares Pantone targets against the LIGHT-backing spectrum, on
the physical argument that a Pantone Coated swatch is opaque ink on white coated
stock. That argument is sound but it was never measured. This script measures it.

For every formula whose name maps to a Pantone code and which has both
backings, it computes CIEDE2000 from the Pantone standard to each backing and
compares them head to head. It also splits the result by how opaque the film is
(the light/dark contrast), because that is where the interesting behaviour is:
a film opaque enough to hide its substrate reads nearly the same over either
backing, so for those the choice should stop mattering.

CPU only -- no model involved, this is pure colorimetry.
"""
from __future__ import annotations

import pathlib

import numpy as np

from src.colorimetry import delta_e_2000, reflectance_to_lab
from src.cxf_parser import WAVELENGTHS, load_dataset
from src.pantone import formula_to_pantone_code, load_pantone_library

ROOT = pathlib.Path(__file__).parent
LIGHT = [f"R_light_{w}" for w in WAVELENGTHS]
DARK = [f"R_dark_{w}" for w in WAVELENGTHS]


def summarise(name, d):
    print(f"  {name:<26} mean {d.mean():6.2f}   median {np.median(d):6.2f}   "
          f"p90 {np.percentile(d, 90):6.2f}   max {d.max():7.2f}")


def main() -> None:
    library = load_pantone_library(ROOT / "Pantone_Coated_V_4.cxf")
    df = load_dataset(str(ROOT / "NPXXXXCPWNUF UFO White PE.cxf"))
    both = df[df["has_light"] & df["has_dark"]].reset_index(drop=True)

    codes = [formula_to_pantone_code(f) for f in both["formula"]]
    keep = [i for i, c in enumerate(codes) if c and c in library]
    sub = both.iloc[keep].reset_index(drop=True)
    tgt = np.array([library[codes[i]].reflectance for i in keep], dtype=float)
    Rl = sub[LIGHT].to_numpy(float)
    Rd = sub[DARK].to_numpy(float)

    lab_t = np.array([reflectance_to_lab(r) for r in tgt])
    lab_l = np.array([reflectance_to_lab(r) for r in Rl])
    lab_d = np.array([reflectance_to_lab(r) for r in Rd])

    d_light = np.array([delta_e_2000(a, b) for a, b in zip(lab_t, lab_l)])
    d_dark = np.array([delta_e_2000(a, b) for a, b in zip(lab_t, lab_d)])

    print(f"MATCHED FORMULAS: {len(sub)} (name maps to a Pantone code, both backings present)\n")

    print("CIEDE2000 FROM THE PANTONE STANDARD")
    summarise("to UFO light backing", d_light)
    summarise("to UFO dark backing", d_dark)

    win_l = int((d_light < d_dark).sum())
    print(f"\nHEAD TO HEAD")
    print(f"  light backing closer : {win_l:5d}  ({win_l / len(sub):6.2%})")
    print(f"  dark backing closer  : {len(sub) - win_l:5d}  ({1 - win_l / len(sub):6.2%})")
    print(f"  median advantage of light: {np.median(d_dark - d_light):.2f} dE00")

    # Spectral, not just perceptual -- RMS over the raw curve.
    rms_l = np.sqrt(((tgt - Rl) ** 2).mean(axis=1))
    rms_d = np.sqrt(((tgt - Rd) ** 2).mean(axis=1))
    print(f"\nRMS REFLECTANCE DIFFERENCE (raw curve shape)")
    print(f"  to light backing     : {rms_l.mean():.4f}")
    print(f"  to dark backing      : {rms_d.mean():.4f}")

    # Opacity: how much the film hides its substrate. A fully opaque film reads
    # the same over either backing, so the two deltas should converge.
    opacity_gap = np.abs(Rl - Rd).mean(axis=1)
    qs = np.quantile(opacity_gap, [0, .25, .5, .75, 1.0])
    print(f"\nBY OPACITY  (mean |R_light - R_dark|; small = opaque film, hides substrate)")
    print(f"  {'quartile':<22}{'n':>6}{'dE00 light':>13}{'dE00 dark':>12}{'light wins':>12}")
    for i in range(4):
        m = (opacity_gap >= qs[i]) & (opacity_gap <= qs[i + 1] if i == 3 else opacity_gap < qs[i + 1])
        lbl = f"Q{i+1} gap {qs[i]:.3f}-{qs[i+1]:.3f}"
        print(f"  {lbl:<22}{int(m.sum()):>6}{d_light[m].mean():>13.2f}"
              f"{d_dark[m].mean():>12.2f}{(d_light[m] < d_dark[m]).mean():>11.1%}")


if __name__ == "__main__":
    main()
