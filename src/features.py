"""Reflectance <-> Kubelka-Munk (K/S) conversions and feature-matrix building.

K/S is computed with the single-constant Kubelka-Munk formula:
    K/S = (1 - R)^2 / (2R)

This is an *approximation* here (see project notes): it assumes an opaque,
fully-hiding layer, which these translucent white-ink films are not. We keep
it anyway as an engineered feature (it linearizes the concentration/color
relationship reasonably well) but we also keep raw R alongside it so models
aren't solely dependent on the approximation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cxf_parser import WAVELENGTHS

# Reflectance is physically in (0, 1]; clip away from the exact 0 boundary
# to avoid division by zero when computing K/S.
_R_EPS = 1e-4


def reflectance_to_ks(r: np.ndarray) -> np.ndarray:
    """Convert reflectance array(s) to K/S using the single-constant KM formula."""
    r = np.clip(np.asarray(r, dtype=float), _R_EPS, 1.0)
    return (1.0 - r) ** 2 / (2.0 * r)


def ks_to_reflectance(ks: np.ndarray) -> np.ndarray:
    """Invert the single-constant Kubelka-Munk formula: K/S -> R."""
    ks = np.clip(np.asarray(ks, dtype=float), 0.0, None)
    return 1.0 + ks - np.sqrt(ks**2 + 2.0 * ks)


FEATURE_GROUPS = (
    "R_light", "R_dark", "KS_light", "KS_dark", "contrast",
    "deriv_light", "deriv_KS_light", "deriv_KS_dark",
)

# Tried and measured, not used: extending with deriv_dark + 2nd derivatives
# of RAW REFLECTANCE + light/dark ratio (354 features total) gave the same
# exact-match rate (80.27%, unchanged) but a *lower* macro-F1 (0.901 vs
# 0.920) -- redundant enough with R/KS/contrast to just add noise for a tree
# ensemble at this dataset size. Reverted; see repo memory for the ablation.
#
# What DID help: the derivative of the K/S-TRANSFORMED curve specifically
# (not the raw-reflectance derivative above, which was already tried and is
# the "deriv_light" group). K/S is designed to linearize the concentration/
# color relationship, so its slope isolates pigment-specific absorption
# changes more cleanly than raw reflectance's slope does. Exact-match
# 80.27% -> 80.60%, macro-F1 0.920 -> 0.930 -- a real, reproducible, if
# modest, gain.


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix: raw R + K/S (light & dark), plus engineered
    groups that measurably improved Stage 1 exact-match accuracy in testing
    (see repo memory for the experiments):
      - contrast: R_light - R_dark per wavelength (36 cols) - directly
        encodes hiding power / opacity, which the model otherwise has to
        infer indirectly from the two separate spectra.
      - deriv_light: first difference of the light-backing REFLECTANCE
        spectrum across adjacent wavelengths (35 cols) - highlights the
        shape/slope of the curve (peaks/valleys), which is often more
        distinctive between similar-looking colorants than raw levels.
      - deriv_KS_light / deriv_KS_dark: first difference of the light/dark
        K/S-TRANSFORMED curves (35 cols each) - a different, and in testing
        additionally useful, view of curve shape: K/S linearizes the
        concentration/color relationship, so its slope highlights
        pigment-specific absorption features that raw reflectance's slope
        (deriv_light, above) partly obscures.

    Requires df to have both `R_light_<wl>` and `R_dark_<wl>` columns fully
    populated (i.e. filtered to `has_light & has_dark` rows beforehand).
    """
    r_light_cols = [f"R_light_{wl}" for wl in WAVELENGTHS]
    r_dark_cols = [f"R_dark_{wl}" for wl in WAVELENGTHS]

    r_light = df[r_light_cols].to_numpy(dtype=float)
    r_dark = df[r_dark_cols].to_numpy(dtype=float)

    ks_light = reflectance_to_ks(r_light)
    ks_dark = reflectance_to_ks(r_dark)
    contrast = r_light - r_dark
    deriv_light = np.diff(r_light, axis=1)
    deriv_ks_light = np.diff(ks_light, axis=1)
    deriv_ks_dark = np.diff(ks_dark, axis=1)

    feature_cols = {}
    for i, wl in enumerate(WAVELENGTHS):
        feature_cols[f"R_light_{wl}"] = r_light[:, i]
    for i, wl in enumerate(WAVELENGTHS):
        feature_cols[f"R_dark_{wl}"] = r_dark[:, i]
    for i, wl in enumerate(WAVELENGTHS):
        feature_cols[f"KS_light_{wl}"] = ks_light[:, i]
    for i, wl in enumerate(WAVELENGTHS):
        feature_cols[f"KS_dark_{wl}"] = ks_dark[:, i]
    for i, wl in enumerate(WAVELENGTHS):
        feature_cols[f"contrast_{wl}"] = contrast[:, i]
    for i in range(deriv_light.shape[1]):
        feature_cols[f"deriv_light_{i}"] = deriv_light[:, i]
    for i in range(deriv_ks_light.shape[1]):
        feature_cols[f"deriv_KS_light_{i}"] = deriv_ks_light[:, i]
    for i in range(deriv_ks_dark.shape[1]):
        feature_cols[f"deriv_KS_dark_{i}"] = deriv_ks_dark[:, i]

    X = pd.DataFrame(feature_cols, index=df.index)
    assert X.shape[1] == 285, f"Expected 285 features, got {X.shape[1]}"
    return X


# Number of features build_single_spectrum_features produces: R + K/S (36 each)
# plus the first difference of both curves (35 each).
N_SINGLE_FEATURES = 36 + 36 + 35 + 35


def build_single_spectrum_features(R, index=None) -> pd.DataFrame:
    """Feature matrix from ONE spectrum per sample, rather than a light/dark pair.

    Needed for the inverse (target colour -> recipe) direction: a Pantone
    standard carries a single reflectance curve, so every feature group that
    depends on having two backings -- `contrast`, `KS_dark`, `deriv_KS_dark` --
    is unavailable by construction. See src/inverse.py.

    Keeps the groups from build_feature_matrix that a single curve supports, in
    the same order and with the same reasoning: raw R, its K/S transform, and
    the first difference of each (curve shape being more discriminative between
    similar colorants than absolute level).

    R: array-like of shape (n_samples, 36), reflectance in (0, 1].
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 1:
        R = R.reshape(1, -1)
    if R.shape[1] != len(WAVELENGTHS):
        raise ValueError(
            f"Expected {len(WAVELENGTHS)} reflectance values per sample "
            f"(380-730nm, 10nm steps), got {R.shape[1]}"
        )

    ks = reflectance_to_ks(R)
    deriv_r = np.diff(R, axis=1)
    deriv_ks = np.diff(ks, axis=1)

    cols = {}
    for i, wl in enumerate(WAVELENGTHS):
        cols[f"R_{wl}"] = R[:, i]
    for i, wl in enumerate(WAVELENGTHS):
        cols[f"KS_{wl}"] = ks[:, i]
    for i in range(deriv_r.shape[1]):
        cols[f"deriv_R_{i}"] = deriv_r[:, i]
    for i in range(deriv_ks.shape[1]):
        cols[f"deriv_KS_{i}"] = deriv_ks[:, i]

    X = pd.DataFrame(cols, index=index if index is not None else range(len(R)))
    assert X.shape[1] == N_SINGLE_FEATURES, \
        f"Expected {N_SINGLE_FEATURES} features, got {X.shape[1]}"
    return X
