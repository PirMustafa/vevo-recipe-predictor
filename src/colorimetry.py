"""Reflectance -> CIELAB -> CIEDE2000, for the 380-730nm/10nm grid this project uses.

Deliberately depends on numpy only. The CIE tables below are embedded rather
than pulled from a colour-science library so the handoff package keeps its slim
dependency list (see requirements.txt) -- they were generated once from
colour-science 0.4.7 and this module's output is verified against it in
tests/test_colorimetry.py.

Everything is fixed to the measurement conditions BOTH CxF files in this
project declare: D50 illuminant, CIE 1931 2-degree standard observer, and the
36-point 380-730nm/10nm sampling. Nothing here generalises to other conditions,
by design -- a resampling/adaptation layer would be dead weight for this repo.
"""
from __future__ import annotations

import numpy as np

# CIE 1931 2-degree standard observer colour-matching functions, 380-730nm/10nm.
_XBAR = (
    0.001368, 0.004243, 0.014310, 0.043510, 0.134380, 0.283900,
    0.348280, 0.336200, 0.290800, 0.195360, 0.095640, 0.032010,
    0.004900, 0.009300, 0.063270, 0.165500, 0.290400, 0.433450,
    0.594500, 0.762100, 0.916300, 1.026300, 1.062200, 1.002600,
    0.854450, 0.642400, 0.447900, 0.283500, 0.164900, 0.087400,
    0.046770, 0.022700, 0.011359, 0.005790, 0.002899, 0.001440,
)
_YBAR = (
    0.000039, 0.000120, 0.000396, 0.001210, 0.004000, 0.011600,
    0.023000, 0.038000, 0.060000, 0.090980, 0.139020, 0.208020,
    0.323000, 0.503000, 0.710000, 0.862000, 0.954000, 0.994950,
    0.995000, 0.952000, 0.870000, 0.757000, 0.631000, 0.503000,
    0.381000, 0.265000, 0.175000, 0.107000, 0.061000, 0.032000,
    0.017000, 0.008210, 0.004102, 0.002091, 0.001047, 0.000520,
)
_ZBAR = (
    0.006450, 0.020050, 0.067850, 0.207400, 0.645600, 1.385600,
    1.747060, 1.772110, 1.669200, 1.287640, 0.812950, 0.465180,
    0.272000, 0.158200, 0.078250, 0.042160, 0.020300, 0.008750,
    0.003900, 0.002100, 0.001650, 0.001100, 0.000800, 0.000340,
    0.000190, 0.000050, 0.000020, 0.000000, 0.000000, 0.000000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
)
# CIE D50 relative spectral power distribution, 380-730nm/10nm.
_D50_SPD = (
    24.488000, 29.871000, 49.308000, 56.513000, 60.034000, 57.818000,
    74.825000, 87.247000, 90.612000, 91.368000, 95.109000, 91.963000,
    95.724000, 96.613000, 97.129000, 102.099000, 100.755000, 102.317000,
    100.000000, 97.735000, 98.918000, 93.499000, 97.688000, 99.269000,
    99.042000, 95.722000, 98.857000, 95.667000, 98.190000, 103.003000,
    99.133000, 87.381000, 91.604000, 92.889000, 76.854000, 86.511000,
)

_X = np.array(_XBAR, dtype=float)
_Y = np.array(_YBAR, dtype=float)
_Z = np.array(_ZBAR, dtype=float)
_S = np.array(_D50_SPD, dtype=float)

N_WAVELENGTHS = 36

# Normalising constant for relative colorimetry: Y of a perfect diffuser = 100.
_K = 100.0 / float(np.sum(_S * _Y))

# White point (perfect diffuser under D50, 2-degree observer).
_WHITE = np.array([
    _K * float(np.sum(_S * _X)),
    _K * float(np.sum(_S * _Y)),
    _K * float(np.sum(_S * _Z)),
])


def reflectance_to_xyz(R: np.ndarray) -> np.ndarray:
    """Reflectance (..., 36) -> CIE XYZ (..., 3), D50/2-degree, Y scaled to 100."""
    R = np.atleast_2d(np.asarray(R, dtype=float))
    if R.shape[-1] != N_WAVELENGTHS:
        raise ValueError(
            f"Expected {N_WAVELENGTHS} reflectance values (380-730nm, 10nm steps), "
            f"got {R.shape[-1]}"
        )
    return np.stack([
        _K * (R * _S * _X).sum(axis=-1),
        _K * (R * _S * _Y).sum(axis=-1),
        _K * (R * _S * _Z).sum(axis=-1),
    ], axis=-1)


def _f(t: np.ndarray) -> np.ndarray:
    """CIELAB nonlinearity, with the linear segment near zero."""
    delta = 6.0 / 29.0
    return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)


def xyz_to_lab(XYZ: np.ndarray) -> np.ndarray:
    """CIE XYZ (..., 3) -> CIELAB (..., 3) against this module's D50 white point."""
    XYZ = np.atleast_2d(np.asarray(XYZ, dtype=float))
    fx, fy, fz = (_f(XYZ[..., i] / _WHITE[i]) for i in range(3))
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def reflectance_to_lab(R: np.ndarray) -> np.ndarray:
    """Reflectance (..., 36) -> CIELAB (..., 3). The conversion used throughout."""
    return xyz_to_lab(reflectance_to_xyz(R))


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference between two sets of Lab values.

    Standard formulation with kL = kC = kH = 1 (the graphic-arts default).
    Broadcasting: pass (N,3) and (N,3) for pairwise, or (N,3) and (1,3) to
    compare many against one.
    """
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=float))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=float))

    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)

    # a* scaling that pulls the near-neutral axis into better agreement
    G = 0.5 * (1.0 - np.sqrt(C_bar ** 7 / (C_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1.0 + G) * a1, (1.0 + G) * a2

    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    C_bar_p = 0.5 * (C1p + C2p)

    h1p = np.degrees(np.arctan2(b1, a1p)) % 360.0
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p

    # Hue difference, taking the short way round; zero when either chroma is 0
    both_chroma = (C1p * C2p) != 0
    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(both_chroma, dhp, 0.0)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(0.5 * dhp))

    L_bar_p = 0.5 * (L1 + L2)

    # Mean hue, again short way round, and again guarded for neutrals
    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    H_bar_p = np.where(
        both_chroma,
        np.where(
            h_diff <= 180.0, 0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
        h_sum,
    )

    T = (1.0
         - 0.17 * np.cos(np.radians(H_bar_p - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * H_bar_p))
         + 0.32 * np.cos(np.radians(3.0 * H_bar_p + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * H_bar_p - 63.0)))

    dtheta = 30.0 * np.exp(-(((H_bar_p - 275.0) / 25.0) ** 2))
    R_C = 2.0 * np.sqrt(C_bar_p ** 7 / (C_bar_p ** 7 + 25.0 ** 7))
    R_T = -R_C * np.sin(np.radians(2.0 * dtheta))

    S_L = 1.0 + (0.015 * (L_bar_p - 50.0) ** 2) / np.sqrt(20.0 + (L_bar_p - 50.0) ** 2)
    S_C = 1.0 + 0.045 * C_bar_p
    S_H = 1.0 + 0.015 * C_bar_p * T

    return np.sqrt(
        (dLp / S_L) ** 2
        + (dCp / S_C) ** 2
        + (dHp / S_H) ** 2
        + R_T * (dCp / S_C) * (dHp / S_H)
    )


def delta_e_2000_spectra(r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """CIEDE2000 straight from two sets of reflectance spectra."""
    return delta_e_2000(reflectance_to_lab(r1), reflectance_to_lab(r2))
