"""The Pantone target library, and its linkage to the UFO formula names.

Discovered 2026-08-17: the two CxF files in this project are a PAIRED dataset,
not two unrelated colour sets. Every UFO formula name encodes the Pantone
colour it was formulated to match:

    NP0325CPWNUF  <->  PANTONE 325 C
    NP7469CPWNUF  <->  PANTONE 7469 C

i.e. `NP` + the Pantone code zero-padded to 4 digits + `PWNUF`. The dataset
filename's own `XXXX` placeholder (`NPXXXXCPWNUF UFO White PE.cxf`) is that
code slot.

Confirmed colorimetrically, not just by string pattern: across 1817 formulas
with both a target and both-backing measurements, name-matched pairs agree to
a mean CIEDE2000 of 0.75, while the same data with the pairing shuffled sits
at 39.5 -- a 53x separation.

Two things this module is careful about:

  - Comparisons use the UFO **light-backing** spectrum. Pantone Coated
    swatches are opaque ink on white coated stock, so the light backing is the
    only fair analogue; comparing against the dark backing would measure the
    substrate, not the formulation.
  - The Pantone file mixes measurement modes (2069 M0, 66 M2 of 2135). M0 and
    M2 differ in UV content, so a small part of any residual difference is
    instrument disagreement rather than formulation error. No correction is
    applied -- callers that report a delta to a user should treat sub-1.0
    values as "at the noise floor".
"""
from __future__ import annotations

import pathlib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .colorimetry import delta_e_2000, reflectance_to_lab

NS = {"cc": "http://colorexchangeformat.com/CxF3-core"}

N_WAVELENGTHS = 36

# Below this, a difference is inside the combined instrument/mode noise of the
# two files and should not be reported as a real miss. Derived from the
# matched-pair median (0.45) and the M0/M2 mode inconsistency in the source.
NOISE_FLOOR_DE = 1.0

# Conventional graphic-arts reading of a CIEDE2000 magnitude.
_BANDS = (
    (1.0, "within instrument noise"),
    (2.0, "commercial match"),
    (3.5, "close, visible to a trained eye"),
    (5.0, "clearly off target"),
    (float("inf"), "different colour"),
)


@dataclass(frozen=True)
class PantoneTarget:
    """One Pantone standard: its name, its measured spectrum, and its Lab."""
    name: str
    code: str
    reflectance: np.ndarray
    lab: np.ndarray


def pantone_code(name: str) -> str:
    """'PANTONE 325 C' -> '325C'.  'PANTONE Black 6 C' -> 'BLACK6C'."""
    return re.sub(r"[^A-Z0-9]", "", name.upper().replace("PANTONE", "").strip())


def formula_to_pantone_code(formula: str) -> str | None:
    """'NP0325CPWNUF' -> '325C'. Returns None if the name isn't in that shape.

    Leading zeros are stripped so the 4-digit padded formula code lines up with
    Pantone's unpadded naming.
    """
    m = re.fullmatch(r"NP(.+?)PWNUF", formula.upper().strip())
    if not m:
        return None
    core = m.group(1)
    m2 = re.fullmatch(r"0*(\d+)([A-Z]*)", core)
    return (m2.group(1) + m2.group(2)) if m2 else core


def load_pantone_library(path: str | pathlib.Path) -> dict[str, PantoneTarget]:
    """Parse a Pantone CxF3 file into {code: PantoneTarget}.

    Returns an empty dict if the file is missing, so the pipeline degrades to
    its previous behaviour rather than failing -- the target library is an
    enhancement, not a hard requirement.
    """
    path = pathlib.Path(path)
    if not path.is_file():
        return {}

    root = ET.parse(path).getroot()
    library: dict[str, PantoneTarget] = {}
    for obj in root.findall(".//cc:Object[@ObjectType='Target']", NS):
        name = obj.get("Name")
        el = obj.find(".//cc:ReflectanceSpectrum", NS)
        if not name or el is None or not el.text:
            continue
        values = [float(v) for v in el.text.split()]
        if len(values) != N_WAVELENGTHS:
            continue
        R = np.array(values, dtype=float)
        code = pantone_code(name)
        library[code] = PantoneTarget(
            name=name, code=code, reflectance=R,
            lab=reflectance_to_lab(R)[0],
        )
    return library


def interpret_delta(dE: float) -> str:
    """Plain-language reading of a CIEDE2000 value, for user-facing output."""
    for limit, label in _BANDS:
        if dE < limit:
            return label
    return _BANDS[-1][1]


def target_for_formula(
    formula: str, library: dict[str, PantoneTarget]
) -> PantoneTarget | None:
    """The Pantone standard a given UFO formula was formulated to match."""
    code = formula_to_pantone_code(formula)
    return library.get(code) if code else None


def target_deltas(
    formulas, r_light: np.ndarray, library: dict[str, PantoneTarget]
) -> np.ndarray:
    """CIEDE2000 of each measured light-backing spectrum vs its own Pantone target.

    Returns an array parallel to `formulas`, with NaN wherever no target exists
    for that formula (so callers can distinguish "no target to check against"
    from "checked and fine").
    """
    formulas = list(formulas)
    r_light = np.asarray(r_light, dtype=float)
    out = np.full(len(formulas), np.nan)
    if not library:
        return out

    rows, targets = [], []
    for i, f in enumerate(formulas):
        t = target_for_formula(f, library)
        if t is not None:
            rows.append(i)
            targets.append(t.lab)
    if not rows:
        return out

    measured = reflectance_to_lab(r_light[rows])
    out[rows] = delta_e_2000(measured, np.array(targets))
    return out


def nearest_targets(
    reflectance: np.ndarray, library: dict[str, PantoneTarget], n: int = 5
) -> list[tuple[PantoneTarget, float]]:
    """The n closest Pantone standards to a measured spectrum, by CIEDE2000.

    `reflectance` should be a light-backing spectrum -- see the module note.
    """
    if not library:
        return []
    targets = list(library.values())
    lab = reflectance_to_lab(np.asarray(reflectance, dtype=float).reshape(1, -1))
    deltas = delta_e_2000(
        np.repeat(lab, len(targets), axis=0),
        np.array([t.lab for t in targets]),
    )
    order = np.argsort(deltas)[:n]
    return [(targets[i], float(deltas[i])) for i in order]


def assess_against_target(
    reflectance: np.ndarray, code: str, library: dict[str, PantoneTarget]
) -> dict | None:
    """How far a measured sample sits from a named Pantone target.

    This answers a question the recipe prediction does NOT: the predicted recipe
    reproduces *the sample that was measured*, which may itself be off-target.
    Returns None if the code isn't in the library.
    """
    target = library.get(pantone_code(code))
    if target is None:
        return None

    R = np.asarray(reflectance, dtype=float).reshape(1, -1)
    measured_lab = reflectance_to_lab(R)[0]
    dE = float(delta_e_2000(measured_lab.reshape(1, -1), target.lab.reshape(1, -1))[0])

    return {
        "target_name": target.name,
        "target_code": target.code,
        "delta_e_2000": round(dE, 2),
        "interpretation": interpret_delta(dE),
        "on_target": bool(dE < NOISE_FLOOR_DE),
        "measured_lab": [round(float(v), 2) for v in measured_lab],
        "target_lab": [round(float(v), 2) for v in target.lab],
    }
