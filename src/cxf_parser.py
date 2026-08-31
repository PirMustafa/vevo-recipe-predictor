"""Parse X-Rite CxF3 (Color Exchange Format) files into a tabular dataset.

Each <cc:Object ObjectType="Trial"> element represents one measurement of one
ink formula, taken over either a light or dark backing (see the "PaperType"
tag). This module extracts, per object:
    - formula name
    - reflectance spectrum (36 values, 380-730nm, 10nm steps)
    - colorant recipe (fractions that sum to 1.0)
    - backing type ("Over light" / "Over dark")

and pivots pairs of (light, dark) measurements for the same formula into a
single wide row, which is the format the rest of the pipeline expects.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NS = {"cc": "http://colorexchangeformat.com/CxF3-core"}

# Fixed, canonical list of colorants seen in the UFO White PE dataset.
# Keeping this list fixed (rather than discovering it per-file) guarantees a
# stable feature/label schema across train/predict runs.
COLORANT_NAMES = [
    "UFO00061 transp. white",
    "UFO10031 yellow",
    "UFO20033 orange 021",
    "UFO30001 warm red",
    "UFO30002 rubine red",
    "UFO30003 rhodamine red",
    "UFO30032 red 032",
    "UFO40011 purple",
    "UFO40013 violet",
    "UFO50021 reflex blue",
    "UFO50022 process blue",
    "UFO50072 blue 072",
    "UFO60051 green",
    "UFO80071 black",
]

WAVELENGTHS = list(range(380, 731, 10))  # 36 points

# ---------------------------------------------------------------------------
# Presence threshold: below this fraction a colorant is treated as trace/noise
# rather than a real active ingredient.
#
# WHY 2%. Two independent reasons pointed the same way: sub-2% amounts barely
# move the measured spectrum, so they are close to undetectable from it; and
# raising the cut from 0% to 2% measurably improved Stage 1 exact-match.
#
# WHAT 2% COSTS. It is not a free win. The client's working minimum is 0.1%
# (occasionally 0.01%), so 2% erases quantities they really do dose. Measured
# on the dataset: at 0.02, 985 of the 8647 non-zero colorant entries (11.4%)
# are zeroed, and that touches 777 of the 2372 formulas (32.8%). A third of
# every formula in the file is therefore being learned in a slightly edited
# form. Retraining at a lower threshold is a genuine experiment, not a tweak --
# which is what the override below exists for.
#
# OVERRIDE via the VEVO_ACTIVE_THRESHOLD environment variable:
#     VEVO_ACTIVE_THRESHOLD=0.001 python train_backing_modes.py --models-root models/exp
#
# READ AT IMPORT TIME -- this matters. Consumers write
# `from .cxf_parser import ACTIVE_THRESHOLD`, which binds the VALUE at import,
# not the module attribute. Setting os.environ from inside a script AFTER its
# imports have run therefore does nothing at all, silently. The variable must
# be exported in the shell BEFORE the Python process starts.
#
# It lives here, in the dataset-vocabulary module, rather than beside the
# classifier that applies it: presence is a property of the DATA, and several
# CPU-only consumers need it (src/evaluation.py, train_inverse.py). Defining it
# next to the model would force them to import the GPU-only tabicl package to
# read one number, or -- worse -- to keep their own copy that drifts.
def _read_active_threshold() -> float:
    """Parse VEVO_ACTIVE_THRESHOLD, failing loudly rather than subtly.

    Unvalidated, a typo takes down every entry point at import time, because
    this constant is read while the module loads: VEVO_ACTIVE_THRESHOLD="0.1%"
    raises a bare ValueError inside app.py, predict.py and train.py alike. And
    a value of 0 -- the natural way to ask for "no threshold at all" -- makes
    every deadband_match() call in src/evaluation.py raise, which in train.py
    happens AFTER the production models have already been written to disk.

    The percent-vs-fraction slip is the easy one to make here, because the
    whole discussion with the client is in percentages: 0.1 means 10%, not
    0.1%. So say that in the error rather than letting float() shrug.
    """
    raw = os.environ.get("VEVO_ACTIVE_THRESHOLD")
    if raw is None:
        return 0.02
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(
            f"VEVO_ACTIVE_THRESHOLD={raw!r} is not a number. Give a FRACTION, "
            "not a percentage: 0.1% is 0.001 and 2% is 0.02."
        ) from None
    if not 0.0 < value < 1.0:
        raise ValueError(
            f"VEVO_ACTIVE_THRESHOLD={value!r} is out of range; expected "
            "0 < t < 1 as a FRACTION (0.1% -> 0.001, 2% -> 0.02). Note 0 is "
            "rejected deliberately: it makes every deadband metric raise."
        )
    return value


ACTIVE_THRESHOLD = _read_active_threshold()


@dataclass
class TrialRecord:
    formula: str
    object_id: str
    backing: str | None
    start_wl: float
    reflectance: list[float]
    colorants: dict[str, float] = field(default_factory=dict)


def _parse_trial_object(obj: ET.Element) -> TrialRecord | None:
    name = obj.get("Name")
    object_id = obj.get("Id")

    spectrum_el = obj.find(".//cc:ReflectanceSpectrum", NS)
    if spectrum_el is None or not spectrum_el.text:
        return None
    reflectance = [float(v) for v in spectrum_el.text.split()]
    start_wl = float(spectrum_el.get("StartWL", "380"))

    colorants: dict[str, float] = {}
    for colorant_el in obj.findall(".//cc:Colorant", NS):
        cname = colorant_el.get("Name")
        value_el = colorant_el.find("cc:Value", NS)
        if cname is not None and value_el is not None and value_el.text:
            colorants[cname] = float(value_el.text)

    backing = None
    for tag_el in obj.findall(".//cc:Tag", NS):
        if tag_el.get("Name") == "PaperType":
            backing = tag_el.get("Value")

    return TrialRecord(
        formula=name,
        object_id=object_id,
        backing=backing,
        start_wl=start_wl,
        reflectance=reflectance,
        colorants=colorants,
    )


def parse_cxf(path: str) -> list[TrialRecord]:
    """Parse a CxF3 file and return one TrialRecord per <cc:Object>."""
    tree = ET.parse(path)
    root = tree.getroot()
    records = []
    for obj in root.findall(".//cc:Object[@ObjectType='Trial']", NS):
        record = _parse_trial_object(obj)
        if record is not None:
            records.append(record)
    return records


def _backing_key(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if "light" in v:
        return "light"
    if "dark" in v:
        return "dark"
    return None


def build_wide_dataset(records: list[TrialRecord]) -> pd.DataFrame:
    """Pivot TrialRecords into one row per formula with light/dark spectra.

    Columns:
        formula
        R_light_<wl>, R_dark_<wl>   (36 columns each, NaN if missing)
        has_light, has_dark         (bool)
        colorant_<name>             (fraction 0-1, 14 columns)
    """
    by_formula: dict[str, dict] = {}

    for rec in records:
        key = _backing_key(rec.backing)
        if key is None:
            continue
        entry = by_formula.setdefault(
            rec.formula,
            {"formula": rec.formula, "colorants": None},
        )
        if len(rec.reflectance) != len(WAVELENGTHS):
            continue
        entry[f"R_{key}"] = np.array(rec.reflectance, dtype=float)
        # Recipe should be identical between light/dark pairs; keep the
        # first one seen, but overwrite if not yet set.
        if entry["colorants"] is None:
            entry["colorants"] = rec.colorants

    rows = []
    for formula, entry in by_formula.items():
        row = {"formula": formula}
        r_light = entry.get("R_light")
        r_dark = entry.get("R_dark")
        row["has_light"] = r_light is not None
        row["has_dark"] = r_dark is not None
        for i, wl in enumerate(WAVELENGTHS):
            row[f"R_light_{wl}"] = r_light[i] if r_light is not None else np.nan
            row[f"R_dark_{wl}"] = r_dark[i] if r_dark is not None else np.nan

        colorants = entry.get("colorants") or {}
        for cname in COLORANT_NAMES:
            row[f"colorant_{cname}"] = colorants.get(cname, 0.0)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_dataset(cxf_path: str) -> pd.DataFrame:
    """Convenience: parse a CxF file straight into the wide DataFrame."""
    records = parse_cxf(cxf_path)
    return build_wide_dataset(records)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else \
        r"e:\Flint_Group_Projects\Vevo_AI_Color_Matching\Data\NPXXXXCPWNUF UFO White PE.cxf"
    df = load_dataset(path)
    print(f"Parsed {len(df)} formulas")
    print(f"  both light+dark: {(df['has_light'] & df['has_dark']).sum()}")
    print(f"  light only: {(df['has_light'] & ~df['has_dark']).sum()}")
    print(f"  dark only: {(~df['has_light'] & df['has_dark']).sum()}")
    print(df.head())
