"""Input validation and portable output paths shared by the analyses."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = [
    "Study_ID",
    "HATMD",
    "Age",
    "Female",
    "Entry_quarter",
    "Symptom_duration_mo",
    "VAS",
    "Bilateral_pain",
    "MUO_mm",
    "Sleep_duration_h",
    "PSQI_global",
    "Poor_sleeper",
    "Stress",
    "Clenching",
    "Bruxism",
] + [f"PSQI_C{i}" for i in range(1, 8)]


def paths(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Original XLSX workbook (Data sheet) or matching CSV",
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        parser.error(f"Input file does not exist: {source}")
    for folder in ["tables", "figures", "private"]:
        (output / folder).mkdir(parents=True, exist_ok=True)
    return source, output


def load_data(source):
    if source.suffix.lower() == ".csv":
        data = pd.read_csv(source, dtype={"Study_ID": str})
    elif source.suffix.lower() == ".xlsx":
        data = pd.read_excel(source, sheet_name="Data", dtype={"Study_ID": str})
    else:
        raise ValueError("Input must be XLSX or CSV")
    absent = sorted(set(REQUIRED) - set(data.columns))
    if absent:
        raise ValueError(f"Missing columns: {absent}")
    data = data.loc[data.Study_ID.notna()].copy()
    if data.Study_ID.duplicated().any():
        raise ValueError("Study_ID must be unique: one baseline row per patient")
    for name in REQUIRED:
        if name not in ["Study_ID", "Entry_quarter"]:
            data[name] = pd.to_numeric(data[name], errors="raise")
    for name in ["HATMD", "Female", "Stress", "Clenching", "Bruxism"]:
        if data[name].isna().any() or not data[name].isin([0, 1]).all():
            raise ValueError(f"{name} must be complete and coded 0/1")
    for name in ["Poor_sleeper", "Bilateral_pain"]:
        if not data[name].dropna().isin([0, 1]).all():
            raise ValueError(f"{name} must be coded 0/1 or missing")
    ranges = {
        "Age": (0, 120),
        "VAS": (0, 10),
        "PSQI_global": (0, 21),
        "Sleep_duration_h": (0, 24),
        "Symptom_duration_mo": (0, np.inf),
    }
    ranges.update({f"PSQI_C{i}": (0, 3) for i in range(1, 8)})
    for name, (low, high) in ranges.items():
        if not data[name].dropna().between(low, high).all():
            raise ValueError(f"Out-of-range values: {name}")
    components = data[[f"PSQI_C{i}" for i in range(1, 8)]]
    for name in components.columns:
        if not components[name].dropna().isin([0, 1, 2, 3]).all():
            raise ValueError(f"{name} must contain integer scores 0–3 or missing")
    complete = components.notna().all(axis=1)
    if not np.array_equal(complete, data.PSQI_global.notna()):
        raise ValueError(
            "PSQI_global must be missing unless all 7 components are available"
        )
    if not np.allclose(
        components.loc[complete].sum(axis=1), data.loc[complete, "PSQI_global"]
    ):
        raise ValueError("PSQI_global must equal the sum of the seven components")
    if not np.array_equal(data.Poor_sleeper.notna(), complete):
        raise ValueError("Poor_sleeper missingness must match PSQI_global")
    if not np.array_equal(
        data.loc[complete, "Poor_sleeper"],
        (data.loc[complete, "PSQI_global"] > 5).astype(int),
    ):
        raise ValueError("Poor_sleeper must equal PSQI_global > 5")
    if data.Entry_quarter.isna().any():
        raise ValueError("Entry_quarter must be complete")
    return data
