from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp",
    "acceleration",
]

OPTIONAL_COLUMNS = [
    "rpm",
    "velocity",
    "label",
    "axis",
]


def load_csv(file) -> pd.DataFrame:
    """
    Load and validate a vibration CSV file.
    """

    df = pd.read_csv(file)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Invalid CSV format.\n\n"
            f"Missing required columns: {missing_columns}\n\n"
            "Required columns: timestamp, acceleration\n"
            "Optional columns: rpm, velocity, label, axis"
        )

    return df