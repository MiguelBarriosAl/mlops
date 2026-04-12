"""
This module is responsible for validating data quality and schema.
"""

import pandas as pd

REQUIRED_COLUMNS = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

COLUMN_CONSTRAINTS = {
    "MedInc": {"min": 0.0, "description": "median income in block group"},
    "HouseAge": {"min": 0.0, "description": "median house age in block group"},
    "AveRooms": {"min": 0.0, "description": "average number of rooms per household"},
    "AveBedrms": {"min": 0.0, "description": "average number of bedrooms per household"},
    "Population": {"min": 0.0, "description": "block group population"},
    "AveOccup": {"min": 0.0, "description": "average number of household members"},
    "Latitude": {"min": -90.0, "max": 90.0, "description": "block group latitude"},
    "Longitude": {"min": -180.0, "max": 180.0, "description": "block group longitude"},
}


def validate_schema(df: pd.DataFrame) -> list[str]:
    """Check that all required columns are present in the DataFrame."""
    errors = []
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")
    return errors


def validate_no_nulls(df: pd.DataFrame) -> list[str]:
    """Check that there are no null values in required columns."""
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            errors.append(f"Column '{col}' has {null_count} null value(s).")
    return errors


def validate_ranges(df: pd.DataFrame) -> list[str]:
    """Check that numeric columns respect their expected value ranges."""
    errors = []
    for col, constraints in COLUMN_CONSTRAINTS.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if "min" in constraints:
            below_min = (df[col] < constraints["min"]).sum()
            if below_min > 0:
                errors.append(
                    f"Column '{col}' ({constraints['description']}) has {below_min} value(s) below minimum ({constraints['min']})."
                )
        if "max" in constraints:
            above_max = (df[col] > constraints["max"]).sum()
            if above_max > 0:
                errors.append(
                    f"Column '{col}' ({constraints['description']}) has {above_max} value(s) above maximum ({constraints['max']})."
                )
    return errors


def validate_data_types(df: pd.DataFrame) -> list[str]:
    """Check that required columns contain numeric data."""
    errors = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}.")
    return errors


def validate(df: pd.DataFrame) -> bool:
    """
    Run all validations on the DataFrame.

    Returns True if validation passes, raises ValueError with details otherwise.
    """
    all_errors: list[str] = []
    all_errors.extend(validate_schema(df))
    all_errors.extend(validate_data_types(df))
    all_errors.extend(validate_no_nulls(df))
    all_errors.extend(validate_ranges(df))

    if all_errors:
        error_summary = "\n".join(f"  - {e}" for e in all_errors)
        raise ValueError(f"Data validation failed with {len(all_errors)} error(s):\n{error_summary}")

    print(f"Data validation passed. Shape: {df.shape}")
    return True
