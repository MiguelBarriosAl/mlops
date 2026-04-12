"""
This module validates the dataset using Pydantic models to ensure data structure
and types before preprocessing.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ValidationError


class HousingData(BaseModel):
    """Pydantic model representing a single row of housing data."""

    MedInc: Optional[float]
    HouseAge: float
    AveRooms: Optional[float]
    Population: float
    Latitude: float
    Longitude: float


def validate_row(row: dict, index: int) -> bool:
    """Try to validate a single row against the HousingData model.

    Args:
        row: A dictionary representing one row of the dataset.
        index: The row index, used for error messages.

    Returns:
        True if the row is valid, False otherwise.
    """
    try:
        HousingData(**row)
        return True
    except ValidationError as e:
        print(f"Row {index} failed validation: {e}")
        return False


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate all rows in the dataset and print a summary.

    Args:
        df: The pandas DataFrame to validate.
    """
    total = len(df)
    valid = 0
    invalid = 0

    for index, row in enumerate(df.to_dict("records")):
        if validate_row(row, index):
            valid += 1
        else:
            invalid += 1

    print("\n--- Validation Summary ---")
    print(f"Total rows:   {total}")
    print(f"Valid rows:   {valid}")
    print(f"Invalid rows: {invalid}")


def main():
    """Load the raw dataset and run validation."""
    path = "data/raw/housing_noisy.csv"
    print(f"Loading dataset from: {path}")
    df = pd.read_csv(path)
    validate_dataset(df)


if __name__ == "__main__":
    main()
