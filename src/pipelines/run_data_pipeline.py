"""
This pipeline orchestrates the data ingestion, validation, and preprocessing steps.
"""

import sys

import pandas as pd

from src.data.validate_data import validate


def run_data_pipeline(input_path: str) -> pd.DataFrame:
    """Load data from *input_path*, validate it, and return the DataFrame."""
    print(f"Loading data from {input_path} ...")
    df = pd.read_csv(input_path)

    print("Running data validation ...")
    validate(df)

    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.pipelines.run_data_pipeline <input_csv_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    validated_df = run_data_pipeline(data_path)
    print(f"Pipeline complete. {len(validated_df)} rows ready for processing.")
