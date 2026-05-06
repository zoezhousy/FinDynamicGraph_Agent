from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_parquet(df: pd.DataFrame, output_file: Path) -> None:
    ensure_dir(output_file.parent)
    df.to_parquet(output_file, index=False)

