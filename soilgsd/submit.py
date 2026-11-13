# Submission writer. Every Kaggle file must pass check_submission() before it is saved.

from pathlib import Path

import numpy as np
import pandas as pd

from soilgsd.config import ID_COLUMN, SIEVE_COLUMNS


# List what is wrong with one curve (11 cumulative percentages).
# Empty list means the curve is valid.
def curve_problems(P_pct, tol=1e-6):
    P = np.asarray(P_pct, dtype=float)
    problems = []
    if P.shape != (len(SIEVE_COLUMNS),):
        return [f"expected {len(SIEVE_COLUMNS)} values, got shape {P.shape}"]
    if not np.all(np.isfinite(P)):
        problems.append("contains NaN or infinity")
        return problems
    if P.min() < -tol or P.max() > 100 + tol:
        problems.append("values outside [0, 100]")
    if np.any(np.diff(P) < -tol):
        problems.append("not non-decreasing")
    if abs(P[-1] - 100.0) > tol:
        problems.append("last value (200 mm) is not 100")
    return problems


# Raise ValueError if the submission table is not acceptable.
# expected_ids: the sample IDs Kaggle expects (from sample_submission.csv).
def check_submission(df, expected_ids):
    expected_columns = [ID_COLUMN] + SIEVE_COLUMNS
    if list(df.columns) != expected_columns:
        raise ValueError(f"columns must be {expected_columns}, got {list(df.columns)}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError("duplicate sample IDs")
    if set(df[ID_COLUMN]) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(df[ID_COLUMN]))
        extra = sorted(set(df[ID_COLUMN]) - set(expected_ids))
        raise ValueError(f"sample IDs do not match. missing: {missing}, unexpected: {extra}")
    for _, row in df.iterrows():
        problems = curve_problems(row[SIEVE_COLUMNS].to_numpy(dtype=float))
        if problems:
            raise ValueError(f"{row[ID_COLUMN]}: {'; '.join(problems)}")


# Check and save a submission file, rows in the order of expected_ids.
# curves_by_id: dict sample_id -> 11 percentages. Returns the DataFrame written.
def write_submission(curves_by_id, expected_ids, out_path):
    missing = [i for i in expected_ids if i not in curves_by_id]
    if missing:
        raise ValueError(f"no prediction for: {missing}")
    rows = [[sample_id] + list(np.asarray(curves_by_id[sample_id], dtype=float)) for sample_id in expected_ids]
    df = pd.DataFrame(rows, columns=[ID_COLUMN] + SIEVE_COLUMNS)
    check_submission(df, expected_ids)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
