"""Generate the Working Plan v2 Stage-1 deliverable for Standard GPA."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.kinetics import build_stage_kinetics_table  # noqa: E402


INPUT_PATH = PROJECT_ROOT / 'notebooks' / 'segmented_dvs_data.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'stage1'
STANDARD_GPA_PATTERN = 'GPA 0.002% 0:90:0 x 3'


def _json_safe(value):
    """Convert NumPy/pandas objects in detailed fit records to JSON values."""
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return None if not np.isfinite(value) else value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main():
    segmented = pd.read_csv(INPUT_PATH)
    standard = segmented[
        segmented['sample_name'].str.contains(
            STANDARD_GPA_PATTERN, case=False, na=False, regex=False
        )
    ]
    if standard.empty:
        raise RuntimeError('Standard GPA rows were not found in the segmented dataset')

    table, details = build_stage_kinetics_table(standard)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUTPUT_DIR / 'standard_gpa_stage_kinetics.csv'
    detail_path = OUTPUT_DIR / 'standard_gpa_fit_diagnostics.json'
    table.to_csv(table_path, index=False)

    serialised = {
        str(stage_id): [
            {key: _json_safe(value) for key, value in record.items()}
            for record in frame.to_dict(orient='records')
        ]
        for stage_id, frame in details.items()
    }
    detail_path.write_text(json.dumps(serialised, indent=2))

    flag_counts = (
        table['flags'].str.split('; ').explode().replace('', np.nan).dropna().value_counts()
    )
    print(f'Wrote {len(table)} Standard GPA stages to {table_path}')
    print(f'Wrote full model diagnostics to {detail_path}')
    print('Most frequent flags: ' + ', '.join(
        f'{flag} ({count})' for flag, count in flag_counts.head(5).items()
    ))
    print('Model winners: ' + ', '.join(
        f'{model} ({count})' for model, count in table['best_model'].value_counts().items()
    ))


if __name__ == '__main__':
    main()
