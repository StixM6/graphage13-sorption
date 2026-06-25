import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, List

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_dvs_excel(file_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses a single DVS Excel file to extract kinetic data and metadata by 
    explicitly targeting the 'DVS Data' sheet and hunting for keyword headers.
    """
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The specified file does not exist: {file_path}")

    logging.info(f"Processing file: {file_path.name}")
    sheet_name = "DVS Data"
    
    # Dynamically select the correct engine based on file extension
    engine = 'openpyxl' if file_path.suffix == '.xlsx' else 'xlrd'

    # --- 1. Top-Down Header Hunt ---
    try:
        # Read only the first 50 rows to rapidly scan for headers without loading the whole file
        meta_df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=50, engine=engine)
    except ValueError as e:
        logging.error(f"Sheet '{sheet_name}' not found in {file_path.name}. Error: {e}")
        return pd.DataFrame(), {"error": f"Sheet '{sheet_name}' not found"}

    header_row_index = -1
    
    # Scan top-down. A good header row will have a high number of text labels.
    # This is more robust than looking for specific keywords.
    best_candidate_row = -1
    max_text_cells = 0
    for idx, row in meta_df.iterrows():
        text_cells = sum(1 for cell in row if isinstance(cell, str) and cell.strip() != '')
        # A header should have at least 2 text labels (e.g., 'Time', 'Mass')
        if text_cells > max_text_cells and text_cells >= 2:
            max_text_cells = text_cells
            best_candidate_row = idx
    header_row_index = best_candidate_row

    if header_row_index == -1:
        logging.error(f"Could not find a suitable header row in {file_path.name}.")
        return pd.DataFrame(), {"error": "Data header not found", 'source_file': file_path.name}

    # --- 2. Metadata Extraction ---
    # Everything above our locked-on header row is metadata
    metadata = {'source_file': file_path.name}
    meta_block = meta_df.iloc[:header_row_index]
    
    for _, row in meta_block.iterrows():
        # Drop empty cells in the row to safely check pairs
        valid_cells = row.dropna().values
        if len(valid_cells) >= 2:
            key = str(valid_cells[0]).strip().lower()
            val = valid_cells[1]
            
            if "sample" in key:
                metadata['sample_name'] = str(val).strip()
            elif "temp" in key:
                try:
                    # Strip out '°C' if it exists in the string and convert to float
                    if isinstance(val, str):
                        val = val.replace('°C', '').replace('C', '').strip()
                    metadata['temperature_c'] = float(val)
                except (ValueError, TypeError):
                    metadata['temperature_c'] = None

    # --- 3. Data Table Extraction ---
    # Now load the full sheet using our exact header index
    data_df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_row_index,
        engine=engine
    )

    # --- 4. Data Transformation (Cleaning) ---
    # Standardize headers: lowercase, replace spaces with underscores, drop weird symbols
    data_df.columns = data_df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]+', '_', regex=True)

    # Find the primary time column (it might be 'time_min_', 'time', etc.)
    time_col = next((col for col in data_df.columns if 'time' in col), None)

    if time_col:
        # Force the time column to numeric (turns text strings like '(min)' into NaN)
        data_df[time_col] = pd.to_numeric(data_df[time_col], errors='coerce')
        # Drop any rows where time is NaN (cleans up unit rows and empty trailing rows automatically)
        data_df.dropna(subset=[time_col], inplace=True)
    else:
        logging.warning(f"No 'time' column identified after cleaning headers in {file_path.name}.")

    logging.info(f"Successfully parsed {len(data_df)} data points.")
    return data_df, metadata

def ingest_sorption_data(data_directory: Path) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Scans a directory for .xlsx files and parses each one."""
    logging.info(f"Starting ingestion from directory: {data_directory}")
    # Find both .xlsx and the older .xls file types
    excel_files = list(data_directory.glob("*.xlsx")) + list(data_directory.glob("*.xls"))
    
    if not excel_files:
        logging.warning("No .xlsx files found in the specified directory.")
        return []

    results = []
    for f in excel_files:
        try:
            results.append(parse_dvs_excel(f))
        except Exception as e:
            # This ensures that if one file is truly corrupt, the whole process doesn't stop.
            logging.error(f"A critical error occurred while parsing {f.name}: {e}")
            # Append a failure record so the notebook can report it.
            results.append((pd.DataFrame(), {'error': f'Critical error: {e}', 'source_file': f.name}))
    return results