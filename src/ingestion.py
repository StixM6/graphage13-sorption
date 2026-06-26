import pandas as pd
from pathlib import Path
import logging
from typing import Tuple, Dict, Any, List

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_dvs_excel(file_path: Path) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Parses a single DVS Excel file to extract both kinetic data (time-series)
    and isotherm data (equilibrium points), plus metadata.
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
        return {'kinetic_data': pd.DataFrame(), 'isotherm_data': pd.DataFrame()}, {"error": f"Sheet '{sheet_name}' not found"}

    header_row_index = -1
    
    # Scan top-down. A good header row will have a high number of text labels.
    best_candidate_row = -1
    max_text_cells = 0
    for idx, row in meta_df.iterrows():
        text_cells = sum(1 for cell in row if isinstance(cell, str) and cell.strip() != '')
        if text_cells > max_text_cells and text_cells >= 2:
            max_text_cells = text_cells
            best_candidate_row = idx
    header_row_index = best_candidate_row

    if header_row_index == -1:
        logging.error(f"Could not find a suitable header row in {file_path.name}.")
        return {'kinetic_data': pd.DataFrame(), 'isotherm_data': pd.DataFrame()}, {"error": "Data header not found", 'source_file': file_path.name}

    # --- 2. Metadata Extraction ---
    metadata = {'source_file': file_path.name}
    meta_block = meta_df.iloc[:header_row_index]
    
    junk_names = ['active reservoir:', 'new baskets', 'unknown_sample', '', 'none']
    
    for _, row in meta_block.iterrows():
        valid_cells = row.dropna().values
        if len(valid_cells) >= 2:
            key = str(valid_cells[0]).strip().lower()
            val = str(valid_cells[1]).strip()
            
            if "sample" in key and not any(junk in val.lower() for junk in junk_names):
                metadata['sample_name'] = val.lstrip(':').strip()
            elif "temp" in key:
                try:
                    val = val.replace('°C', '').replace('C', '').strip()
                    metadata['temperature_c'] = float(val)
                except (ValueError, TypeError):
                    metadata['temperature_c'] = None

    # --- 2.5 Fallback Mechanism ---
    if 'sample_name' not in metadata or any(junk in metadata['sample_name'].lower() for junk in junk_names):
        raw_filename = file_path.stem
        clean_name = raw_filename.replace('_', ' ').replace('-', ':')
        metadata['sample_name'] = clean_name

    # --- 3. Data Table Extraction ---
    dvs_data_df = pd.read_excel(
        file_path, sheet_name=sheet_name, header=header_row_index, engine=engine
    )
    
    try:
        iso_report_df = pd.read_excel(
            file_path, sheet_name="Iso Report", header=0, engine=engine
        )
    except Exception as e:
        logging.warning(f"Could not parse 'Iso Report' sheet in {file_path.name}: {e}")
        iso_report_df = pd.DataFrame()

    # --- 4. Data Transformation (Cleaning) ---
    dvs_data_df.columns = dvs_data_df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]+', '_', regex=True)
    
    if not iso_report_df.empty:
        iso_report_df.columns = iso_report_df.columns.astype(str).str.lower().str.strip().str.replace(r'[^a-z0-9_]+', '_', regex=True)
    
    time_col = next((col for col in dvs_data_df.columns if 'time' in col), None)
    if time_col:
        dvs_data_df[time_col] = pd.to_numeric(dvs_data_df[time_col], errors='coerce')
        dvs_data_df.dropna(subset=[time_col], inplace=True)

    data_package = {
        'kinetic_data': dvs_data_df,
        'isotherm_data': iso_report_df
    }

    logging.info(f"Successfully parsed {len(dvs_data_df)} kinetic points and {len(iso_report_df)} isotherm points.")
    return data_package, metadata

def ingest_sorption_data(data_directory: Path) -> List[Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]]:
    """Scans a directory for .xlsx and .xls files and parses each one."""
    logging.info(f"Starting ingestion from directory: {data_directory}")
    excel_files = list(data_directory.glob("*.xlsx")) + list(data_directory.glob("*.xls"))
    
    if not excel_files:
        logging.warning("No Excel files found in the specified directory.")
        return []

    results = []
    for f in excel_files:
        try:
            results.append(parse_dvs_excel(f))
        except Exception as e:
            logging.error(f"A critical error occurred while parsing {f.name}: {e}")
            empty_package = {'kinetic_data': pd.DataFrame(), 'isotherm_data': pd.DataFrame()}
            results.append((empty_package, {'error': f'Critical error: {e}', 'source_file': f.name}))
    return results