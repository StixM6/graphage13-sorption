# GraPhage13 DVS Sorption Kinetics Project 

## Project Structure

* `data/raw/` — Contains raw DVS files (`*.xlsx` and legacy `*.xls` formats).
* `notebooks/` — Development sandbox.
    * `data_ingestion.ipynb` — Pipeline checkpoint. Ingests all raw files, handles exception boundaries, and subsets data into a pristine 7-column matrix (`kinetic_df`).
* `src/` — Production-ready modular script files.
    * `ingestion.py` — Core parser featuring dynamic keyword-matching engine and multi-format engine switching (`openpyxl` / `xlrd`).

---

## Engineering Log & Recent Updates

### [24 to 25 Jun 2026] Ingestion Pipeline Refactoring & Multi-Format Support
* **Issue:** The initial "bottom-up" row parser threw `Data header not found` errors, tripping over empty trailing cells, summary statistics blocks, and non-numeric unit rows in multi-tab DVS workbooks.
* **Resolution:** 1. Rewrote `src/ingestion.py` to use a top-down keyword search engine that targets the `DVS Data` sheet and locks onto the row containing both `"Time"` and `"Mass"`.
  2. Implemented an automated type-coercion layer (`pd.to_numeric`) that cleanly turns formatting rows (like unit definitions) into `NaN` values and strips them out.
  3. Created a feature-isolation step in the master notebook, successfully condensing an expanded 94-column structural concatenation down to a pristine 7-column kinetics DataFrame (`30,076` clean entries).
* **Legacy Format Expansion:** Expanded the script loop to find and parse older `*.xls` files alongside standard `*.xlsx` variants. Added an extension check to dynamically switch underlying spreadsheet engines (`openpyxl` for modern xml files, `xlrd` for legacy binary files).