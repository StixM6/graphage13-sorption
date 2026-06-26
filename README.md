# GraPhage13 DVS Sorption Kinetics Project 

## Project Structure

* `data/raw/` — Contains raw DVS files (`*.xlsx` and legacy `*.xls` formats).
* `notebooks/` — Development sandbox.
    * `sorption_kinetics_pipeline.ipynb` — Pipeline checkpoint. Ingests all raw files, handles exception boundaries, and subsets data into a pristine 7-column matrix (`kinetic_df`).
* `src/` — Production-ready modular script files.
    * `ingestion.py` — Core parser featuring dynamic keyword-matching engine and multi-format engine switching (`openpyxl` / `xlrd`).

---

## 📂 Repository Architecture

This repository contains an automated data pipeline for processing Dynamic Vapor Sorption (DVS) experimental data and fitting kinetic models.

```text
├── data/
│   ├── raw/            # Raw .xls and .xlsx DVS data files
│   └── processed/      # Exported kinetic parameters and pore data
├── notebooks/
│   └── sorption_kinetics_pipeline.ipynb  # Main pipeline workspace
├── src/
│   ├── __init__.py
│   └── ingestion.py    # Multi-sheet Excel parsing engine (xlrd/openpyxl)
├── README.md
└── requirements.txt