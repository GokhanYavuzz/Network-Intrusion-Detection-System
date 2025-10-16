# Copilot Instructions for nids-adversarial

## Project Overview
This project is a Network Intrusion Detection System (NIDS) research codebase focused on adversarial machine learning. It uses the UNSW-NB15 dataset and related CSVs for experiments in the `data/` directory. Notebooks in `notebooks/` are used for exploratory data analysis (EDA) and model building.

## Key Directories
- `data/`: Contains all raw and processed datasets (e.g., `UNSW_NB15_train.csv`, `clear_data_full.csv`).
- `notebooks/`: Jupyter notebooks for EDA and model development. Main workflows are in `EDA.ipynb` and `ML_Model_Building_.ipynb`.
- `src/`: (If present) Place for reusable Python modules, model code, and utilities.
- `configs/`: (If present) Store experiment or model configuration files.
- `reports/`: (If present) Store generated reports, figures, or results.

## Data Flow & Architecture
- Data is loaded from `data/` CSVs, preprocessed in notebooks or `src/` scripts, and used for model training/testing.
- Notebooks are the primary interface for experimentation; scripts in `src/` may be imported for reusable logic.

## Developer Workflows
- **Environment:** Install dependencies from `requirements.txt` (e.g., `pip install -r requirements.txt`).
- **Notebook Execution:** Run and modify notebooks in `notebooks/` for EDA, feature engineering, and model training.
- **Data:** Do not commit large datasets; only use the provided CSVs in `data/`.
- **Testing:** (If present) Tests should be placed in a `tests/` directory or as notebook cells.

## Project Conventions
- Use clear, descriptive variable names for features and models.
- Prefer pandas for data manipulation and scikit-learn for ML models.
- Keep all experiment-specific code in notebooks; reusable code should go in `src/`.
- Document any new datasets or scripts in a markdown cell or a new file in `reports/`.

## Integration & Dependencies
- All dependencies must be listed in `requirements.txt`.
- Notebooks may use additional pip installs in cells, but these should be added to `requirements.txt` for reproducibility.

## Examples
- To add a new model, create a new notebook or add a section to `ML_Model_Building_.ipynb` and, if reusable, place model code in `src/`.
- For new datasets, add them to `data/` and document their schema in a markdown cell in the relevant notebook.

---

For questions or unclear conventions, review existing notebooks for examples or ask the maintainers.
