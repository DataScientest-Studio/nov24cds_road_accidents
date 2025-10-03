# Road Accidents ML Project

This project analyzes and models French road accident data (2005–2023).  
It follows a modular pipeline for **data processing, feature engineering, model training, prediction, and explainability**.

---

## Project Structure

```
├── src                <- Source code
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── features       <- Feature engineering scripts
│   │   └── build_features.py
│   │
│   ├── models         <- Model training and prediction
│   │   ├── train_model.py
│   │   └── predict_model.py
│   │
│   ├── visualization  <- Explainability and plots
│   │   └── visualize.py
│   │
├── data
│   ├── raw            <- Original datasets (2005–2023)
│   ├── processed      <- Intermediate & final processed files
│   │   ├── accidents_merged_2005_2023.csv
│   │   ├── accidents_processed.csv
│   │   ├── accidents_cleaned.csv
│   │   └── df_for_ml.csv
│   │
├── models             <- Saved trained models & results
├── reports
│   ├── figures        <- Confusion matrices & plots
│   ├── shap           <- SHAP visualizations
```

---

## Code Descriptions

### `src/features/build_features.py`
- Loads yearly accident datasets and merges them.  
- Cleans types (categorical, string, numeric, datetime).  
- Fixes GPS coordinates, INSEE codes, and missing values.  
- Engineers new features:
  - `hour`, `dayofweek`, `rush_hour`, `season`, `age_bin`  
  - `agg_catr` (urban × road type)  
  - `catv_group` (vehicle grouping)  
  - `belt_status`, `belt_user_type_code`  
- Saves:  
  - `accidents_processed.csv`  
  - `accidents_cleaned.csv`  
  - `df_for_ml.csv` (final ML-ready dataset).  

### `src/models/train_model.py`
- Trains a **Balanced Random Forest** classifier.  
- Uses `ColumnTransformer` preprocessing:  
  - Numeric → imputation + scaling  
  - Categorical → imputation + one-hot encoding + variance filter  
- Tunes hyperparameters with `GridSearchCV`.  
- Threshold tuning for best Recall/F1 balance.  
- Saves:  
  - Model (`.joblib`)  
  - Confusion matrices (PNG)  
  - Metrics and params (JSON).  

### `src/models/predict_model.py`
- Loads a trained pipeline.  
- Applies preprocessing to new input data.  
- Outputs predictions + probabilities.  
- Can generate new CSVs for unseen datasets.  

### `src/visualization/visualize.py`
- Generates **SHAP explainability visualizations**:  
  - Summary, bar, violin, heatmap, dependence & interaction plots  
  - Force plots for FN/FP/TP/high-risk cases  
- Exports:  
  - PNG figures  
  - `shap_streamlit_summary.json`.  

---

## How to Run

```bash
conda activate roadacc
cd C:\Users\jl\Documents\02_DataScience_Course_2024-2025\nov24cds_road_accidents
```

1. **Feature Engineering**  
   Build processed datasets for ML:
   ```bash
   python -m src.features.build_features
   ```

2. **Train Model**  
   Train Balanced Random Forest on processed data:
   ```bash
   python -m src.models.train_model
   ```

3. **Predict on Data**  
   Run predictions using saved model:
   ```bash
   python -m src.models.predict_model --input data/processed/df_for_ml.csv --output reports/predictions.csv
   ```

4. **Generate SHAP Visualizations**  
   Explain model predictions:
   ```bash
   python -m src.visualization.visualize
   ```

---

## Data Source
- Accidents corporels de la circulation routière 2005–2023 (data.gouv.fr)  
- Communes de France – Base des codes postaux  

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
