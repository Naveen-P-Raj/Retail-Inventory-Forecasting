# Retail Inventory Forecasting

Machine-learning based retail sales forecasting using time-series feature engineering and LightGBM.

## Problem

Retailers need reliable sales estimates to reduce overstocking and avoid stockouts. This project uses historical store/product data to predict **Units Sold** while incorporating calendar, categorical, and lag-based information.

## Approach

1. Load the retail inventory dataset.
2. Convert `Date` to a datetime feature.
3. Create calendar features: `Year`, `Month`, and `DayOfWeek`.
4. Sort records by `Store ID`, `Product ID`, and `Date`.
5. Create 1-, 2-, and 3-step sales lag features.
6. Use a chronological train/test split at `2023-01-01`.
7. Train a LightGBM regression model.
8. Evaluate with MAE, RMSE, and R².
9. Inspect numerical correlations and prepare forecasting visualizations.

## Model

**LightGBM Regressor**

- `num_leaves = 31`
- `learning_rate = 0.05`
- `n_estimators = 300`
- `random_state = 42`

Categorical variables are represented using pandas categorical dtype for LightGBM, including Store ID, Product ID, Category, Region, Weather Condition, Holiday/Promotion, and Seasonality.

## Reported result

The supplied notebook reports the following held-out test performance:

| Metric | Value |
|---|---:|
| MAE | 7.2278 |
| RMSE | 8.4653 |
| R² | 0.99394 |
|
The chronological split contains **36,500 training rows** and **36,600 testing rows**.

These numbers are reported from the supplied notebook and should not be interpreted as a separately reproduced benchmark until the dataset is available and the notebook is rerun from a clean environment.

## Dataset

The project notebook loads `retail_store_inventory.csv`. The dataset itself is intentionally **not included** in this repository package because it was not supplied with the project files.

Place the dataset locally as:

```text
data/retail_store_inventory.csv
```

The original notebook currently uses the Google Colab path `/content/sample_data/retail_store_inventory.csv`; update that path when running locally.

## Repository structure

```text
Retail-Inventory-Forecasting/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── PROJECT_STATUS.md
├── notebooks/
│   └── retail_inventory_forecasting_original.ipynb
├── src/
│   └── train_lightgbm.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── PROJECT_DESCRIPTION.md
├── data/
│   └── README.md
└── results/
    └── README.md
```

## Reproducing the training workflow

```bash
pip install -r requirements.txt
python src/train_lightgbm.py --data data/retail_store_inventory.csv
```

The script implements the documented preprocessing, chronological split, LightGBM training, and evaluation workflow.

## Limitations

- The supplied project material does not include the dataset, so the reported metrics are preserved from the original notebook rather than newly generated here.
- The original notebook contains a future-forecast section that references `future_df` and `future_clean` without defining them in the supplied cells. It is therefore not presented as a fully reproducible 30-day forecasting pipeline in this repository.
- The model predicts `Units Sold`; it is not a direct inventory-level optimization system.

## Technologies

Python · Pandas · NumPy · LightGBM · Scikit-learn · Matplotlib · Seaborn · Jupyter

## Project context

The accompanying project report describes the broader objective as reducing inventory losses and preventing lost sales through daily sales forecasting, with lagged sales and promotional/calendar variables identified as useful predictors.
