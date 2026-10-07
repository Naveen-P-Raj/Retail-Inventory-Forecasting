"""Train and evaluate the retail sales forecasting model.

Usage:
    python src/train_lightgbm.py --data data/retail_store_inventory.csv
"""

import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CATEGORICAL_COLS = [
    "Store ID", "Product ID", "Category", "Region",
    "Weather Condition", "Holiday/Promotion", "Seasonality"
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df = df.sort_values(["Store ID", "Product ID", "Date"])

    group = df.groupby(["Store ID", "Product ID"])["Units Sold"]
    df["Units Sold_lag1"] = group.shift(1)
    df["Units Sold_lag2"] = group.shift(2)
    df["Units Sold_lag3"] = group.shift(3)
    return df.fillna(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to retail_store_inventory.csv")
    parser.add_argument("--split-date", default="2023-01-01")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df = build_features(df)

    split_date = pd.Timestamp(args.split_date)
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]

    y_train = train["Units Sold"]
    y_test = test["Units Sold"]
    X_train = train.drop(columns=["Units Sold", "Date"])
    X_test = test.drop(columns=["Units Sold", "Date"])

    for col in CATEGORICAL_COLS:
        X_train[col] = X_train[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    model = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    print(f"Training rows: {len(train)}")
    print(f"Testing rows : {len(test)}")
    print("\n--- MODEL PERFORMANCE ---")
    print(f"MAE : {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²  : {r2:.6f}")


if __name__ == "__main__":
    main()
