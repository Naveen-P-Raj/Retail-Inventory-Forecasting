# Architecture

```text
Retail CSV
   │
   ▼
Load + datetime conversion
   │
   ├── Year
   ├── Month
   └── DayOfWeek
   │
   ▼
Sort by Store + Product + Date
   │
   ▼
Lag features
   ├── Units Sold_lag1
   ├── Units Sold_lag2
   └── Units Sold_lag3
   │
   ▼
Chronological split
   │  before 2023-01-01 → train
   │  2023-01-01 onward → test
   ▼
LightGBM Regressor
   │
   ▼
Predicted Units Sold
   │
   ▼
MAE / RMSE / R²
```

The architecture above follows the supplied notebook's implemented training workflow.
