# Statistical Model — ReturnGuard

## Overview

The **statistical model** (`Backend/risk/return_model.py`) is the first-layer defense in the ReturnGuard pipeline. It predicts whether an individual return transaction is abusive using behavioral, transactional, temporal, and geographical features extracted from the dataset.

This is **not** the graph model. Graph/network risk scoring is handled separately in `graph_model.py`.

---

## Model Choice

| Property | Value |
|---|---|
| Algorithm | `LogisticRegression` (scikit-learn) |
| Class weighting | `balanced` (auto-adjusts for ~8% abuse rate) |
| Solver | `lbfgs` |
| Regularization | L2, `C=1.0` |
| Max iterations | 1000 |

A simple logistic regression is used as the baseline. It is fast, interpretable (coefficients map directly to feature importance), and production-friendly. Future iterations may upgrade to gradient boosting or ensemble methods.

---

## Data Flow

```
transformed_data.csv
        │
        ▼
┌─────────────────────────┐
│  split column filter    │  ← train / validation / test (temporal split)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Drop leakage columns   │  ← abuse_type, abuse_label, split, IDs, raw dates
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LabelEncode categoricals│  ← return_reason, product_category, payment_method, city, state
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Impute (median) + Scale │  ← StandardScaler fit on train only
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  LogisticRegression.fit  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  W&B: log metrics +     │
│  save model artifact     │
└─────────────────────────┘
```

---

## Feature Columns (130)

The model consumes all numeric and label-encoded columns from `transformed_data.csv` **except** the following excluded groups:

### Excluded (leakage / non-predictive)

| Group | Columns |
|---|---|
| Target | `abuse_label` |
| Direct leakage | `abuse_type` |
| Split indicator | `split` |
| IDs | `return_id`, `order_id`, `customer_id`, `device_id`, `ip_id`, `address_id`, `payment_id` |
| Raw dates | `return_date`, `purchase_date`, `account_creation_date` |
| Constant | `country` |

### Included feature groups

- **Transaction**: `order_amount`, `refund_amount`, `discount_percentage`, `log_order_amount`, `refund_ratio`, `category_risk_score`, `payment_risk_score`, ...
- **Customer history**: `number_of_previous_orders`, `number_of_previous_returns`, `historical_return_rate`, `avg_order_value`, `avg_return_value`, `previous_chargebacks`, ...
- **Temporal windows**: `orders_last_7_days`, `orders_last_30_days`, `returns_last_7_days`, `returns_last_30_days`, `return_rate_30d`, `return_rate_90d`, ...
- **Behavioral**: `order_frequency`, `return_to_order_ratio`, `refund_to_order_ratio`, `return_consistency`, `fast_purchaser`, ...
- **Geographical**: `latitude`, `longitude`, `distance_from_shipping_address`, `unique_locations_30d`
- **Graph-lite**: `linked_account_count`, `shared_device_count`, `shared_ip_count`, `shared_address_count`, `shared_payment_count`
- **Engineered interactions**: `days_amount_interaction`, `return_rate_amount`, `age_return_interaction`, `tenure_returns_ratio`, ...
- **Encoded categoricals**: `return_reason` (encoded), `product_category` (encoded), `payment_method` (encoded), `city` (encoded), `state` (encoded)

---

## API Reference

### `ReturnRiskModel`

```python
from Backend.risk.return_model import ReturnRiskModel

model = ReturnRiskModel(
    wandb_project="returnguard",   # W&B project name
    wandb_entity=None,             # optional W&B team/user
    C=1.0,                         # LogisticRegression inverse regularization
    max_iter=1000,
    random_state=42,
)
```

#### `fit(train_df, validation_df=None, run_name=None)`

Trains the model on the provided DataFrame(s). Internally filters by the `split` column.

```python
import pandas as pd
df = pd.read_csv("dataHub/transformed_data.csv")
model.fit(df)
```

Logs to W&B:
- Train / validation / test metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- Confusion matrices
- Feature coefficient table
- Model artifact (`return-risk-model:latest`)

#### `predict_proba(df) -> np.ndarray`

Returns the probability of abuse (class 1) for each row. Gracefully handles missing or extra columns.

#### `predict(df, threshold=0.5) -> np.ndarray`

Binary predictions. Default threshold is 0.5; tune for precision/recall tradeoff.

#### `evaluate(df, threshold=0.5) -> dict`

```python
{
    "accuracy": 0.8654,
    "precision": 0.8515,
    "recall": 0.8113,
    "f1": 0.8309,
    "roc_auc": 0.9235,
    "pr_auc": 0.8611,
    "true_negatives": 139,
    "false_positives": 15,
    "false_negatives": 20,
    "true_positives": 86,
}
```

#### `save(path)` / `load(path)`

Local persistence via `joblib`. Saves the full bundle: model, scaler, encoders, feature list, metadata.

```python
model.save("dataHub/return_risk_model.joblib")
loaded = ReturnRiskModel.load("dataHub/return_risk_model.joblib")
```

#### `load_from_wandb(project, entity, artifact_name, tag)`

Downloads the model artifact from W&B and returns a ready-to-use `ReturnRiskModel`.

```python
model = ReturnRiskModel.load_from_wandb(project="returnguard", tag="production")
model.predict_proba(new_data)
```

---

## W&B Integration

| Item | Detail |
|---|---|
| API key source | `WANDB` env var in `.env` |
| Project | `returnguard` |
| Artifact name | `return-risk-model` |
| Artifact aliases | `latest`, `production` |
| Logged metrics | `train/*`, `val/*`, `test/*` (accuracy, precision, recall, f1, roc_auc, pr_auc, confusion matrix) |
| Logged tables | `feature_coefficients` |

---

## Test Results (real dataset)

```
Test split: 260 samples
Abuse rate: ~41% in test set

accuracy:  0.8654
precision: 0.8515
recall:    0.8113
f1:        0.8309
roc_auc:   0.9235
pr_auc:    0.8611
```

---

## Running Tests

```bash
# Local unit tests (no W&B required)
python -m unittest tests.test_risk.TestReturnRiskModelLocal -v

# Signature verification tests
python -m unittest tests.test_risk.TestReturnRiskModelSignature -v

# W&B integration test (requires WANDB env var)
set WANDB=your_wandb_key
python -m unittest tests.test_risk.TestWandbArtifactDownload -v
```

---

## Files

| File | Purpose |
|---|---|
| `Backend/risk/return_model.py` | Model class, preprocessing, W&B integration |
| `Backend/risk/features.py` | Feature engineering pipeline (used by `build_data.py`) |
| `Backend/risk/build_data.py` | Dataset generation (produces `transformed_data.csv`) |
| `tests/test_risk.py` | Unit + integration tests |
| `dataHub/transformed_data.csv` | Pre-engineered dataset consumed by the model |
