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
| `Backend/risk/return_model.py` | Behavioral model class, preprocessing, W&B integration |
| `Backend/risk/graph_model.py` | Graph-based network risk model |
| `Backend/risk/features.py` | Feature engineering pipeline (used by `build_data.py`) |
| `Backend/risk/build_data.py` | Dataset generation (produces `transformed_data.csv`) |
| `tests/test_risk.py` | Unit + integration tests |
| `dataHub/transformed_data.csv` | Pre-engineered dataset consumed by the models |

## FlowChart
```

                 Return Model
                      │
       ┌──────────────┴──────────────┐
       │                             │
 Behavioral features          Transaction features
       │                             │
       └──────────────┬──────────────┘
                      ↓
                 ML classifier
                      ↓
                Return Risk
```

---

# Graph Model — ReturnGuard

## Overview

The **graph model** (`Backend/risk/graph_model.py`) is the second-layer defense in the ReturnGuard pipeline. It determines whether a customer is part of a suspicious network based on shared entities (devices, IPs, addresses, payments) with other customers.

While the behavioral model (`return_model.py`) analyzes individual return patterns, the graph model answers: **"Does this customer have an unusually suspicious relationship with other customers/entities?"**

---

## Architecture

```
Orders DataFrame (customer_id, device_id, ip_id, address_id, payment_id)
        │
        ▼
┌──────────────────────────────────────────────┐
│  build_entity_index(orders)                  │  ← inverted index: entity → {customers}
│  O(E) where E = number of orders             │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  build_customer_graph(entity_index)          │  ← customer↔customer via shared entities
│  Edge attrs: shared_entity_types, n_shared   │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  extract_customer_features(                  │  ← 18 features per customer
│    customer_graph, customer_entities, orders)│     No target labels
│  )                                           │
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  Temporal graph snapshots per split          │  ← train graph, val graph, test graph
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  RandomForestClassifier                      │  ← class_weight="balanced"
│  + threshold optimization                    │     F1 or cost-based selection
└──────────────────┬───────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────┐
│  W&B logging + artifact save                 │  ← artifact: "graph-risk-model"
└──────────────────────────────────────────────┘
```

---

## Model Choice

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier` (scikit-learn) |
| Class weighting | `balanced` (auto-adjusts for class imbalance) |
| n_estimators | 100 |
| max_depth | 5 |
| n_jobs | -1 (parallel) |

A random forest is used as the baseline. It handles non-linear relationships, is robust to feature scaling, and provides feature importances. The model is trained on customer-level graph features only — no behavioral or transaction features.

---

## Graph Features (18 total)

| Category | Features |
|---|---|
| Entity counts (4) | `device_count`, `ip_count`, `address_count`, `payment_count` |
| Shared-entity customer counts (4) | `shared_device_customers`, `shared_ip_customers`, `shared_address_customers`, `shared_payment_customers` |
| Max sharing (4) | `max_device_sharing`, `max_ip_sharing`, `max_address_sharing`, `max_payment_sharing` |
| Neighborhood (2) | `one_hop_customer_count`, `two_hop_customer_count` |
| Community (2) | `community_size`, `community_density` |
| Structural (2) | `customer_degree`, `local_clustering_coefficient` |

### Feature Descriptions

- **Entity counts**: Number of distinct devices/IPs/addresses/payments used by the customer.
- **Shared-entity customer counts**: How many OTHER customers share at least one entity of each type.
- **Max sharing**: Maximum number of customers sharing any single entity (e.g., if one device is used by 5 customers, max_device_sharing = 5).
- **One-hop customers**: Customers directly connected via a shared entity.
- **Two-hop customers**: Customers reachable through another customer/entity (BFS depth=2).
- **Community size**: Size of the customer's community detected via greedy modularity.
- **Community density**: Internal edge density of the community.
- **Customer degree**: Total number of entity connections.
- **Local clustering coefficient**: NetworkX clustering coefficient for the customer node.

### Excluded by Design

- `abuse_label`, `abuse_type`, `scenario`, `ring_id` — ground truth / metadata
- Return-level and order-level outcomes
- Behavioral features (reserved for `return_model.py`)

---

## Temporal Strategy

Graph snapshots are built per split using only orders within that split's time period:

```
For each split (train / validation / test):
  1. Filter orders to that split
  2. Build entity_index from those orders
  3. Build customer_graph from entity_index
  4. Extract features for all customers in the graph
```

This avoids temporal leakage because each split's graph only contains orders from that split's time period. For production inference, build the graph from all available historical orders.

---

## API Reference

### `GraphRiskModel`

```python
from Backend.risk.graph_model import GraphRiskModel

model = GraphRiskModel(
    wandb_project="returnguard",
    wandb_entity=None,
    max_iter=100,        # number of trees
    max_depth=5,
    random_state=42,
)
```

#### `fit(orders, returns_df, validation_orders=None, validation_returns=None, run_name=None)`

Trains the graph model. Builds graph from orders, extracts customer features, trains classifier.

```python
import pandas as pd
df = pd.read_csv("dataHub/data_v2.csv")
train_df = df[df["split"] == "train"]
model.fit(orders=train_df, returns_df=train_df)
```

Logs to W&B:
- Train / validation metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- Confusion matrices
- Feature importances
- Model artifact (`graph-risk-model:latest`)

#### `predict_proba(orders, returns_df=None) -> DataFrame`

Returns a DataFrame with `customer_id` and `network_risk_probability`.

#### `predict(orders, returns_df=None, threshold=None) -> DataFrame`

Returns a DataFrame with `customer_id`, `network_risk_probability`, and `network_risk_label`.

#### `evaluate(orders, returns_df, threshold=None) -> dict`

Returns a dict of evaluation metrics.

#### `find_best_threshold(orders, returns_df, fp_cost=1.0, fn_cost=5.0, criterion="f1") -> float`

Finds the optimal threshold on validation data. Supports F1 maximization or cost minimization.

#### `get_customer_risk_explanation(customer_id, orders) -> dict`

Returns a structured explanation suitable for UI display:

```python
{
    "customer_id": "C1",
    "network_risk_probability": 0.87,
    "threshold": 0.5,
    "is_high_risk": True,
    "entity_counts": {"devices": 2, "ips": 3, "addresses": 1, "payments": 1},
    "sharing": {"shared_device_customers": 4, "shared_ip_customers": 6, ...},
    "neighborhood": {"one_hop_customer_count": 8, "two_hop_customer_count": 23},
    "community": {"community_size": 14, "community_density": 0.31},
    "structural": {"customer_degree": 7, "local_clustering_coefficient": 0.42},
}
```

#### `save(path)` / `load(path)`

Local persistence via `joblib`.

#### `load_from_wandb(project, entity, artifact_name, tag)`

Downloads the model artifact from W&B.

---

## W&B Integration

| Item | Detail |
|---|---|
| API key source | `WANDB` env var in `.env` |
| Project | `returnguard` |
| Artifact name | `graph-risk-model` |
| Artifact aliases | `latest`, `production` |
| Logged metrics | `train/*`, `val/*` (accuracy, precision, recall, f1, roc_auc, pr_auc, confusion matrix) |
| Logged tables | `feature_importances` |

---

## Test Results (real dataset)

```
Dataset: data_v2.csv (1823 rows, 507 customers)
Train: 1367 rows (436 customers)
Val:   182 rows (130 customers)
Test:  274 rows (177 customers)

Test Metrics:
  accuracy:  0.4350
  precision: 0.3667
  recall:    0.9167
  f1:        0.5238
  roc_auc:   0.6907
  pr_auc:    0.5437
  false_positive_rate: 0.8120
```

### Interpretation

The graph model achieves high recall (0.9167) but with a high false positive rate (0.8120). This is expected given the dataset limitations:

1. **Entity sharing is ubiquitous**: 507 customers compete for 56 devices (~9 customers/device on average). Random entity collision creates dense sharing patterns regardless of abuse status.
2. **Ring is tiny**: Only 2 customers out of 507 are ring members (0.7%). The graph cannot learn meaningful ring-detection patterns from this.
3. **The model learns from noise**: Entity sharing patterns in the synthetic data reflect random assignment, not coordinated abuse behavior.

**This performance must not be interpreted as evidence of real-world ring detection capability.** It is a synthetic-experiment result only.

---

## Limitations of `build_data.py` v2

| Limitation | Impact |
|---|---|
| Ring members are only 2 customers (0.7%) | Community detection and shared-entity patterns are too weak to learn |
| `ring_id` column is empty in CSV | Cannot verify ring grouping |
| Entity assignment is random per customer | Graph structure reflects random collision, not behavior |
| Non-ring customers also share entities randomly | Dense graph with no clear abuse signal |

### Recommendations

1. **Increase ring size**: Make ring members 20-30% of abusers (currently ~3%).
2. **Make entity sharing deterministic per scenario**: Ring members should share entities more heavily than normal customers.
3. **Add `ring_id` to the CSV output**: Fix the merge in `_create_final_dataset`.
4. **Reduce entity pool size**: Fewer devices/IPs per customer creates more meaningful sharing patterns.

---

## Running Tests

```bash
# Graph model unit tests (no W&B)
python -m unittest tests.test_risk.TestGraphRiskModelLocal -v

# W&B integration test (requires WANDB env var)
set WANDB=your_wandb_key
python -m unittest tests.test_risk.TestGraphRiskModelWandb -v

# All tests
python -m unittest tests.test_risk -v
```

---

## Files

| File | Purpose |
|---|---|
| `Backend/risk/graph_model.py` | Graph model class, feature extraction, W&B integration |
| `Backend/risk/return_model.py` | Behavioral model (for comparison) |
| `Backend/risk/build_data.py` | Dataset generation with entity assignment |
| `tests/test_risk.py` | Unit + integration tests |
| `dataHub/data_v2.csv` | Raw dataset with orders, returns, and entities |
| `dataHub/transformed_data.csv` | Feature-engineered dataset |
