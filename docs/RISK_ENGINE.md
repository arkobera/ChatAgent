# Risk Engine

`Backend.risk.RiskEngine` combines the existing behavioral return-risk model
and graph network-risk model into one return-level decision. It orchestrates
already trained models only; it does not train models, construct graphs, or
perform retrieval or UI work.

## Changes added

- Added `Backend/risk/risk_engine.py` with the typed `RiskEngine` class.
- Added dependency-injected construction with a `ReturnRiskModel` and a
  `GraphRiskModel`, keeping both existing model interfaces unchanged.
- Added `RiskEngine.from_wandb()`, which loads `.env`, requires the `WANDB`
  environment variable, and loads the `production` W&B artifacts for both
  models.
- Added configurable behavioral and graph weights, defaulting to `0.6` and
  `0.4`. Weights must be finite, non-negative, and sum to one.
- Added configurable overall-risk thresholds, defaulting to `0.30` for REVIEW
  and `0.70` for HIGH. Thresholds must be finite, in `[0, 1]`, and ordered.
- Added `assess(returns_df, orders_df)`, returning one row per return with:
  `return_id`, `customer_id`, `return_risk`, `network_risk`, `overall_risk`,
  `risk_level`, and `recommended_action`.
- Combined prediction formula:

  ```text
  overall_risk = 0.6 * return_risk + 0.4 * network_risk
  ```

- Added decision mapping:

  | Overall risk | Level | Action |
  | --- | --- | --- |
  | `< 0.30` | LOW | APPROVE |
  | `>= 0.30` and `< 0.70` | REVIEW | MANUAL_REVIEW |
  | `>= 0.70` | HIGH | HOLD_AND_REVIEW |

- Added validation for required IDs, model-output probabilities, duplicate graph
  scores, and customers that have no graph risk score.
- Added `evaluate(returns_df, orders_df)`, which treats REVIEW and HIGH as
  positive flags, uses `abuse_label` as ground truth, and returns standard
  classification metrics. ROC-AUC and PR-AUC are `None` when a dataset has only
  one label class.
- Exported `RiskEngine` from `Backend.risk` and added focused unit tests in
  `tests/test_risk.py` for scoring, thresholds/actions, validation, W&B loading,
  and evaluation.

## Inputs

Use `dataHub/transformed_data.csv` as `returns_df`; it contains the engineered
features required by `ReturnRiskModel`. Use `dataHub/data_v2.csv` as
`orders_df`; it contains the customer/entity fields required by
`GraphRiskModel`. For evaluation, pass matching data windows or matching
`split` values to both frames.

## Recorded evaluation result

The supplied evaluation used an overall-risk flag threshold of `0.30`.

| Metric | Value |
| --- | ---: |
| Accuracy | 0.8617663193 |
| Precision | 0.7466529351 |
| Recall | 0.9917920657 |
| F1 | 0.8519388954 |
| ROC-AUC | 0.9430054669 |
| PR-AUC | 0.8764630770 |
| True negatives | 846 |
| False positives | 246 |
| False negatives | 6 |
| True positives | 725 |

These values describe the combined engine at the configured REVIEW-or-HIGH
operating point. They do not establish an improvement over either baseline
model. To determine improvement or non-improvement, evaluate each baseline and
the combined engine on the identical held-out rows using the same positive
decision policy and compare the resulting metrics and error counts.
