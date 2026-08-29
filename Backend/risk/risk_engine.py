"""Decision orchestration for return and network risk predictions.

``RiskEngine`` combines the probabilities produced by the existing behavioral
and graph models.  It intentionally does not train models or construct graphs;
those responsibilities remain with ``ReturnRiskModel`` and ``GraphRiskModel``.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from graph_model import GraphRiskModel
from return_model import ReturnRiskModel


class RiskEngine:
    """Combine behavioral and network risk scores into return decisions.

    Parameters
    ----------
    return_model:
        A loaded :class:`ReturnRiskModel` used to generate behavioral risk.
    graph_model:
        A loaded :class:`GraphRiskModel` used to generate customer network risk.
    behavioral_weight, graph_weight:
        Non-negative weights that must sum to one.
    review_threshold, high_threshold:
        Overall-risk cutoffs. Scores below ``review_threshold`` are LOW, scores
        below ``high_threshold`` are REVIEW, and remaining scores are HIGH.
    """

    WEIGHT_TOLERANCE = 1e-9
    REQUIRED_RESULT_COLUMNS = (
        "return_id",
        "customer_id",
        "return_risk",
        "network_risk",
        "overall_risk",
        "risk_level",
        "recommended_action",
    )

    def __init__(
        self,
        return_model: ReturnRiskModel,
        graph_model: GraphRiskModel,
        behavioral_weight: float = 0.6,
        graph_weight: float = 0.4,
        review_threshold: float = 0.3,
        high_threshold: float = 0.7,
    ) -> None:
        self.return_model = return_model
        self.graph_model = graph_model
        self.behavioral_weight = float(behavioral_weight)
        self.graph_weight = float(graph_weight)
        self.review_threshold = float(review_threshold)
        self.high_threshold = float(high_threshold)
        self._validate_configuration()

    @classmethod
    def from_wandb(
        cls,
        project: str = "returnguard",
        entity: Optional[str] = None,
        tag: str = "production",
        return_artifact_name: str = "return-risk-model",
        graph_artifact_name: str = "graph-risk-model",
        **engine_kwargs: float,
    ) -> "RiskEngine":
        """Load both deployed models from Weights & Biases.

        The model classes read their API key from ``WANDB``. Loading dotenv here
        makes the factory usable by applications that have not done so already.
        """
        load_dotenv()
        if not os.getenv("WANDB"):
            raise EnvironmentError(
                "WANDB must be configured to load RiskEngine models from W&B."
            )

        return_model = ReturnRiskModel.load_from_wandb(
            artifact_name=return_artifact_name,
            project=project,
            entity=entity,
            tag=tag,
        )
        graph_model = GraphRiskModel.load_from_wandb(
            artifact_name=graph_artifact_name,
            project=project,
            entity=entity,
            tag=tag,
        )
        return cls(return_model, graph_model, **engine_kwargs)

    def assess(self, returns_df: pd.DataFrame, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Score returns and assign an operational risk decision.

        Each return receives its behavioral probability and the probability for
        its customer from the graph model. Every scored return must have both
        probabilities; otherwise a decision is not produced.
        """
        self._require_columns(returns_df, {"return_id", "customer_id"}, "returns_df")

        return_scores = np.asarray(self.return_model.predict_proba(returns_df))
        if return_scores.ndim != 1 or len(return_scores) != len(returns_df):
            raise ValueError(
                "ReturnRiskModel.predict_proba must return one probability per return."
            )
        self._validate_probabilities(return_scores, "return risk")

        graph_scores = self.graph_model.predict_proba(orders_df, returns_df)
        self._require_columns(
            graph_scores,
            {"customer_id", "network_risk_probability"},
            "GraphRiskModel prediction output",
        )
        if graph_scores["customer_id"].duplicated().any():
            raise ValueError("GraphRiskModel returned multiple scores for a customer.")
        self._validate_probabilities(
            graph_scores["network_risk_probability"].to_numpy(), "network risk"
        )

        result = returns_df.loc[:, ["return_id", "customer_id"]].copy()
        result["return_risk"] = return_scores.astype(float)
        result = result.merge(
            graph_scores.loc[:, ["customer_id", "network_risk_probability"]],
            on="customer_id",
            how="left",
            validate="many_to_one",
            sort=False,
        )
        missing_customers = result.loc[
            result["network_risk_probability"].isna(), "customer_id"
        ].drop_duplicates()
        if not missing_customers.empty:
            customers = ", ".join(map(str, missing_customers.tolist()))
            raise ValueError(f"No network risk score available for customer(s): {customers}")

        result = result.rename(columns={"network_risk_probability": "network_risk"})
        result["overall_risk"] = (
            self.behavioral_weight * result["return_risk"]
            + self.graph_weight * result["network_risk"]
        )
        result["risk_level"] = np.select(
            [
                result["overall_risk"] < self.review_threshold,
                result["overall_risk"] < self.high_threshold,
            ],
            ["LOW", "REVIEW"],
            default="HIGH",
        )
        result["recommended_action"] = result["risk_level"].map(
            {
                "LOW": "APPROVE",
                "REVIEW": "MANUAL_REVIEW",
                "HIGH": "HOLD_AND_REVIEW",
            }
        )
        return result.loc[:, self.REQUIRED_RESULT_COLUMNS]

    def evaluate(
        self, returns_df: pd.DataFrame, orders_df: pd.DataFrame
    ) -> dict[str, float | int | None]:
        """Evaluate the combined score against ``returns_df.abuse_label``.

        REVIEW and HIGH are treated as positive operational flags. AUC metrics
        are returned as ``None`` when the supplied labels contain only one class.
        """
        self._require_columns(returns_df, {"abuse_label"}, "returns_df")
        y_true = returns_df["abuse_label"]
        if y_true.isna().any() or not y_true.isin([0, 1, False, True]).all():
            raise ValueError("returns_df.abuse_label must contain only binary 0/1 labels.")
        y_true_array = y_true.astype(int).to_numpy()

        assessed = self.assess(returns_df, orders_df)
        y_pred = (assessed["overall_risk"].to_numpy() >= self.review_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()

        result: dict[str, float | int | None] = {
            "evaluation_threshold": self.review_threshold,
            "accuracy": float(accuracy_score(y_true_array, y_pred)),
            "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
            "roc_auc": None,
            "pr_auc": None,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }
        if np.unique(y_true_array).size == 2:
            overall_risk = assessed["overall_risk"].to_numpy()
            result["roc_auc"] = float(roc_auc_score(y_true_array, overall_risk))
            result["pr_auc"] = float(average_precision_score(y_true_array, overall_risk))
        return result

    def _validate_configuration(self) -> None:
        """Validate weights and risk thresholds at construction time."""
        weights = (self.behavioral_weight, self.graph_weight)
        if not all(math.isfinite(weight) and weight >= 0 for weight in weights):
            raise ValueError("Risk weights must be finite, non-negative numbers.")
        if not math.isclose(sum(weights), 1.0, abs_tol=self.WEIGHT_TOLERANCE):
            raise ValueError("behavioral_weight and graph_weight must sum to 1.")

        thresholds = (self.review_threshold, self.high_threshold)
        if not all(math.isfinite(threshold) and 0 <= threshold <= 1 for threshold in thresholds):
            raise ValueError("Risk thresholds must be finite numbers between 0 and 1.")
        if self.review_threshold >= self.high_threshold:
            raise ValueError("review_threshold must be less than high_threshold.")

    @staticmethod
    def _require_columns(
        dataframe: pd.DataFrame, required: set[str], name: str
    ) -> None:
        missing = sorted(required.difference(dataframe.columns))
        if missing:
            raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")

    @staticmethod
    def _validate_probabilities(values: np.ndarray, name: str) -> None:
        """Reject invalid model outputs before they can drive a decision."""
        try:
            numeric_values = values.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name.capitalize()} predictions must be numeric.") from error
        if not np.isfinite(numeric_values).all() or not np.all(
            (numeric_values >= 0) & (numeric_values <= 1)
        ):
            raise ValueError(f"{name.capitalize()} predictions must be finite probabilities in [0, 1].")


if __name__ == '__main__':

    import pandas as pd
    from dotenv import load_dotenv
    import os
    load_dotenv()
    PATH = os.getenv('DATA_STORAGE_PATH')

    returns_path = os.path.join(PATH,'transformed_data.csv') #type: ignore
    orders_path = os.path.join(PATH,'data_v2.csv') #type: ignore
    returns_df = pd.read_csv(returns_path)
    orders_df = pd.read_csv(orders_path)

    engine = RiskEngine.from_wandb()
    print("Models Loaded Successfully")
    result = engine.evaluate(returns_df=returns_df,orders_df=orders_df)
    print(result)
