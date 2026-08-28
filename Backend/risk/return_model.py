"""
ReturnGuard - Behavioral Abuse Detection Model
Trains a LogisticRegression on pre-engineered features from transformed_data.csv.
Logs metrics and saves the model artifact to Weights & Biases.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Columns that must never be used as features
_LEAKY_COLS = {"abuse_type", "abuse_label", "split"}

# ID / metadata columns that carry no predictive signal
_ID_COLS = {
    "return_id",
    "order_id",
    "customer_id",
    "device_id",
    "ip_id",
    "address_id",
    "payment_id",
}

# Raw date columns (already decomposed into engineered features)
_DATE_COLS = {
    "return_date",
    "purchase_date",
    "account_creation_date",
    "delivery_date",
}

# Constant columns (e.g. country == "India" for every row)
_CONST_COLS = {"country"}


def _feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the list of columns safe to use as model features."""
    exclude = _LEAKY_COLS | _ID_COLS | _DATE_COLS | _CONST_COLS
    return [c for c in df.columns if c not in exclude]


def _prepare_splits(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the transformed dataframe into train / validation / test using the
    ``split`` column produced by ``build_data.py``.

    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    def _xy(sub: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return sub[feature_cols], sub["abuse_label"]

    X_train, y_train = _xy(train_df)
    X_val, y_val = _xy(val_df)
    X_test, y_test = _xy(test_df)

    return X_train, y_train, X_val, y_val, X_test, y_test #type: ignore


class ReturnRiskModel:
    """Logistic-Regression wrapper for return-abuse detection.

    The class is self-contained: it handles preprocessing (imputation,
    encoding, scaling), training, evaluation, W&B logging, and artifact
    persistence.

    Parameters
    ----------
    wandb_project : str
        W&B project name. Defaults to ``"returnguard"``.
    wandb_entity : str | None
        Optional W&B entity (team / user).
    C : float
        Inverse regularisation strength for LogisticRegression.
    max_iter : int
        Maximum solver iterations.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        wandb_project: str = "returnguard",
        wandb_entity: Optional[str] = None,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state

        # Fitted state (populated by fit / load)
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Preprocessing helpers
    # ------------------------------------------------------------------ #
    def _detect_column_types(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Identify categorical and numeric columns among the feature set."""
        categorical_cols: List[str] = []
        numeric_cols: List[str] = []
        for col in self.feature_cols:
            if col not in df.columns:
                continue
            if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
                categorical_cols.append(col)
            else:
                numeric_cols.append(col)
        return categorical_cols, numeric_cols

    def _fit_preprocessors(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit encoders + scaler on training data and return transformed X."""
        self.categorical_cols, self.numeric_cols = self._detect_column_types(X)
        X = X.copy()

        # Label-encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            vals = X[col].astype(str).fillna("__MISSING__")
            le.fit(vals)
            self.label_encoders[col] = le
            X[col] = le.transform(vals) #type: ignore

        # Impute numeric NaNs with median, then scale
        if self.numeric_cols:
            medians = {}
            for col in self.numeric_cols:
                med = X[col].median()
                medians[col] = med if not np.isnan(med) else 0.0
                X[col] = X[col].fillna(medians[col])
            self.metadata["medians"] = medians

        self.scaler = StandardScaler()
        X[self.numeric_cols] = self.scaler.fit_transform(X[self.numeric_cols])
        return X

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted encoders + scaler to new data."""
        X = X.copy()

        # Label-encode categoricals (handle unseen labels gracefully)
        for col in self.categorical_cols:
            le = self.label_encoders.get(col)
            if col not in X.columns or le is None:
                X[col] = 0
                continue
            vals = X[col].astype(str).fillna("__MISSING__")
            known = set(le.classes_)
            if "__MISSING__" not in known:
                vals = vals.apply(lambda v: v if v in known else le.classes_[0]) #type: ignore
            X[col] = le.transform(vals) #type: ignore

        # Ensure all expected columns exist
        for col in self.numeric_cols:
            if col not in X.columns:
                X[col] = 0.0

        # Impute numeric NaNs
        medians = self.metadata.get("medians", {})
        for col in self.numeric_cols:
            X[col] = X[col].fillna(medians.get(col, 0.0))

        # Scale
        if self.scaler is not None and self.numeric_cols:
            X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])

        return X

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_df: pd.DataFrame,
        validation_df: Optional[pd.DataFrame] = None,
        run_name: Optional[str] = None,
    ) -> "ReturnRiskModel":
        """Train the model and log to W&B.

        Parameters
        ----------
        train_df : DataFrame
            Training split (must contain ``abuse_label`` and ``split``).
        validation_df : DataFrame | None
            Optional held-out validation split for early stopping / logging.
        run_name : str | None
            Custom W&B run name.
        """
        self.feature_cols = _feature_columns(train_df)

        X_train, y_train, X_val, y_val, X_test, y_test = _prepare_splits(
            train_df if validation_df is None else pd.concat([train_df, validation_df]),
            self.feature_cols,
        )

        # If caller passed an explicit validation_df that isn't in train_df,
        # use it. Otherwise fall back to the split column.
        if validation_df is not None and len(validation_df) > 0:
            X_val, y_val = validation_df[self.feature_cols], validation_df["abuse_label"]

        X_train = self._fit_preprocessors(X_train)
        X_val = self._transform(X_val) if len(X_val) > 0 else X_val

        # Train Logistic Regression
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        )
        self.model.fit(X_train, y_train)

        # ------------------------------------------------------------------ #
        # W&B logging
        # ------------------------------------------------------------------ #
        wandb_api_key = os.getenv("WANDB")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name,
            reinit="return_previous",
            config={
                "model": "LogisticRegression",
                "C": self.C,
                "max_iter": self.max_iter,
                "class_weight": "balanced",
                "solver": "lbfgs",
                "random_state": self.random_state,
                "n_features": len(self.feature_cols),
                "n_train": len(X_train),
                "n_val": len(X_val),
                "abuse_rate_train": float(y_train.mean()),
            },
        )

        # -- evaluation on train --
        train_metrics = self._compute_metrics(y_train, self.model.predict(X_train), self.model.predict_proba(X_train)[:, 1])
        for k, v in train_metrics.items():
            wandb.log({f"train/{k}": v})

        # -- evaluation on validation --
        if len(X_val) > 0:
            val_pred = self.model.predict(X_val)
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_metrics = self._compute_metrics(y_val, val_pred, val_proba)
            for k, v in val_metrics.items():
                wandb.log({f"val/{k}": v})

        # -- evaluation on test (if split exists) --
        if len(X_test) > 0:
            X_test_p = self._transform(X_test)
            test_pred = self.model.predict(X_test_p)
            test_proba = self.model.predict_proba(X_test_p)[:, 1]
            test_metrics = self._compute_metrics(y_test, test_pred, test_proba)
            for k, v in test_metrics.items():
                wandb.log({f"test/{k}": v})

        # -- feature coefficients as a table --
        coef_table = wandb.Table(columns=["feature", "coefficient"])
        for feat, coef in zip(self.feature_cols, self.model.coef_[0]):
            coef_table.add_data(feat, float(coef))
        wandb.log({"feature_coefficients": coef_table})

        # -- confusion matrices --
        for split_name, (y_true, y_pred) in {
            "train": (y_train, self.model.predict(X_train)),
            "val": (y_val, self.model.predict(X_val)) if len(X_val) > 0 else (None, None),
            "test": (y_test, self.model.predict(self._transform(X_test))) if len(X_test) > 0 else (None, None),
        }.items():
            if y_true is not None:
                cm = confusion_matrix(y_true, y_pred)
                wandb.log({f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true.tolist(),
                    preds=y_pred.tolist(),
                    class_names=["legitimate", "abusive"],
                )})

        # -- save model as W&B artifact --
        self.metadata.update({
            "feature_cols": self.feature_cols,
            "categorical_cols": self.categorical_cols,
            "numeric_cols": self.numeric_cols,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics if len(X_val) > 0 else {},
        })

        artifact = wandb.Artifact(
            name="return-risk-model",
            type="model",
            metadata=self.metadata,
        )

        # Save model + preprocessor bundle to a temp dir
        bundle_path = Path(wandb.config.tmp_dir) / "return_risk_model.joblib" if hasattr(wandb.config, "tmp_dir") else Path("return_risk_model.joblib")
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_cols": self.feature_cols,
                "categorical_cols": self.categorical_cols,
                "numeric_cols": self.numeric_cols,
                "metadata": self.metadata,
            },
            bundle_path,
        )
        artifact.add_file(str(bundle_path))
        run.log_artifact(artifact, name="return-risk-model", aliases=["latest", "production"])

        wandb.finish()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return probability of abuse (class 1) for each row."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")
        X = df.copy()
        # Ensure all expected feature columns exist (fill missing with 0)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]
        X = self._transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions at the given threshold."""
        proba = self.predict_proba(df)
        return (proba >= threshold).astype(int)

    def evaluate(
        self, df: pd.DataFrame, threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Return a dict of evaluation metrics."""
        y_true = df["abuse_label"]
        y_pred = self.predict(df, threshold)
        y_proba = self.predict_proba(df)
        return self._compute_metrics(y_true, y_pred, y_proba)

    @staticmethod
    def _compute_metrics(
        y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Aggregate classification metrics."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
            "pr_auc": float(average_precision_score(y_true, y_proba)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """Persist the full bundle (model + preprocessor) to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "scaler": self.scaler,
                "label_encoders": self.label_encoders,
                "feature_cols": self.feature_cols,
                "categorical_cols": self.categorical_cols,
                "numeric_cols": self.numeric_cols,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ReturnRiskModel":
        """Load a previously saved bundle from disk."""
        bundle = joblib.load(path)
        instance = cls()
        instance.model = bundle["model"]
        instance.scaler = bundle["scaler"]
        instance.label_encoders = bundle["label_encoders"]
        instance.feature_cols = bundle["feature_cols"]
        instance.categorical_cols = bundle["categorical_cols"]
        instance.numeric_cols = bundle["numeric_cols"]
        instance.metadata = bundle["metadata"]
        return instance

    @classmethod
    def load_from_wandb(
        cls,
        artifact_name: str = "return-risk-model",
        project: str = "returnguard",
        entity: Optional[str] = None,
        tag: str = "latest",
    ) -> "ReturnRiskModel":
        """Download and load the model artifact from W&B.

        Parameters
        ----------
        artifact_name : str
            Name of the W&B artifact (default ``"return-risk-model"``).
        project : str
            W&B project.
        entity : str | None
            Optional W&B entity.
        tag : str
            Alias to fetch (e.g. ``"latest"``, ``"production"``).
        """
        wandb_api_key = os.getenv("WANDB")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        api = wandb.Api()
        artifact_ref = api.artifact(
            f"{entity + '/' if entity else ''}{project}/{artifact_name}:{tag}"
        )
        artifact_dir = artifact_ref.download()
        # Find the .joblib file inside the downloaded directory
        joblib_files = list(Path(artifact_dir).glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(
                f"No .joblib file found in downloaded artifact at {artifact_dir}"
            )
        return cls.load(joblib_files[0])


if __name__ == '__main__':
    import pandas as pd
    from dotenv import load_dotenv
    import os
    load_dotenv()

    ver = 2
    PATH = os.getenv('DATA_STORAGE_PATH')
    req_path = os.path.join(PATH,'transformed_data.csv') #type: ignore
    df = pd.read_csv(req_path)
    model = ReturnRiskModel(wandb_project="returnguard")
    model.fit(df)
