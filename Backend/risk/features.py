"""
Feature Engineering Script for Return Abuse Detection
Transforms raw dataset into features ready for model training
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Known categories from build_data.py for stable dummy creation
KNOWN_CATEGORIES = [
    "electronics",
    "clothing",
    "books",
    "home_appliances",
    "beauty",
    "sports",
    "jewelry",
    "grocery",
]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Comprehensive feature engineering pipeline for return abuse detection.
    sklearn compatible: fit/transform/prepare_for_modeling.
    """

    def __init__(self, target_col="abuse_label", random_state=42):
        self.target_col = target_col
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.selector = None
        self.encoders = {}
        self.dummy_columns = None
        self.selected_features = None
        self.numeric_cols_ = None
        self.feature_importance = None
        self.is_fitted = False
        # thresholds learned on train to avoid leakage
        self._high_order_threshold = None
        self._fast_purchaser_threshold = 7  # days, fixed to avoid quantile leakage

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main method to engineer all features (unsupervised, no target)."""
        df = df.copy()

        df = self._create_time_features(df)
        df = self._create_behavior_features(df)
        df = self._create_transaction_features(df)
        df = self._create_risk_scores(df)
        df = self._create_aggregated_features(df)
        df = self._create_interaction_features(df)
        df = self._create_normalization_features(df)
        df = self._create_temporal_patterns(df)

        return df

    def fit(self, df: pd.DataFrame, y=None):
        """Fit encoders / imputer / scaler / selector on engineered data."""
        # engineer first (unsupervised)
        engineered = self.engineer_features(df)
        # delegate to prepare logic to learn params
        # we call internal _fit_transform without splitting, to learn global transformers
        # For sklearn pipeline, y may be passed separately
        if y is None and self.target_col in engineered.columns:
            y = engineered[self.target_col]
            X = engineered.drop(columns=[self.target_col])
        else:
            X = engineered.drop(columns=[self.target_col], errors="ignore")

        # drop split col if present - never a feature
        if "split" in X.columns:
            X = X.drop(columns=["split"])
        # also drop abuse_type if present
        for col in ["abuse_type"]:
            if col in X.columns:
                X = X.drop(columns=[col])

        X = self._fit_encoders_imputer_selector_scaler(X, y)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Transform new data using fitted transformers."""
        if not self.is_fitted:
            warnings.warn("FeatureEngineer.transform called before fit; fitting on given data.")
            return self.fit_transform(df)

        # engineer
        engineered = self.engineer_features(df)
        # remove target/split if present
        for col in [self.target_col, "split", "abuse_type"]:
            if col in engineered.columns:
                engineered = engineered.drop(columns=[col])

        X = engineered

        # align dummy columns / categorical encodings
        X = self._apply_encoders(X)
        # ensure all selected_features present
        X = self._apply_imputer(X)
        # select
        if self.selected_features is not None:
            # add missing cols as 0
            for col in self.selected_features:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.selected_features]
        # scale
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
            return X_scaled
        return X

    def fit_transform(self, df: pd.DataFrame, y=None, **fit_params):
        return self.fit(df, y).transform(df)

    # ------------------------------------------------------------------ #
    # Time features
    # ------------------------------------------------------------------ #
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        for col in ["purchase_date", "return_date", "delivery_date", "account_creation_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        if "return_date" in df.columns and not df["return_date"].isna().all():
            df["return_day_of_week"] = df["return_date"].dt.dayofweek
            df["return_day_of_month"] = df["return_date"].dt.day
            df["return_month"] = df["return_date"].dt.month
            df["return_quarter"] = df["return_date"].dt.quarter
            df["return_weekend"] = df["return_date"].dt.dayofweek.isin([5, 6]).astype(int)
            # deterministic hour extraction, not random
            df["return_hour"] = df["return_date"].dt.hour.fillna(-1).astype(int)

        if "purchase_date" in df.columns and not df["purchase_date"].isna().all():
            df["purchase_day_of_week"] = df["purchase_date"].dt.dayofweek
            df["purchase_month"] = df["purchase_date"].dt.month
            df["purchase_weekend"] = df["purchase_date"].dt.dayofweek.isin([5, 6]).astype(int)

        if "return_date" in df.columns and "purchase_date" in df.columns:
            df["days_since_purchase"] = (df["return_date"] - df["purchase_date"]).dt.days
            df["days_since_purchase"] = df["days_since_purchase"].fillna(0).clip(lower=0)

        if "return_date" in df.columns and "delivery_date" in df.columns:
            df["days_since_delivery"] = (df["return_date"] - df["delivery_date"]).dt.days
            df["days_since_delivery"] = df["days_since_delivery"].fillna(0).clip(lower=0)

        if "account_creation_date" in df.columns and "return_date" in df.columns:
            df["account_age_days"] = (df["return_date"] - df["account_creation_date"]).dt.days.fillna(0).clip(lower=0)
            df["account_age_months"] = df["account_age_days"] / 30.0
            df["account_age_years"] = df["account_age_days"] / 365.0

        if "customer_tenure_days" in df.columns and "account_age_days" in df.columns:
            df["customer_tenure_ratio"] = df["account_age_days"] / df["customer_tenure_days"].clip(lower=1)

        if "return_month" in df.columns:
            df["is_holiday_season"] = df["return_month"].isin([11, 12]).astype(int)
            df["is_summer"] = df["return_month"].isin([6, 7, 8]).astype(int)

        return df

    # ------------------------------------------------------------------ #
    # Behavior
    # ------------------------------------------------------------------ #
    def _create_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer behavior features"""
        if "historical_return_rate" in df.columns:
            df["return_rate_category"] = pd.cut(
                df["historical_return_rate"],
                bins=[-np.inf, 0.05, 0.15, 0.30, 0.50, 0.80, np.inf],
                labels=["very_low", "low", "moderate", "high", "very_high", "extreme"],
            )
            if "returns_last_7_days" in df.columns:
                df["return_velocity_7d"] = df["returns_last_7_days"] / 7.0
            if "returns_last_30_days" in df.columns:
                df["return_velocity_30d"] = df["returns_last_30_days"] / 30.0

        if "number_of_previous_orders" in df.columns:
            if "customer_tenure_days" in df.columns:
                df["order_frequency"] = df["number_of_previous_orders"] / df["customer_tenure_days"].clip(lower=1)
            else:
                df["order_frequency"] = df["number_of_previous_orders"] / 30.0
            # Fixed threshold to avoid leakage (previously used global quantile)
            df["high_order_frequency"] = (df["order_frequency"] > 0.1).astype(int)

        if "number_of_previous_returns" in df.columns and "number_of_previous_orders" in df.columns:
            df["return_to_order_ratio"] = df["number_of_previous_returns"] / df["number_of_previous_orders"].clip(lower=1)
        else:
            df["return_to_order_ratio"] = 0

        if "historical_refund_amount" in df.columns and "order_amount" in df.columns:
            df["refund_to_order_ratio"] = df["historical_refund_amount"] / df["order_amount"].clip(lower=1)
        else:
            df["refund_to_order_ratio"] = 0

        if "historical_refund_amount" in df.columns and "refund_amount" in df.columns:
            df["refund_to_return_ratio"] = df["historical_refund_amount"] / df["refund_amount"].clip(lower=1)
        else:
            df["refund_to_return_ratio"] = 0

        # Customer consistency - per customer, not global rolling
        if "historical_return_rate" in df.columns and "customer_id" in df.columns:
            df["return_consistency"] = (
                df.groupby("customer_id")["historical_return_rate"].transform(lambda x: x.std() if len(x) > 1 else 0).fillna(0)
            )
        elif "historical_return_rate" in df.columns:
            df["return_consistency"] = 0

        if "number_of_previous_orders" in df.columns and "customer_id" in df.columns:
            df["order_consistency"] = (
                df.groupby("customer_id")["number_of_previous_orders"].transform(lambda x: x.std() if len(x) > 1 else 0).fillna(0)
            )
        elif "number_of_previous_orders" in df.columns:
            df["order_consistency"] = 0

        return df

    # ------------------------------------------------------------------ #
    # Transaction
    # ------------------------------------------------------------------ #
    def _create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transaction-based features"""
        if "product_category" in df.columns:
            # Stable dummy creation for known categories (avoids schema drift)
            for cat in KNOWN_CATEGORIES:
                df[f"cat_{cat}"] = (df["product_category"] == cat).astype(int)

            category_risk = {
                "electronics": 0.7,
                "clothing": 0.3,
                "books": 0.1,
                "home_appliances": 0.6,
                "beauty": 0.4,
                "sports": 0.3,
                "jewelry": 0.8,
                "grocery": 0.1,
            }
            df["category_risk_score"] = df["product_category"].map(category_risk).fillna(0.5)

        if "payment_method" in df.columns:
            payment_risk = {
                "credit_card": 0.3,
                "debit_card": 0.2,
                "upi": 0.4,
                "net_banking": 0.3,
                "cash_on_delivery": 0.7,
            }
            df["payment_risk_score"] = df["payment_method"].map(payment_risk).fillna(0.5)

        if "order_amount" in df.columns:
            df["log_order_amount"] = np.log1p(df["order_amount"].clip(lower=0))

            df["order_amount_bin"] = pd.cut(
                df["order_amount"],
                bins=[0, 500, 2000, 5000, 10000, 50000, np.inf],
                labels=["micro", "small", "medium", "large", "xlarge", "xxlarge"],
            )

            if "product_category" in df.columns:
                # robust per-category zscore
                def _zscore(x):
                    std = x.std()
                    if pd.isna(std) or std == 0:
                        return pd.Series(0, index=x.index)
                    return (x - x.mean()) / std

                df["amount_relative_to_category"] = (
                    df.groupby("product_category")["order_amount"].transform(_zscore).fillna(0).replace([np.inf, -np.inf], 0)
                )
            else:
                df["amount_relative_to_category"] = 0

            if "refund_amount" in df.columns:
                df["refund_ratio"] = df["refund_amount"] / df["order_amount"].clip(lower=1)

        if "discount_percentage" in df.columns:
            df["high_discount"] = (df["discount_percentage"] > 20).astype(int)
            if "order_amount" in df.columns:
                df["discount_impact"] = df["discount_percentage"] * df["order_amount"] / 100.0
            else:
                df["discount_impact"] = df["discount_percentage"] * 0

        return df

    # ------------------------------------------------------------------ #
    # Risk scores
    # ------------------------------------------------------------------ #
    def _create_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk scores"""
        risk_factors = []

        if "historical_return_rate" in df.columns:
            risk_factors.append(df["historical_return_rate"] / 0.1)

        if "number_of_previous_returns" in df.columns and "number_of_previous_orders" in df.columns:
            risk_factors.append(df["number_of_previous_returns"] / df["number_of_previous_orders"].clip(lower=1))

        if "order_amount" in df.columns:
            risk_factors.append(df["order_amount"] / 10000.0)

        if "days_to_return" in df.columns:
            risk_factors.append(1 / ((df["days_to_return"] / 7).clip(lower=0.1) + 1))

        if risk_factors:
            risk_matrix = pd.concat(risk_factors, axis=1)
            df["composite_risk_score"] = risk_matrix.mean(axis=1)
            min_val = df["composite_risk_score"].min()
            max_val = df["composite_risk_score"].max()
            denom = max_val - min_val
            if denom == 0 or pd.isna(denom):
                df["risk_score_normalized"] = 0.0
            else:
                df["risk_score_normalized"] = (df["composite_risk_score"] - min_val) / (denom + 1e-6)
        else:
            df["composite_risk_score"] = 0
            df["risk_score_normalized"] = 0

        return_reason_risk = {
            "defective": 0.2,
            "not_as_described": 0.3,
            "wrong_item": 0.4,
            "changed_mind": 0.6,
            "quality_issue": 0.3,
            "size_issue": 0.2,
            "late_delivery": 0.1,
            "damaged_delivery": 0.1,
        }
        if "return_reason" in df.columns:
            df["return_reason_risk"] = df["return_reason"].map(return_reason_risk).fillna(0.3)

        return df

    # ------------------------------------------------------------------ #
    # Aggregated
    # ------------------------------------------------------------------ #
    def _create_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features"""
        if "customer_id" in df.columns:
            agg_dict = {}
            if "order_amount" in df.columns:
                agg_dict["order_amount"] = ["mean", "std", "max", "min"]
            if "historical_return_rate" in df.columns:
                agg_dict["historical_return_rate"] = "mean"
            if "days_to_return" in df.columns:
                agg_dict["days_to_return"] = ["mean", "std"]

            if agg_dict:
                customer_agg = df.groupby("customer_id").agg(agg_dict).reset_index()
                new_cols = ["customer_id"]
                for col in agg_dict:
                    if isinstance(agg_dict[col], list):
                        for agg_func in agg_dict[col]:
                            new_cols.append(f"cust_{col}_{agg_func}")
                    else:
                        new_cols.append(f"cust_{col}")
                customer_agg.columns = new_cols
                df = df.merge(customer_agg, on="customer_id", how="left")

        # Daily aggregations - use floor date, not exact timestamp
        if "return_date" in df.columns:
            try:
                return_date_floor = pd.to_datetime(df["return_date"]).dt.floor("D")
                # avoid overwriting original return_date
                tmp = pd.DataFrame({"_return_date_floor": return_date_floor, "order_amount": df.get("order_amount", 0), "historical_return_rate": df.get("historical_return_rate", 0)})
                daily_agg = tmp.groupby("_return_date_floor").agg({"order_amount": "mean", "historical_return_rate": "mean"}).reset_index()
                daily_agg.columns = ["_return_date_floor", "daily_avg_amount", "daily_avg_return_rate"]
                df["_return_date_floor"] = return_date_floor
                df = df.merge(daily_agg, on="_return_date_floor", how="left")
                df = df.drop(columns=["_return_date_floor"])
            except Exception as e:
                warnings.warn(f"Could not create daily aggregations: {e}")

        # Ensure return_week still created without affecting merge
        if "return_date" in df.columns:
            try:
                df["return_week"] = pd.to_datetime(df["return_date"]).dt.isocalendar().week.astype("Int64").fillna(0).astype(int)
            except Exception:
                df["return_week"] = 0

        return df

    # ------------------------------------------------------------------ #
    # Interactions
    # ------------------------------------------------------------------ #
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features (no target leakage)"""
        if "days_to_return" in df.columns and "order_amount" in df.columns:
            df["days_amount_interaction"] = df["days_to_return"] * df["order_amount"]
            df["days_amount_ratio"] = df["days_to_return"] / (df["order_amount"] / 1000).clip(lower=0.001)

        if "historical_return_rate" in df.columns and "order_amount" in df.columns:
            df["return_rate_amount"] = df["historical_return_rate"] * df["order_amount"]

        if "number_of_previous_returns" in df.columns and "discount_percentage" in df.columns:
            df["returns_discount_interaction"] = df["number_of_previous_returns"] * df["discount_percentage"]

        if "age" in df.columns and "historical_return_rate" in df.columns:
            df["age_return_interaction"] = df["age"] * df["historical_return_rate"]

        if "customer_tenure_days" in df.columns and "number_of_previous_returns" in df.columns:
            df["tenure_returns_ratio"] = df["number_of_previous_returns"] / (df["customer_tenure_days"] / 30).clip(lower=0.01)

        # Removed target-leakage features: city_category_risk, device_category_risk
        # Keep payment_amount_risk as unsupervised per-payment zscore
        if "payment_method" in df.columns and "order_amount" in df.columns:
            try:
                def _zscore(x):
                    std = x.std()
                    if pd.isna(std) or std == 0:
                        return pd.Series(0, index=x.index)
                    return (x - x.mean()) / std

                df["payment_amount_risk"] = df.groupby("payment_method")["order_amount"].transform(_zscore).fillna(0).replace([np.inf, -np.inf], 0)
            except Exception:
                df["payment_amount_risk"] = 0

        return df

    # ------------------------------------------------------------------ #
    # Normalization
    # ------------------------------------------------------------------ #
    def _create_normalization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create normalized and scaled features"""
        for col in ["order_amount", "refund_amount", "historical_refund_amount"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                df[f"{col}_rank"] = df[col].rank(pct=True)
                mean = df[col].mean()
                std = df[col].std()
                if pd.isna(std) or std == 0:
                    df[f"{col}_zscore"] = 0.0
                else:
                    df[f"{col}_zscore"] = (df[col] - mean) / std
                    df[f"{col}_zscore"] = df[f"{col}_zscore"].replace([np.inf, -np.inf], 0).fillna(0)

        for col in ["number_of_previous_orders", "number_of_previous_returns", "orders_last_7_days", "orders_last_30_days", "customer_tenure_days"]:
            if col in df.columns:
                df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))
                df[f"{col}_rank"] = df[col].rank(pct=True)

        for col in ["historical_return_rate", "return_rate_30d", "return_rate_90d"]:
            if col in df.columns:
                rate = df[col].clip(lower=0.001, upper=0.999)
                df[f"{col}_logit"] = np.log(rate / (1 - rate)).replace([np.inf, -np.inf], 0).fillna(0) #type: ignore

        # Percentile ranks - avoid duplicate if _rank already exists
        for col in ["order_amount", "historical_return_rate", "age", "customer_tenure_days"]:
            if col in df.columns:
                perc_col = f"{col}_percentile"
                rank_col = f"{col}_rank"
                # if rank already computed for this col, reuse
                if rank_col in df.columns:
                    df[perc_col] = df[rank_col]
                else:
                    df[perc_col] = df[col].rank(pct=True)

        return df

    # ------------------------------------------------------------------ #
    # Temporal patterns
    # ------------------------------------------------------------------ #
    def _create_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal pattern features"""
        if "return_date" in df.columns and "customer_id" in df.columns:
            try:
                return_dt = pd.to_datetime(df["return_date"])
                df["return_trend"] = df.groupby("customer_id")[return_dt.name if return_dt.name else "return_date"].rank(pct=True) if return_dt.name in df.columns else df.groupby("customer_id")["return_date"].rank(pct=True) #type: ignore
                # fallback if above fails, use generic rank
                if df["return_trend"].isna().all():
                    df["return_trend"] = df.groupby("customer_id")["return_date"].rank(pct=True)
            except Exception:
                # simple per-customer rank pct
                try:
                    df["return_date_dt"] = pd.to_datetime(df["return_date"])
                    df["return_trend"] = df.groupby("customer_id")["return_date_dt"].rank(pct=True)
                    df = df.drop(columns=["return_date_dt"])
                except Exception:
                    df["return_trend"] = 0

        if "purchase_date" in df.columns and "customer_id" in df.columns:
            try:
                purchase_dt = pd.to_datetime(df["purchase_date"])
                df["purchase_velocity"] = df.groupby("customer_id")[purchase_dt.name if purchase_dt.name else "purchase_date"].diff().dt.days.fillna(0) if False else df.groupby("customer_id")["purchase_date"].transform(lambda x: pd.to_datetime(x).diff().dt.days.fillna(0))
                # vectorized alternative
                df["purchase_velocity"] = df.groupby("customer_id")["purchase_date"].transform(lambda x: pd.to_datetime(x).diff().dt.days.fillna(0))
                df["purchase_velocity_rank"] = df["purchase_velocity"].rank(pct=True)
                # fixed threshold instead of global quantile (leakage)
                df["fast_purchaser"] = (df["purchase_velocity"] < self._fast_purchaser_threshold).astype(int)
            except Exception:
                df["purchase_velocity"] = 0
                df["purchase_velocity_rank"] = 0
                df["fast_purchaser"] = 0

        if "return_reason" in df.columns and "customer_id" in df.columns:
            try:
                df["return_reason_consistency"] = df.groupby("customer_id")["return_reason"].transform(
                    lambda x: x.value_counts().get(x.iloc[0] if len(x) > 0 else "unknown", 1) / len(x) if len(x) > 0 else 0
                )
            except Exception:
                df["return_reason_consistency"] = 0

        return df

    # ------------------------------------------------------------------ #
    # Internal helpers for fit/transform
    # ------------------------------------------------------------------ #
    def _fit_encoders_imputer_selector_scaler(self, X: pd.DataFrame, y):
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

        # Remove datetime cols from features (already extracted)
        X = X.drop(columns=datetime_cols, errors="ignore")
        for col in datetime_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)
            if col in categorical_cols:
                categorical_cols.remove(col)

        # Encode categoricals - fit only on train data passed to this method
        # Caller ensures X is train-split
        for col in categorical_cols:
            if col in X.columns and X[col].nunique() < 50:
                le = LabelEncoder()
                # handle Categorical dtype that may not contain "unknown"
                vals = X[col].astype(object).where(X[col].notna(), "unknown").astype(str)
                vals = vals.replace("nan", "unknown")
                le.fit(vals)
                self.encoders[col] = le
                X[col + "_encoded"] = le.transform(vals) #type: ignore
                numeric_cols.append(col + "_encoded")
            # high cardinality cols are dropped (not added)

        # Store dummy columns for stable schema - not needed with LabelEncoder path
        # but keep for backward compat with cat_* dummies

        # Imputer fit on train numeric
        self.imputer = SimpleImputer(strategy="median")
        # Ensure numeric_cols are present in X
        numeric_cols = [c for c in numeric_cols if c in X.columns]
        self.numeric_cols_ = numeric_cols

        if numeric_cols:
            # fit imputer on numeric subset
            self.imputer.fit(X[numeric_cols])
            X_imputed = pd.DataFrame(self.imputer.transform(X[numeric_cols]), columns=numeric_cols, index=X.index) #type: ignore
            X[numeric_cols] = X_imputed

            # Feature selection fit on train
            if y is not None and len(numeric_cols) > 0:
                k = min(50, len(numeric_cols))
                self.selector = SelectKBest(score_func=f_classif, k=k)
                # y may be DataFrame/Series with non-numeric; ensure aligned
                y_aligned = y.loc[X.index] if hasattr(y, "loc") else y
                self.selector.fit(X[numeric_cols], y_aligned)
                selected_indices = self.selector.get_support(indices=True)
                self.selected_features = [numeric_cols[i] for i in selected_indices]
                self.feature_importance = pd.DataFrame({"feature": numeric_cols, "score": self.selector.scores_}).sort_values("score", ascending=False)
            else:
                self.selected_features = numeric_cols
                self.selector = None
                self.feature_importance = None

            # Scaler fit on selected features
            if self.selected_features:
                X_selected = X[self.selected_features]
                self.scaler = StandardScaler()
                self.scaler.fit(X_selected)
        else:
            self.selected_features = []
            self.feature_importance = None

        return X

    def _apply_encoders(self, X: pd.DataFrame) -> pd.DataFrame:
        # drop datetime
        datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        X = X.drop(columns=datetime_cols, errors="ignore")

        for col, le in self.encoders.items():
            if col in X.columns:
                vals = X[col].astype(object).where(X[col].notna(), "unknown").astype(str)
                vals = vals.replace("nan", "unknown")
                # handle unseen labels -> map to 'unknown' if present else most frequent
                known = set(le.classes_)
                # replace unseen with 'unknown' if encoder knows it, else first class
                def map_val(v):
                    return v if v in known else ("unknown" if "unknown" in known else le.classes_[0])

                vals = vals.apply(map_val)
                X[col + "_encoded"] = le.transform(vals)
            else:
                # column missing in transform data, create encoded col as 0
                X[col + "_encoded"] = 0
        return X

    def _apply_imputer(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.imputer is None or self.numeric_cols_ is None:
            return X
        # Ensure all numeric_cols present
        for col in self.numeric_cols_:
            if col not in X.columns:
                X[col] = np.nan
        # also include any encoded cols that are in numeric_cols_
        cols_to_impute = [c for c in self.numeric_cols_ if c in X.columns]
        # Align X subset
        try:
            X[cols_to_impute] = self.imputer.transform(X[cols_to_impute]) #type: ignore
        except Exception:
            # fallback: fill with median per col
            for col in cols_to_impute:
                X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)
        return X

    # ------------------------------------------------------------------ #
    # Prepare for modeling - respects temporal split
    # ------------------------------------------------------------------ #
    def prepare_for_modeling(self, df: pd.DataFrame, target_col=None, test_size=0.2, use_temporal_split=True):
        """
        Prepare data for model training.
        Respects temporal split column 'split' if present (train/validation/test from build_data.py).
        Otherwise falls back to stratified train_test_split.
        Fits encoders/imputer/selector/scaler ONLY on train.
        """
        if target_col is None:
            target_col = self.target_col

        # Separate features and target first
        if target_col in df.columns:
            y = df[target_col]
            X_full = df.drop(columns=[target_col])
        else:
            y = None
            X_full = df.copy()

        # Remove abuse_type if present (leaky auxiliary label)
        if "abuse_type" in X_full.columns:
            X_full = X_full.drop(columns=["abuse_type"])

        # Identify split strategy
        has_temporal = use_temporal_split and "split" in X_full.columns and X_full["split"].nunique() > 1

        if has_temporal:
            # Use build_data temporal split
            train_mask = X_full["split"] == "train"
            test_mask = X_full["split"].isin(["test", "validation"])
            # Edge: if only train/test without validation, fallback
            if test_mask.sum() == 0:
                test_mask = X_full["split"] == "test"
            # Drop split col before processing
            X_full = X_full.drop(columns=["split"])
            X_train_raw = X_full[train_mask].copy()
            X_test_raw = X_full[test_mask].copy()
            if y is not None:
                y_train = y[train_mask]
                y_test = y[test_mask]
            else:
                y_train = y_test = None
            # If train or test empty, fallback to random split
            if len(X_train_raw) == 0 or len(X_test_raw) == 0:
                warnings.warn("Temporal split resulted in empty train/test, falling back to random split.")
                has_temporal = False

        if not has_temporal:
            # Fallback: random stratified split - but we need to engineer before split?
            # For pipeline correctness, we split raw X_full before fitting transformers
            # So here we split indices first, then fit on train only
            if y is not None:
                X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                    X_full, y, test_size=test_size, random_state=self.random_state, stratify=y
                )
            else:
                # no y, just return engineered scaled data
                X_train_raw = X_full
                X_test_raw = None
                y_train = y_test = None

        # Now fit transformers on train only
        # Build numeric/categorical detection on train
        # Drop datetime cols from train/test consistently
        datetime_cols = X_train_raw.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()
        X_train_raw = X_train_raw.drop(columns=datetime_cols, errors="ignore")
        if X_test_raw is not None:
            X_test_raw = X_test_raw.drop(columns=datetime_cols, errors="ignore")

        # Categorical handling - fit on train
        categorical_cols = X_train_raw.select_dtypes(include=["object", "category"]).columns.tolist()
        numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()

        # Remove any datetime that leaked into numeric (should already be dropped)
        for col in datetime_cols:
            if col in numeric_cols:
                numeric_cols.remove(col)

        self.encoders = {}
        for col in categorical_cols:
            if col in X_train_raw.columns and X_train_raw[col].nunique() < 50:
                le = LabelEncoder()
                vals = X_train_raw[col].astype(object).where(X_train_raw[col].notna(), "unknown").astype(str)
                vals = vals.replace("nan", "unknown")
                le.fit(vals)
                self.encoders[col] = le
                X_train_raw[col + "_encoded"] = le.transform(vals)
                if X_test_raw is not None and col in X_test_raw.columns:
                    # apply to test with unseen handling
                    test_vals = X_test_raw[col].astype(object).where(X_test_raw[col].notna(), "unknown").astype(str)
                    test_vals = test_vals.replace("nan", "unknown")
                    known = set(le.classes_)
                    test_vals = test_vals.apply(lambda v: v if v in known else ("unknown" if "unknown" in known else le.classes_[0]))
                    X_test_raw[col + "_encoded"] = le.transform(test_vals)
                elif X_test_raw is not None:
                    X_test_raw[col + "_encoded"] = 0
                numeric_cols.append(col + "_encoded")

        # After encoding, keep only numeric columns for modeling
        # Note: original categorical cols remain but are not used
        numeric_cols = [c for c in numeric_cols if c in X_train_raw.columns]

        # Imputer fit on train
        self.imputer = SimpleImputer(strategy="median")
        self.numeric_cols_ = numeric_cols
        if numeric_cols:
            self.imputer.fit(X_train_raw[numeric_cols])
            X_train_raw[numeric_cols] = self.imputer.transform(X_train_raw[numeric_cols])
            if X_test_raw is not None:
                # ensure test has same cols
                for col in numeric_cols:
                    if col not in X_test_raw.columns:
                        X_test_raw[col] = 0
                X_test_raw[numeric_cols] = self.imputer.transform(X_test_raw[numeric_cols])

        # Feature selection fit on train only
        if y_train is not None and len(numeric_cols) > 0:
            k = min(50, len(numeric_cols))
            self.selector = SelectKBest(score_func=f_classif, k=k)
            self.selector.fit(X_train_raw[numeric_cols], y_train)
            selected_indices = self.selector.get_support(indices=True)
            self.selected_features = [numeric_cols[i] for i in selected_indices]
            self.feature_importance = pd.DataFrame({"feature": numeric_cols, "score": self.selector.scores_}).sort_values("score", ascending=False)
            print(f"Selected {len(self.selected_features)} features out of {len(numeric_cols)}")
        else:
            self.selected_features = numeric_cols
            self.selector = None
            self.feature_importance = None

        # Final feature sets
        if self.selected_features:
            X_train_final = X_train_raw[self.selected_features]
            X_test_final = X_test_raw[self.selected_features] if X_test_raw is not None else None
        else:
            X_train_final = X_train_raw[numeric_cols] if numeric_cols else X_train_raw
            X_test_final = X_test_raw[numeric_cols] if X_test_raw is not None and numeric_cols else X_test_raw

        # Scale fit on train
        if y is not None:
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train_final)
            X_test_scaled = self.scaler.transform(X_test_final) if X_test_final is not None else None
            self.is_fitted = True
            return X_train_scaled, X_test_scaled, y_train, y_test, self.selected_features
        else:
            self.scaler = StandardScaler()
            # For inference path, X_train_raw actually holds all data
            X_scaled = self.scaler.fit_transform(X_train_final)
            self.is_fitted = True
            return X_scaled, self.selected_features

    def get_feature_importance(self):
        """Get feature importance scores"""
        return self.feature_importance


# Main execution script
if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    PATH = os.getenv("DATA_STORAGE_PATH")
    if not PATH:
        PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        warnings.warn(f"DATA_STORAGE_PATH not set, using fallback: {PATH}")
    os.makedirs(PATH, exist_ok=True)

    print("Loading dataset...")
    dataset_path = os.path.join(PATH, "data.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Generate it via build_data.py first.")
    dataset = pd.read_csv(dataset_path)

    print(f"Original dataset shape: {dataset.shape}")
    print(f"Columns: {dataset.columns.tolist()}")

    fe = FeatureEngineer(target_col="abuse_label")

    print("\nPerforming feature engineering...")
    engineered_df = fe.engineer_features(dataset)

    print(f"Engineered dataset shape: {engineered_df.shape}")
    print(f"New columns: {len(engineered_df.columns)}")

    print("\nPreparing data for modeling...")
    X_train, X_test, y_train, y_test, selected_features = fe.prepare_for_modeling(engineered_df, test_size=0.2)  # type: ignore

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")  # type: ignore

    print("\nTop 20 most important features:")
    imp = fe.get_feature_importance()
    if imp is not None:
        print(imp.head(20))  # type: ignore
    else:
        print("No feature importance (no target).")

    print("\nSaving processed data...")
    np.save(os.path.join(PATH, "X_train.npy"), X_train)
    np.save(os.path.join(PATH, "X_test.npy"), X_test) #type: ignore
    np.save(os.path.join(PATH, "y_train.npy"), y_train) #type: ignore
    np.save(os.path.join(PATH, "y_test.npy"), y_test) #type: ignore

    with open(os.path.join(PATH, "selected_features.txt"), "w") as f:
        for feat in selected_features:
            f.write(f"{feat}\n")

    if imp is not None:
        imp.to_csv(os.path.join(PATH, "fe_imp.csv"), index=False)

    engineered_df.to_csv(os.path.join(PATH, "transformed_data.csv"), index=False)

    print("\nFeature engineering complete!")
    print("Files saved:")
    print(f"  - {os.path.join(PATH, 'X_train.npy')}, {os.path.join(PATH, 'X_test.npy')}, {os.path.join(PATH, 'y_train.npy')}, {os.path.join(PATH, 'y_test.npy')}")
    print(f"  - {os.path.join(PATH, 'selected_features.txt')}")
    print(f"  - {os.path.join(PATH, 'fe_imp.csv')}")
    print(f"  - {os.path.join(PATH, 'transformed_data.csv')}")

    print("\nDataset Statistics:")
    print(f"  - Total samples: {len(engineered_df)}")
    print(f"  - Features: {len(selected_features)}")
    if y_train is not None and y_test is not None:
        print(f"  - Abuse rate: {y_train.mean():.2%} (train), {y_test.mean():.2%} (test)")
        print(f"  - Train/Test split: {len(X_train)} / {len(X_test)}") #type: ignore
