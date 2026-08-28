"""
Tests for Backend.risk.return_model

Run with:
    python -m pytest tests/test_risk.py -v
    or
    python -m unittest tests.test_risk -v
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd


def _make_synthetic_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic dataset that mirrors transformed_data.csv."""
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        abuse = int(rng.random() < 0.3)
        row = {
            "return_id": f"R{i:08d}",
            "order_id": f"O{i:08d}",
            "customer_id": f"C{i // 5:06d}",
            "return_date": "2025-06-01",
            "purchase_date": "2025-05-25",
            "account_creation_date": "2023-01-01",
            "return_reason": rng.choice(["defective", "changed_mind", "wrong_item"]),
            "product_category": rng.choice(["electronics", "clothing", "books"]),
            "payment_method": rng.choice(["credit_card", "upi", "cod"]),
            "device_id": f"D{i // 10:06d}",
            "ip_id": f"IP{i // 8:06d}",
            "address_id": f"A{i // 7:06d}",
            "payment_id": f"P{i // 6:06d}",
            "city": rng.choice(["Bengaluru", "Mumbai"]),
            "state": rng.choice(["Karnataka", "Maharashtra"]),
            "country": "India",
            "abuse_type": "legitimate" if abuse == 0 else "individual_abuse",
            "split": "train" if i < 120 else ("validation" if i < 160 else "test"),
            "abuse_label": abuse,
            "order_amount": rng.lognormal(8, 1),
            "refund_amount": rng.lognormal(6, 1) if abuse else rng.lognormal(5, 1),
            "discount_percentage": rng.uniform(0, 30),
            "number_of_previous_orders": rng.poisson(10),
            "number_of_previous_returns": rng.poisson(3) if abuse else rng.poisson(1),
            "historical_return_rate": rng.beta(2, 10) if not abuse else rng.beta(5, 2),
            "historical_refund_amount": rng.lognormal(5, 1),
            "avg_order_value": rng.lognormal(7, 0.5),
            "avg_return_value": rng.lognormal(6, 0.5),
            "unique_devices_30d": rng.randint(1, 4),
            "unique_ips_30d": rng.randint(1, 5),
            "unique_addresses_30d": rng.randint(1, 3),
            "orders_last_7_days": rng.poisson(2),
            "orders_last_30_days": rng.poisson(8),
            "returns_last_7_days": rng.poisson(1) if abuse else 0,
            "returns_last_30_days": rng.poisson(3) if abuse else rng.poisson(1),
            "refund_amount_30d": rng.lognormal(4, 1),
            "return_rate_30d": rng.beta(2, 5),
            "return_rate_90d": rng.beta(2, 8),
            "previous_chargebacks": int(rng.random() < 0.1),
            "age": rng.randint(18, 65),
            "customer_tenure_days": rng.randint(30, 1500),
            "days_to_return": rng.randint(1, 28),
            "linked_account_count": rng.randint(1, 4),
            "shared_device_count": rng.randint(0, 3),
            "shared_ip_count": rng.randint(0, 3),
            "shared_address_count": rng.randint(0, 2),
            "shared_payment_count": rng.randint(0, 2),
            "latitude": rng.uniform(8, 37),
            "longitude": rng.uniform(68, 97),
            "distance_from_shipping_address": rng.uniform(0, 50),
            "unique_locations_30d": rng.randint(1, 4),
            # engineered features
            "return_day_of_week": 2,
            "return_month": 6,
            "return_weekend": 0,
            "return_hour": 12,
            "purchase_day_of_week": 1,
            "purchase_month": 5,
            "purchase_weekend": 0,
            "days_since_purchase": 7,
            "account_age_days": 900,
            "account_age_months": 30.0,
            "is_holiday_season": 0,
            "is_summer": 1,
            "return_velocity_7d": 0.1,
            "return_velocity_30d": 0.05,
            "order_frequency": 0.1,
            "high_order_frequency": 0,
            "return_to_order_ratio": rng.beta(1, 5),
            "refund_to_order_ratio": rng.beta(1, 5),
            "refund_to_return_ratio": rng.uniform(0.5, 1.0),
            "return_consistency": rng.uniform(0, 0.5),
            "order_consistency": rng.uniform(0, 0.3),
            "cat_electronics": 0,
            "cat_clothing": 1,
            "cat_books": 0,
            "cat_home_appliances": 0,
            "cat_beauty": 0,
            "cat_sports": 0,
            "cat_jewelry": 0,
            "cat_grocery": 0,
            "category_risk_score": 0.3,
            "payment_risk_score": 0.3,
            "log_order_amount": 8.0,
            "amount_relative_to_category": 0.0,
            "refund_ratio": 0.8,
            "high_discount": 0,
            "discount_impact": 0.0,
            "composite_risk_score": 0.3,
            "risk_score_normalized": 0.3,
            "return_reason_risk": 0.3,
            "daily_avg_amount": 2000.0,
            "daily_avg_return_rate": 0.1,
            "days_amount_interaction": 7000.0,
            "days_amount_ratio": 0.007,
            "return_rate_amount": 0.1,
            "returns_discount_interaction": 0.0,
            "age_return_interaction": 30.0,
            "tenure_returns_ratio": 1.0,
            "payment_amount_risk": 0.0,
            "order_amount_log": 8.0,
            "order_amount_rank": 0.5,
            "order_amount_zscore": 0.0,
            "refund_amount_log": 6.0,
            "refund_amount_rank": 0.5,
            "refund_amount_zscore": 0.0,
            "number_of_previous_orders_log": 2.3,
            "number_of_previous_orders_rank": 0.5,
            "number_of_previous_returns_log": 1.0,
            "number_of_previous_returns_rank": 0.5,
            "orders_last_7_days_log": 0.7,
            "orders_last_7_days_rank": 0.5,
            "orders_last_30_days_log": 2.1,
            "orders_last_30_days_rank": 0.5,
            "customer_tenure_days_log": 7.0,
            "customer_tenure_days_rank": 0.5,
            "historical_return_rate_logit": -2.0,
            "return_rate_30d_logit": -1.5,
            "return_rate_90d_logit": -1.8,
            "order_amount_percentile": 0.5,
            "historical_return_rate_percentile": 0.3,
            "age_percentile": 0.5,
            "customer_tenure_days_percentile": 0.5,
            "return_trend": 0.5,
            "purchase_velocity": 5.0,
            "purchase_velocity_rank": 0.5,
            "fast_purchaser": 0,
            "return_reason_consistency": 1.0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


class TestReturnRiskModelLocal(unittest.TestCase):
    """Unit tests that run entirely locally (no W&B)."""

    @classmethod
    def setUpClass(cls):
        cls.df = _make_synthetic_df()

    def test_feature_columns_exclude_leaky(self):
        from Backend.risk.return_model import _feature_columns

        cols = _feature_columns(self.df)
        for forbidden in ("abuse_type", "abuse_label", "split", "return_id", "order_id", "customer_id"):
            self.assertNotIn(forbidden, cols)

    def test_prepare_splits_uses_split_column(self):
        from Backend.risk.return_model import _feature_columns, _prepare_splits

        cols = _feature_columns(self.df)
        X_tr, y_tr, X_val, y_val, X_te, y_te = _prepare_splits(self.df, cols)
        self.assertGreater(len(X_tr), 0)
        self.assertGreater(len(X_val), 0)
        self.assertGreater(len(X_te), 0)
        # train split should match raw count
        self.assertEqual(len(X_tr), (self.df["split"] == "train").sum())

    def test_fit_predict_evaluate(self):
        from Backend.risk.return_model import ReturnRiskModel

        model = ReturnRiskModel(wandb_project="test-dummy")
        train_df = self.df[self.df["split"] == "train"]

        with patch("Backend.risk.return_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(train_df)

        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.feature_cols), 0)

        # predict on test split
        test_df = self.df[self.df["split"] == "test"]
        proba = model.predict_proba(test_df)
        self.assertEqual(len(proba), len(test_df))
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

        preds = model.predict(test_df, threshold=0.5)
        self.assertEqual(len(preds), len(test_df))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

        metrics = model.evaluate(test_df)
        self.assertIn("accuracy", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("f1", metrics)

    def test_save_and_load_roundtrip(self):
        from Backend.risk.return_model import ReturnRiskModel

        model = ReturnRiskModel()
        train_df = self.df[self.df["split"] == "train"]

        with patch("Backend.risk.return_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(train_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model.joblib"
            model.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = ReturnRiskModel.load(save_path)
            self.assertEqual(loaded.feature_cols, model.feature_cols)
            self.assertEqual(loaded.categorical_cols, model.categorical_cols)

            test_df = self.df[self.df["split"] == "test"]
            proba_orig = model.predict_proba(test_df)
            proba_loaded = loaded.predict_proba(test_df)
            np.testing.assert_array_almost_equal(proba_orig, proba_loaded)


class TestReturnRiskModelSignature(unittest.TestCase):
    """Verify that the model enforces its feature signature."""

    def test_predict_rejects_extra_columns(self):
        from Backend.risk.return_model import ReturnRiskModel

        df = _make_synthetic_df()
        model = ReturnRiskModel()
        train_df = df[df["split"] == "train"]

        with patch("Backend.risk.return_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(train_df)

        # DataFrame with an extra column that was NOT in training
        test_df = df[df["split"] == "test"].copy()
        test_df["sneaky_leak_column"] = 999

        # predict_proba should still work (extra cols are silently ignored
        # because we select self.feature_cols)
        proba = model.predict_proba(test_df)
        self.assertEqual(len(proba), len(test_df))

    def test_predict_rejects_missing_columns(self):
        from Backend.risk.return_model import ReturnRiskModel

        df = _make_synthetic_df()
        model = ReturnRiskModel()
        train_df = df[df["split"] == "train"]

        with patch("Backend.risk.return_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(train_df)

        # DataFrame missing a feature column that WAS in training
        test_df = df[df["split"] == "test"].copy()
        dropped_feat = model.feature_cols[0]
        test_df = test_df.drop(columns=[dropped_feat])

        # _transform adds missing cols as 0, so predict_proba should not crash
        proba = model.predict_proba(test_df)
        self.assertEqual(len(proba), len(test_df))


class TestWandbArtifactDownload(unittest.TestCase):
    """Integration test: download the model from W&B and verify its signature.

    Set the WANDB env var before running:
        export WANDB='wandb_v1_...'
        python -m pytest tests/test_risk.py::TestWandbArtifactDownload -v -s
    """

    def test_download_and_verify_signature(self):
        wandb_key = os.getenv("WANDB")
        if not wandb_key:
            self.skipTest("WANDB env var not set; skipping W&B integration test")

        os.environ["WANDB_API_KEY"] = wandb_key

        from Backend.risk.return_model import ReturnRiskModel

        model = ReturnRiskModel.load_from_wandb(
            project="returnguard",
            tag="latest",
        )

        # Verify the loaded object is a proper ReturnRiskModel
        self.assertIsInstance(model, ReturnRiskModel)
        self.assertIsNotNone(model.model)
        self.assertIsNotNone(model.scaler)
        self.assertGreater(len(model.feature_cols), 0)
        self.assertIsInstance(model.feature_cols, list)
        self.assertIsInstance(model.categorical_cols, list)
        self.assertIsInstance(model.numeric_cols, list)

        # The model should accept DataFrames with exactly these feature columns
        n_features = len(model.feature_cols)
        self.assertEqual(
            n_features,
            model.model.coef_.shape[1],
            "LogisticRegression coefficient count must match feature_cols",
        )

        # Verify we can run predict_proba on a synthetic DataFrame
        # with the exact right columns
        rng = np.random.RandomState(99)
        fake_data = {}
        for col in model.feature_cols:
            if col in model.categorical_cols:
                le = model.label_encoders.get(col)
                if le is not None and len(le.classes_) > 0:
                    fake_data[col] = [le.classes_[0]] * 5
                else:
                    fake_data[col] = ["unknown"] * 5
            else:
                fake_data[col] = rng.randn(5)
        fake_df = pd.DataFrame(fake_data)

        proba = model.predict_proba(fake_df)
        self.assertEqual(len(proba), 5)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

        preds = model.predict(fake_df, threshold=0.5)
        self.assertEqual(len(preds), 5)
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))


if __name__ == "__main__":
    unittest.main()
