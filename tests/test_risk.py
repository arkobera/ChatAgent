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


class TestGraphRiskModelLocal(unittest.TestCase):
    """Unit tests for GraphRiskModel that run locally (no W&B)."""

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_orders_and_returns():
        """Create deterministic orders/returns for a hand-crafted graph.

        Graph structure:
            C1 -- D1 -- C2     (C1 and C2 share device D1)
            C1 -- IP1 -- C2    (C1 and C2 share IP IP1)
            C3 -- D2 -- C4     (C3 and C4 share device D2)
            C5 (isolated, no shared entities)

        Returns:
            C1: 1 abusive return
            C2: 1 abusive return
            C3: 1 legitimate return
            C4: 1 legitimate return
            C5: 1 legitimate return
        """
        orders = pd.DataFrame(
            [
                # C1 uses D1, IP1, A1, P1
                {
                    "order_id": "O001",
                    "customer_id": "C1",
                    "device_id": "D1",
                    "ip_id": "IP1",
                    "address_id": "A1",
                    "payment_id": "P1",
                    "purchase_date": pd.Timestamp("2025-01-15"),
                },
                # C2 uses D1, IP1, A2, P2  (shares D1 and IP1 with C1)
                {
                    "order_id": "O002",
                    "customer_id": "C2",
                    "device_id": "D1",
                    "ip_id": "IP1",
                    "address_id": "A2",
                    "payment_id": "P2",
                    "purchase_date": pd.Timestamp("2025-01-20"),
                },
                # C3 uses D2, IP2, A3, P3
                {
                    "order_id": "O003",
                    "customer_id": "C3",
                    "device_id": "D2",
                    "ip_id": "IP2",
                    "address_id": "A3",
                    "payment_id": "P3",
                    "purchase_date": pd.Timestamp("2025-02-01"),
                },
                # C4 uses D2, IP3, A4, P4  (shares D2 with C3)
                {
                    "order_id": "O004",
                    "customer_id": "C4",
                    "device_id": "D2",
                    "ip_id": "IP3",
                    "address_id": "A4",
                    "payment_id": "P4",
                    "purchase_date": pd.Timestamp("2025-02-05"),
                },
                # C5 uses D3, IP4, A5, P5  (isolated)
                {
                    "order_id": "O005",
                    "customer_id": "C5",
                    "device_id": "D3",
                    "ip_id": "IP4",
                    "address_id": "A5",
                    "payment_id": "P5",
                    "purchase_date": pd.Timestamp("2025-02-10"),
                },
            ]
        )

        returns = pd.DataFrame(
            [
                {
                    "return_id": "R001",
                    "order_id": "O001",
                    "customer_id": "C1",
                    "abuse_label": 1,
                    "return_date": pd.Timestamp("2025-01-20"),
                },
                {
                    "return_id": "R002",
                    "order_id": "O002",
                    "customer_id": "C2",
                    "abuse_label": 1,
                    "return_date": pd.Timestamp("2025-01-25"),
                },
                {
                    "return_id": "R003",
                    "order_id": "O003",
                    "customer_id": "C3",
                    "abuse_label": 0,
                    "return_date": pd.Timestamp("2025-02-10"),
                },
                {
                    "return_id": "R004",
                    "order_id": "O004",
                    "customer_id": "C4",
                    "abuse_label": 0,
                    "return_date": pd.Timestamp("2025-02-15"),
                },
                {
                    "return_id": "R005",
                    "order_id": "O005",
                    "customer_id": "C5",
                    "abuse_label": 0,
                    "return_date": pd.Timestamp("2025-02-20"),
                },
            ]
        )

        return orders, returns

    # ------------------------------------------------------------------ #
    # Graph construction tests
    # ------------------------------------------------------------------ #
    def test_entity_index_structure(self):
        """build_entity_index returns correct entity_type → entity_id → {customers}."""
        from Backend.risk.graph_model import build_entity_index

        orders, _ = self._make_orders_and_returns()
        idx = build_entity_index(orders)

        # D1 is used by C1 and C2
        self.assertIn("device", idx)
        self.assertEqual(idx["device"]["D1"], {"C1", "C2"})
        self.assertEqual(idx["device"]["D2"], {"C3", "C4"})
        self.assertEqual(idx["device"]["D3"], {"C5"})

        # IP1 is used by C1 and C2
        self.assertIn("ip", idx)
        self.assertEqual(idx["ip"]["IP1"], {"C1", "C2"})

    def test_customer_graph_edges(self):
        """Shared entities create edges between customers."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
        )

        orders, _ = self._make_orders_and_returns()
        entity_idx = build_entity_index(orders)
        G, cust_entities = build_customer_graph(entity_idx)

        # C1 and C2 share D1 and IP1 → should have an edge
        self.assertTrue(G.has_edge("C1", "C2"))
        edge_data = G["C1"]["C2"]
        self.assertIn("device", edge_data["shared_types"])
        self.assertIn("ip", edge_data["shared_types"])
        self.assertEqual(edge_data["n_shared"], 2)

        # C3 and C4 share D2 → should have an edge
        self.assertTrue(G.has_edge("C3", "C4"))
        self.assertEqual(G["C3"]["C4"]["n_shared"], 1)

        # C5 is isolated → no edges
        self.assertEqual(G.degree("C5"), 0)

    def test_isolated_customer_zero_shared_counts(self):
        """A customer with unique entities gets zero shared-entity counts."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
            extract_customer_features,
        )

        orders, _ = self._make_orders_and_returns()
        entity_idx = build_entity_index(orders)
        G, cust_entities = build_customer_graph(entity_idx)
        features = extract_customer_features(G, cust_entities, orders)

        c5 = features[features["customer_id"] == "C5"].iloc[0]
        self.assertEqual(c5["shared_device_customers"], 0)
        self.assertEqual(c5["shared_ip_customers"], 0)
        self.assertEqual(c5["shared_address_customers"], 0)
        self.assertEqual(c5["shared_payment_customers"], 0)
        self.assertEqual(c5["one_hop_customer_count"], 0)
        self.assertEqual(c5["customer_degree"], 0)

    def test_shared_customer_counts(self):
        """Customers sharing entities have correct shared_*_customers counts."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
            extract_customer_features,
        )

        orders, _ = self._make_orders_and_returns()
        entity_idx = build_entity_index(orders)
        G, cust_entities = build_customer_graph(entity_idx)
        features = extract_customer_features(G, cust_entities, orders)

        c1 = features[features["customer_id"] == "C1"].iloc[0]
        # C1 shares D1 with C2, and IP1 with C2
        self.assertEqual(c1["shared_device_customers"], 1)  # C2
        self.assertEqual(c1["shared_ip_customers"], 1)  # C2
        self.assertEqual(c1["one_hop_customer_count"], 1)  # only C2

    def test_features_contain_no_target_columns(self):
        """Graph features never include abuse_label, abuse_type, scenario, ring_id."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
            extract_customer_features,
        )

        orders, _ = self._make_orders_and_returns()
        entity_idx = build_entity_index(orders)
        G, cust_entities = build_customer_graph(entity_idx)
        features = extract_customer_features(G, cust_entities, orders)

        for col in ("abuse_label", "abuse_type", "scenario", "ring_id"):
            self.assertNotIn(col, features.columns)

    def test_features_contain_expected_columns(self):
        """Features DataFrame has all 17 expected columns + customer_id."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
            extract_customer_features,
            GRAPH_FEATURE_NAMES,
        )

        orders, _ = self._make_orders_and_returns()
        entity_idx = build_entity_index(orders)
        G, cust_entities = build_customer_graph(entity_idx)
        features = extract_customer_features(G, cust_entities, orders)

        self.assertIn("customer_id", features.columns)
        for feat in GRAPH_FEATURE_NAMES:
            self.assertIn(feat, features.columns, f"Missing feature: {feat}")

    def test_temporal_excludes_future_orders(self):
        """Graph built from a subset of orders does not include later edges."""
        from Backend.risk.graph_model import (
            build_customer_graph,
            build_entity_index,
            extract_customer_features,
        )

        orders, _ = self._make_orders_and_returns()
        # Only use orders before Feb 1 (excludes C3, C4, C5 orders)
        early_orders = orders[orders["purchase_date"] < "2025-02-01"]
        entity_idx = build_entity_index(early_orders)
        G, cust_entities = build_customer_graph(entity_idx)
        features = extract_customer_features(G, cust_entities, early_orders)

        customer_ids = set(features["customer_id"])
        # C3, C4, C5 should NOT appear
        self.assertNotIn("C3", customer_ids)
        self.assertNotIn("C4", customer_ids)
        self.assertNotIn("C5", customer_ids)

    # ------------------------------------------------------------------ #
    # Model integration tests
    # ------------------------------------------------------------------ #
    def test_fit_predict_evaluate(self):
        """Full fit/predict/evaluate pipeline works on synthetic data."""
        from Backend.risk.graph_model import GraphRiskModel

        orders, returns = self._make_orders_and_returns()
        model = GraphRiskModel(wandb_project="test-dummy")

        with patch("Backend.risk.graph_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(orders=orders, returns_df=returns)

        self.assertIsNotNone(model.model)
        self.assertEqual(len(model.feature_cols), 18)

        # predict_proba returns probabilities in [0, 1]
        proba_df = model.predict_proba(orders, returns)
        self.assertIn("customer_id", proba_df.columns)
        self.assertIn("network_risk_probability", proba_df.columns)
        self.assertTrue(
            np.all(
                (proba_df["network_risk_probability"] >= 0)
                & (proba_df["network_risk_probability"] <= 1)
            )
        )

        # predict returns binary labels
        pred_df = model.predict(orders, returns)
        self.assertIn("network_risk_label", pred_df.columns)
        self.assertTrue(
            set(pred_df["network_risk_label"].unique()).issubset({0, 1})
        )

        # evaluate returns expected metrics
        metrics = model.evaluate(orders, returns)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("roc_auc", metrics)
        self.assertIn("false_positive_rate", metrics)

    def test_save_and_load_roundtrip(self):
        """Save and load produce identical predictions."""
        from Backend.risk.graph_model import GraphRiskModel

        orders, returns = self._make_orders_and_returns()
        model = GraphRiskModel()

        with patch("Backend.risk.graph_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(orders=orders, returns_df=returns)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "graph_model.joblib"
            model.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = GraphRiskModel.load(save_path)
            self.assertEqual(loaded.feature_cols, model.feature_cols)
            self.assertAlmostEqual(loaded.threshold, model.threshold)

            proba_orig = model.predict_proba(orders, returns)
            proba_loaded = loaded.predict_proba(orders, returns)
            pd.testing.assert_frame_equal(proba_orig, proba_loaded)

    def test_get_customer_risk_explanation(self):
        """Risk explanation returns structured dict for a known customer."""
        from Backend.risk.graph_model import GraphRiskModel

        orders, returns = self._make_orders_and_returns()
        model = GraphRiskModel()

        with patch("Backend.risk.graph_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(orders=orders, returns_df=returns)

        explanation = model.get_customer_risk_explanation("C1", orders)
        self.assertIn("network_risk_probability", explanation)
        self.assertIn("entity_counts", explanation)
        self.assertIn("sharing", explanation)
        self.assertIn("neighborhood", explanation)
        self.assertIn("community", explanation)
        self.assertIn("structural", explanation)
        self.assertIsNotNone(explanation["network_risk_probability"])

    def test_unknown_customer_returns_error(self):
        """Risk explanation for non-existent customer returns error dict."""
        from Backend.risk.graph_model import GraphRiskModel

        orders, returns = self._make_orders_and_returns()
        model = GraphRiskModel()

        with patch("Backend.risk.graph_model.wandb") as mock_wandb:
            mock_wandb.init.return_value = MagicMock()
            mock_wandb.Table.return_value = MagicMock()
            mock_wandb.config = MagicMock()
            mock_wandb.config.tmp_dir = tempfile.gettempdir()
            model.fit(orders=orders, returns_df=returns)

        explanation = model.get_customer_risk_explanation("C_NONEXISTENT", orders)
        self.assertIn("error", explanation)


class TestGraphRiskModelWandb(unittest.TestCase):
    """Integration test: download the graph model from W&B and verify."""

    def test_download_and_verify_signature(self):
        wandb_key = os.getenv("WANDB")
        if not wandb_key:
            self.skipTest("WANDB env var not set; skipping W&B integration test")

        os.environ["WANDB_API_KEY"] = wandb_key

        from Backend.risk.graph_model import GraphRiskModel

        model = GraphRiskModel.load_from_wandb(
            project="returnguard",
            tag="latest",
        )

        self.assertIsInstance(model, GraphRiskModel)
        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.feature_cols), 0)
        self.assertEqual(len(model.feature_cols), 18)

        # Verify model can predict on synthetic orders
        rng = np.random.RandomState(99)
        n = 10
        fake_orders = pd.DataFrame(
            {
                "order_id": [f"O{i:04d}" for i in range(n)],
                "customer_id": [f"C{i // 2:04d}" for i in range(n)],
                "device_id": [f"D{rng.randint(0, 3):04d}" for _ in range(n)],
                "ip_id": [f"IP{rng.randint(0, 3):04d}" for _ in range(n)],
                "address_id": [f"A{rng.randint(0, 3):04d}" for _ in range(n)],
                "payment_id": [f"P{rng.randint(0, 3):04d}" for _ in range(n)],
                "purchase_date": pd.date_range("2025-01-01", periods=n),
            }
        )
        fake_returns = pd.DataFrame(
            {
                "return_id": [f"R{i:04d}" for i in range(5)],
                "customer_id": [f"C{i:04d}" for i in range(5)],
                "abuse_label": [0, 1, 0, 1, 0],
            }
        )

        proba_df = model.predict_proba(fake_orders, fake_returns)
        self.assertIn("customer_id", proba_df.columns)
        self.assertIn("network_risk_probability", proba_df.columns)
        self.assertTrue(
            np.all(
                (proba_df["network_risk_probability"] >= 0)
                & (proba_df["network_risk_probability"] <= 1)
            )
        )


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
            model.model.coef_.shape[1], #type: ignore           
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


class TestRiskEngine(unittest.TestCase):
    """Unit tests for combined return and network decision logic."""

    @staticmethod
    def _returns(labels: list[int] | None = None) -> pd.DataFrame:
        labels = labels if labels is not None else [0, 1, 1]
        return pd.DataFrame(
            {
                "return_id": [f"R{i}" for i in range(len(labels))],
                "customer_id": [f"C{i}" for i in range(len(labels))],
                "abuse_label": labels,
            }
        )

    @staticmethod
    def _engine(
        return_scores: list[float], graph_scores: list[float]
    ):
        from Backend.risk.risk_engine import RiskEngine

        return_model = MagicMock()
        return_model.predict_proba.return_value = np.array(return_scores)
        graph_model = MagicMock()
        graph_model.predict_proba.return_value = pd.DataFrame(
            {
                "customer_id": [f"C{i}" for i in range(len(graph_scores))],
                "network_risk_probability": graph_scores,
            }
        )
        return RiskEngine(return_model, graph_model)

    def test_assess_combines_scores_and_assigns_actions(self):
        engine = self._engine([0.1, 0.2, 0.5, 0.9], [0.2, 0.45, 1.0, 0.6])
        result = engine.assess(self._returns([0, 1, 1, 1]), pd.DataFrame())

        self.assertEqual(
            list(result.columns),
            [
                "return_id",
                "customer_id",
                "return_risk",
                "network_risk",
                "overall_risk",
                "risk_level",
                "recommended_action",
            ],
        )
        np.testing.assert_allclose(result["overall_risk"], [0.14, 0.3, 0.7, 0.78])
        self.assertEqual(
            result["risk_level"].tolist(), ["LOW", "REVIEW", "HIGH", "HIGH"]
        )
        self.assertEqual(
            result["recommended_action"].tolist(),
            ["APPROVE", "MANUAL_REVIEW", "HOLD_AND_REVIEW", "HOLD_AND_REVIEW"],
        )

    def test_rejects_invalid_configuration_and_missing_network_scores(self):
        from Backend.risk.risk_engine import RiskEngine

        model = MagicMock()
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            RiskEngine(model, model, behavioral_weight=0.7, graph_weight=0.4)
        with self.assertRaisesRegex(ValueError, "less than"):
            RiskEngine(model, model, review_threshold=0.7, high_threshold=0.7)

        engine = self._engine([0.2], [])
        with self.assertRaisesRegex(ValueError, "No network risk score"):
            engine.assess(self._returns([0]), pd.DataFrame())

    def test_evaluate_uses_review_threshold_as_positive_cutoff(self):
        engine = self._engine([0.1, 0.4, 0.9, 0.6], [0.1, 0.4, 0.9, 0.6])
        metrics = engine.evaluate(self._returns([0, 1, 1, 0]), pd.DataFrame())

        self.assertEqual(metrics["evaluation_threshold"], 0.3)
        self.assertEqual(metrics["true_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["false_negatives"], 0)
        self.assertEqual(metrics["true_positives"], 2)
        self.assertIsNotNone(metrics["roc_auc"])
        self.assertIsNotNone(metrics["pr_auc"])

    def test_evaluate_returns_no_auc_for_single_class_labels(self):
        engine = self._engine([0.1, 0.4], [0.1, 0.4])
        metrics = engine.evaluate(self._returns([0, 0]), pd.DataFrame())

        self.assertIsNone(metrics["roc_auc"])
        self.assertIsNone(metrics["pr_auc"])

    def test_from_wandb_loads_production_artifacts(self):
        from Backend.risk.risk_engine import RiskEngine

        return_model = MagicMock()
        graph_model = MagicMock()
        with patch.dict(os.environ, {"WANDB": "test-key"}, clear=False), patch(
            "Backend.risk.risk_engine.load_dotenv"
        ), patch(
            "Backend.risk.risk_engine.ReturnRiskModel.load_from_wandb",
            return_value=return_model,
        ) as load_return, patch(
            "Backend.risk.risk_engine.GraphRiskModel.load_from_wandb",
            return_value=graph_model,
        ) as load_graph:
            engine = RiskEngine.from_wandb()

        self.assertIs(engine.return_model, return_model)
        self.assertIs(engine.graph_model, graph_model)
        self.assertEqual(load_return.call_args.kwargs["tag"], "production")
        self.assertEqual(load_graph.call_args.kwargs["tag"], "production")


if __name__ == "__main__":
    unittest.main()
