"""
ReturnGuard — Graph-based Network Risk Model

Builds a heterogeneous entity graph (customer ↔ device / IP / address / payment),
projects it to a customer relationship graph, extracts structural features, and
trains a HistGradientBoostingClassifier to predict coordinated return abuse.

Architecture:
    Heterogeneous entity graph
            ↓
    customer relationship graph (via shared entities)
            ↓
    graph features (17 per customer)
            ↓
    HistGradientBoostingClassifier
            ↓
    network risk score

Temporal strategy:
    Graph snapshots are built per split using only orders within that split's
    time period. No future information leaks into historical features.

Limitations documented:
    - Ring members in the synthetic dataset are very few (~0.7% of customers).
    - Entity sharing is ubiquitous (507 customers, 56 devices → ~9 customers/device).
    - The graph model learns from entity-sharing patterns, not true coordinated fraud.
    - Reported performance must be interpreted as synthetic-experiment results only.
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import wandb
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Entity types that connect customers
ENTITY_TYPES = ("device", "ip", "address", "payment")

# Columns in orders DataFrame that map customer → entity
_ORDER_ENTITY_COLS: Dict[str, str] = {
    "device": "device_id",
    "ip": "ip_id",
    "address": "address_id",
    "payment": "payment_id",
}

# Columns that must NEVER appear as graph features
_LEAKY_COLS = frozenset({
    "abuse_label",
    "abuse_type",
    "split",
    "scenario",
    "ring_id",
})

_ID_COLS = frozenset({
    "return_id",
    "order_id",
    "customer_id",
    "device_id",
    "ip_id",
    "address_id",
    "payment_id",
})

_DATE_COLS = frozenset({
    "return_date",
    "purchase_date",
    "account_creation_date",
    "delivery_date",
})

# Graph feature names (17 total)
GRAPH_FEATURE_NAMES: List[str] = [
    # Entity counts (4)
    "device_count",
    "ip_count",
    "address_count",
    "payment_count",
    # Shared-entity customer counts (4)
    "shared_device_customers",
    "shared_ip_customers",
    "shared_address_customers",
    "shared_payment_customers",
    # Max sharing per entity type (4)
    "max_device_sharing",
    "max_ip_sharing",
    "max_address_sharing",
    "max_payment_sharing",
    # Neighborhood (2)
    "one_hop_customer_count",
    "two_hop_customer_count",
    # Community (2)
    "community_size",
    "community_density",
    # Structural (2)
    "customer_degree",
    "local_clustering_coefficient",
]

# ---------------------------------------------------------------------------
# Graph construction helpers
# ---------------------------------------------------------------------------


def build_entity_index(
    orders: pd.DataFrame,
) -> Dict[str, Dict[str, Set[str]]]:
    """Build an inverted index: entity_type → entity_id → {customer_ids}.

    This is O(E) where E = number of orders, and avoids constructing the full
    heterogeneous graph just to derive customer relationships.
    """
    index: Dict[str, Dict[str, Set[str]]] = {
        etype: defaultdict(set) for etype in ENTITY_TYPES
    }

    for _, row in orders.iterrows():
        cust = row.get("customer_id")
        if pd.isna(cust):
            continue
        for etype, col in _ORDER_ENTITY_COLS.items():
            eid = row.get(col)
            if pd.notna(eid):
                index[etype][str(eid)].add(str(cust))

    return index


def build_customer_graph(
    entity_index: Dict[str, Dict[str, Set[str]]],
) -> Tuple[nx.Graph, Dict[str, Dict[str, Set[str]]]]:
    """Build a customer-to-customer relationship graph from shared entities.

    Returns:
        customer_graph: nx.Graph with customer nodes and weighted edges.
        customer_entities: Dict mapping customer_id → {entity_type: set of entity_ids}.
    """
    G = nx.Graph()
    customer_entities: Dict[str, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    # Track which entities each customer uses
    for etype, entity_map in entity_index.items():
        for eid, customers in entity_map.items():
            for cust in customers:
                customer_entities[cust][etype].add(eid)

    # Create customer nodes
    all_customers = set(customer_entities.keys())
    G.add_nodes_from(all_customers)

    # Build edges via shared entities
    # For each entity, connect all customers who share it
    edge_data: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for etype, entity_map in entity_index.items():
        for eid, customers in entity_map.items():
            cust_list = sorted(customers)
            if len(cust_list) < 2:
                continue
            # Connect all pairs sharing this entity
            for i in range(len(cust_list)):
                for j in range(i + 1, len(cust_list)):
                    c1, c2 = cust_list[i], cust_list[j]
                    key = (c1, c2) if c1 < c2 else (c2, c1)
                    if key not in edge_data:
                        edge_data[key] = {
                            "shared_types": set(),
                            "n_shared": 0,
                        }
                    edge_data[key]["shared_types"].add(etype)
                    edge_data[key]["n_shared"] += 1

    # Add edges to graph
    for (c1, c2), data in edge_data.items():
        G.add_edge(
            c1,
            c2,
            shared_types=data["shared_types"],
            n_shared=data["n_shared"],
            weight=data["n_shared"],
        )

    return G, dict(customer_entities)


def extract_customer_features(
    customer_graph: nx.Graph,
    customer_entities: Dict[str, Dict[str, Set[str]]],
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Extract 17 graph features per customer.

    Features are purely structural — no target labels, no return outcomes.
    """
    customers = sorted(customer_graph.nodes())
    features: List[Dict[str, Any]] = []

    # Pre-compute community detection
    communities_map: Dict[str, int] = {}
    community_sizes: Dict[int, int] = {}
    community_internal_edges: Dict[int, int] = {}
    community_nodes: Dict[int, List[str]] = {}

    if len(customers) >= 2:
        try:
            comm_gen = greedy_modularity_communities(customer_graph, resolution=1.0)
            for idx, comm in enumerate(comm_gen):
                members = sorted(comm)
                community_nodes[idx] = members
                community_sizes[idx] = len(members)
                for c in members:
                    communities_map[c] = idx

                # Count internal edges
                subgraph = customer_graph.subgraph(members)
                community_internal_edges[idx] = subgraph.number_of_edges()
        except Exception:
            # Fallback: no communities
            pass

    # Pre-compute clustering coefficients
    try:
        clustering = nx.clustering(customer_graph)
    except Exception:
        clustering = {c: 0.0 for c in customers}

    # Pre-compute entity-to-customer mapping for shared counts
    # Inverted: entity_type → entity_id → customer count
    entity_customer_count: Dict[str, Dict[str, int]] = {
        etype: {} for etype in ENTITY_TYPES
    }
    entity_max_sharing: Dict[str, int] = {etype: 0 for etype in ENTITY_TYPES}

    for etype in ENTITY_TYPES:
        for eid, custs in _iter_entity_customers(customer_entities, etype):
            count = len(custs)
            entity_customer_count[etype][eid] = count
            if count > entity_max_sharing[etype]:
                entity_max_sharing[etype] = count

    for cust in customers:
        # --- Entity counts ---
        ent = customer_entities.get(cust, {})
        device_count = len(ent.get("device", set()))
        ip_count = len(ent.get("ip", set()))
        address_count = len(ent.get("address", set()))
        payment_count = len(ent.get("payment", set()))

        # --- Shared-entity customer counts ---
        # How many OTHER customers share at least one entity of this type
        shared_device_customers = _count_shared_customers(
            customer_entities, cust, "device"
        )
        shared_ip_customers = _count_shared_customers(
            customer_entities, cust, "ip"
        )
        shared_address_customers = _count_shared_customers(
            customer_entities, cust, "address"
        )
        shared_payment_customers = _count_shared_customers(
            customer_entities, cust, "payment"
        )

        # --- Max sharing per entity type ---
        max_device_sharing = _max_entity_sharing(customer_entities, cust, "device")
        max_ip_sharing = _max_entity_sharing(customer_entities, cust, "ip")
        max_address_sharing = _max_entity_sharing(
            customer_entities, cust, "address"
        )
        max_payment_sharing = _max_entity_sharing(
            customer_entities, cust, "payment"
        )

        # --- Neighborhood ---
        one_hop = set(customer_graph.neighbors(cust))
        one_hop_customer_count = len(one_hop)

        two_hop: Set[str] = set()
        for neighbor in one_hop:
            two_hop.update(customer_graph.neighbors(neighbor))
        two_hop.discard(cust)
        two_hop_customer_count = len(two_hop)

        # --- Community ---
        comm_idx = communities_map.get(cust)
        if comm_idx is not None:
            community_size = community_sizes[comm_idx]
            n_internal = community_internal_edges[comm_idx]
            n_members = community_sizes[comm_idx]
            max_possible = n_members * (n_members - 1) // 2 if n_members > 1 else 1
            community_density = n_internal / max_possible if max_possible > 0 else 0.0
        else:
            community_size = 1
            community_density = 0.0

        # --- Structural ---
        customer_degree = customer_graph.degree(cust)
        local_clustering_coefficient = clustering.get(cust, 0.0) #type: ignore

        features.append(
            {
                "customer_id": cust,
                "device_count": device_count,
                "ip_count": ip_count,
                "address_count": address_count,
                "payment_count": payment_count,
                "shared_device_customers": shared_device_customers,
                "shared_ip_customers": shared_ip_customers,
                "shared_address_customers": shared_address_customers,
                "shared_payment_customers": shared_payment_customers,
                "max_device_sharing": max_device_sharing,
                "max_ip_sharing": max_ip_sharing,
                "max_address_sharing": max_address_sharing,
                "max_payment_sharing": max_payment_sharing,
                "one_hop_customer_count": one_hop_customer_count,
                "two_hop_customer_count": two_hop_customer_count,
                "community_size": community_size,
                "community_density": community_density,
                "customer_degree": customer_degree,
                "local_clustering_coefficient": local_clustering_coefficient,
            }
        )

    return pd.DataFrame(features)


def _iter_entity_customers(
    customer_entities: Dict[str, Dict[str, Set[str]]], etype: str
):
    """Yield (entity_id, set_of_customers) for a given entity type."""
    eid_to_custs: Dict[str, Set[str]] = defaultdict(set)
    for cust, entities in customer_entities.items():
        for eid in entities.get(etype, set()):
            eid_to_custs[eid].add(cust)
    return eid_to_custs.items()


def _count_shared_customers(
    customer_entities: Dict[str, Dict[str, Set[str]]],
    target_cust: str,
    etype: str,
) -> int:
    """Count how many OTHER customers share at least one entity of etype."""
    target_eids = customer_entities.get(target_cust, {}).get(etype, set())
    if not target_eids:
        return 0

    shared_custs: Set[str] = set()
    for cust, entities in customer_entities.items():
        if cust == target_cust:
            continue
        cust_eids = entities.get(etype, set())
        if cust_eids & target_eids:  # intersection
            shared_custs.add(cust)
    return len(shared_custs)


def _max_entity_sharing(
    customer_entities: Dict[str, Dict[str, Set[str]]],
    target_cust: str,
    etype: str,
) -> int:
    """Max number of customers sharing any single entity of etype."""
    target_eids = customer_entities.get(target_cust, {}).get(etype, set())
    if not target_eids:
        return 0

    # Build inverted index for this entity type on the fly
    eid_to_count: Dict[str, int] = defaultdict(int)
    for cust, entities in customer_entities.items():
        for eid in entities.get(etype, set()):
            eid_to_count[eid] += 1

    max_count = 0
    for eid in target_eids:
        cnt = eid_to_count.get(eid, 0)
        if cnt > max_count:
            max_count = cnt
    return max_count


# ---------------------------------------------------------------------------
# Temporal graph construction
# ---------------------------------------------------------------------------


def build_graph_snapshot(
    orders: pd.DataFrame,
) -> Tuple[nx.Graph, Dict[str, Dict[str, Set[str]]]]:
    """Build a customer graph from a set of orders.

    This is the main entry point for temporal graph construction.
    Pass orders filtered to the relevant time window.
    """
    entity_index = build_entity_index(orders)
    customer_graph, customer_entities = build_customer_graph(entity_index)
    return customer_graph, customer_entities


# ---------------------------------------------------------------------------
# Label aggregation
# ---------------------------------------------------------------------------


def aggregate_customer_labels(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate return-level labels to customer-level.

    A customer is labeled abusive (1) if ANY of their returns is abusive.
    Uses max() aggregation on abuse_label.
    """
    if "customer_id" not in returns_df.columns:
        raise ValueError("returns_df must contain 'customer_id'")
    if "abuse_label" not in returns_df.columns:
        raise ValueError("returns_df must contain 'abuse_label'")

    customer_labels = (
        returns_df.groupby("customer_id")["abuse_label"]
        .max()
        .reset_index()
        .rename(columns={"abuse_label": "customer_abuse_label"})
    )
    return customer_labels


# ---------------------------------------------------------------------------
# GraphRiskModel
# ---------------------------------------------------------------------------


class GraphRiskModel:
    """Graph-based network risk model for return-abuse detection.

    Builds a customer relationship graph from shared entities (devices, IPs,
    addresses, payments), extracts structural features, and trains a
    HistGradientBoostingClassifier to predict coordinated abuse.

    Parameters
    ----------
    wandb_project : str
        W&B project name.
    wandb_entity : str | None
        Optional W&B entity (team / user).
    max_iter : int
        Number of trees in the forest.
    learning_rate : float
        (unused, kept for API compatibility).
    max_depth : int | None
        Maximum tree depth. None uses the default.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        wandb_project: str = "returnguard",
        wandb_entity: Optional[str] = None,
        max_iter: int = 200,
        learning_rate: float = 0.1,
        max_depth: Optional[int] = 5,
        random_state: int = 42,
    ) -> None:
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        # Fitted state
        self.model: Optional[HistGradientBoostingClassifier] = None #type: ignore
        self.feature_cols: List[str] = list(GRAPH_FEATURE_NAMES)
        self.threshold: float = 0.5
        self.metadata: Dict[str, Any] = {}

        # Graph state (populated during fit)
        self._customer_graph: Optional[nx.Graph] = None
        self._customer_entities: Optional[Dict[str, Dict[str, Set[str]]]] = None
        self._customer_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        orders: pd.DataFrame,
        returns_df: pd.DataFrame,
        validation_orders: Optional[pd.DataFrame] = None,
        validation_returns: Optional[pd.DataFrame] = None,
        run_name: Optional[str] = None,
    ) -> "GraphRiskModel":
        """Train the graph model.

        Parameters
        ----------
        orders : DataFrame
            Orders for training (must contain customer_id, device_id, ip_id,
            address_id, payment_id).
        returns_df : DataFrame
            Returns for training (must contain customer_id, abuse_label).
        validation_orders : DataFrame | None
            Optional validation orders for threshold selection.
        validation_returns : DataFrame | None
            Optional validation returns for threshold selection.
        run_name : str | None
            Custom W&B run name.
        """
        # Build training graph
        logger.info("Building training graph...")
        self._customer_graph, self._customer_entities = build_graph_snapshot(orders)

        # Extract features for all customers in the graph
        logger.info("Extracting customer features...")
        self._customer_features = extract_customer_features(
            self._customer_graph, self._customer_entities, orders
        )

        # Aggregate labels to customer level
        customer_labels = aggregate_customer_labels(returns_df)

        # Merge features with labels
        train_df = self._customer_features.merge(
            customer_labels, on="customer_id", how="inner"
        )

        # Filter to customers with features
        X_train = train_df[self.feature_cols].fillna(0)
        y_train = train_df["customer_abuse_label"]

        logger.info(
            f"Training set: {len(X_train)} customers, "
            f"abuse rate: {y_train.mean():.3f}"
        )

        # Train classifier
        self.model = RandomForestClassifier(
            n_estimators=self.max_iter,
            max_depth=self.max_depth,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train) #type: ignore

        # Store metadata
        self.metadata.update(
            {
                "n_train_customers": len(X_train),
                "train_abuse_rate": float(y_train.mean()),
                "n_graph_nodes": self._customer_graph.number_of_nodes(),
                "n_graph_edges": self._customer_graph.number_of_edges(),
                "max_iter": self.max_iter,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            }
        )

        # Threshold selection on validation set
        if validation_orders is not None and validation_returns is not None:
            self.threshold = self.find_best_threshold(
                validation_orders, validation_returns
            )
            self.metadata["threshold"] = self.threshold
            logger.info(f"Selected threshold: {self.threshold:.2f}")
        else:
            self.metadata["threshold"] = self.threshold

        # ------------------------------------------------------------------ #
        # W&B logging
        # ------------------------------------------------------------------ #
        wandb_api_key = os.getenv("WANDB")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        run = wandb.init(
            project=self.wandb_project,
            entity=self.wandb_entity,
            name=run_name or "graph-risk-model",
            reinit="return_previous",
            config={
                "model": "RandomForestClassifier",
                "n_estimators": self.max_iter,
                "max_depth": self.max_depth,
                "class_weight": "balanced",
                "random_state": self.random_state,
                "n_features": len(self.feature_cols),
                "n_train_customers": len(X_train),
                "train_abuse_rate": float(y_train.mean()),
                "threshold": self.threshold,
            },
        )

        # -- train metrics --
        train_pred = self.model.predict(X_train) #type: ignore
        train_proba = self.model.predict_proba(X_train)[:, 1] #type: ignore
        train_metrics = self._compute_metrics(y_train, train_pred, train_proba)
        for k, v in train_metrics.items():
            wandb.log({f"train/{k}": v})

        # -- validation metrics --
        if validation_orders is not None and validation_returns is not None:
            val_metrics = self.evaluate(
                validation_orders, validation_returns, threshold=self.threshold
            )
            for k, v in val_metrics.items():
                wandb.log({f"val/{k}": v})

        # -- feature importances --
        importances = self.model.feature_importances_ #type: ignore
        feat_table = wandb.Table(columns=["feature", "importance"])
        for feat, imp in sorted(
            zip(self.feature_cols, importances), key=lambda x: -x[1]
        ):
            feat_table.add_data(feat, float(imp))
        wandb.log({"feature_importances": feat_table})

        # -- confusion matrices --
        for split_name, (y_true, y_pred) in {
            "train": (y_train, train_pred),
        }.items():
            cm = confusion_matrix(y_true, y_pred)
            wandb.log(
                {
                    f"{split_name}/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true.tolist(),
                        preds=y_pred.tolist(),
                        class_names=["legitimate", "abusive"],
                    )
                }
            )

        # -- save model artifact --
        artifact = wandb.Artifact(
            name="graph-risk-model",
            type="model",
            metadata=self.metadata,
        )

        bundle_path = (
            Path(wandb.config.tmp_dir) / "graph_risk_model.joblib"
            if hasattr(wandb.config, "tmp_dir")
            else Path("graph_risk_model.joblib")
        )
        self.save(bundle_path)
        artifact.add_file(str(bundle_path))
        run.log_artifact(artifact, name="graph-risk-model", aliases=["latest", "production"])

        wandb.finish()
        return self

    def predict_proba(
        self,
        orders: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Predict network risk for customers.

        Parameters
        ----------
        orders : DataFrame
            Orders to build graph from.
        returns_df : DataFrame | None
            Optional returns to get customer list. If None, uses all
            customers from orders.

        Returns
        -------
        DataFrame with columns: customer_id, network_risk_probability
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() or load() first.")

        # Build graph from provided orders
        customer_graph, customer_entities = build_graph_snapshot(orders)
        customer_features = extract_customer_features(
            customer_graph, customer_entities, orders
        )

        # Get customer list
        if returns_df is not None and "customer_id" in returns_df.columns:
            customers = returns_df["customer_id"].unique()
            customer_features = customer_features[
                customer_features["customer_id"].isin(customers)
            ]

        if len(customer_features) == 0:
            return pd.DataFrame(columns=["customer_id", "network_risk_probability"])

        X = customer_features[self.feature_cols].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]

        result = customer_features[["customer_id"]].copy()
        result["network_risk_probability"] = proba
        return result.reset_index(drop=True)

    def predict(
        self,
        orders: pd.DataFrame,
        returns_df: Optional[pd.DataFrame] = None,
        threshold: Optional[float] = None,
    ) -> pd.DataFrame:
        """Predict binary network risk labels.

        Returns
        -------
        DataFrame with columns: customer_id, network_risk_label
        """
        thr = threshold if threshold is not None else self.threshold
        proba_df = self.predict_proba(orders, returns_df)
        proba_df["network_risk_label"] = (
            proba_df["network_risk_probability"] >= thr
        ).astype(int)
        return proba_df

    def evaluate(
        self,
        orders: pd.DataFrame,
        returns_df: pd.DataFrame,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate the model on a set of orders and returns.

        Parameters
        ----------
        orders : DataFrame
            Orders to build graph from.
        returns_df : DataFrame
            Returns with ground-truth abuse_label.
        threshold : float | None
            Classification threshold. Uses self.threshold if None.
        """
        thr = threshold if threshold is not None else self.threshold

        # Get predictions
        proba_df = self.predict_proba(orders, returns_df)

        # Merge with ground truth
        customer_labels = aggregate_customer_labels(returns_df)
        eval_df = proba_df.merge(customer_labels, on="customer_id", how="inner")

        if len(eval_df) == 0:
            return {"error": "no overlapping customers"}

        y_true = eval_df["customer_abuse_label"]
        y_proba = eval_df["network_risk_probability"]
        y_pred = (y_proba >= thr).astype(int)

        return self._compute_metrics(y_true, y_pred, y_proba)

    def find_best_threshold(
        self,
        orders: pd.DataFrame,
        returns_df: pd.DataFrame,
        fp_cost: float = 1.0,
        fn_cost: float = 5.0,
        criterion: str = "f1",
    ) -> float:
        """Find the optimal classification threshold on validation data.

        Parameters
        ----------
        orders : DataFrame
            Validation orders.
        returns_df : DataFrame
            Validation returns with abuse_label.
        fp_cost : float
            Cost of a false positive.
        fn_cost : float
            Cost of a false negative.
        criterion : str
            "f1" to maximize F1, "cost" to minimize expected cost.
        """
        proba_df = self.predict_proba(orders, returns_df)
        customer_labels = aggregate_customer_labels(returns_df)
        eval_df = proba_df.merge(customer_labels, on="customer_id", how="inner")

        if len(eval_df) == 0:
            return 0.5

        y_true = eval_df["customer_abuse_label"].values
        y_proba = eval_df["network_risk_probability"].values

        thresholds = np.arange(0.05, 0.95, 0.05)
        best_score = -1.0
        best_threshold = 0.5

        for thr in thresholds:
            y_pred = (y_proba >= thr).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
            else:
                tn = fp = fn = tp = 0

            if criterion == "cost":
                score = -(fp_cost * fp + fn_cost * fn)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            if score > best_score:
                best_score = score
                best_threshold = float(thr)

        return best_threshold

    def get_customer_risk_explanation(
        self,
        customer_id: str,
        orders: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Get an explanation of why a customer has high/low network risk.

        Returns a dict suitable for display in a UI.
        """
        customer_graph, customer_entities = build_graph_snapshot(orders)
        customer_features = extract_customer_features(
            customer_graph, customer_entities, orders
        )

        row = customer_features[customer_features["customer_id"] == customer_id]
        if len(row) == 0:
            return {"error": f"customer {customer_id} not found in graph"}

        row = row.iloc[0]

        # Get risk probability
        X = row[self.feature_cols].values.reshape(1, -1)
        proba = self.model.predict_proba(X)[0, 1] if self.model else None

        # Build explanation
        explanation: Dict[str, Any] = {
            "customer_id": customer_id,
            "network_risk_probability": float(proba) if proba is not None else None,
            "threshold": self.threshold,
            "is_high_risk": bool(proba >= self.threshold) if proba is not None else None,
            "entity_counts": {
                "devices": int(row["device_count"]),
                "ips": int(row["ip_count"]),
                "addresses": int(row["address_count"]),
                "payments": int(row["payment_count"]),
            },
            "sharing": {
                "shared_device_customers": int(row["shared_device_customers"]),
                "shared_ip_customers": int(row["shared_ip_customers"]),
                "shared_address_customers": int(row["shared_address_customers"]),
                "shared_payment_customers": int(row["shared_payment_customers"]),
            },
            "max_sharing": {
                "max_device_sharing": int(row["max_device_sharing"]),
                "max_ip_sharing": int(row["max_ip_sharing"]),
                "max_address_sharing": int(row["max_address_sharing"]),
                "max_payment_sharing": int(row["max_payment_sharing"]),
            },
            "neighborhood": {
                "one_hop_customer_count": int(row["one_hop_customer_count"]),
                "two_hop_customer_count": int(row["two_hop_customer_count"]),
            },
            "community": {
                "community_size": int(row["community_size"]),
                "community_density": float(row["community_density"]),
            },
            "structural": {
                "customer_degree": int(row["customer_degree"]),
                "local_clustering_coefficient": float(
                    row["local_clustering_coefficient"]
                ),
            },
        }

        return explanation

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(
        y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, Any]:
        """Aggregate classification metrics."""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        }

        # AUC metrics (require at least 2 classes)
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

        # False-positive rate
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        return metrics

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the full bundle to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_cols": self.feature_cols,
                "threshold": self.threshold,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "GraphRiskModel":
        """Load a previously saved bundle from disk."""
        bundle = joblib.load(path)
        instance = cls()
        instance.model = bundle["model"]
        instance.feature_cols = bundle["feature_cols"]
        instance.threshold = bundle["threshold"]
        instance.metadata = bundle["metadata"]
        return instance

    @classmethod
    def load_from_wandb(
        cls,
        artifact_name: str = "graph-risk-model",
        project: str = "returnguard",
        entity: Optional[str] = None,
        tag: str = "latest",
    ) -> "GraphRiskModel":
        """Download and load the model artifact from W&B."""
        wandb_api_key = os.getenv("WANDB")
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key

        api = wandb.Api()
        artifact_ref = api.artifact(
            f"{entity + '/' if entity else ''}{project}/{artifact_name}:{tag}"
        )
        artifact_dir = artifact_ref.download()
        joblib_files = list(Path(artifact_dir).glob("*.joblib"))
        if not joblib_files:
            raise FileNotFoundError(
                f"No .joblib file found in downloaded artifact at {artifact_dir}"
            )
        return cls.load(joblib_files[0])


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    PATH = os.getenv("DATA_STORAGE_PATH", "./data")

    print("Loading dataset...")
    orders_path = os.path.join(PATH, "data_v2.csv")
    transformed_path = os.path.join(PATH, "transformed_data.csv")

    if not os.path.exists(orders_path):
        raise FileNotFoundError(f"Dataset not found at {orders_path}")

    full_df = pd.read_csv(orders_path)
    print(f"Dataset shape: {full_df.shape}")

    # Split
    train_df = full_df[full_df["split"] == "train"]
    val_df = full_df[full_df["split"] == "validation"]
    test_df = full_df[full_df["split"] == "test"]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model = GraphRiskModel(wandb_project="returnguard")
    model.fit(
        orders=train_df,
        returns_df=train_df,
        validation_orders=val_df,
        validation_returns=val_df,
        run_name="graph-model-v1",
    )

    # Evaluate on test
    test_metrics = model.evaluate(test_df, test_df, threshold=model.threshold)
    print("\nTest metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save locally
    save_path = os.path.join(PATH, "graph_risk_model.joblib")
    model.save(save_path)
    print(f"\nModel saved to {save_path}")


if __name__ == '__main__':
    import pandas as pd
    from dotenv import load_dotenv
    import os
    load_dotenv()

    PATH = os.getenv('DATA_STORAGE_PATH')
    req_path = os.path.join(PATH,'data_v2.csv') #type: ignore
    df = pd.read_csv(req_path)
    train = df[df['split']=='train']
    model = GraphRiskModel(wandb_project="returnguard")
    model.fit(orders=train,returns_df=train)
    proba = model.predict_proba(orders=test_df,returns_df=test_df)
    explanation = model.get_customer_risk_explanation("C000001",orders=df)
    print(explanation)
    # model = GraphRiskModel.load_from_wandb(project='returnguard')