"""
Unit tests for build_data.py (Return Abuse Dataset Generator)
"""

import pytest
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime
from Backend.risk.build_data import ReturnAbuseDatasetGenerator, generate_dataset


class TestReturnAbuseDatasetGenerator:
    """Test all major components of the dataset generator."""

    def test_entity_generation(self):
        """Check that entities are created with correct structure."""
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        entities = gen._generate_entities(100)

        assert len(entities['customers']) == 100
        assert len(entities['devices']) > 0
        assert len(entities['ips']) > 0
        assert len(entities['addresses']) > 0
        assert len(entities['payments']) > 0

        for cust in entities['customers']:
            assert 'customer_id' in cust
            assert 'age' in cust
            assert 'account_creation_date' in cust
            assert isinstance(cust['account_creation_date'], datetime)

    def test_scenario_assignment(self):
        """Verify scenario distributions and ring creation."""
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        entities = gen._generate_entities(100)
        scenarios, rings = gen._assign_scenarios(
            entities, 100, abuse_rate=0.08, ring_rate=0.03
        )

        total = sum(len(v) for v in scenarios.values())
        assert total == 100
        assert set(scenarios.keys()) == {
            'normal', 'individual_abuser', 'legitimate_high_return', 'ring_member'
        }

        # Rings should have 3-7 members
        for ring_id, members in rings.items():
            assert 3 <= len(members) <= 7

        # All ring members must appear in the ring_member scenario
        ring_member_ids = set(scenarios['ring_member'])
        for members in rings.values():
            for m in members:
                assert m in ring_member_ids

    def test_infrastructure_assignment(self):
        """Ensure each customer gets entities, and ring members share infrastructure."""
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        entities = gen._generate_entities(50)
        scenarios, rings = gen._assign_scenarios(
            entities, 50, abuse_rate=0.20, ring_rate=0.30
        )
        gen._assign_infrastructure(entities, scenarios, rings)

        # Every customer must have at least one of each entity type
        for cust in entities['customers']:
            assert 'devices' in cust and len(cust['devices']) > 0
            assert 'ips' in cust and len(cust['ips']) > 0
            assert 'addresses' in cust and len(cust['addresses']) > 0
            assert 'payments' in cust and len(cust['payments']) > 0

        # For each ring, check that members share at least one device or IP
        for ring_id, members in rings.items():
            first = members[0]
            first_cust = next(c for c in entities['customers'] if c['customer_id'] == first)
            first_devices = {d['device_id'] for d in first_cust['devices']}
            first_ips = {ip['ip_id'] for ip in first_cust['ips']}
            first_addresses = {a['address_id'] for a in first_cust['addresses']}
            first_payments = {p['payment_id'] for p in first_cust['payments']}

            for member in members[1:]:
                mc = next(c for c in entities['customers'] if c['customer_id'] == member)
                m_devices = {d['device_id'] for d in mc['devices']}
                m_ips = {ip['ip_id'] for ip in mc['ips']}
                m_addresses = {a['address_id'] for a in mc['addresses']}
                m_payments = {p['payment_id'] for p in mc['payments']}

                shared = (
                    len(first_devices & m_devices) +
                    len(first_ips & m_ips) +
                    len(first_addresses & m_addresses) +
                    len(first_payments & m_payments)
                )
                assert shared > 0, f"No shared entity between {first} and {member}"

    def test_full_generation_small(self):
        """Smoke test: generate a tiny dataset and validate structure."""
        dataset, graph = generate_dataset(
            n_customers=50,
            n_orders=100,
            abuse_rate=0.08,
            ring_rate=0.03,
            random_state=42
        )

        assert isinstance(dataset, pd.DataFrame)
        assert isinstance(graph, nx.Graph)
        assert len(dataset) > 0

        # Required columns (including metadata)
        required = [
            'return_id', 'order_id', 'customer_id', 'return_date',
            'return_reason', 'refund_amount', 'abuse_label', 'abuse_type',
            'split', 'scenario', 'ring_id'
        ]
        for col in required:
            assert col in dataset.columns

        # Abuse labels are binary
        assert set(dataset['abuse_label'].unique()).issubset({0, 1})

        # Temporal split exists
        assert set(dataset['split'].unique()).issubset({'train', 'validation', 'test'})

        # Graph contains all customers as nodes
        customer_ids = set(dataset['customer_id'].unique())
        graph_customers = {
            n for n, data in graph.nodes(data=True)
            if data.get('type') == 'customer'
        }
        assert customer_ids == graph_customers
        assert graph.number_of_edges() > 0

    def test_temporal_split(self):
        """Check that temporal split proportions are roughly correct."""
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        df = pd.DataFrame({'return_date': dates, 'dummy': range(100)})
        df = gen._add_temporal_split(df)

        assert 'split' in df.columns
        assert (df['split'] == 'train').sum() >= 75
        assert (df['split'] == 'validation').sum() >= 10
        assert (df['split'] == 'test').sum() >= 15

    def test_build_graph_no_keyerror(self):
        """
        Regression test: ensure _build_graph does not raise KeyError
        when iterating over columns.
        """
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        entities = gen._generate_entities(20)
        scenarios, rings = gen._assign_scenarios(
            entities, 20, abuse_rate=0.20, ring_rate=0.30
        )
        gen._assign_infrastructure(entities, scenarios, rings)

        # Create a small orders DataFrame manually
        orders_data = []
        for cust in entities['customers']:
            for _ in range(2):
                order = {
                    'order_id': f'O{len(orders_data)+1:08d}',
                    'customer_id': cust['customer_id'],
                    'device_id': np.random.choice([d['device_id'] for d in cust['devices']]),
                    'ip_id': np.random.choice([ip['ip_id'] for ip in cust['ips']]),
                    'address_id': np.random.choice([a['address_id'] for a in cust['addresses']]),
                    'payment_id': np.random.choice([p['payment_id'] for p in cust['payments']]),
                }
                orders_data.append(order)

        orders_df = pd.DataFrame(orders_data)
        graph = gen._build_graph(orders_df, entities)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_returns_generation_logic(self):
        """Verify that return generation respects scenario parameters."""
        gen = ReturnAbuseDatasetGenerator(random_state=42)
        entities = gen._generate_entities(30)
        scenarios, rings = gen._assign_scenarios(
            entities, 30, abuse_rate=0.30, ring_rate=0.20
        )
        gen._assign_infrastructure(entities, scenarios, rings)

        start = datetime(2025, 1, 1)
        end = datetime(2025, 12, 31)
        orders, returns = gen._generate_transactions(
            entities, scenarios, n_orders=50, start=start, end=end
        )

        # At least some returns should be generated
        assert len(returns) > 0

        # Check that abusers have higher return rates (heuristic)
        abuser_ids = set(scenarios['individual_abuser'] + scenarios['ring_member'])
        normal_ids = set(scenarios['normal'])

        # Compute per-customer return rate from generated returns
        ret_counts = returns.groupby('customer_id').size()
        ord_counts = orders.groupby('customer_id').size()

        abuser_return_rates = []
        normal_return_rates = []
        for cust_id in set(orders['customer_id']):
            n_ord = ord_counts.get(cust_id, 0)
            n_ret = ret_counts.get(cust_id, 0)
            rate = n_ret / max(1, n_ord)
            if cust_id in abuser_ids:
                abuser_return_rates.append(rate)
            else:
                normal_return_rates.append(rate)

        # On average, abusers should return more (allow some noise)
        if abuser_return_rates and normal_return_rates:
            assert np.mean(abuser_return_rates) > np.mean(normal_return_rates)