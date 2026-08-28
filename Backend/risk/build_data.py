"""
Dataset Generator for Return Abuse Detection - Version 2
Behavioral scenario-driven with realistic graph relationships.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import random
from collections import defaultdict
import networkx as nx
import pickle
import json
import os

class ReturnAbuseDatasetGenerator:
    def __init__(self, random_state: int = 42):
        np.random.seed(random_state)
        random.seed(random_state)
        self.random_state = random_state

        # Product categories and their typical prices
        self.categories = {
            'electronics': {'avg_price': 15000, 'return_rate': 0.15},
            'clothing': {'avg_price': 2500, 'return_rate': 0.30},
            'books': {'avg_price': 600, 'return_rate': 0.10},
            'home_appliances': {'avg_price': 12000, 'return_rate': 0.12},
            'beauty': {'avg_price': 1500, 'return_rate': 0.18},
            'sports': {'avg_price': 4000, 'return_rate': 0.14},
            'jewelry': {'avg_price': 20000, 'return_rate': 0.08},
            'grocery': {'avg_price': 800, 'return_rate': 0.05}
        }

        self.payment_methods = ['credit_card', 'debit_card', 'upi', 'net_banking', 'cash_on_delivery']
        self.return_reasons = ['defective', 'not_as_described', 'wrong_item', 'changed_mind',
                               'quality_issue', 'size_issue', 'late_delivery', 'damaged_delivery']

        self.locations = [
            {'city': 'Bengaluru', 'state': 'Karnataka', 'lat': 12.9716, 'lon': 77.5946},
            {'city': 'Mumbai', 'state': 'Maharashtra', 'lat': 19.0760, 'lon': 72.8777},
            {'city': 'Delhi', 'state': 'Delhi', 'lat': 28.7041, 'lon': 77.1025},
            {'city': 'Kolkata', 'state': 'West Bengal', 'lat': 22.5726, 'lon': 88.3639},
            {'city': 'Chennai', 'state': 'Tamil Nadu', 'lat': 13.0827, 'lon': 80.2707},
            {'city': 'Hyderabad', 'state': 'Telangana', 'lat': 17.3850, 'lon': 78.4867},
            {'city': 'Pune', 'state': 'Maharashtra', 'lat': 18.5204, 'lon': 73.8567},
            {'city': 'Ahmedabad', 'state': 'Gujarat', 'lat': 23.0225, 'lon': 72.5714}
        ]

    def generate_dataset(
        self,
        n_customers: int = 50000,
        n_orders: int = 500000,
        abuse_rate: float = 0.08,
        ring_rate: float = 0.03,
        start_date: str = "2025-01-01",
        end_date: str = "2025-12-31",
    ) -> Tuple[pd.DataFrame, nx.Graph]:
        """
        Main dataset generation function.
        Returns:
            dataset: DataFrame with all orders/returns and features
            graph: NetworkX graph of relationships
        """
        print(f"Generating dataset with {n_customers} customers and {n_orders} orders...")
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Step 1: Generate entities (customers and infrastructure)
        print("Step 1: Generating entities...")
        entities = self._generate_entities(n_customers)

        # Step 2: Assign behavioral scenarios (normal, abuser, ring, high-return)
        print("Step 2: Assigning behavioral scenarios...")
        scenarios, rings = self._assign_scenarios(entities, n_customers, abuse_rate, ring_rate)

        # Step 3: Assign devices, IPs, addresses, payments to customers based on scenarios
        print("Step 3: Assigning infrastructure to customers...")
        self._assign_infrastructure(entities, scenarios, rings)

        # Step 4: Generate orders and returns
        print("Step 4: Generating orders and returns...")
        orders, returns = self._generate_transactions(entities, scenarios, n_orders, start, end)

        # Step 5: Build graph from actual transactions
        print("Step 5: Building graph...")
        graph = self._build_graph(orders, entities)

        # Step 6: Calculate graph-based and temporal features
        print("Step 6: Calculating features...")
        features = self._calculate_features(orders, returns, entities, graph)

        # Step 7: Create final dataset
        print("Step 7: Creating final dataset...")
        dataset = self._create_final_dataset(orders, returns, features, entities, scenarios)

        # Step 8: Add temporal split
        print("Step 8: Adding temporal split...")
        dataset = self._add_temporal_split(dataset)

        print("Dataset generation complete!")
        return dataset, graph

    # ----------------------------------------------------------------------
    # Entity generation
    # ----------------------------------------------------------------------
    def _generate_entities(self, n_customers: int) -> Dict:
        """Generate base entities: customers, devices, IPs, addresses, payments."""
        entities = {
            'customers': [],
            'devices': [],
            'ips': [],
            'addresses': [],
            'payments': []
        }

        # Customers
        for i in range(n_customers):
            customer = {
                'customer_id': f'C{str(i+1).zfill(6)}',
                'age': np.random.randint(18, 70),
                'account_creation_date': self._random_date(
                    datetime(2020, 1, 1),
                    datetime(2024, 12, 31)
                ),
                'city': np.random.choice([loc['city'] for loc in self.locations])
            }
            customer['customer_tenure_days'] = (datetime(2025, 12, 31) - customer['account_creation_date']).days
            entities['customers'].append(customer)

        # Devices (5-10% of customer count)
        n_devices = max(1, int(n_customers * np.random.uniform(0.05, 0.10)))
        for i in range(n_devices):
            entities['devices'].append({
                'device_id': f'D{str(i+1).zfill(6)}',
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'])
            })

        # IPs (10-20% of customer count)
        n_ips = max(1, int(n_customers * np.random.uniform(0.10, 0.20)))
        for i in range(n_ips):
            ip_parts = [str(np.random.randint(0, 255)) for _ in range(4)]
            entities['ips'].append({
                'ip_id': f'IP{str(i+1).zfill(6)}',
                'ip_address': '.'.join(ip_parts),
                'city': np.random.choice([loc['city'] for loc in self.locations])
            })

        # Addresses (15-25% of customer count)
        n_addresses = max(1, int(n_customers * np.random.uniform(0.15, 0.25)))
        for i in range(n_addresses):
            loc = np.random.choice(self.locations) #type: ignore
            entities['addresses'].append({
                'address_id': f'A{str(i+1).zfill(6)}',
                'city': loc['city'],
                'state': loc['state'],
                'latitude': loc['lat'] + np.random.normal(0, 0.1),
                'longitude': loc['lon'] + np.random.normal(0, 0.1)
            })

        # Payments (10-15% of customer count)
        n_payments = max(1, int(n_customers * np.random.uniform(0.10, 0.15)))
        for i in range(n_payments):
            entities['payments'].append({
                'payment_id': f'P{str(i+1).zfill(6)}',
                'payment_method': np.random.choice(self.payment_methods),
                'payment_token': hashlib.md5(f"token_{i}".encode()).hexdigest()[:16]
            })

        return entities

    # ----------------------------------------------------------------------
    # Scenario assignment
    # ----------------------------------------------------------------------
    def _assign_scenarios(self, entities: Dict, n_customers: int,
                          abuse_rate: float, ring_rate: float) -> Tuple[Dict, Dict]:
        """
        Assign each customer a scenario: normal, individual_abuser,
        legitimate_high_return, or ring_member.
        Returns:
            scenarios: dict with scenario -> list of customer_ids
            rings: dict ring_id -> list of customer_ids (for ring members)
        """
        customers = entities['customers']
        customer_ids = [c['customer_id'] for c in customers]
        np.random.shuffle(customer_ids)

        n_abusers = int(n_customers * abuse_rate)
        n_ring_members = int(n_abusers * ring_rate)  # ring members are abusers
        n_individual_abusers = n_abusers - n_ring_members

        # We'll also add some legitimate high-return (3-5% of normal)
        n_normal = n_customers - n_abusers
        n_high_return = int(n_normal * np.random.uniform(0.03, 0.05))

        # Allocate
        ring_member_ids = customer_ids[:n_ring_members]
        remaining = customer_ids[n_ring_members:]
        individual_abuser_ids = remaining[:n_individual_abusers]
        remaining = remaining[n_individual_abusers:]
        high_return_ids = remaining[:n_high_return]
        normal_ids = remaining[n_high_return:]

        scenarios = {
            'normal': normal_ids,
            'individual_abuser': individual_abuser_ids,
            'legitimate_high_return': high_return_ids,
            'ring_member': ring_member_ids
        }

        # Create rings: group ring members into rings of size 3-7
        rings = {}
        if ring_member_ids:
            np.random.shuffle(ring_member_ids)
            ring_sizes = np.random.randint(3, 8, size=len(ring_member_ids) // 3 + 1)
            ring_sizes = ring_sizes[ring_sizes <= len(ring_member_ids)]
            idx = 0
            for i, size in enumerate(ring_sizes):
                ring_id = f'RING{str(i+1).zfill(4)}'
                rings[ring_id] = ring_member_ids[idx:idx+size]
                idx += size
            # leftovers: assign to last ring or create new
            if idx < len(ring_member_ids):
                # add to last ring if possible
                if rings:
                    last_ring = list(rings.keys())[-1]
                    rings[last_ring].extend(ring_member_ids[idx:])
                else:
                    # shouldn't happen
                    pass

        return scenarios, rings

    # ----------------------------------------------------------------------
    # Infrastructure assignment based on scenarios
    # ----------------------------------------------------------------------
    def _assign_infrastructure(self, entities: Dict, scenarios: Dict, rings: Dict):
        """
        For each customer, assign a set of devices, IPs, addresses, payments.
        For ring members, share some infrastructure within the ring.
        """
        customers = entities['customers']
        devices = entities['devices']
        ips = entities['ips']
        addresses = entities['addresses']
        payments = entities['payments']

        # Pre-shuffle entity lists for random selection
        np.random.shuffle(devices)
        np.random.shuffle(ips)
        np.random.shuffle(addresses)
        np.random.shuffle(payments)

        # Keep track of assignments per customer
        for cust in customers:
            cust_id = cust['customer_id']
            # Default: each customer gets 1-2 devices, 1-3 IPs, 1-2 addresses, 1-2 payments
            cust['devices'] = np.random.choice(devices, size=np.random.randint(1, 3), replace=False).tolist()
            cust['ips'] = np.random.choice(ips, size=np.random.randint(1, 4), replace=False).tolist()
            cust['addresses'] = np.random.choice(addresses, size=np.random.randint(1, 3), replace=False).tolist()
            cust['payments'] = np.random.choice(payments, size=np.random.randint(1, 3), replace=False).tolist()

        # For ring members, override with shared infrastructure
        for ring_id, member_ids in rings.items():
            # Decide how many shared devices, IPs, addresses, payments for this ring
            n_shared_devices = np.random.randint(1, 3)
            n_shared_ips = np.random.randint(1, 3)
            n_shared_addresses = np.random.randint(1, 2)
            n_shared_payments = np.random.randint(1, 2)

            # Select shared entities (could be a subset of existing entities)
            shared_devices = np.random.choice(devices, size=min(n_shared_devices, len(devices)), replace=False).tolist()
            shared_ips = np.random.choice(ips, size=min(n_shared_ips, len(ips)), replace=False).tolist()
            shared_addresses = np.random.choice(addresses, size=min(n_shared_addresses, len(addresses)), replace=False).tolist()
            shared_payments = np.random.choice(payments, size=min(n_shared_payments, len(payments)), replace=False).tolist()

            for cust_id in member_ids:
                # Find customer object
                cust = next(c for c in customers if c['customer_id'] == cust_id)
                # Replace some of their entities with shared ones
                # Each ring member gets a mix of personal and shared entities
                # We'll keep some personal and add shared
                # Keep at least 1 personal of each type, and add shared ones
                personal_devices = cust['devices']
                personal_ips = cust['ips']
                personal_addresses = cust['addresses']
                personal_payments = cust['payments']

                # Combine: personal + shared, but ensure some shared are present
                # For devices: keep 1 personal, add shared
                new_devices = personal_devices[:1] + shared_devices
                # If too many, limit to 4
                cust['devices'] = new_devices[:4] if len(new_devices) > 4 else new_devices

                new_ips = personal_ips[:1] + shared_ips
                cust['ips'] = new_ips[:5] if len(new_ips) > 5 else new_ips

                new_addresses = personal_addresses[:1] + shared_addresses
                cust['addresses'] = new_addresses[:3] if len(new_addresses) > 3 else new_addresses

                new_payments = personal_payments[:1] + shared_payments
                cust['payments'] = new_payments[:3] if len(new_payments) > 3 else new_payments

                # Also store ring_id for later
                cust['ring_id'] = ring_id

        # For non-ring members, ensure ring_id is None
        for cust in customers:
            if 'ring_id' not in cust:
                cust['ring_id'] = None

    # ----------------------------------------------------------------------
    # Generate orders and returns
    # ----------------------------------------------------------------------
    def _generate_transactions(self, entities: Dict, scenarios: Dict,
                               n_orders: int, start: datetime, end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate orders and returns based on scenario parameters.
        Each order randomly picks one of the customer's assigned entities.
        """
        customers = {c['customer_id']: c for c in entities['customers']}
        orders_data = []
        returns_data = []

        # Define scenario parameters
        scenario_params = {}
        for cust_id in scenarios['normal']:
            scenario_params[cust_id] = {
                'return_rate': np.clip(np.random.normal(0.08, 0.04), 0.01, 0.30),
                'order_frequency': max(1, int(np.random.normal(10, 3))),
                'avg_order_value': max(100, np.random.normal(2000, 500)),
                'suspicious': False,
                'legit_high_return': False
            }
        for cust_id in scenarios['individual_abuser']:
            scenario_params[cust_id] = {
                'return_rate': np.clip(np.random.uniform(0.70, 0.95), 0.50, 0.95),
                'order_frequency': max(1, int(np.random.normal(15, 5))),
                'avg_order_value': max(500, np.random.uniform(5000, 30000)),
                'suspicious': True,
                'legit_high_return': False
            }
        for cust_id in scenarios['legitimate_high_return']:
            scenario_params[cust_id] = {
                'return_rate': np.clip(np.random.uniform(0.60, 0.80), 0.40, 0.85),
                'order_frequency': max(1, int(np.random.normal(20, 5))),
                'avg_order_value': max(100, np.random.normal(2500, 600)),
                'suspicious': False,
                'legit_high_return': True
            }
        for cust_id in scenarios['ring_member']:
            scenario_params[cust_id] = {
                'return_rate': np.clip(np.random.uniform(0.60, 0.85), 0.50, 0.95),
                'order_frequency': max(1, int(np.random.normal(12, 4))),
                'avg_order_value': max(500, np.random.uniform(3000, 25000)),
                'suspicious': True,
                'legit_high_return': False,
                'is_ring_member': True
            }

        # Generate orders per customer
        order_counter = 0
        for cust_id, customer in customers.items():
            params = scenario_params.get(cust_id)
            if params is None:
                # Fallback for any missing customer (should not happen)
                params = {
                    'return_rate': 0.08,
                    'order_frequency': 5,
                    'avg_order_value': 2000,
                    'suspicious': False,
                    'legit_high_return': False
                }
            n_orders_for_customer = max(1, np.random.poisson(params['order_frequency']))

            # Cap total orders
            if order_counter + n_orders_for_customer > n_orders:
                n_orders_for_customer = max(1, n_orders - order_counter)
                if n_orders_for_customer <= 0:
                    break

            # Pick entities for this customer
            cust_devices = [d['device_id'] for d in customer['devices']]
            cust_ips = [ip['ip_id'] for ip in customer['ips']]
            cust_addresses = [a['address_id'] for a in customer['addresses']]
            cust_payments = [p['payment_id'] for p in customer['payments']]

            for i in range(n_orders_for_customer):
                order_date = self._random_date(start, end)
                category = np.random.choice(list(self.categories.keys()))
                cat_info = self.categories[category]

                # Order amount
                base_value = np.random.lognormal(np.log(max(100, params['avg_order_value'])), 0.5)
                if params['suspicious'] and not params['legit_high_return']:
                    base_value *= np.random.uniform(1.5, 3.0)
                order_amount = max(50, base_value)

                # Randomly pick one of each entity type
                device_id = np.random.choice(cust_devices) if cust_devices else None
                ip_id = np.random.choice(cust_ips) if cust_ips else None
                address_id = np.random.choice(cust_addresses) if cust_addresses else None
                payment_id = np.random.choice(cust_payments) if cust_payments else None

                order = {
                    'order_id': f'O{str(order_counter+1).zfill(8)}',
                    'customer_id': cust_id,
                    'purchase_date': order_date,
                    'order_amount': order_amount,
                    'product_category': category,
                    'discount_percentage': max(0, min(30, np.random.uniform(0, 30))),
                    'payment_method': np.random.choice(self.payment_methods),
                    'device_id': device_id,
                    'ip_id': ip_id,
                    'address_id': address_id,
                    'payment_id': payment_id,
                    'delivery_date': order_date + timedelta(days=np.random.randint(1, 7))
                }
                orders_data.append(order)
                order_counter += 1

        orders_df = pd.DataFrame(orders_data)

        # Generate returns based on return rate
        for _, order in orders_df.iterrows():
            cust_id = order['customer_id']
            params = scenario_params.get(cust_id)
            if params is None:
                continue
            return_prob = params['return_rate']

            # Slight randomness: not all orders with high return rate are returned
            if np.random.random() < return_prob:
                purchase_date = order['purchase_date']
                days_to_return = max(1, min(29, int(np.random.normal(14, 7))))
                return_date = purchase_date + timedelta(days=days_to_return)
                if return_date > order['delivery_date']:
                    refund_amount = order['order_amount'] * max(0.5, min(1.0, np.random.uniform(0.7, 1.0)))

                    # Determine abuse label: for abusers (individual or ring), most returns are abusive (85%)
                    if cust_id in scenarios['ring_member'] or cust_id in scenarios['individual_abuser']:
                        abuse_label = 1 if np.random.random() < 0.85 else 0
                        abuse_type = 'coordinated_ring' if cust_id in scenarios['ring_member'] else 'individual_abuse'
                    elif cust_id in scenarios['legitimate_high_return']:
                        abuse_label = 0
                        abuse_type = 'legitimate'
                    else:
                        # normal customers, small chance of false positive
                        abuse_label = 1 if np.random.random() < 0.02 else 0
                        abuse_type = 'legitimate' if abuse_label == 0 else 'individual_abuse'

                    return_record = {
                        'return_id': f'R{str(len(returns_data)+1).zfill(8)}',
                        'order_id': order['order_id'],
                        'customer_id': cust_id,
                        'return_date': return_date,
                        'return_reason': np.random.choice(self.return_reasons),
                        'refund_amount': refund_amount,
                        'abuse_label': abuse_label,
                        'abuse_type': abuse_type
                    }
                    returns_data.append(return_record)

        return orders_df, pd.DataFrame(returns_data)

    # ----------------------------------------------------------------------
    # Build graph from transactions
    # ----------------------------------------------------------------------
    def _build_graph(self, orders: pd.DataFrame, entities: Dict) -> nx.Graph:
        """Build graph with nodes: customers, devices, IPs, addresses, payments."""
        G = nx.Graph()

        # Add all nodes
        for cust in entities['customers']:
            G.add_node(cust['customer_id'], type='customer')
        for dev in entities['devices']:
            G.add_node(dev['device_id'], type='device')
        for ip in entities['ips']:
            G.add_node(ip['ip_id'], type='ip')
        for addr in entities['addresses']:
            G.add_node(addr['address_id'], type='address')
        for pay in entities['payments']:
            G.add_node(pay['payment_id'], type='payment')

        # Add edges based on orders (only if both nodes exist)
        for _, order in orders.iterrows():
            cust = order['customer_id']
            for col,node_type in [('device_id', 'device'),
                               ('ip_id', 'ip'),
                               ('address_id', 'address'),
                               ('payment_id', 'payment')]:
                node_id = order[col]
                if pd.notna(node_id) and node_id in G:
                    # Edge weight: count of times this customer used this entity
                    if G.has_edge(cust, node_id):
                        G[cust][node_id]['weight'] += 1
                    else:
                        G.add_edge(cust, node_id, weight=1)

        return G

    # ----------------------------------------------------------------------
    # Feature calculation (graph-based + temporal)
    # ----------------------------------------------------------------------
    def _calculate_features(self, orders: pd.DataFrame, returns: pd.DataFrame,
                            entities: Dict, graph: nx.Graph) -> pd.DataFrame:
        """
        Compute features per customer:
        - Temporal: order/return counts, amounts, rates in windows
        - Graph-based: shared entity counts, degree, etc.
        """
        all_customer_ids = [c['customer_id'] for c in entities['customers']]
        features_list = []

        # Pre-group orders and returns
        orders_group = orders.groupby('customer_id')
        returns_group = returns.groupby('customer_id')

        for cust_id in all_customer_ids:
            cust_orders = orders[orders['customer_id'] == cust_id]
            cust_returns = returns[returns['customer_id'] == cust_id]

            # Basic counts
            n_orders = len(cust_orders)
            n_returns = len(cust_returns)
            total_refund = cust_returns['refund_amount'].sum() if n_returns > 0 else 0
            avg_order_value = cust_orders['order_amount'].mean() if n_orders > 0 else 0
            avg_return_value = cust_returns['refund_amount'].mean() if n_returns > 0 else 0

            # Historical return rate (over all time)
            hist_return_rate = n_returns / max(1, n_orders)

            # Time-window features (relative to last order date)
            if n_orders > 0:
                max_date = cust_orders['purchase_date'].max()
                last_7d = max_date - timedelta(days=7)
                last_30d = max_date - timedelta(days=30)
                last_90d = max_date - timedelta(days=90)

                orders_7d = cust_orders[cust_orders['purchase_date'] >= last_7d]
                orders_30d = cust_orders[cust_orders['purchase_date'] >= last_30d]
                orders_90d = cust_orders[cust_orders['purchase_date'] >= last_90d]

                returns_7d = cust_returns[cust_returns['return_date'] >= last_7d] if n_returns > 0 else pd.DataFrame()
                returns_30d = cust_returns[cust_returns['return_date'] >= last_30d] if n_returns > 0 else pd.DataFrame()
                returns_90d = cust_returns[cust_returns['return_date'] >= last_90d] if n_returns > 0 else pd.DataFrame()

                n_orders_7d = len(orders_7d)
                n_orders_30d = len(orders_30d)
                n_orders_90d = len(orders_90d)
                n_returns_7d = len(returns_7d)
                n_returns_30d = len(returns_30d)
                n_returns_90d = len(returns_90d)

                refund_30d = returns_30d['refund_amount'].sum() if n_returns_30d > 0 else 0
                refund_90d = returns_90d['refund_amount'].sum() if n_returns_90d > 0 else 0

                return_rate_30d = n_returns_30d / max(1, n_orders_30d)
                return_rate_90d = n_returns_90d / max(1, n_orders_90d)
            else:
                n_orders_7d = n_orders_30d = n_orders_90d = 0
                n_returns_7d = n_returns_30d = n_returns_90d = 0
                refund_30d = refund_90d = 0
                return_rate_30d = return_rate_90d = 0

            # Graph-based features
            # Get neighbors of this customer (devices, IPs, addresses, payments)
            if cust_id in graph:
                neighbors = list(graph.neighbors(cust_id))
                # Count number of shared entities per type
                shared_device_count = 0
                shared_ip_count = 0
                shared_address_count = 0
                shared_payment_count = 0
                total_degree = len(neighbors)

                for nb in neighbors:
                    if graph.nodes[nb].get('type') == 'device':
                        shared_device_count += 1
                    elif graph.nodes[nb].get('type') == 'ip':
                        shared_ip_count += 1
                    elif graph.nodes[nb].get('type') == 'address':
                        shared_address_count += 1
                    elif graph.nodes[nb].get('type') == 'payment':
                        shared_payment_count += 1
            else:
                shared_device_count = shared_ip_count = shared_address_count = shared_payment_count = 0
                total_degree = 0

            # Feature dict
            feature = {
                'customer_id': cust_id,
                'number_of_previous_orders': n_orders,
                'number_of_previous_returns': n_returns,
                'historical_return_rate': hist_return_rate,
                'historical_refund_amount': total_refund,
                'avg_order_value': avg_order_value,
                'avg_return_value': avg_return_value,
                'orders_last_7_days': n_orders_7d,
                'orders_last_30_days': n_orders_30d,
                'orders_last_90_days': n_orders_90d,
                'returns_last_7_days': n_returns_7d,
                'returns_last_30_days': n_returns_30d,
                'returns_last_90_days': n_returns_90d,
                'refund_amount_30d': refund_30d,
                'refund_amount_90d': refund_90d,
                'return_rate_30d': return_rate_30d,
                'return_rate_90d': return_rate_90d,
                'graph_degree': total_degree,
                'shared_device_count': shared_device_count,
                'shared_ip_count': shared_ip_count,
                'shared_address_count': shared_address_count,
                'shared_payment_count': shared_payment_count,
            }
            features_list.append(feature)

        return pd.DataFrame(features_list)

    # ----------------------------------------------------------------------
    # Create final dataset
    # ----------------------------------------------------------------------
    def _create_final_dataset(self, orders: pd.DataFrame, returns: pd.DataFrame,
                              features: pd.DataFrame, entities: Dict,
                              scenarios: Dict) -> pd.DataFrame:
        """
        Merge orders, returns, features, and customer demographics.
        Also include scenario and ring_id as metadata (not features).
        """
        # Start with returns
        dataset = returns.copy()

        # Add order-level info
        order_cols = ['order_id', 'purchase_date', 'order_amount', 'product_category',
                      'discount_percentage', 'payment_method', 'device_id', 'ip_id',
                      'address_id', 'payment_id']
        dataset = dataset.merge(orders[order_cols], on='order_id', how='left')

        # Add customer features
        dataset = dataset.merge(features, on='customer_id', how='left')

        # Add customer demographics
        cust_info = pd.DataFrame(entities['customers'])
        cust_info = cust_info[['customer_id', 'age', 'account_creation_date', 'customer_tenure_days', 'city', 'ring_id']]
        dataset = dataset.merge(cust_info, on='customer_id', how='left')

        # Add scenario as metadata
        scenario_map = {}
        for scenario, ids in scenarios.items():
            for cid in ids:
                scenario_map[cid] = scenario
        dataset['scenario'] = dataset['customer_id'].map(scenario_map)

        # Calculate days_to_return
        dataset['days_to_return'] = (dataset['return_date'] - dataset['purchase_date']).dt.days

        # Add geographical features (simplified)
        dataset['latitude'] = np.random.uniform(8, 37, len(dataset))
        dataset['longitude'] = np.random.uniform(68, 97, len(dataset))
        dataset['state'] = np.random.choice([loc['state'] for loc in self.locations], len(dataset))
        dataset['country'] = 'India'

        return dataset

    # ----------------------------------------------------------------------
    # Temporal split
    # ----------------------------------------------------------------------
    def _add_temporal_split(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add split column based on return_date: train (Jan-Sep), validation (Oct), test (Nov-Dec)."""
        if len(dataset) == 0:
            dataset['split'] = 'train'
            return dataset

        # Sort by return_date
        dataset = dataset.sort_values('return_date').reset_index(drop=True)
        n = len(dataset)
        train_idx = int(n * 0.75)   # Jan-Sep
        val_idx = int(n * 0.85)     # Jan-Oct

        dataset['split'] = 'train'
        dataset.loc[train_idx:val_idx-1, 'split'] = 'validation'
        dataset.loc[val_idx:, 'split'] = 'test'
        return dataset

    # ----------------------------------------------------------------------
    # Helper
    # ----------------------------------------------------------------------
    def _random_date(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        return start + timedelta(days=random_days)


# --------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------
def generate_dataset(
    n_customers: int = 50000,
    n_orders: int = 500000,
    abuse_rate: float = 0.08,
    ring_rate: float = 0.03,
    random_state: int = 42
) -> Tuple[pd.DataFrame, nx.Graph]:
    """Wrapper function."""
    generator = ReturnAbuseDatasetGenerator(random_state)
    return generator.generate_dataset(
        n_customers=n_customers,
        n_orders=n_orders,
        abuse_rate=abuse_rate,
        ring_rate=ring_rate
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import os

    PATH = os.getenv('DATA_STORAGE_PATH')
    if not PATH:
        PATH = './data'

    # Generate a dataset (small for testing, adjust for final)
    dataset, graph = generate_dataset(
        n_customers=1000,
        n_orders=10000,
        abuse_rate=0.08,
        ring_rate=0.03
    )

    # Save dataset
    dataset.to_csv(os.path.join(PATH, 'data_v2.csv'), index=False)

    # Save graph as JSON
    graph_json = {
        'nodes': [{'id': str(n), 'type': data.get('type', 'unknown')}
                  for n, data in graph.nodes(data=True)],
        'edges': [{'source': str(u), 'target': str(v), 'weight': data.get('weight', 1)}
                  for u, v, data in graph.edges(data=True)]
    }
    with open(os.path.join(PATH, 'return_abuse_graph_v2.json'), 'w') as f:
        json.dump(graph_json, f, indent=2)

    print(f"Dataset shape: {dataset.shape}")
    print("Abuse distribution:")
    print(dataset['abuse_label'].value_counts())
    print("Split distribution:")
    print(dataset['split'].value_counts())
    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())