"""
Dataset Generator for Return Abuse Detection
Based on the "dataset-first" approach with behavioral scenario generation
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import hashlib
import random
from collections import defaultdict
import networkx as nx
from sklearn.model_selection import train_test_split
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
        
        # Payment methods
        self.payment_methods = ['credit_card', 'debit_card', 'upi', 'net_banking', 'cash_on_delivery']
        
        # Return reasons
        self.return_reasons = ['defective', 'not_as_described', 'wrong_item', 'changed_mind', 
                              'quality_issue', 'size_issue', 'late_delivery', 'damaged_delivery']
        
        # Cities and states for geographical data
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
        Main dataset generation function
        """
        print(f"Generating dataset with {n_customers} customers and {n_orders} orders...")
        
        # Step 1: Generate entities
        print("Step 1: Generating entities...")
        entities = self._generate_entities(n_customers)
        
        # Step 2: Generate behavioral scenarios
        print("Step 2: Generating behavioral scenarios...")
        scenarios = self._assign_scenarios(
            entities, 
            n_customers, 
            abuse_rate, 
            ring_rate
        )
        
        # Step 3: Generate orders and returns
        print("Step 3: Generating orders and returns...")
        orders, returns = self._generate_transactions(
            entities, 
            scenarios, 
            n_orders, 
            start_date, 
            end_date
        )
        
        # Step 4: Calculate historical features
        print("Step 4: Calculating features...")
        features = self._calculate_features(orders, returns, entities, scenarios)
        
        # Step 5: Build graph
        print("Step 5: Building graph...")
        graph = self._build_graph(entities, orders)
        
        # Step 6: Create final dataset
        print("Step 6: Creating final dataset...")
        dataset = self._create_final_dataset(orders, returns, features, entities)
        
        # Step 7: Temporal split
        print("Step 7: Creating temporal split...")
        dataset = self._add_temporal_split(dataset)
        
        print("Dataset generation complete!")
        return dataset, graph

    def _generate_entities(self, n_customers: int) -> Dict:
        """Generate base entities: customers, devices, IPs, addresses, payments"""
        entities = {
            'customers': [],
            'devices': [],
            'ips': [],
            'addresses': [],
            'payments': []
        }
        
        # Generate customers
        for i in range(n_customers):
            customer = {
                'customer_id': f'C{str(i+1).zfill(6)}',
                'age': np.random.randint(18, 70),
                'account_creation_date': self._random_date(
                    datetime(2020, 1, 1), 
                    datetime(2024, 12, 31)
                ),
                'city': np.random.choice(self.locations)['city'] #type: ignore
            }
            # Calculate tenure
            customer['customer_tenure_days'] = (
                datetime(2025, 12, 31) - customer['account_creation_date']
            ).days
            entities['customers'].append(customer)
        
        # Generate devices (5-10% of customer count)
        n_devices = max(1, int(n_customers * np.random.uniform(0.05, 0.10)))
        for i in range(n_devices):
            entities['devices'].append({
                'device_id': f'D{str(i+1).zfill(6)}',
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet'])
            })
        
        # Generate IPs (10-20% of customer count)
        n_ips = max(1, int(n_customers * np.random.uniform(0.10, 0.20)))
        for i in range(n_ips):
            ip_parts = [str(np.random.randint(0, 255)) for _ in range(4)]
            entities['ips'].append({
                'ip_id': f'IP{str(i+1).zfill(6)}',
                'ip_address': '.'.join(ip_parts),
                'city': np.random.choice(self.locations)['city'] #type: ignore
            })
        
        # Generate addresses (15-25% of customer count)
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
        
        # Generate payments (10-15% of customer count)
        n_payments = max(1, int(n_customers * np.random.uniform(0.10, 0.15)))
        for i in range(n_payments):
            entities['payments'].append({
                'payment_id': f'P{str(i+1).zfill(6)}',
                'payment_method': np.random.choice(self.payment_methods),
                'payment_token': hashlib.md5(f"token_{i}".encode()).hexdigest()[:16]
            })
        
        return entities

    def _assign_scenarios(
        self, 
        entities: Dict, 
        n_customers: int, 
        abuse_rate: float,
        ring_rate: float
    ) -> Dict:
        """Assign behavioral scenarios to customers"""
        customers = entities['customers']
        n_abusers = int(n_customers * abuse_rate)
        n_ring_members = int(n_abusers * ring_rate)
        
        scenarios = {
            'normal': [],
            'individual_abuser': [],
            'ring_member': [],
            'legitimate_high_return': []
        }
        
        # Shuffle customers for assignment
        shuffled_customers = customers.copy()
        np.random.shuffle(shuffled_customers)
        
        # Assign ring members (they're also abusers)
        ring_members = shuffled_customers[:n_ring_members]
        for customer in ring_members:
            scenarios['ring_member'].append(customer['customer_id'])
        
        # Assign individual abusers
        remaining_customers = shuffled_customers[n_ring_members:]
        n_individual_abusers = n_abusers - n_ring_members
        individual_abusers = remaining_customers[:n_individual_abusers]
        for customer in individual_abusers:
            scenarios['individual_abuser'].append(customer['customer_id'])
        
        # Assign legitimate high-return customers (3-5% of normal customers)
        remaining_customers = remaining_customers[n_individual_abusers:]
        n_high_return = int(len(remaining_customers) * np.random.uniform(0.03, 0.05))
        legitimate_high_return = remaining_customers[:n_high_return]
        for customer in legitimate_high_return:
            scenarios['legitimate_high_return'].append(customer['customer_id'])
        
        # Rest are normal customers
        remaining_customers = remaining_customers[n_high_return:]
        for customer in remaining_customers:
            scenarios['normal'].append(customer['customer_id'])
        
        return scenarios

    def _generate_transactions(
        self, 
        entities: Dict, 
        scenarios: Dict, 
        n_orders: int,
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate orders and returns based on scenarios"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        customers = {c['customer_id']: c for c in entities['customers']}
        devices = entities['devices']
        ips = entities['ips']
        addresses = entities['addresses']
        payments = entities['payments']
        
        orders_data = []
        returns_data = []
        
        # Create scenario parameters with validation
        scenario_params = {}
        for customer_id in scenarios['normal']:
            # Ensure positive values with minimum thresholds
            scenario_params[customer_id] = {
                'return_rate': max(0.01, min(0.30, np.random.normal(0.08, 0.04))),
                'order_frequency': max(1, int(np.random.normal(10, 3))),
                'avg_order_value': max(100, np.random.normal(2000, 500)),
                'suspicious_behavior': False
            }
        
        for customer_id in scenarios['individual_abuser']:
            scenario_params[customer_id] = {
                'return_rate': max(0.50, min(0.95, np.random.uniform(0.70, 0.90))),
                'order_frequency': max(1, int(np.random.normal(15, 5))),
                'avg_order_value': max(500, np.random.uniform(5000, 30000)),
                'suspicious_behavior': True
            }
        
        for customer_id in scenarios['ring_member']:
            scenario_params[customer_id] = {
                'return_rate': max(0.50, min(0.95, np.random.uniform(0.60, 0.85))),
                'order_frequency': max(1, int(np.random.normal(12, 4))),
                'avg_order_value': max(500, np.random.uniform(3000, 25000)),
                'suspicious_behavior': True,
                'is_ring_member': True
            }
        
        for customer_id in scenarios['legitimate_high_return']:
            scenario_params[customer_id] = {
                'return_rate': max(0.40, min(0.85, np.random.uniform(0.60, 0.80))),
                'order_frequency': max(1, int(np.random.normal(20, 5))),
                'avg_order_value': max(100, np.random.normal(2500, 600)),
                'suspicious_behavior': False,
                'is_legitimate_high_return': True
            }
        
        # Ensure all customers have parameters with valid values
        for customer_id in customers:
            if customer_id not in scenario_params:
                # Generate valid parameters for normal customers
                scenario_params[customer_id] = {
                    'return_rate': max(0.01, min(0.30, np.random.normal(0.08, 0.04))),
                    'order_frequency': max(1, int(np.random.normal(10, 3))),
                    'avg_order_value': max(100, np.random.normal(2000, 500)),
                    'suspicious_behavior': False
                }
        
        # Generate orders
        order_counter = 0
        for customer_id, customer in customers.items():
            params = scenario_params[customer_id]
            
            # Ensure order_frequency is a positive integer
            order_frequency = max(1, int(params['order_frequency']))
            n_orders_for_customer = max(1, np.random.poisson(order_frequency))
            
            # Ensure total orders constraint
            if order_counter + n_orders_for_customer > n_orders:
                break
            
            for i in range(n_orders_for_customer):
                order_date = self._random_date(start, end)
                category = np.random.choice(list(self.categories.keys()))
                category_info = self.categories[category]
                
                # Adjust order value based on scenario
                avg_value = params['avg_order_value']
                base_value = np.random.lognormal(np.log(max(100, avg_value)), 0.5)
                
                if params['suspicious_behavior'] and not params.get('is_legitimate_high_return', False):
                    # Abusers tend to order higher value items
                    base_value *= np.random.uniform(1.5, 3.0)
                
                # Ensure order amount is reasonable
                order_amount = max(50, base_value)
                
                order = {
                    'order_id': f'O{str(order_counter+1).zfill(8)}',
                    'customer_id': customer_id,
                    'purchase_date': order_date,
                    'order_amount': order_amount,
                    'product_category': category,
                    'discount_percentage': max(0, min(30, np.random.uniform(0, 30))),
                    'payment_method': np.random.choice(self.payment_methods),
                    'device_id': np.random.choice(devices)['device_id'] if devices else None,
                    'ip_id': np.random.choice(ips)['ip_id'] if ips else None,
                    'address_id': np.random.choice(addresses)['address_id'] if addresses else None,
                    'payment_id': np.random.choice(payments)['payment_id'] if payments else None,
                    'delivery_date': order_date + timedelta(days=np.random.randint(1, 7))
                }
                orders_data.append(order)
                order_counter += 1
        
        # Generate returns based on return rates
        for order in orders_data:
            customer_id = order['customer_id']
            params = scenario_params[customer_id]
            
            # Determine if this order is returned
            return_probability = max(0.01, min(0.95, params['return_rate']))
            
            if params.get('is_legitimate_high_return', False):
                # High return customers return often but legitimately
                return_probability = max(0.50, min(0.85, np.random.uniform(0.60, 0.80)))
            
            if np.random.random() < return_probability:
                purchase_date = order['purchase_date']
                days_to_return = max(1, min(29, int(np.random.normal(14, 7))))  # Ensure between 1-29 days
                return_date = purchase_date + timedelta(days=days_to_return)
                
                # Ensure return is after delivery
                if return_date > order['delivery_date']:
                    refund_amount = order['order_amount'] * max(0.5, min(1.0, np.random.uniform(0.7, 1.0)))
                    
                    return_record = {
                        'return_id': f'R{str(len(returns_data)+1).zfill(8)}',
                        'order_id': order['order_id'],
                        'customer_id': customer_id,
                        'return_date': return_date,
                        'return_reason': np.random.choice(self.return_reasons),
                        'refund_amount': refund_amount,
                        'abuse_label': self._determine_abuse_label(
                            customer_id, 
                            scenarios, 
                            params,
                            order
                        )
                    }
                    
                    # Add abuse type for abusers
                    if return_record['abuse_label'] == 1:
                        if customer_id in scenarios['ring_member']:
                            return_record['abuse_type'] = 'coordinated_ring'
                        elif customer_id in scenarios['individual_abuser']:
                            return_record['abuse_type'] = 'individual_abuse'
                        else:
                            return_record['abuse_type'] = 'individual_abuse'
                    else:
                        return_record['abuse_type'] = 'legitimate'
                    
                    returns_data.append(return_record)
        
        return pd.DataFrame(orders_data), pd.DataFrame(returns_data)

    def _determine_abuse_label(self, customer_id: str, scenarios: Dict, params: Dict, order: Dict) -> int:
        """Determine if a return is abusive"""
        # Ring members and individual abusers are abusive
        if customer_id in scenarios['ring_member'] or customer_id in scenarios['individual_abuser']:
            # Not all returns are abusive - sometimes they're genuine
            # But abusers mostly abuse
            return 1 if np.random.random() < 0.85 else 0
        
        # Legitimate high-return customers are not abusive
        if customer_id in scenarios['legitimate_high_return']:
            return 0
        
        # Normal customers - mostly legitimate, some false positives
        return 1 if np.random.random() < 0.02 else 0

    def _calculate_features(
        self, 
        orders: pd.DataFrame, 
        returns: pd.DataFrame,
        entities: Dict,
        scenarios: Dict
    ) -> pd.DataFrame:
        """Calculate historical features for each customer"""
        features = []
        
        # Group orders by customer
        customer_orders = orders.groupby('customer_id')
        
        # Get all unique customers
        all_customers = [c['customer_id'] for c in entities['customers']]
        
        for customer_id in all_customers:
            customer_returns = returns[returns['customer_id'] == customer_id]
            
            # Get orders for this customer
            cust_orders = orders[orders['customer_id'] == customer_id]
            
            feature_dict = {
                'customer_id': customer_id,
                'number_of_previous_orders': len(cust_orders),
                'number_of_previous_returns': len(customer_returns),
                'historical_return_rate': len(customer_returns) / max(1, len(cust_orders)),
                'historical_refund_amount': customer_returns['refund_amount'].sum() if len(customer_returns) > 0 else 0,
                'avg_order_value': cust_orders['order_amount'].mean() if len(cust_orders) > 0 else 0,
                'avg_return_value': customer_returns['refund_amount'].mean() if len(customer_returns) > 0 else 0,
                'unique_devices_30d': 0,
                'unique_ips_30d': 0,
                'unique_addresses_30d': 0,
                'orders_last_7_days': 0,
                'orders_last_30_days': 0,
                'returns_last_7_days': 0,
                'returns_last_30_days': 0,
                'refund_amount_30d': 0,
                'return_rate_30d': 0,
                'return_rate_90d': 0,
                'previous_chargebacks': 0
            }
            
            # Calculate time-based features
            if len(cust_orders) > 0:
                max_date = cust_orders['purchase_date'].max()
                last_7_days = max_date - timedelta(days=7)
                last_30_days = max_date - timedelta(days=30)
                
                feature_dict['orders_last_7_days'] = len(cust_orders[cust_orders['purchase_date'] >= last_7_days])
                feature_dict['orders_last_30_days'] = len(cust_orders[cust_orders['purchase_date'] >= last_30_days])
                
                # Returns in last 30 days
                if len(customer_returns) > 0:
                    feature_dict['returns_last_7_days'] = len(
                        customer_returns[customer_returns['return_date'] >= last_7_days]
                    )
                    feature_dict['returns_last_30_days'] = len(
                        customer_returns[customer_returns['return_date'] >= last_30_days]
                    )
                    feature_dict['refund_amount_30d'] = customer_returns[
                        customer_returns['return_date'] >= last_30_days
                    ]['refund_amount'].sum()
            
            # Calculate return rates
            if feature_dict['orders_last_30_days'] > 0:
                feature_dict['return_rate_30d'] = feature_dict['returns_last_30_days'] / feature_dict['orders_last_30_days']
            
            # Unique devices, IPs, addresses (simplified)
            feature_dict['unique_devices_30d'] = np.random.randint(1, 4)
            feature_dict['unique_ips_30d'] = np.random.randint(1, 5)
            feature_dict['unique_addresses_30d'] = np.random.randint(1, 3)
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)

    def _build_graph(self, entities: Dict, orders: pd.DataFrame) -> nx.Graph:
        """Build the relationship graph"""
        G = nx.Graph()
        
        # Add nodes
        for customer in entities['customers']:
            G.add_node(customer['customer_id'], type='customer')
        
        for device in entities['devices']:
            G.add_node(device['device_id'], type='device')
        
        for ip in entities['ips']:
            G.add_node(ip['ip_id'], type='ip')
        
        for address in entities['addresses']:
            G.add_node(address['address_id'], type='address')
        
        for payment in entities['payments']:
            G.add_node(payment['payment_id'], type='payment')
        
        # Add edges based on orders
        for _, order in orders.iterrows():
            if pd.notna(order['customer_id']) and pd.notna(order['device_id']):
                G.add_edge(order['customer_id'], order['device_id'], weight=1)
            if pd.notna(order['customer_id']) and pd.notna(order['ip_id']):
                G.add_edge(order['customer_id'], order['ip_id'], weight=1)
            if pd.notna(order['customer_id']) and pd.notna(order['address_id']):
                G.add_edge(order['customer_id'], order['address_id'], weight=1)
            if pd.notna(order['customer_id']) and pd.notna(order['payment_id']):
                G.add_edge(order['customer_id'], order['payment_id'], weight=1)
        
        return G

    def _create_final_dataset(
        self, 
        orders: pd.DataFrame, 
        returns: pd.DataFrame,
        features: pd.DataFrame,
        entities: Dict
    ) -> pd.DataFrame:
        """Create the final dataset by merging all components"""
        # Merge returns with order information
        dataset = returns.merge(
            orders[['order_id', 'purchase_date', 'order_amount', 'product_category', 
                   'discount_percentage', 'payment_method', 'device_id', 'ip_id', 
                   'address_id', 'payment_id']],
            on='order_id',
            how='left'
        )
        
        # Add customer features
        dataset = dataset.merge(features, on='customer_id', how='left')
        
        # Add customer demographic info
        customer_info = pd.DataFrame(entities['customers'])
        dataset = dataset.merge(
            customer_info[['customer_id', 'age', 'account_creation_date', 'customer_tenure_days']],
            on='customer_id',
            how='left'
        )
        
        # Calculate days_to_return
        dataset['days_to_return'] = (
            dataset['return_date'] - dataset['purchase_date']
        ).dt.days
        
        # Add graph-based features (simplified)
        dataset['linked_account_count'] = np.random.randint(1, 6)
        dataset['shared_device_count'] = np.random.randint(0, 4)
        dataset['shared_ip_count'] = np.random.randint(0, 4)
        dataset['shared_address_count'] = np.random.randint(0, 3)
        dataset['shared_payment_count'] = np.random.randint(0, 3)
        
        # Add geographical features
        dataset = self._add_geographical_features(dataset)
        
        return dataset

    def _add_geographical_features(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add geographical features to the dataset"""
        # Simulate location data
        dataset['latitude'] = np.random.uniform(8, 37, len(dataset))
        dataset['longitude'] = np.random.uniform(68, 97, len(dataset))
        dataset['city'] = np.random.choice([loc['city'] for loc in self.locations], len(dataset))
        dataset['state'] = np.random.choice([loc['state'] for loc in self.locations], len(dataset))
        dataset['country'] = 'India'
        dataset['distance_from_shipping_address'] = np.random.uniform(0, 100, len(dataset))
        dataset['unique_locations_30d'] = np.random.randint(1, 5, len(dataset))
        
        return dataset

    def _add_temporal_split(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Add temporal split based on return date"""
        if len(dataset) == 0:
            dataset['split'] = 'train'
            return dataset
            
        dates = dataset['return_date'].sort_values()
        n = len(dates)
        
        # Split: Jan-Sep train, Oct validation, Nov-Dec test
        train_idx = int(n * 0.75)  # Jan-Sep
        val_idx = int(n * 0.85)    # Jan-Oct
        
        dataset['split'] = 'train'
        if len(dataset) > train_idx:
            dataset.iloc[train_idx:val_idx, dataset.columns.get_loc('split')] = 'validation' #type: ignore
        if len(dataset) > val_idx:
            dataset.iloc[val_idx:, dataset.columns.get_loc('split')] = 'test' #type: ignore
        
        return dataset

    def _random_date(self, start: datetime, end: datetime) -> datetime:
        """Generate a random date between start and end"""
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        return start + timedelta(days=random_days)


def generate_dataset(
    n_customers: int = 50000,
    n_orders: int = 500000,
    abuse_rate: float = 0.08,
    ring_rate: float = 0.03,
    random_state: int = 42
) -> Tuple[pd.DataFrame, nx.Graph]:
    """
    Main function to generate the return abuse detection dataset
    
    Args:
        n_customers: Number of customers to generate
        n_orders: Number of orders to generate
        abuse_rate: Proportion of customers who are abusers
        ring_rate: Proportion of abusers who are in rings
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (dataset DataFrame, graph NetworkX object)
    """
    generator = ReturnAbuseDatasetGenerator(random_state)
    return generator.generate_dataset(
        n_customers=n_customers,
        n_orders=n_orders,
        abuse_rate=abuse_rate,
        ring_rate=ring_rate
    )

if __name__ == "__main__":
    # Generate a small dataset for testing
    dataset, graph = generate_dataset(
        n_customers=1000,
        n_orders=10000,
        abuse_rate=0.08,
        ring_rate=0.03
    )
    dataset.to_csv('return_abuse_dataset.csv', index=False)
    graph_json = {
    'nodes': [{'id': str(n), 'type': data.get('type', 'unknown')} for n, data in graph.nodes(data=True)],
    'edges': [{'source': str(u), 'target': str(v), 'weight': data.get('weight', 1)} 
              for u, v, data in graph.edges(data=True)]
            }
    with open('return_abuse_graph.json', 'w') as f:
        json.dump(graph_json, f, indent=2)
    print("\nDataset shape:", dataset.shape)
    print("\nDataset columns:", dataset.columns.tolist())
    print("\nAbuse distribution:")
    print(dataset['abuse_label'].value_counts())
    print("\nAbuse type distribution:")
    print(dataset['abuse_type'].value_counts())
    print("\nSplit distribution:")
    print(dataset['split'].value_counts())
    print("\nSample data:")
    print(dataset.head())
    print("\nGraph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())