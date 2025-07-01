import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
from src.proxy_target import calculate_rfm, cluster_customers, assign_high_risk_label, create_proxy_target

class TestProxyTarget(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C3', 'C3'],
            'TransactionStartTime': [
                '2023-01-01', '2023-01-10', '2023-01-05',
                '2022-12-01', '2022-12-15', '2023-01-20'
            ],
            'Amount': [100, 150, 200, 50, 60, 70]
        })

    def test_calculate_rfm(self):
        rfm = calculate_rfm(self.df, snapshot_date=pd.Timestamp('2023-01-31'))
        self.assertIn('Recency', rfm.columns)
        self.assertIn('Frequency', rfm.columns)
        self.assertIn('Monetary', rfm.columns)
        self.assertEqual(len(rfm), 3)

    def test_cluster_and_label(self):
        rfm = calculate_rfm(self.df, snapshot_date=pd.Timestamp('2023-01-31'))
        rfm = cluster_customers(rfm, n_clusters=2, random_state=0)
        self.assertIn('Cluster', rfm.columns)
        rfm = assign_high_risk_label(rfm)
        self.assertIn('is_high_risk', rfm.columns)
        self.assertTrue(set(rfm['is_high_risk']).issubset({0,1}))

    def test_create_proxy_target(self):
        proxy_df = create_proxy_target(self.df)
        self.assertIn('CustomerId', proxy_df.columns)
        self.assertIn('is_high_risk', proxy_df.columns)
        self.assertEqual(proxy_df['is_high_risk'].nunique(), 2)  

if __name__ == '__main__':
    unittest.main()
