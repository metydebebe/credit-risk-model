import unittest
import pandas as pd
from src.data_processing import build_feature_engineering_pipeline, categorical_cols, numerical_cols


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C3', 'C3'],
            'Amount': [100, 200, 300, 400, 500, 600],
            'Value': [50, 100, 150, 200, 250, 300],
            'TransactionStartTime': pd.to_datetime([
                '2023-01-01 10:00', '2023-01-02 11:00',
                '2023-02-01 12:00', '2023-03-01 13:00',
                '2023-03-02 14:00', '2023-03-03 15:00'
            ]),
            'ProductCategory': ['A', 'B', 'A', 'B', 'B', 'A'],
            'ProviderId': ['P1', 'P2', 'P1', 'P2', 'P2', 'P1'],
            'ChannelId': ['Ch1', 'Ch2', 'Ch1', 'Ch2', 'Ch2', 'Ch1'],
            'PricingStrategy': [1, 2, 1, 2, 2, 1]
        })

        self.target = pd.Series([0, 1, 0, 1, 0, 1])

    def test_pipeline_runs(self):
        pipeline = build_feature_engineering_pipeline()
        pipeline.fit(self.df, self.target)
        X = pipeline.transform(self.df)
        self.assertEqual(X.shape[0], self.df.shape[0])

    def test_feature_names_and_shape(self):
        pipeline = build_feature_engineering_pipeline()
        pipeline.fit(self.df, self.target)
        X = pipeline.transform(self.df)

        feature_count = len(pipeline.named_steps['column_transformer'].get_feature_names_out())
        self.assertEqual(X.shape[1], feature_count)

    def test_missing_value_imputation(self):
        df_missing = self.df.copy()
        df_missing.loc[0, 'Amount'] = None

        pipeline = build_feature_engineering_pipeline()
        pipeline.fit(df_missing, self.target)
        X_missing = pipeline.transform(df_missing)

        self.assertFalse(pd.isnull(X_missing).any())


if __name__ == '__main__':
    unittest.main()
