import unittest
import pandas as pd
from src.data_processing import (
    build_feature_engineering_pipeline,
    categorical_cols,
    numerical_cols,
)


class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'CustomerId': ['C1', 'C1', 'C2', 'C3', 'C3', 'C3'],
            'Amount': [100, 200, 300, 400, 500, 600],
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
        # Fit WoE encoder with target before fitting pipeline
        cat_pipe = pipeline.named_steps['column_transformer'].named_transformers_['cat']
        woe_encoder = cat_pipe.named_steps['woe_encoder']
        woe_encoder.fit(self.df[categorical_cols], self.target)
        pipeline.fit(self.df)
        X_transformed = pipeline.transform(self.df)
        self.assertEqual(X_transformed.shape[0], self.df.shape[0])

    def test_feature_names_and_shape(self):
        pipeline = build_feature_engineering_pipeline()
        cat_pipe = pipeline.named_steps['column_transformer'].named_transformers_['cat']
        woe_encoder = cat_pipe.named_steps['woe_encoder']
        woe_encoder.fit(self.df[categorical_cols], self.target)
        pipeline.fit(self.df)
        X_transformed = pipeline.transform(self.df)
        expected_num_features = len(numerical_cols) + len(categorical_cols)
        self.assertEqual(X_transformed.shape[1], expected_num_features)

    def test_missing_value_imputation(self):
        df_missing = self.df.copy()
        df_missing.loc[0, 'Amount'] = None
        pipeline = build_feature_engineering_pipeline()
        cat_pipe = pipeline.named_steps['column_transformer'].named_transformers_['cat']
        woe_encoder = cat_pipe.named_steps['woe_encoder']
        woe_encoder.fit(df_missing[categorical_cols], self.target)
        pipeline.fit(df_missing)
        X_transformed = pipeline.transform(df_missing)
        # Check no NaNs after imputation
        self.assertFalse(pd.isnull(X_transformed).any())


if __name__ == '__main__':
    unittest.main()
