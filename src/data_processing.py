import pandas as pd
# import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from feature_engine.encoding import WoEEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Define categorical and numerical columns globally for reuse
categorical_cols = [
    'ProductCategory',
    'ProviderId',
    'ChannelId',
    'PricingStrategy'
]

numerical_cols = [
    'Amount',
    'Value',
    'std_amount',
    'avg_amount',
    'total_amount',
    'transaction_count',
    'transaction_hour',
    'transaction_day',
    'transaction_month',
    'transaction_year'
]


class AggregateFeatures(BaseEstimator, TransformerMixin):
    """
    Create aggregate features per customer such as total amount, average amount,
    transaction count, and standard deviation of amounts.
    """

    def __init__(self, groupby_col='CustomerId', amount_col='Amount'):
        self.groupby_col = groupby_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby(self.groupby_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            transaction_count='count',
            std_amount='std'
        ).reset_index()
        X = X.merge(agg, on=self.groupby_col, how='left')
        X['std_amount'] = X['std_amount'].fillna(0)
        return X


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract datetime features from the transaction start time.
    """

    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors='coerce')
        X['transaction_hour'] = X[self.datetime_col].dt.hour
        X['transaction_day'] = X[self.datetime_col].dt.day
        X['transaction_month'] = X[self.datetime_col].dt.month
        X['transaction_year'] = X[self.datetime_col].dt.year
        return X


def build_feature_engineering_pipeline():
    """
    Build a two-stage sklearn pipeline:
    - Stage 1: Aggregate and datetime features (no target needed)
    - Stage 2: ColumnTransformer with numerical and categorical pipelines including WoE encoding
    """

    # Stage 1: Aggregate and datetime feature extraction
    preprocessor = Pipeline([
        ('aggregate_features', AggregateFeatures()),
        ('datetime_features', DateTimeFeatures())
    ])


    # Stage 2: ColumnTransformer for numerical and categorical processing
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('woe_encoder', WoEEncoder(variables=categorical_cols))
    ])

    column_transformer = ColumnTransformer([
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ], remainder='drop')

    # Full pipeline: preprocessor followed by column transformer
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('column_transformer', column_transformer)
    ])

    return full_pipeline


def process_data(df, target=None):
    pipeline = build_feature_engineering_pipeline()
    # 1. Fit and transform preprocessor (aggregate + datetime features)
    pipeline.named_steps['preprocessor'].fit(df)
    df_transformed = pipeline.named_steps['preprocessor'].transform(df)

    # 2. Fit column_transformer to initialize its transformers
    pipeline.named_steps['column_transformer'].fit(df_transformed)

    # 3. Access WoE encoder and fit it with target
    cat_pipe = pipeline.named_steps['column_transformer'].named_transformers_['cat']
    woe_encoder = cat_pipe.named_steps['woe_encoder']

    if target is not None:
        woe_encoder.fit(df_transformed[categorical_cols], target)

    # 4. Fit column_transformer again to update with fitted WoE encoder
    pipeline.named_steps['column_transformer'].fit(df_transformed)

    # 5. Transform the full data through the pipeline
    X_transformed = pipeline.transform(df)

    feature_names = numerical_cols + categorical_cols
    return pd.DataFrame(X_transformed, columns=feature_names, index=df.index)