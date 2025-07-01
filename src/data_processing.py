import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from feature_engine.encoding import WoEEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Column definitions
categorical_cols = ['ProductCategory', 'ProviderId', 'ChannelId', 'PricingStrategy']
numerical_cols = [
    'Amount', 'Value', 'std_amount', 'avg_amount', 'total_amount',
    'transaction_count', 'transaction_hour', 'transaction_day',
    'transaction_month', 'transaction_year'
]

class AggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, groupby_col='CustomerId', amount_col='Amount'):
        self.groupby_col = groupby_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = df.groupby(self.groupby_col)[self.amount_col].agg(
            total_amount='sum',
            avg_amount='mean',
            transaction_count='count',
            std_amount='std'
        ).reset_index()
        agg['std_amount'] = agg['std_amount'].fillna(0)
        df = df.merge(agg, on=self.groupby_col, how='left')
        return df

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col='TransactionStartTime'):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_col] = pd.to_datetime(df[self.datetime_col], errors='coerce')
        df['transaction_hour'] = df[self.datetime_col].dt.hour
        df['transaction_day'] = df[self.datetime_col].dt.day
        df['transaction_month'] = df[self.datetime_col].dt.month
        df['transaction_year'] = df[self.datetime_col].dt.year
        return df

def build_feature_engineering_pipeline():
    return Pipeline([
        ('aggregate_features', AggregateFeatures()),
        ('datetime_features', DateTimeFeatures())
    ])

def build_encoding_pipeline():
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('woe_encoder', WoEEncoder(variables=categorical_cols))
    ])

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformer = ColumnTransformer([
        ('cat', cat_pipe, categorical_cols),
        ('num', num_pipe, numerical_cols)
    ], remainder='drop')

    return transformer

def process_data(df: pd.DataFrame, target=None) -> pd.DataFrame:
    """
    Run feature engineering, then encoding/transformation.
    """
    # Feature engineering
    feature_engineer = build_feature_engineering_pipeline()
    df_fe = feature_engineer.fit_transform(df)

    # Ensure categorical columns are strings (required for WoEEncoder)
    for col in categorical_cols:
        if col in df_fe.columns:
            df_fe[col] = df_fe[col].astype(str)

    # Validate required columns exist
    missing = [col for col in categorical_cols + numerical_cols if col not in df_fe.columns]
    if missing:
        raise KeyError(f"Missing expected columns after feature engineering: {missing}")

    # Fit and transform with encoding pipeline
    transformer = build_encoding_pipeline()
    if target is not None:
        X_array = transformer.fit_transform(df_fe, target)
    else:
        X_array = transformer.transform(df_fe)

    # Get feature names if possible
    try:
        feature_names = transformer.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]

    return pd.DataFrame(X_array, columns=feature_names, index=df.index)
