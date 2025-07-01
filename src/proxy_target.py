import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_rfm(df, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount', snapshot_date=None):
    """
    Calculate Recency, Frequency, Monetary (RFM) metrics per customer.
    Args:
        df: DataFrame with transactions
        customer_id_col: column with customer IDs
        date_col: transaction date column (datetime)
        amount_col: transaction amount column
        snapshot_date: datetime to calculate recency from (default: max date in dataset + 1 day)
    Returns:
        DataFrame with columns: CustomerId, Recency, Frequency, Monetary
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if snapshot_date is None:
        snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id_col).agg(
        Recency = (date_col, lambda x: (snapshot_date - x.max()).days),
        Frequency = (date_col, 'count'),
        Monetary = (amount_col, 'sum')
    ).reset_index()

    return rfm

def cluster_customers(rfm_df, n_clusters=3, random_state=42):
    """
    Cluster customers using KMeans on scaled RFM features.
    Args:
        rfm_df: DataFrame with RFM columns
        n_clusters: number of clusters
        random_state: for reproducibility
    Returns:
        rfm_df with cluster labels in 'Cluster' column
    """
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary']])

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    rfm_df['Cluster'] = kmeans.fit_predict(rfm_scaled)

    return rfm_df

def assign_high_risk_label(rfm_df):
    """
    Identify the high-risk cluster and assign binary label.
    Assumes high-risk cluster has highest Recency, lowest Frequency and Monetary.
    Args:
        rfm_df: DataFrame with RFM and Cluster columns
    Returns:
        rfm_df with 'is_high_risk' column (1 = high risk, 0 = others)
    """
    cluster_summary = rfm_df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })

    # Heuristic: high-risk cluster has highest Recency, lowest Frequency and Monetary
    cluster_summary['risk_score'] = (
        cluster_summary['Recency'].rank(ascending=False) + 
        cluster_summary['Frequency'].rank(ascending=True) + 
        cluster_summary['Monetary'].rank(ascending=True)
    )

    high_risk_cluster = cluster_summary['risk_score'].idxmax()

    rfm_df['is_high_risk'] = (rfm_df['Cluster'] == high_risk_cluster).astype(int)

    return rfm_df

def create_proxy_target(df, customer_id_col='CustomerId', date_col='TransactionStartTime', amount_col='Amount'):
    """
    Full pipeline to create proxy target variable.
    Args:
        df: raw transaction DataFrame
    Returns:
        DataFrame with CustomerId and is_high_risk label
    """
    rfm = calculate_rfm(df, customer_id_col, date_col, amount_col)
    rfm = cluster_customers(rfm)
    rfm = assign_high_risk_label(rfm)

    return rfm[[customer_id_col, 'is_high_risk']]
