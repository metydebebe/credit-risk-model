import pandas as pd
from src.data_processing import process_data
from src.proxy_target import create_proxy_target  
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Loading raw data...")
    df = pd.read_csv('data/raw/data.csv')

    # Create proxy target variable (Task 4)
    logger.info("Creating proxy target variable (is_high_risk)...")
    proxy_target_df = create_proxy_target(df)

    # Merge proxy target back into main DataFrame
    df = df.merge(proxy_target_df, on='CustomerId', how='left')

    # Process features with Task 3 pipeline, passing the new target
    logger.info("Processing features...")
    X = process_data(df, target=df['is_high_risk'])
    y = df['is_high_risk']

    logger.info("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Predicting on validation set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f'Validation AUC: {auc:.4f}')

    logger.info("Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/logistic_model.joblib')

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
