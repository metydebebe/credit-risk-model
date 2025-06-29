import pandas as pd
from src.data_processing import process_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading raw data...")
    df = pd.read_csv('..data/raw/data.csv')
    target = df['FraudResult']

    logger.info("Processing features...")
    X = process_data(df, target=target)

    logger.info("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, target, test_size=0.2, random_state=42, stratify=target)

    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    logger.info("Predicting on validation set...")
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    logger.info(f'Validation AUC: {auc:.4f}')

    logger.info("Saving model and pipeline...")
    joblib.dump(model, 'models/logistic_model.joblib')

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
