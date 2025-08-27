# train.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import json

def train_model():
    # Load data
    df = pd.read_csv("sales_data.csv")

    # Create features and target
    df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
    df['month'] = pd.to_datetime(df['date']).dt.month

    X = df[['store_id', 'product_id', 'day_of_week', 'month', 'units_sold']]
    y = df['sales_amount']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training R²: {train_score:.3f}")
    print(f"Testing R²: {test_score:.3f}")

    # Save model
    joblib.dump(model, "model.pkl")

    # Save metrics
    metrics = {
        "train_r2": float(train_score),
        "test_r2": float(test_score)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Model saved to model.pkl")
    print("Metrics saved to metrics.json")

if __name__ == "__main__":
    train_model()