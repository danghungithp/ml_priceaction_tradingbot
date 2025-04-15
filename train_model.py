import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib # For saving the model

def train_evaluate_model(data_path="features_data.csv", model_save_path="trading_model.joblib"):
    """
    Loads data, trains a Logistic Regression model, evaluates it, and saves it.

    Args:
        data_path (str): Path to the CSV file containing features and target.
        model_save_path (str): Path to save the trained model.

    Returns:
        None
    """
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file '{data_path}' not found. Please run feature_engineering.py first.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Data Preparation ---
    print("Preparing data for training...")
    # Define features (X) and target (y)
    # Drop original OHLCV columns and any other non-feature columns if necessary
    # Keep only the calculated indicators and price action features
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    X = df[features]
    y = df['target']

    print(f"Features selected ({len(features)}): {features}")
    print(f"Target variable: 'target'")

    # Check if data is sufficient
    if len(X) < 100: # Arbitrary threshold, adjust as needed
        print("Error: Not enough data to train the model after cleaning.")
        return

    # --- Time Series Split ---
    # Use TimeSeriesSplit for cross-validation appropriate for time series data
    # For simplicity in this initial step, we'll do a simple train/test split based on time
    split_ratio = 0.8
    split_index = int(len(df) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"Data split into training and testing sets:")
    print(f"  Training set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"  Testing set shape: X={X_test.shape}, y={y_test.shape}")

    # --- Feature Scaling ---
    # Scale features for better performance, especially for Logistic Regression
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on training data

    # --- Model Training ---
    print("Training Logistic Regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000) # Increased max_iter for convergence
    try:
        model.fit(X_train_scaled, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # --- Model Evaluation ---
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Price Down/Stay', 'Price Up']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save Model and Scaler ---
    print(f"\nSaving trained model to {model_save_path}...")
    try:
        joblib.dump(model, model_save_path)
        # Also save the scaler, as it's needed for predicting on new data
        scaler_save_path = model_save_path.replace(".joblib", "_scaler.joblib")
        joblib.dump(scaler, scaler_save_path)
        print(f"Model saved to {model_save_path}")
        print(f"Scaler saved to {scaler_save_path}")
    except Exception as e:
        print(f"Error saving model or scaler: {e}")

if __name__ == "__main__":
    train_evaluate_model()
