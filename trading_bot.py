import time
import requests
import pandas as pd
import json # Added for potential JSON errors
from datetime import datetime, timedelta
# import schedule # Removed unused schedule library
import pandas_ta as ta # Import pandas-ta
from sklearn.model_selection import train_test_split # TimeSeriesSplit not used in current train logic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib # For saving the model
import smtplib
import ssl
from email.message import EmailMessage
import os # To potentially use environment variables later
import logging # Added for signal logging

# --- Constants ---
SYMBOL = "BTC_USDT"
INTERVAL = "15m" # Must match the interval used for training
API_BASE_URL = "https://spot-markets.goonus.io/candlesticks"
YEARS_OF_DATA = 1 # How many years of historical data to fetch for training
HISTORICAL_DATA_CSV = f"{SYMBOL}_{INTERVAL}_{YEARS_OF_DATA}y_hist.csv" # File to save raw historical data
FEATURES_DATA_CSV = f"{SYMBOL}_{INTERVAL}_features.csv" # File to save features data
MODEL_SAVE_PATH = f"{SYMBOL}_{INTERVAL}_model.joblib" # File for trained model
SCALER_SAVE_PATH = f"{SYMBOL}_{INTERVAL}_scaler.joblib" # File for scaler
SIGNALS_LOG_FILE = "signals.log" # File to log signals for the web UI
SUBSCRIBERS_FILE = "subscribers.txt" # File containing recipient emails
LOOKAHEAD_PERIODS = 5 # Predict trend over next N * interval (e.g., 5 * 15 minutes = 75 minutes)
REQUIRED_HISTORY_FOR_LIVE_FEATURES = 60 # Candles needed for live feature calculation (adjust based on longest lookback + buffer)
POLL_INTERVAL_SECONDS = 60 # How often to check if it's time for a new candle (e.g., every minute)

# --- Global variable to track last signal to avoid duplicates ---
last_signal_sent = None
last_signal_timestamp = None

# --- Configuration (User needs to replace placeholders securely!) ---
# WARNING: Do NOT hardcode your real password here.
# Option 1: Use Environment Variables (Recommended)
# Example: Set environment variables 'GMAIL_SENDER' and 'GMAIL_PASSWORD'
# SENDER_EMAIL = os.environ.get("GMAIL_SENDER")
# SENDER_PASSWORD = os.environ.get("GMAIL_PASSWORD")

# Option 2: Replace placeholders directly (Less Secure - Use Environment Variables!)
SENDER_EMAIL = "mcpsoftware@gmail.com"  # <<< REPLACE THIS with your Gmail address
SENDER_PASSWORD = "cdtz inwd vfwr dqgu"  # <<< REPLACE THIS with your Gmail App Password or regular password
# RECIPIENT_EMAIL is now read from subscribers.txt

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587 # For starttls

# --- Setup Logging ---
# Basic configuration for signal logging
logging.basicConfig(
    filename=SIGNALS_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s;%(message)s', # Simple format: timestamp;SIGNAL;PRICE
    datefmt='%Y-%m-%d %H:%M:%S'
)

# List of features used during training (must match the order in train_model.py)
# Get this list dynamically or ensure it's consistent
# --- Global variable to track last signal to avoid duplicates ---
last_signal_sent = None
last_signal_timestamp = None
# Global variable to track the start time of the last processed interval in the live loop
last_processed_interval_start = None
# Global list to store feature names after creation
# This will be populated by create_features and used by train_evaluate_model and calculate_latest_features
FEATURE_NAMES = []


def get_interval_milliseconds(interval_str):
    """Converts interval string (e.g., '1m', '15m', '1h', '1d') to milliseconds."""
    unit = interval_str[-1].lower()
    value = int(interval_str[:-1])
    if unit == 'm':
        return value * 60 * 1000
    elif unit == 'h':
        return value * 60 * 60 * 1000
    elif unit == 'd':
        return value * 24 * 60 * 60 * 1000
    else:
        raise ValueError(f"Unsupported interval unit: {unit}")

def get_historical_data(symbol, interval, years_back=1):
    """
    Fetches historical candlestick data for a given symbol and interval.

    Args:
        symbol (str): The trading pair symbol (e.g., 'BTC_USDT').
        interval (str): The candle interval (e.g., '15m').
        years_back (int): How many years of historical data to fetch.

    Returns:
        pandas.DataFrame: DataFrame containing the historical data, or None if failed.
    """
    base_url = "https://spot-markets.goonus.io/candlesticks"
    end_time = int(time.time() * 1000) # Current time in milliseconds
    start_time = int((datetime.now() - timedelta(days=years_back * 365)).timestamp() * 1000)

    print(f"Fetching {years_back} year(s) of data for {symbol} ({interval}) from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

    all_data = []
    fetch_start_time = start_time
    interval_ms = get_interval_milliseconds(interval)
    # API limit might vary, 500 is a reasonable guess. Adjust if needed.
    api_limit = 500
    print(f"Fetching data in chunks (limit per request: {api_limit})...")

    while fetch_start_time < end_time:
        # Calculate the end time for this chunk request
        fetch_end_time = min(fetch_start_time + api_limit * interval_ms, end_time)

        params = {
            "symbol_name": symbol, # API uses symbol_name
            "interval": interval,
            "from": fetch_start_time,
            "to": fetch_end_time,
            # Some APIs use 'limit' instead of 'to', check API docs if needed. This one uses 'from'/'to'.
        }
        # print(f"  Fetching chunk from {datetime.fromtimestamp(fetch_start_time/1000)} to {datetime.fromtimestamp(fetch_end_time/1000)}")

        try:
            response = requests.get(base_url, params=params, timeout=60) # Increased timeout for potentially larger requests
            response.raise_for_status()
            data_chunk = response.json()

            if isinstance(data_chunk, list) and len(data_chunk) > 0:
                all_data.extend(data_chunk)
                # print(f"    Fetched {len(data_chunk)} candlesticks. Total fetched: {len(all_data)}")
                # Update the start time for the next chunk based on the last timestamp received
                last_timestamp = int(data_chunk[-1]['t']) # Timestamp of the last candle in the chunk
                fetch_start_time = last_timestamp + interval_ms # Start next fetch *after* the last candle

                # Break if we received fewer candles than the limit, might be at the end
                if len(data_chunk) < api_limit:
                     print(f"    Fetched {len(data_chunk)} (less than limit {api_limit}), assuming end of data for this range.")
                     break
            elif isinstance(data_chunk, list) and len(data_chunk) == 0:
                print("    API returned an empty list for this chunk, stopping.")
                break # Stop if we get an empty list
            else:
                print(f"    Warning: Unexpected API response format for chunk: {data_chunk}")
                # Decide how to handle: break, retry, or skip? For now, break.
                break

            # Add a small delay to avoid hitting API rate limits
            time.sleep(0.5) # Sleep for 500ms

        except requests.exceptions.RequestException as e:
            print(f"Error fetching chunk: {e}")
            # Decide how to handle: break, retry, or skip? For now, break.
            return None # Or implement retry logic
        except json.JSONDecodeError:
            print(f"Error decoding JSON response for chunk: {response.text}") # Log the problematic text
            return None
        except Exception as e:
            print(f"An unexpected error occurred during chunk fetching: {e}") # Catch other potential errors
            return None

    # --- Post-fetching processing ---
    if not all_data:
        print("No data fetched after attempting chunking.")
        return None

    print(f"\nTotal candlesticks fetched before processing: {len(all_data)}")
    if not all_data:
        return None

    df = pd.DataFrame(all_data)

    # Basic validation
    if 't' not in df.columns:
        print("Error: 't' (timestamp) column not found in fetched data.")
        return None

    # Rename columns
    df.rename(columns={
        't': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'q': 'volume'
    }, inplace=True)

    # Select and convert necessary columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in fetched data. Found: {df.columns.tolist()}")
        return None
    df = df[required_cols]

    # Convert timestamp and numeric columns
    try:
        df['timestamp'] = pd.to_numeric(df['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
    except Exception as e:
        print(f"Error converting data types: {e}")
        return None

    # Sort by time, remove duplicates, set index
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
    df.set_index('timestamp', inplace=True)

    # Filter data to the originally requested range (optional, but good practice)
    start_dt = pd.to_datetime(start_time, unit='ms')
    end_dt = pd.to_datetime(end_time, unit='ms')
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    print(f"Historical data processed. Final shape: {df.shape}")
    return df

def create_features(df, n_periods=5):
    """
    Adds technical indicators and price action features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data, indexed by timestamp.
        n_periods (int): Number of periods to look ahead for target definition.

    Returns:
        pd.DataFrame: DataFrame with added features and target variable, or None on failure.
    """
    global FEATURE_NAMES # Declare intent to modify the global variable
    print("Starting feature engineering...")

    # --- Technical Indicators using pandas_ta ---
    # Calculate SMA
    print("Calculating SMAs...")
    print("  Calculating SMAs...")
    df.ta.sma(length=10, append=True) # SMA_10
    df.ta.sma(length=20, append=True) # SMA_20
    df.ta.sma(length=50, append=True) # SMA_50

    # Calculate RSI
    print("  Calculating RSI...")
    df.ta.rsi(length=14, append=True) # RSI_14

    # Calculate MACD
    print("  Calculating MACD...")
    df.ta.macd(append=True) # Defaults: fast=12, slow=26, signal=9 -> MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # Calculate Bollinger Bands
    print("  Calculating Bollinger Bands...")
    # Using length 5 as identified previously from training output. Confirm this is desired.
    df.ta.bbands(length=5, std=2.0, append=True) # -> BBL_5_2.0, BBM_5_2.0, BBU_5_2.0, BBB_5_2.0, BBP_5_2.0

    # --- Price Action Features ---
    print("  Calculating Price Action features...")
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['price_change'] = df['close'].diff()
    df['volume_change'] = df['volume'].diff()

    # --- Target Variable ---
    print(f"  Defining target variable (price increase after {n_periods} periods)...")
    df['target'] = (df['close'].shift(-n_periods) > df['close']).astype(int)

    # --- Identify Feature Columns ---
    # Exclude OHLCV and target
    potential_features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    print(f"  Potential features identified: {potential_features}")

    # --- Clean up ---
    print("  Cleaning data (removing NaNs)...")
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"  Removed {initial_rows - final_rows} rows with NaN values.")

    # --- Store Feature Names ---
    # Store the final list of feature names *after* dropna, ensuring they exist in the cleaned df
    FEATURE_NAMES = [col for col in potential_features if col in df.columns]
    if not FEATURE_NAMES:
         print("Error: No feature columns remaining after cleaning!")
         return None

    print("Feature engineering completed.")
    print(f"Final feature DataFrame shape: {df.shape}")
    print(f"Features used for training ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    return df


def train_evaluate_model(features_df, model_path, scaler_path):
    """
    Trains a Logistic Regression model, evaluates it, and saves the model and scaler.

    Args:
        features_df (pd.DataFrame): DataFrame containing features and the 'target' column.
        model_path (str): Path to save the trained model.
        scaler_path (str): Path to save the fitted scaler.

    Returns:
        bool: True if training and saving were successful, False otherwise.
    """
    global FEATURE_NAMES # Access the global list defined in create_features
    if not FEATURE_NAMES:
        print("Error: FEATURE_NAMES list is empty. Cannot train model.")
        return False
    if 'target' not in features_df.columns:
        print("Error: 'target' column not found in features DataFrame.")
        return False

    print("Preparing data for training...")
    X = features_df[FEATURE_NAMES]
    y = features_df['target']

    # Check if data is sufficient
    if len(X) < 100: # Arbitrary threshold
        print(f"Error: Not enough data ({len(X)} rows) to train the model after cleaning.")
        return False

    # --- Time Series Split (Simple Train/Test) ---
    # Using a simple time-based split for now. TimeSeriesSplit CV is more robust.
    split_ratio = 0.8
    split_index = int(len(features_df) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    print(f"Data split into training ({len(X_train)} rows) and testing ({len(X_test)} rows).")

    # --- Feature Scaling ---
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Use the scaler fitted on training data

    # --- Model Training ---
    print("Training Logistic Regression model...")
    # Consider adding class_weight='balanced' if data is imbalanced
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    try:
        model.fit(X_train_scaled, y_train)
        print("Model training completed.")
    except Exception as e:
        print(f"Error during model training: {e}")
        return False

    # --- Model Evaluation ---
    print("\nEvaluating model performance on the test set...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f}")
    print("  Classification Report:")
    # Use zero_division=0 to avoid warnings if a class has no predicted samples
    print(classification_report(y_test, y_pred, target_names=['Down/Stay', 'Up'], zero_division=0))
    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Save Model and Scaler ---
    print(f"\nSaving trained model to {model_path}...")
    try:
        joblib.dump(model, model_path)
        print(f"Saving scaler to {scaler_path}...")
        joblib.dump(scaler, scaler_path)
        print("Model and scaler saved successfully.")
        return True
    except Exception as e:
        print(f"Error saving model or scaler: {e}")
        return False


def get_subscribers(filename=SUBSCRIBERS_FILE):
    """Reads subscriber emails from a file, one email per line."""
    subscribers = []
    try:
        with open(filename, 'r') as f:
            subscribers = [line.strip() for line in f if line.strip() and '@' in line.strip()]
        if not subscribers:
            print(f"Warning: No valid subscriber emails found in {filename}.")
        else:
            print(f"Loaded {len(subscribers)} subscribers from {filename}.")
    except FileNotFoundError:
        print(f"Warning: Subscribers file '{filename}' not found. No emails will be sent.")
    except Exception as e:
        print(f"Error reading subscribers file '{filename}': {e}")
    return subscribers

def send_email_notification(signal, timestamp, price, recipients):
    """Sends an email notification for a trading signal to multiple recipients."""

    if not recipients:
        # print("No recipients provided, skipping email notification.")
        return False # No need to proceed if no one to send to

    if not SENDER_EMAIL or SENDER_EMAIL == "your_email@gmail.com" or not SENDER_PASSWORD or SENDER_PASSWORD == "your_app_password":
        print("Error: Sender email or password not configured. Cannot send email.")
        print("Please replace the placeholder values securely (or use environment variables).")
        return False

    subject = f"Trading Signal Alert: {signal} {SYMBOL}"
    body = f"""
    A trading signal has been generated:

    Signal:      {signal}
    Timestamp:   {timestamp}
    Price (approx): {price:.2f} USDT

    This is an automated notification based on the ML model prediction.
    Please review the charts and apply your own analysis before taking action.
    """

    # Create message object once
    em = EmailMessage()
    em['From'] = SENDER_EMAIL
    # em['To'] = ", ".join(recipients) # Set 'To' header for display, but send individually
    em['Subject'] = subject
    em.set_content(body)
    # Note: Sending individually is generally better to avoid exposing emails to all recipients
    # and to handle individual send failures. Setting 'To' header is optional.

    context = ssl.create_default_context()
    all_sent = True
    try:
        print(f"Attempting to send email notification to {len(recipients)} subscriber(s)...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls(context=context)
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            # Send to each recipient individually
            for recipient in recipients:
                try:
                    # Create a new message object or clear recipients for each send
                    msg = EmailMessage()
                    msg['From'] = SENDER_EMAIL
                    msg['To'] = recipient # Send to one recipient at a time
                    msg['Subject'] = subject
                    msg.set_content(body)
                    smtp.sendmail(SENDER_EMAIL, recipient, msg.as_string())
                    print(f"  Successfully sent to {recipient}")
                except smtplib.SMTPException as e:
                    print(f"  Failed to send to {recipient}: {e}")
                    all_sent = False # Mark failure if any recipient fails
        print("Email sending process completed.")
        return all_sent
    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP Authentication failed. Check sender email/password (or App Password).")
        print("If not using App Password, ensure 'Less secure app access' is ON (NOT RECOMMENDED): https://myaccount.google.com/lesssecureapps")
        return False
    except smtplib.SMTPServerDisconnected:
        print("Error: SMTP server disconnected unexpectedly.")
        return False
    except smtplib.SMTPException as e:
        print(f"Error sending email (SMTPException): {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during email sending: {e}")
        return False


def fetch_recent_candles(symbol, interval, limit):
    """Fetches the most recent 'limit' candles for live prediction."""
    # This API uses from/to, not limit. We need to calculate 'from'.
    interval_ms = get_interval_milliseconds(interval)
    end_time = int(time.time() * 1000)
    # Fetch slightly more than needed to ensure calculations work near the edges
    start_time = end_time - (limit + 5) * interval_ms

    params = {
        "symbol_name": symbol,
        "interval": interval,
        "from": start_time,
        "to": end_time
    }
    print(f"Fetching recent ~{limit} candles for {symbol} ({interval}) for live prediction...")
    try:
        response = requests.get(API_BASE_URL, params=params, timeout=20) # Increased timeout slightly
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            # Rename, select, convert types (same logic as historical fetch)
            df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'q': 'volume'}, inplace=True)
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                print("Error: Missing required columns in recent data.")
                return None
            df = df[required_cols]
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            df.sort_values('timestamp', inplace=True)
            df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True)
            df.set_index('timestamp', inplace=True)
            # Return the last 'limit' rows after processing
            print(f"Fetched and processed {len(df)} recent candles. Using last {limit}.")
            return df.tail(limit) # Return only the required number of candles
        elif isinstance(data, list) and len(data) == 0:
             print("Warning: API returned no recent data.")
             return None
        else:
            print(f"Warning: API returned unexpected format for recent data: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recent candles: {e}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON for recent candles: {response.text}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching recent candles: {e}")
        return None

def calculate_latest_features(ohlcv_df):
    """
    Calculates the necessary features for the latest data point(s).
    This function mirrors the feature calculation in feature_engineering.py
    but should be optimized for calculating only the needed recent features.

    Args:
        ohlcv_df (pd.DataFrame): DataFrame containing recent OHLCV data.
                                 Needs enough historical data to calculate indicators.

    Returns:
        pd.DataFrame: DataFrame with features calculated for the latest row(s), or None if calculation fails.
    """
    global FEATURE_NAMES # Use the globally defined feature names
    if not FEATURE_NAMES:
        print("Error: FEATURE_NAMES list is empty. Cannot calculate latest features.")
        return None
    if ohlcv_df is None or ohlcv_df.empty:
        print("Error: Input DataFrame is empty. Cannot calculate latest features.")
        return None

    print(f"Calculating features for the latest {len(ohlcv_df)} data points...")
    try:
        # Calculate all features (similar to create_features but without target)
        # Ensure calculations match those in create_features exactly
        ohlcv_df.ta.sma(length=10, append=True)
        ohlcv_df.ta.sma(length=20, append=True)
        ohlcv_df.ta.sma(length=50, append=True)
        ohlcv_df.ta.rsi(length=14, append=True)
        ohlcv_df.ta.macd(append=True) # Defaults: 12, 26, 9
        ohlcv_df.ta.bbands(length=5, std=2.0, append=True) # Match training

        ohlcv_df['body_size'] = abs(ohlcv_df['close'] - ohlcv_df['open'])
        ohlcv_df['upper_wick'] = ohlcv_df['high'] - ohlcv_df[['open', 'close']].max(axis=1)
        ohlcv_df['lower_wick'] = ohlcv_df[['open', 'close']].min(axis=1) - ohlcv_df['low']
        ohlcv_df['price_change'] = ohlcv_df['close'].diff()
        ohlcv_df['volume_change'] = ohlcv_df['volume'].diff()

        # Select only the required features and the latest row
        latest_row = ohlcv_df.iloc[-1:]

        # Check if all required features are present in the latest row's columns
        if not all(f in latest_row.columns for f in FEATURE_NAMES):
             missing = [f for f in FEATURE_NAMES if f not in latest_row.columns]
             print(f"Error: Missing expected feature columns after calculation: {missing}")
             return None

        latest_features_df = latest_row[FEATURE_NAMES]

        # Check for NaNs in the final feature set for the latest row
        if latest_features_df.isnull().any().any():
             print("Warning: NaN values found in the latest calculated features row. Prediction cannot proceed.")
             print(latest_features_df[latest_features_df.isnull().any(axis=1)]) # Show row with NaNs
             return None # Cannot predict with NaNs

        return latest_features_df

    except Exception as e:
        print(f"Error calculating latest features: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None

def generate_signal(latest_features_df, model_path, scaler_path):
    """
    Generates a trading signal based on the latest features using the loaded model.

    Args:
        latest_features_df (pd.DataFrame): DataFrame containing the features for the *single* latest data point.
        model_path (str): Path to the saved model file.
        scaler_path (str): Path to the saved scaler file.

    Returns:
        str: "BUY", "SELL", or "HOLD". Returns "HOLD" on failure.
    """
    global FEATURE_NAMES # Use the globally defined feature names
    if not FEATURE_NAMES:
        print("Error: FEATURE_NAMES list is empty. Cannot generate signal.")
        return "HOLD"
    if latest_features_df is None or latest_features_df.empty:
        print("Cannot generate signal: No valid latest features provided.")
        return "HOLD"
    if len(latest_features_df) != 1:
         print(f"Warning: generate_signal expected 1 row, got {len(latest_features_df)}. Using the last one.")
         latest_features_df = latest_features_df.iloc[-1:]

    try:
        # Load the model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # Ensure features are in the same order as during training
        # The calculate_latest_features should already return them in the correct order if FEATURE_NAMES is correct
        if not all(f in latest_features_df.columns for f in FEATURE_NAMES):
             missing = [f for f in FEATURE_NAMES if f not in latest_features_df.columns]
             print(f"Error: Missing expected features in input for prediction: {missing}")
             return "HOLD"
        latest_features_ordered = latest_features_df[FEATURE_NAMES]


        # Scale the features
        latest_features_scaled = scaler.transform(latest_features_ordered)

        # Make prediction
        prediction = model.predict(latest_features_scaled)[0] # [0] because predict returns an array
        probabilities = model.predict_proba(latest_features_scaled)[0] # [prob_0, prob_1]

        print(f"  ML Prediction: {prediction} (0: Down/Stay, 1: Up)")
        print(f"  Probabilities: [P(Down/Stay)={probabilities[0]:.4f}, P(Up)={probabilities[1]:.4f}]")

        # --- Signal Logic ---
        # Basic logic: Predict 1 -> BUY, Predict 0 -> SELL
        # TODO: Enhance with probability thresholds, price action confirmation, risk management rules.
        signal = "HOLD" # Default
        if prediction == 1:
            # Basic: Predict 1 -> BUY
            print("  ML predicts UP -> BUY Signal (Simple Logic)")
            signal = "BUY"
        else: # prediction == 0
            # Basic: Predict 0 -> SELL
            print("  ML predicts DOWN/STAY -> SELL Signal (Simple Logic)")
            signal = "SELL"

        return signal

    except FileNotFoundError:
        print(f"Error: Model '{model_path}' or Scaler '{scaler_path}' not found.")
        return "HOLD"
    except Exception as e:
        print(f"Error generating signal: {e}")
        import traceback
        traceback.print_exc()
        return "HOLD"


def run_live_trading_cycle(model_path, scaler_path):
    """Performs one cycle of fetching data, calculating features, and generating signal."""
    global last_signal_sent, last_signal_timestamp
    print(f"\n--- Running Live Cycle: {datetime.now()} ---")

    # 1. Fetch recent data
    # Fetch enough history for feature calculation
    ohlcv_df = fetch_recent_candles(SYMBOL, INTERVAL, REQUIRED_HISTORY_FOR_LIVE_FEATURES)
    if ohlcv_df is None or len(ohlcv_df) < REQUIRED_HISTORY_FOR_LIVE_FEATURES:
        print("Failed to fetch sufficient recent data for feature calculation. Skipping cycle.")
        return
    if ohlcv_df.index.duplicated().any():
        print("Warning: Duplicate timestamps found in fetched recent data. Dropping duplicates.")
        ohlcv_df = ohlcv_df[~ohlcv_df.index.duplicated(keep='first')]


    # Get the timestamp and price of the latest *complete* candle
    # The API might return a partial candle for the current interval, features might be unreliable.
    # We should ideally only predict on fully closed candles.
    # Let's assume the second to last row is the last fully completed candle.
    if len(ohlcv_df) < 2:
        print("Not enough data points after fetching to determine the last completed candle. Skipping cycle.")
        return

    last_complete_candle = ohlcv_df.iloc[-2] # Use the second to last row
    latest_candle_timestamp = last_complete_candle.name # Index is the timestamp
    latest_price = last_complete_candle['close']
    print(f"Processing based on last completed candle: Timestamp={latest_candle_timestamp}, Close Price={latest_price:.2f}")

    # 2. Calculate Features for the latest data point
    # We need to pass the dataframe ending with the candle we want to predict *for*
    # So, we pass the dataframe up to and including the second-to-last candle
    features_df_for_prediction = ohlcv_df[:-1] # Exclude the potentially incomplete latest candle
    if len(features_df_for_prediction) < REQUIRED_HISTORY_FOR_LIVE_FEATURES:
         print(f"Not enough historical data ({len(features_df_for_prediction)}) before the latest candle to calculate features reliably. Skipping.")
         return

    print(f"Calculating features using data up to {latest_candle_timestamp}...")
    latest_features = calculate_latest_features(features_df_for_prediction.copy()) # Pass a copy

    if latest_features is None:
        print("Failed to calculate features for the latest completed candle. Skipping cycle.")
        return

    # 3. Generate Signal using the trained model
    print("Generating signal...")
    current_signal = generate_signal(latest_features, model_path, scaler_path)
    print(f"---> Generated Signal: {current_signal}")
    print(f"     Based on data up to: {latest_candle_timestamp}")


    # 4. Log Signal and Send Notification
    if current_signal in ["BUY", "SELL"]:
        # Log the signal: timestamp;SIGNAL;price
        log_message = f"{latest_candle_timestamp};{current_signal};{latest_price:.8f}" # Log with more precision
        logging.info(log_message)
        print(f"Signal logged: {log_message}")

        # Check if it's a new signal for this specific candle timestamp before sending email
        if current_signal != last_signal_sent or latest_candle_timestamp != last_signal_timestamp:
            print(f"New signal ({current_signal}) detected for {latest_candle_timestamp}. Preparing email notification...")
            subscribers = get_subscribers() # Get current list of subscribers
            if subscribers:
                success = send_email_notification(current_signal, latest_candle_timestamp, latest_price, subscribers)
                if success:
                    print("Email notifications sent (or attempted).")
                    last_signal_sent = current_signal # Update last sent signal
                    last_signal_timestamp = latest_candle_timestamp
                else:
                    print("Failed to send one or more email notifications.")
            else:
                print("No subscribers found, skipping email notifications.")
        else:
            print(f"Signal ({current_signal}) is the same as the last one sent for this timestamp ({latest_candle_timestamp}). No new notification sent.")
    else: # HOLD signal
        print("Signal is HOLD. No log entry or notification.")
        # Resetting last signal on HOLD means the next BUY/SELL for the *same* timestamp would trigger again if the loop runs fast.
        # It's better to only compare against the last signal *sent* for a *specific* timestamp.
        # The current logic already handles this by checking timestamp inequality.

    print(f"--- Live Cycle Completed: {datetime.now()} ---")


def live_mode_loop(model_path, scaler_path):
    """Main loop for continuous live signal generation."""
    global last_processed_interval_start # Use the global variable
    print("\n===================================")
    print(" Entering Live Signal Generation Mode ")
    print(f" Checking every {POLL_INTERVAL_SECONDS}s for new {INTERVAL} candle completion.")
    print("===================================")

    while True:
        now = datetime.now()
        interval_duration = timedelta(milliseconds=get_interval_milliseconds(INTERVAL))

        # Calculate the start time of the *most recently completed* interval
        # Example: If interval is 15m and time is 10:23, the last completed interval started at 10:15.
        # If time is 10:15:05, the last completed interval started at 10:00.
        current_minute = now.minute
        minutes_past_interval = current_minute % int(INTERVAL[:-1]) # e.g., 23 % 15 = 8
        last_interval_start_minute = current_minute - minutes_past_interval
        last_completed_interval_start_time = now.replace(minute=last_interval_start_minute, second=0, microsecond=0)

        # Check if this completed interval is newer than the last one we processed
        if last_processed_interval_start is None or last_completed_interval_start_time > last_processed_interval_start:
            print(f"\n*** New {INTERVAL} candle completed at {last_completed_interval_start_time}. Running live cycle... ***")
            run_live_trading_cycle(model_path, scaler_path)
            last_processed_interval_start = last_completed_interval_start_time # Update the last processed time
        # else:
            # Optional: print a message indicating it's not time yet
            # print(f".", end='', flush=True) # Simple dot indicator

        # Wait before checking again
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    print("--- Trading Bot Initialization ---")

    # 1. Fetch Historical Data
    print(f"\n[Step 1/4] Fetching {YEARS_OF_DATA} year(s) of historical data for {SYMBOL} ({INTERVAL})...")
    historical_df = get_historical_data(SYMBOL, INTERVAL, years_back=YEARS_OF_DATA)

    if historical_df is None or historical_df.empty:
        print("Failed to fetch historical data. Exiting.")
        exit()
    else:
        # Optional: Save the raw historical data
        # historical_df.to_csv(HISTORICAL_DATA_CSV)
        # print(f"Raw historical data saved to {HISTORICAL_DATA_CSV}")
        print("Historical data fetched successfully.")

    # 2. Create Features
    print(f"\n[Step 2/4] Creating features and target variable (lookahead={LOOKAHEAD_PERIODS})...")
    features_df = create_features(historical_df.copy(), n_periods=LOOKAHEAD_PERIODS) # Pass a copy

    if features_df is None or features_df.empty:
        print("Failed to create features. Exiting.")
        exit()
    else:
        # Save features data
        features_df.to_csv(FEATURES_DATA_CSV)
        print(f"Features data saved to {FEATURES_DATA_CSV}")
        print("Features created successfully.")

    # 3. Train Model
    print(f"\n[Step 3/4] Training model...")
    training_successful = train_evaluate_model(features_df, MODEL_SAVE_PATH, SCALER_SAVE_PATH)

    if not training_successful:
        print("Model training failed. Exiting.")
        exit()
    else:
        print("Model training completed successfully.")

    # 4. Start Live Mode
    print("\n[Step 4/4] Initial setup complete. Starting live signal generation loop...")
    # Check if model and scaler files were actually created before starting live mode
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
         live_mode_loop(MODEL_SAVE_PATH, SCALER_SAVE_PATH)
    else:
         print(f"Error: Model ({MODEL_SAVE_PATH}) or Scaler ({SCALER_SAVE_PATH}) not found after training. Cannot start live mode.")
         exit()
