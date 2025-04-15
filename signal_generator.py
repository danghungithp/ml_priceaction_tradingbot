import pandas as pd
import joblib
import pandas_ta as ta # Needed for feature calculation on new data if not already done

# --- Constants ---
MODEL_PATH = "trading_model.joblib"
SCALER_PATH = "trading_model_scaler.joblib"
FEATURES_DATA_PATH = "features_data.csv" # To get latest data for simulation

# List of features used during training (must match the order in train_model.py)
# Get this list dynamically or ensure it's consistent
# Example: Manually define based on train_model.py output
TRAINED_FEATURES = [
    'SMA_10', 'SMA_20', 'SMA_50', 'RSI_14', 'MACD_12_26_9',
    'MACDh_12_26_9', 'MACDs_12_26_9', 'BBL_5_2.0', 'BBM_5_2.0',
    'BBU_5_2.0', 'BBB_5_2.0', 'BBP_5_2.0', 'body_size', 'upper_wick',
    'lower_wick', 'price_change', 'volume_change'
]

def calculate_latest_features(ohlcv_df):
    """
    Calculates the necessary features for the latest data point(s).
    This function mirrors the feature calculation in feature_engineering.py
    but should be optimized for calculating only the needed recent features.

    Args:
        ohlcv_df (pd.DataFrame): DataFrame containing recent OHLCV data.
                                 Needs enough historical data to calculate indicators.

    Returns:
        pd.DataFrame: DataFrame with features calculated for the latest row(s).
                      Returns None if calculation fails.
    """
    try:
        # Ensure enough data for longest lookback (e.g., SMA_50 needs 50 periods)
        required_rows = 50 # Based on SMA_50
        if len(ohlcv_df) < required_rows:
            print(f"Error: Not enough historical data ({len(ohlcv_df)}) to calculate features (need {required_rows}).")
            return None

        # Calculate indicators using pandas_ta
        ohlcv_df.ta.sma(length=10, append=True)
        ohlcv_df.ta.sma(length=20, append=True)
        ohlcv_df.ta.sma(length=50, append=True)
        ohlcv_df.ta.rsi(length=14, append=True)
        ohlcv_df.ta.macd(append=True)
        # Note: Using bbands length 5 as per train_model output, adjust if needed
        ohlcv_df.ta.bbands(length=5, std=2.0, append=True) # Match training features

        # Calculate Price Action features
        ohlcv_df['body_size'] = abs(ohlcv_df['close'] - ohlcv_df['open'])
        ohlcv_df['upper_wick'] = ohlcv_df['high'] - ohlcv_df[['open', 'close']].max(axis=1)
        ohlcv_df['lower_wick'] = ohlcv_df[['open', 'close']].min(axis=1) - ohlcv_df['low']
        ohlcv_df['price_change'] = ohlcv_df['close'].diff()
        ohlcv_df['volume_change'] = ohlcv_df['volume'].diff()

        # Return only the latest row(s) with all features calculated
        # The last row will have all features needed for prediction
        latest_features = ohlcv_df.iloc[-1:] # Get the last row as a DataFrame
        
        # Check for NaNs in the latest features row
        if latest_features[TRAINED_FEATURES].isnull().any().any():
             print("Warning: NaN values found in the latest calculated features. Prediction might be unreliable.")
             # Optionally, handle NaNs (e.g., fill with 0 or mean, though risky)
             # latest_features.fillna(0, inplace=True) # Example: fill NaNs with 0 - use with caution!
             return None # Safer to return None if NaNs are present

        return latest_features[TRAINED_FEATURES] # Return only the required feature columns

    except Exception as e:
        print(f"Error calculating latest features: {e}")
        return None


def generate_signal(latest_features_df):
    """
    Generates a trading signal based on the latest features using the loaded model.

    Args:
        latest_features_df (pd.DataFrame): DataFrame containing the features for the latest data point.

    Returns:
        str: "BUY", "SELL", or "HOLD" signal. Returns "HOLD" if prediction fails.
    """
    if latest_features_df is None or latest_features_df.empty:
        print("Cannot generate signal: No valid features provided.")
        return "HOLD"

    try:
        # Load the model and scaler
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        # Ensure features are in the correct order
        latest_features_ordered = latest_features_df[TRAINED_FEATURES]

        # Scale the features
        latest_features_scaled = scaler.transform(latest_features_ordered)

        # Make prediction (predicts class 0 or 1)
        prediction = model.predict(latest_features_scaled)[0]
        # predict_proba gives probability for each class: [prob_class_0, prob_class_1]
        # probabilities = model.predict_proba(latest_features_scaled)[0]

        print(f"Prediction for latest data: {prediction}") # 0: Down/Stay, 1: Up
        # print(f"Probabilities: [P(Down/Stay)={probabilities[0]:.4f}, P(Up)={probabilities[1]:.4f}]")

        # --- Basic Signal Logic ---
        # This is a very simple example. Real-world logic would be more complex,
        # incorporating probabilities, thresholds, and Price Action rules.
        if prediction == 1:
            # Potential BUY signal based on ML prediction
            # TODO: Add Price Action confirmation rules here
            # Example PA rule: Is the last candle bullish (close > open)?
            # last_open = latest_ohlcv['open'].iloc[-1] # Need OHLCV data passed here too
            # last_close = latest_ohlcv['close'].iloc[-1]
            # if last_close > last_open: # Simple bullish candle check
            #     print("ML predicts UP + Bullish Candle -> BUY Signal")
            #     return "BUY"
            # else:
            #     print("ML predicts UP but candle not bullish -> HOLD Signal")
            #     return "HOLD"
            print("ML predicts UP -> BUY Signal (Simple Logic)")
            return "BUY"
        else: # prediction == 0
            # Potential SELL/HOLD signal based on ML prediction
            # TODO: Add Price Action confirmation rules here
            print("ML predicts DOWN/STAY -> SELL/HOLD Signal (Simple Logic)")
            # For now, let's just treat 0 as HOLD/SELL (no shorting assumed yet)
            return "SELL" # Or "HOLD" depending on strategy

    except FileNotFoundError:
        print(f"Error: Model '{MODEL_PATH}' or Scaler '{SCALER_PATH}' not found.")
        return "HOLD"
    except Exception as e:
        print(f"Error generating signal: {e}")
        return "HOLD"

if __name__ == "__main__":
    print("--- Signal Generation Simulation ---")
    try:
        # --- Simulate getting new data ---
        # In a real bot, you'd fetch the latest OHLCV data from the API.
        # Here, we load the original data and take the last N rows needed for feature calculation.
        print("Loading historical data to simulate latest data...")
        # Load original OHLCV + volume (adjust columns if needed)
        # Need enough rows to calculate longest lookback (e.g., 50 for SMA_50)
        required_rows_for_calc = 60 # A bit more than 50 for safety margin with diff() etc.
        all_ohlcv = pd.read_csv("BTC_USDT_15m_1y.csv", index_col='timestamp', parse_dates=True)
        latest_ohlcv_chunk = all_ohlcv.iloc[-required_rows_for_calc:].copy() # Get last N rows
        print(f"Using latest {len(latest_ohlcv_chunk)} data points for feature calculation.")

        # --- Calculate Features for the latest point ---
        print("\nCalculating features for the most recent data point...")
        latest_features = calculate_latest_features(latest_ohlcv_chunk)

        # --- Generate Signal ---
        if latest_features is not None:
            print("\nGenerating signal...")
            signal = generate_signal(latest_features)
            print(f"\n---> Generated Signal: {signal}")
            print(f"Based on data up to: {latest_features.index[0]}")
        else:
            print("\nCould not generate signal due to feature calculation issues.")

    except FileNotFoundError:
        print("Error: Data file 'BTC_USDT_15m_1y.csv' not found. Run fetch_data.py.")
    except Exception as e:
        print(f"An error occurred during simulation: {e}")
