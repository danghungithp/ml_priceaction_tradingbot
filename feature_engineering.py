import pandas as pd
import pandas_ta as ta # Import pandas-ta

def create_features(df, n_periods=5):
    """
    Adds technical indicators and price action features to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data, indexed by timestamp.
        n_periods (int): Number of periods to look ahead for target definition.

    Returns:
        pd.DataFrame: DataFrame with added features and target variable.
    """
    print("Starting feature engineering...")

    # --- Technical Indicators using pandas_ta ---
    # Calculate SMA (Simple Moving Average) for different periods
    print("Calculating SMAs...")
    df.ta.sma(length=10, append=True) # SMA_10
    df.ta.sma(length=20, append=True) # SMA_20
    df.ta.sma(length=50, append=True) # SMA_50

    # Calculate RSI (Relative Strength Index)
    print("Calculating RSI...")
    df.ta.rsi(length=14, append=True) # RSI_14

    # Calculate MACD (Moving Average Convergence Divergence)
    print("Calculating MACD...")
    df.ta.macd(append=True) # Uses default lengths (12, 26, 9) -> MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

    # Calculate Bollinger Bands
    print("Calculating Bollinger Bands...")
    df.ta.bbands(append=True) # Uses default length 20, std 2 -> BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

    # --- Price Action Features ---
    print("Calculating Price Action features...")
    # Candle body size
    df['body_size'] = abs(df['close'] - df['open'])
    # Upper wick size
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower wick size
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    # Price change from previous candle
    df['price_change'] = df['close'].diff()
    # Volume change from previous candle
    df['volume_change'] = df['volume'].diff()

    # --- Target Variable ---
    # Predict if the price will increase N periods later
    print(f"Defining target variable (price increase after {n_periods} periods)...")
    df['target'] = (df['close'].shift(-n_periods) > df['close']).astype(int)

    # --- Clean up ---
    # Remove rows with NaN values created by indicators/diff()
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"Removed {initial_rows - final_rows} rows with NaN values.")

    print("Feature engineering completed.")
    print(f"Final DataFrame shape: {df.shape}")
    print("\nColumns added:")
    print(df.columns) # Print all columns to see the new features

    return df

if __name__ == "__main__":
    input_filename = "BTC_USDT_15m_1y.csv"
    output_filename = "features_data.csv"
    lookahead_periods = 5 # Predict trend over next 5 * 15 minutes = 75 minutes

    try:
        print(f"Reading data from {input_filename}...")
        # Ensure 'timestamp' column is parsed as datetime and set as index
        data_df = pd.read_csv(input_filename, index_col='timestamp', parse_dates=True)
        print(f"Initial data shape: {data_df.shape}")

        # Create features
        features_df = create_features(data_df.copy(), n_periods=lookahead_periods) # Use copy to avoid modifying original df

        if features_df is not None and not features_df.empty:
            print("\nSample of data with features:")
            print(features_df.head())

            # Save the features DataFrame
            print(f"\nSaving features data to {output_filename}...")
            features_df.to_csv(output_filename)
            print("Features data saved successfully.")
        else:
            print("\nFailed to create features or resulting DataFrame is empty.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please run fetch_data.py first.")
    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")
