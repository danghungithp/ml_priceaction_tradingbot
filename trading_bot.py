import time
import requests
import pandas as pd
from datetime import datetime, timedelta
import schedule # For scheduling tasks

# Import functions from our other modules
from fetch_data import get_interval_milliseconds # For timing calculations
from signal_generator import calculate_latest_features, generate_signal, TRAINED_FEATURES
from email_notifier import send_email_notification

# --- Constants ---
SYMBOL = "BTC_USDT"
INTERVAL = "15m" # Must match the interval used for training
INTERVAL_MS = get_interval_milliseconds(INTERVAL)
API_BASE_URL = "https://spot-markets.goonus.io/candlesticks"
REQUIRED_HISTORY_FOR_FEATURES = 60 # Number of candles needed for feature calculation (adjust based on longest lookback + buffer)
POLL_INTERVAL_SECONDS = 60 # How often to check if it's time for a new candle (e.g., every minute)

# --- Global variable to track last signal to avoid duplicates ---
last_signal_sent = None
last_signal_timestamp = None

def fetch_recent_candles(symbol, interval, limit):
    """Fetches the most recent 'limit' candles."""
    params = {
        "symbol_name": symbol,
        "interval": interval,
        "limit": limit # Assuming the API supports a 'limit' parameter for recent candles
    }
    # If API doesn't support limit, we might need to fetch based on time (more complex)
    # end_time = int(time.time() * 1000)
    # start_time = end_time - limit * INTERVAL_MS
    # params = {"symbol_name": symbol, "interval": interval, "from": start_time, "to": end_time}

    print(f"Fetching last {limit} candles for {symbol} ({interval})...")
    try:
        response = requests.get(API_BASE_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            # Rename and select columns (similar to fetch_data.py)
            df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'q': 'volume'}, inplace=True)
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df['timestamp'] = pd.to_numeric(df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col])
            df.sort_values('timestamp', inplace=True)
            df.set_index('timestamp', inplace=True)
            print(f"Fetched {len(df)} candles successfully.")
            return df
        else:
            print(f"Warning: API returned no data or unexpected format: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching recent candles: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching recent candles: {e}")
        return None

def run_bot_logic():
    """The main logic to run on each interval."""
    global last_signal_sent, last_signal_timestamp
    print(f"\n--- Running Bot Logic at {datetime.now()} ---")

    # 1. Fetch recent data
    ohlcv_df = fetch_recent_candles(SYMBOL, INTERVAL, REQUIRED_HISTORY_FOR_FEATURES)
    if ohlcv_df is None or len(ohlcv_df) < REQUIRED_HISTORY_FOR_FEATURES:
        print("Failed to fetch sufficient recent data. Skipping cycle.")
        return

    # Get the timestamp of the latest complete candle
    latest_candle_timestamp = ohlcv_df.index[-1]
    latest_price = ohlcv_df['close'].iloc[-1]
    print(f"Latest candle timestamp: {latest_candle_timestamp}, Price: {latest_price:.2f}")

    # 2. Calculate Features
    print("Calculating latest features...")
    latest_features = calculate_latest_features(ohlcv_df.copy()) # Pass a copy

    if latest_features is None:
        print("Failed to calculate features. Skipping cycle.")
        return

    # 3. Generate Signal
    print("Generating signal...")
    current_signal = generate_signal(latest_features)
    print(f"Generated signal: {current_signal}")

    # 4. Send Notification (if signal is BUY/SELL and different from last sent)
    if current_signal in ["BUY", "SELL"]:
        if current_signal != last_signal_sent or latest_candle_timestamp != last_signal_timestamp:
            print(f"New signal ({current_signal}) detected. Sending notification...")
            success = send_email_notification(current_signal, latest_candle_timestamp, latest_price)
            if success:
                last_signal_sent = current_signal # Update last sent signal
                last_signal_timestamp = latest_candle_timestamp
            else:
                print("Failed to send email notification.")
        else:
            print(f"Signal ({current_signal}) is the same as the last one sent for this timestamp. No notification sent.")
    else: # HOLD signal
        print("Signal is HOLD. No notification sent.")
        # Optionally reset last_signal if you want a notification on the next BUY/SELL
        # last_signal_sent = None
        # last_signal_timestamp = None

    print(f"--- Bot Logic Cycle Completed at {datetime.now()} ---")


def main_loop():
    """Main loop checking time and running bot logic."""
    print("Starting Trading Bot...")
    print(f"Running checks every {POLL_INTERVAL_SECONDS} seconds for the start of the next {INTERVAL} candle.")

    while True:
        now = datetime.now()
        # Calculate the timestamp of the start of the *next* 15-minute interval
        next_run_minute = (now.minute // 15 + 1) * 15
        if next_run_minute >= 60:
            next_run_hour = now.hour + 1
            next_run_minute = 0
            if next_run_hour >= 24: # Rollover to next day
                 next_run_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                 next_run_time = now.replace(hour=next_run_hour, minute=0, second=0, microsecond=0)
        else:
            next_run_time = now.replace(minute=next_run_minute, second=0, microsecond=0)

        # Calculate the timestamp of the start of the *current* or *last completed* 15-minute interval
        current_interval_minute = (now.minute // 15) * 15
        current_interval_start_time = now.replace(minute=current_interval_minute, second=0, microsecond=0)

        # Check if we are near the start of the next interval (e.g., within the polling interval)
        # and if the last completed interval is different from the last time we ran the logic
        # This prevents running multiple times within the same candle interval
        time_to_run = (now >= next_run_time - timedelta(seconds=POLL_INTERVAL_SECONDS))
        
        # We need a variable to track the timestamp of the last interval processed
        global last_processed_interval_start
        try:
            last_processed_interval_start
        except NameError:
            last_processed_interval_start = None # Initialize if it doesn't exist

        if time_to_run and current_interval_start_time != last_processed_interval_start :
            print(f"\n*** Detected start of new interval ({current_interval_start_time}). Running logic... ***")
            run_bot_logic()
            last_processed_interval_start = current_interval_start_time # Update last processed time
            # Wait a bit longer after running to ensure we don't double-trigger
            wait_time = (next_run_time - datetime.now()).total_seconds() + 5 # Wait until 5s past next interval start
            if wait_time > 0:
                 print(f"Waiting {wait_time:.0f} seconds for next check cycle after run...")
                 time.sleep(wait_time)
            else:
                 # If calculation took too long, just proceed to next check cycle
                 time.sleep(POLL_INTERVAL_SECONDS)

        else:
            # Wait for the polling interval before checking again
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    # Initialize the variable for tracking processed intervals
    last_processed_interval_start = None
    # You might want to run the logic once immediately on startup
    # print("Running initial logic cycle on startup...")
    # run_bot_logic()
    # last_processed_interval_start = datetime.now().replace(minute=(datetime.now().minute // 15) * 15, second=0, microsecond=0)

    main_loop() # Start the main checking loop

    # --- Alternative using 'schedule' library (requires 'pip install schedule') ---
    # import schedule
    # print("Scheduling bot logic to run every 15 minutes...")
    # schedule.every(15).minutes.at(":01").do(run_bot_logic) # Run at :01, :16, :31, :46 past the hour
    #
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
    # ---------------------------------------------------------------------------
