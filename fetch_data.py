import requests
import time
import json
import pandas as pd
from datetime import datetime, timedelta

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
    start_time = int((datetime.now() - timedelta(days=years_back * 90)).timestamp() * 1000)

    print(f"Fetching data for {symbol} ({interval}) from {datetime.fromtimestamp(start_time/1000)} to {datetime.fromtimestamp(end_time/1000)}")

    all_data = []
    fetch_start_time = start_time
    limit = 500 # Assume a limit of 500 candles per request, adjust if needed
    print(f"Fetching data in chunks (limit per request: {limit})...")

    while fetch_start_time < end_time:
        # Calculate the end time for this chunk request
        # Note: The API might have specific behavior for 'to' timestamp (inclusive/exclusive)
        # We fetch slightly overlapping ranges just in case, duplicates will be handled later.
        fetch_end_time = min(fetch_start_time + limit * get_interval_milliseconds(interval), end_time)

        params = {
            "symbol_name": symbol,
            "interval": interval,
            "from": fetch_start_time,
            "to": fetch_end_time,
            # Some APIs use a 'limit' parameter instead of 'to', check API docs if needed
            # "limit": limit
        }
        print(f"  Fetching chunk from {datetime.fromtimestamp(fetch_start_time/1000)} to {datetime.fromtimestamp(fetch_end_time/1000)}")

        try:
            response = requests.get(base_url, params=params, timeout=60) # Increased timeout for potentially larger requests
            response.raise_for_status()
            data_chunk = response.json()

            if isinstance(data_chunk, list) and len(data_chunk) > 0:
                all_data.extend(data_chunk)
                print(f"    Fetched {len(data_chunk)} candlesticks. Total fetched: {len(all_data)}")
                # Update the start time for the next chunk based on the last timestamp received
                # Ensure timestamps are integers
                last_timestamp = int(data_chunk[-1]['t']) # Get the timestamp of the last candle in the chunk
                fetch_start_time = last_timestamp + get_interval_milliseconds(interval) # Start next fetch after the last candle

                # Break if we received fewer candles than the limit, indicating we might be at the end
                if len(data_chunk) < limit:
                     print("    Reached end of data (received less than limit).")
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
            print(f"Error decoding JSON response for chunk: {response.text}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during chunk fetching: {e}")
            return None

    # --- Post-fetching processing ---
    if not all_data:
        print("No data fetched after attempting chunking.")
        return None

    print(f"\nTotal candlesticks fetched: {len(all_data)}")
    df = pd.DataFrame(all_data)

    # Rename columns for clarity based on sample data
    # "s": symbol, "i": interval, "o": open, "h": high, "l": low, "c": close,
    # "b": base volume?, "q": quote volume?, "t": open time, "u": close time?
    df.rename(columns={
        't': 'timestamp', # Assuming 't' is the standard open time
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'q': 'volume' # Assuming 'q' (quote volume) is more relevant than 'b'
    }, inplace=True)

    # Select and convert necessary columns
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Fix FutureWarning: Explicitly convert timestamp column to numeric before to_datetime
    df['timestamp'] = pd.to_numeric(df['timestamp'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])

    # Sort by time and remove duplicates (important when fetching chunks)
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset=['timestamp'], keep='first', inplace=True) # Keep the first occurrence

    df.set_index('timestamp', inplace=True)

    # Filter data to ensure it's within the original requested range (start_time to end_time)
    # Convert start_time and end_time to datetime objects for comparison
    start_dt = pd.to_datetime(start_time, unit='ms')
    end_dt = pd.to_datetime(end_time, unit='ms')
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]

    print(f"Data processed into DataFrame. Shape: {df.shape}")
    return df

# Removed the incorrect except blocks from here, they belong outside the main processing block

# The following except blocks should be at the same indentation level as the 'while' loop's 'try'
# This part seems to have been misplaced or duplicated during the previous replace operation.
# The correct structure should have the main data processing within the 'while' loop's 'try',
# and broader exceptions handled outside or differently.
# Let's assume the original structure intended these exceptions to catch errors *outside* the loop
# or during the final processing steps after the loop.
# Re-adding them at the correct level if they were meant to catch errors during final processing.
# However, the previous code structure had them indented incorrectly.
# Let's remove the faulty except blocks entirely for now, as the inner try-except handles chunk errors.
# A broader try-except around the whole function might be better if needed.

# Correcting the structure by removing the misplaced except blocks:
# The code block starting from 'except requests.exceptions.RequestException as e:'
# down to 'return None' before the 'get_interval_milliseconds' function definition
# seems to be the source of the indentation errors and logical misplacement.
# We will rely on the try-except block inside the while loop for fetching errors.

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


if __name__ == "__main__":
    symbol_to_fetch = "BTC_USDT" # Changed from ONX_USDT
    interval_to_fetch = "15m"    # Changed from 5m
    # Try fetching 3 months of data instead of 1 year to test API limits
    data_df = get_historical_data(symbol_to_fetch, interval_to_fetch, years_back=0.25)

    if data_df is not None:
        print("\nSample of fetched data:")
        print(data_df.head())
        print(data_df.tail())

        # Save to CSV
        output_filename = f"{symbol_to_fetch}_{interval_to_fetch}_1y.csv"
        try:
            data_df.to_csv(output_filename)
            print(f"\nData saved successfully to {output_filename}")
        except Exception as e:
            print(f"\nError saving data to CSV: {e}")
    else:
        print("\nFailed to fetch or process data.")
