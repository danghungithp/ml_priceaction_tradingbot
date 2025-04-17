import os
import pandas as pd
import plotly
import plotly.graph_objects as go
import json
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from datetime import datetime

# --- Configuration ---
# Use constants defined in trading_bot.py if possible, or redefine here
SYMBOL = "BTC_USDT"
INTERVAL = "15m"
FEATURES_DATA_CSV = f"{SYMBOL}_{INTERVAL}_features.csv" # Data for chart
SIGNALS_LOG_FILE = "signals.log" # Log file with signals
SUBSCRIBERS_FILE = "subscribers.txt" # File for email subscribers
CHART_DATA_POINTS = 200 # Number of recent data points to show on the chart

# --- Flask App Setup ---
app = Flask(__name__)
# Secret key for flashing messages (important for security)
app.secret_key = os.urandom(24) # Generates a random secret key

# --- Helper Functions ---

def get_chart_data(csv_path=FEATURES_DATA_CSV, num_points=CHART_DATA_POINTS):
    """Loads historical data and prepares it for Plotly chart."""
    try:
        df = pd.read_csv(csv_path, index_col='timestamp', parse_dates=True)
        # Select recent data points for better performance/visualization
        df_recent = df.tail(num_points)
        return df_recent
    except FileNotFoundError:
        print(f"Error: Chart data file '{csv_path}' not found.")
        return pd.DataFrame() # Return empty DataFrame
    except Exception as e:
        print(f"Error reading chart data from '{csv_path}': {e}")
        return pd.DataFrame()

def get_signals_from_log(log_path=SIGNALS_LOG_FILE):
    """Reads signals from the log file."""
    signals = []
    try:
        with open(log_path, 'r') as f:
            for line in f:
                parts = line.strip().split(';')
                if len(parts) == 3:
                    # Format: LogTimestamp;SignalTimestamp;SIGNAL;PRICE
                    log_ts_str, signal_info = parts[0], parts[1] + ";" + parts[2] # Keep original signal log format
                    signal_parts = signal_info.split(';')
                    if len(signal_parts) == 3:
                         signal_ts_str, signal_type, price_str = signal_parts
                         try:
                             # Attempt to parse timestamp for sorting/display
                             signal_dt = datetime.strptime(signal_ts_str, '%Y-%m-%d %H:%M:%S')
                             signals.append({
                                 'timestamp': signal_dt,
                                 'signal': signal_type,
                                 'price': float(price_str)
                             })
                         except (ValueError, TypeError) as parse_error:
                             print(f"Warning: Could not parse signal line: {line.strip()} - Error: {parse_error}")
                             # Append raw if parsing fails? Or skip? Skipping for now.
    except FileNotFoundError:
        print(f"Info: Signals log file '{log_path}' not found. No signals to display yet.")
    except Exception as e:
        print(f"Error reading signals log '{log_path}': {e}")

    # Sort signals by timestamp, most recent first
    signals.sort(key=lambda x: x['timestamp'], reverse=True)
    return signals

def add_subscriber(email, filename=SUBSCRIBERS_FILE):
    """Adds an email to the subscribers file if not already present."""
    try:
        # Read existing subscribers to avoid duplicates
        existing_subscribers = set()
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                existing_subscribers = {line.strip() for line in f if line.strip()}

        if email not in existing_subscribers:
            with open(filename, 'a') as f:
                f.write(email + '\n')
            return True # Added successfully
        else:
            return False # Already exists
    except Exception as e:
        print(f"Error adding subscriber to '{filename}': {e}")
        return False # Indicate failure

# --- Flask Routes ---

@app.route('/')
def index():
    """Main page route."""
    # Get current time
    current_time = datetime.utcnow() # Using UTC for consistency

    # Prepare chart data
    df_chart = get_chart_data()
    signals = get_signals_from_log()

    fig = go.Figure()

    if not df_chart.empty:
        # Add Candlestick trace
        fig.add_trace(go.Candlestick(x=df_chart.index,
                                     open=df_chart['open'],
                                     high=df_chart['high'],
                                     low=df_chart['low'],
                                     close=df_chart['close'],
                                     name='Price'))

        # Add signals to the chart
        buy_signals = [s for s in signals if s['signal'] == 'BUY' and s['timestamp'] in df_chart.index]
        sell_signals = [s for s in signals if s['signal'] == 'SELL' and s['timestamp'] in df_chart.index]

        if buy_signals:
            fig.add_trace(go.Scatter(
                x=[s['timestamp'] for s in buy_signals],
                y=[s['price'] for s in buy_signals],
                mode='markers',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                name='Buy Signal'
            ))
        if sell_signals:
             fig.add_trace(go.Scatter(
                x=[s['timestamp'] for s in sell_signals],
                y=[s['price'] for s in sell_signals],
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                name='Sell Signal'
            ))

        fig.update_layout(
            title=f'{SYMBOL} Price Chart ({INTERVAL}) with Signals',
            xaxis_title='Timestamp',
            yaxis_title='Price (USDT)',
            xaxis_rangeslider_visible=False, # Hide range slider for cleaner look
            template='plotly_dark' # Use a dark theme
        )
    else:
         fig.update_layout(title='Chart Data Not Available', template='plotly_dark')


    # Convert plot to JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Get signals for the table (already fetched)
    signals_table = signals # Use the full list for the table

    return render_template('index.html',
                           graphJSON=graphJSON,
                           signals=signals_table,
                           symbol=SYMBOL,
                           now=current_time) # Pass 'now' to the template

@app.route('/subscribe', methods=['POST'])
def subscribe():
    """Handles email subscription form submission."""
    email = request.form.get('email')
    if email and '@' in email: # Basic validation
        added = add_subscriber(email)
        if added:
            flash(f'Email {email} subscribed successfully!', 'success')
        else:
            flash(f'Email {email} is already subscribed or an error occurred.', 'info')
    else:
        flash('Please enter a valid email address.', 'danger')
    return redirect(url_for('index')) # Redirect back to the main page

# --- API Routes (Optional - could be used for dynamic updates later) ---

@app.route('/api/signals')
def api_signals():
    """API endpoint to get signals data."""
    signals = get_signals_from_log()
    # Convert datetime objects to strings for JSON serialization
    for s in signals:
        s['timestamp'] = s['timestamp'].isoformat()
    return jsonify(signals)

# --- Main Execution ---
if __name__ == '__main__':
    # Make sure the subscribers file exists
    if not os.path.exists(SUBSCRIBERS_FILE):
        with open(SUBSCRIBERS_FILE, 'w') as f:
            pass # Create empty file if it doesn't exist
        print(f"Created empty subscribers file: {SUBSCRIBERS_FILE}")

    # Run the Flask app
    # Use host='0.0.0.0' to make it accessible externally (e.g., within Docker)
    app.run(debug=True, host='0.0.0.0', port=5000)
