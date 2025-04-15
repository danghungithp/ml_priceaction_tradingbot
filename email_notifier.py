import smtplib
import ssl
from email.message import EmailMessage
from datetime import datetime
import os # To potentially use environment variables later

# --- Configuration (User needs to replace placeholders securely!) ---
# WARNING: Do NOT hardcode your real password here.
# Option 1: Use Environment Variables (Recommended)
# Example: Set environment variables 'GMAIL_SENDER' and 'GMAIL_PASSWORD'
# SENDER_EMAIL = os.environ.get("GMAIL_SENDER")
# SENDER_PASSWORD = os.environ.get("GMAIL_PASSWORD")

# Option 2: Replace placeholders directly (Less Secure - Ensure this file is NOT committed to Git)
# If using Gmail with 2FA, generate an App Password: https://myaccount.google.com/apppasswords
SENDER_EMAIL = "mcpsoftware@gmail.com"  # <<< REPLACE THIS with your Gmail address
SENDER_PASSWORD = "cdtz inwd vfwr dqgu"  # <<< REPLACE THIS with your Gmail App Password or regular password
RECIPIENT_EMAIL = "mcpsoftware@gmail.com" # User-provided recipient

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587 # For starttls

def send_email_notification(signal, timestamp, price):
    """Sends an email notification for a trading signal."""

    if not SENDER_EMAIL or SENDER_EMAIL == "your_email@gmail.com" or not SENDER_PASSWORD or SENDER_PASSWORD == "your_app_password":
        print("Error: Sender email or password not configured in email_notifier.py. Cannot send email.")
        print("Please replace the placeholder values securely.")
        return False

    subject = f"Trading Signal Alert: {signal} BTC/USDT"
    body = f"""
    A trading signal has been generated:

    Signal:      {signal}
    Timestamp:   {timestamp}
    Price (approx): {price:.2f} USDT

    This is an automated notification based on the ML model prediction.
    Please review the charts and apply your own analysis before taking action.
    """

    em = EmailMessage()
    em['From'] = SENDER_EMAIL
    em['To'] = RECIPIENT_EMAIL
    em['Subject'] = subject
    em.set_content(body)

    # Add SSL layer
    context = ssl.create_default_context()

    try:
        print(f"Attempting to send email notification to {RECIPIENT_EMAIL}...")
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls(context=context) # Secure the connection
            smtp.login(SENDER_EMAIL, SENDER_PASSWORD)
            smtp.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, em.as_string())
        print("Email notification sent successfully.")
        return True
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

# Example usage (for testing purposes - run 'python email_notifier.py')
if __name__ == "__main__":
    print("--- Testing Email Notifier ---")
    # IMPORTANT: Replace placeholders above before running this test directly.
    if SENDER_EMAIL == "your_email@gmail.com" or SENDER_PASSWORD == "your_app_password":
        print("\n*** WARNING: Please replace placeholder email credentials in email_notifier.py before testing! ***\n")
    else:
        print("Attempting to send a test email...")
        test_signal = "TEST_BUY"
        test_timestamp = datetime.now()
        test_price = 99999.99 # Example price
        success = send_email_notification(test_signal, test_timestamp, test_price)
        if success:
            print("Test email function executed successfully (check recipient's inbox).")
        else:
            print("Test email function failed.")
