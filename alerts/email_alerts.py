import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from loguru import logger
from signal_engine.models import Signal, SignalType


def create_email_html(signal: Signal) -> str:
    """Render clean HTML email template for trading signal."""
    reasons_li = "".join(f"<li>{r}</li>" for r in signal.reasons)
    color = "#10B981" if "BUY" in signal.signal_type.value else ("#EF4444" if "SELL" in signal.signal_type.value else "#F59E0B")

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
            .header {{ background-color: {color}; color: white; padding: 20px; text-align: center; }}
            .content {{ padding: 24px; }}
            .badge {{ font-size: 24px; font-weight: bold; }}
            .metrics {{ background-color: #f9fafb; padding: 16px; border-radius: 6px; margin: 16px 0; }}
            .disclaimer {{ font-size: 11px; color: #6b7280; margin-top: 24px; border-top: 1px solid #eee; padding-top: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="badge">{signal.signal_type.value}</div>
                <h2>{signal.symbol}</h2>
            </div>
            <div class="content">
                <div class="metrics">
                    <p><strong>Current Price:</strong> ${signal.current_price:.2f}</p>
                    <p><strong>Confidence:</strong> {signal.confidence}%</p>
                    <p><strong>RSI:</strong> {signal.rsi or 'N/A'}</p>
                    <p><strong>Historical Win Rate:</strong> {signal.win_rate or 'N/A'}%</p>
                </div>
                <h3>Technical Indicators Confluence:</h3>
                <ul>{reasons_li}</ul>
                <div class="disclaimer">
                    Automated technical signal from Finance AI. This is analysis based on historical patterns, not financial advice.
                </div>
            </div>
        </div>
    </body>
    </html>
    """


def send_email_alert(
    signal: Signal,
    to_email: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
) -> bool:
    """Send trade signal alert via SMTP email."""
    if not (to_email and smtp_host and smtp_user and smtp_password):
        return False

    subject = f"[{signal.signal_type.value}] {signal.symbol} @ ${signal.current_price:.2f} (Confidence: {signal.confidence}%)"
    html_content = create_email_html(signal)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        logger.info(f"Email alert sent to {to_email} for {signal.symbol}")
        return True
    except Exception as e:
        logger.error(f"Error sending email alert: {e}")
        return False
