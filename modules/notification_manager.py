"""
Notification Management Module

Provides SMTP email delivery and notification formatting for the AI Pricing
Intelligence Platform without changing existing analysis or reporting logic.
"""

from __future__ import annotations

import os
import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from typing import Dict, List, Optional, Tuple


def send_email_alert(
    subject: str,
    body: str,
    recipient: str,
    sender: str,
    smtp_server: str,
    smtp_port: int,
    smtp_password: str,
    attachments: Optional[List[Tuple[str, bytes, str]]] = None,
) -> Tuple[bool, str]:
    """Send a notification email using SMTP and return status.

    Args:
        subject: Email subject line.
        body: Plain text email body.
        recipient: Recipient email address.
        sender: Sender email address.
        smtp_server: SMTP host.
        smtp_port: SMTP port.
        smtp_password: SMTP app password.

    Returns:
        Tuple indicating success and a status message.
    """
    if not recipient:
        return False, "Recipient email address is required."
    if not sender:
        return False, "Sender email address is required."
    if not smtp_server:
        return False, "SMTP server is not configured."
    if not smtp_port:
        return False, "SMTP port is not configured."
    if not smtp_password:
        return False, "SMTP password is required."

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = recipient
        msg.set_content(body)
        for filename, content, mime_type in attachments or []:
            maintype, _, subtype = mime_type.partition("/")
            msg.add_attachment(
                content,
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=filename,
            )

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port, timeout=20) as smtp:
            smtp.ehlo()
            smtp.starttls(context=context)
            smtp.ehlo()
            smtp.login(sender, smtp_password)
            smtp.send_message(msg)

        return True, "Email sent successfully."
    except Exception as exc:
        return False, str(exc)


def load_env_smtp_settings() -> Dict[str, str]:
    """Return SMTP settings from environment variables."""
    return {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
        "smtp_port": os.getenv("SMTP_PORT", "587"),
        "smtp_email": os.getenv("SMTP_EMAIL", ""),
        "smtp_password": os.getenv("SMTP_PASSWORD", ""),
        "recipient_email": os.getenv("RECIPIENT_EMAIL", ""),
    }


def format_notification_body(title: str, lines: list[str]) -> str:
    """Create a clean plain-text notification email body."""
    body_lines = ["AI Pricing Intelligence Platform Notification", "", title, ""]
    for line in lines:
        body_lines.append(f"• {line}")
    body_lines.append("")
    body_lines.append("Generated automatically from pricing analysis.")
    return "\n".join(body_lines)
