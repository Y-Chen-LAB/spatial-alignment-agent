"""Send a paper2bench notification.

Delivery is tried in this order:
  1. SMTP email, if the full SMTP env is configured (recommended for Gmail via app password).
  2. macOS `display notification` via osascript (always available on darwin).
  3. Stderr warning (no-op).

Config via environment:
  P2B_NOTIFY_TO            destination email (enables SMTP attempt)
  P2B_NOTIFY_FROM          from address; defaults to P2B_NOTIFY_SMTP_USER
  P2B_NOTIFY_SMTP_HOST     e.g. smtp.gmail.com
  P2B_NOTIFY_SMTP_PORT     default 587 (STARTTLS)
  P2B_NOTIFY_SMTP_USER     SMTP username
  P2B_NOTIFY_SMTP_PASS     SMTP password (Gmail: generate an App Password)

Usage:
  python scripts/notify.py --subject "SUBJECT" --body "BODY"
  python scripts/notify.py --subject "SUBJECT" --body-file path/to/body.txt
"""

from __future__ import annotations

import argparse
import os
import platform
import smtplib
import subprocess
import sys
from email.message import EmailMessage
from pathlib import Path


def _smtp_configured() -> bool:
    return all(os.environ.get(k) for k in ("P2B_NOTIFY_TO", "P2B_NOTIFY_SMTP_HOST", "P2B_NOTIFY_SMTP_USER", "P2B_NOTIFY_SMTP_PASS"))


def _send_smtp(subject: str, body: str) -> None:
    host = os.environ["P2B_NOTIFY_SMTP_HOST"]
    port = int(os.environ.get("P2B_NOTIFY_SMTP_PORT", "587"))
    user = os.environ["P2B_NOTIFY_SMTP_USER"]
    password = os.environ["P2B_NOTIFY_SMTP_PASS"]
    to_addr = os.environ["P2B_NOTIFY_TO"]
    from_addr = os.environ.get("P2B_NOTIFY_FROM", user)

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.send_message(msg)


def _send_osascript(subject: str, body: str) -> None:
    snippet = body.splitlines()[0] if body else ""
    if len(snippet) > 200:
        snippet = snippet[:197] + "..."
    title = subject.replace('"', '\\"')
    text = snippet.replace('"', '\\"')
    subprocess.run(
        ["osascript", "-e", f'display notification "{text}" with title "{title}"'],
        check=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a paper2bench notification via SMTP or macOS notification.")
    parser.add_argument("--subject", required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--body", default="")
    group.add_argument("--body-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    body = args.body_file.read_text(encoding="utf-8") if args.body_file else args.body

    if _smtp_configured():
        try:
            _send_smtp(args.subject, body)
            print(f"[notify] emailed {os.environ['P2B_NOTIFY_TO']}")
            return
        except Exception as exc:
            print(f"[notify] SMTP failed: {exc}", file=sys.stderr)

    if platform.system() == "Darwin":
        _send_osascript(args.subject, body)
        print("[notify] posted macOS notification")
        return

    print("[notify] no delivery method available; set P2B_NOTIFY_SMTP_* env vars to enable email.", file=sys.stderr)


if __name__ == "__main__":
    main()
