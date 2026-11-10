"""Alerts: always written to logs/alerts.log; also sent to Discord if configured.

To get Discord alerts on your phone: in a Discord server you own, open
Server Settings, Integrations, Webhooks, New Webhook, copy the URL, then set
it as the environment variable DISCORD_WEBHOOK_URL.
"""
from __future__ import annotations

import json
import os
import urllib.request
from datetime import datetime
from pathlib import Path

ICONS = {"INFO": "🟢", "WARN": "🟡", "HALT": "🔴"}


class Alerter:
    def __init__(self, log_path: Path, webhook: str | None = None, enabled: bool = True):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.webhook = webhook if webhook is not None else os.environ.get("DISCORD_WEBHOOK_URL")
        self.enabled = enabled

    def send(self, level: str, message: str) -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} [{level}] {message}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        if not (self.enabled and self.webhook):
            return
        body = json.dumps({"content": f"{ICONS.get(level, '')} **{level}** {message}"[:1900]}).encode()
        req = urllib.request.Request(self.webhook, data=body,
                                     headers={"Content-Type": "application/json", "User-Agent": "powerbot"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception as e:  # an alert failing must never stop the desk
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S} [WARN] Discord alert failed: {e}\n")
