from __future__ import annotations

import os
import threading
from collections.abc import Iterable
from datetime import datetime, timezone


class Heartbeat:
    """Append-only progress log on the network volume, glanceable from a phone.

    Redacts any known secret substring so a token can never leak into the log."""

    def __init__(self, path: str, secrets: Iterable[str] = ()) -> None:
        self._path = path
        self._secrets = [s for s in secrets if s]
        self._step = 0
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "***")
        return text

    def emit(self, stage: str, message: str = "") -> None:
        # Locked so the watchdog thread and the main eval thread can both call
        # emit() without racing on self._step or interleaving partial writes.
        with self._lock:
            self._step += 1
            # UTC timestamp so stage durations are readable post-mortem - "ran
            # ~1 min" was pure inference before this; a load phase that hangs vs
            # crashes looks very different once each line is stamped.
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            prefix = f"{ts} [{self._step:03d}] {stage}"
            line = self.redact(f"{prefix} | {message}" if message else prefix)
            with open(self._path, "a", encoding="ascii", errors="replace") as fh:
                fh.write(line + "\n")
