"""Application logging bootstrap.

Configures stdlib logging once per process with either human-readable console
records or machine-parseable single-line JSON (for aggregators), according to
``Settings.log_json`` / ``Settings.log_level``.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_RESERVED_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs", "message",
    "msg", "name", "pathname", "process", "processName", "relativeCreated",
    "stack_info", "stacklevel", "thread", "threadName", "taskName",
})


class JsonFormatter(logging.Formatter):
    """Render a record as one JSON object per line, extras merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(json_logs: bool = False, level: str = "INFO") -> None:
    """Install the application log handlers on the root logger (idempotent)."""
    root = logging.getLogger()
    level_value = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)

    if json_logs:
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    stream = logging.StreamHandler(stream=sys.stdout)
    stream.setFormatter(formatter)

    root.handlers = [stream]  # replace, never stack up on re-configuration
    root.setLevel(level_value)
