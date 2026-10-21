"""Tests for the logging bootstrap (supply_twin.logging_setup)."""

from __future__ import annotations

import json
import logging
import sys

from supply_twin.logging_setup import JsonFormatter, configure_logging


def test_json_formatter_emits_parseable_record():
    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="supply_twin.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="what-if complete",
        args=(),
        exc_info=None,
    )
    record.run_id = "run-123"  # type: ignore[attr-defined]
    payload = json.loads(formatter.format(record))
    assert payload["logger"] == "supply_twin.api"
    assert payload["level"] == "INFO"
    assert payload["message"] == "what-if complete"
    assert payload["run_id"] == "run-123"
    assert payload["ts"].endswith("+00:00")


def test_json_formatter_includes_exception():
    formatter = JsonFormatter()
    try:
        raise RuntimeError("sim crashed")
    except RuntimeError:
        record = logging.LogRecord(
            name="supply_twin.api",
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="handled",
            args=(),
            exc_info=sys.exc_info(),
        )
    payload = json.loads(formatter.format(record))
    assert "RuntimeError: sim crashed" in payload["exception"]


def test_json_stream_mode(capsys):
    configure_logging(json_logs=True, level="INFO")
    logging.getLogger("test.json").info("hello json")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["message"] == "hello json"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.json"


def test_console_mode_not_json(capsys):
    configure_logging(json_logs=False, level="INFO")
    logging.getLogger("test.console").warning("plain line")
    line = capsys.readouterr().out.strip()
    try:
        json.loads(line)
    except json.JSONDecodeError:
        pass  # expected: console format is not JSON
    else:
        raise AssertionError("console format must not be JSON")
    assert "WARNING" in line
    assert "plain line" in line


def test_configure_logging_idempotent():
    for _ in range(3):
        configure_logging(json_logs=True, level="INFO")
    assert len(logging.getLogger().handlers) == 1


def test_configure_logging_level_filter(capsys):
    configure_logging(json_logs=True, level="WARNING")
    logging.getLogger("test.filter").debug("suppressed")
    logging.getLogger("test.filter").error("kept")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["message"] == "kept"
