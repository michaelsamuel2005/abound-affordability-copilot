"""Structured JSON logging + per-stage timing.

Every log line is a single JSON object with a fixed envelope:
    ts, level, logger, event, request_id, applicant_id, ...event fields

Redaction policy: transaction DESCRIPTIONS and customer names are never logged —
only transaction IDs, counts and amounts-in-aggregate. Logs go to stdout
(12-factor); in production they would be shipped to a log store — that shipping,
plus metrics/traces/dashboards, is future work and this module does not claim it.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from contextlib import contextmanager

from config import ENGINE_VERSION

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
applicant_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("applicant_id", default="-")

_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
            "applicant_id": applicant_id_var.get(),
            "engine_version": ENGINE_VERSION,
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            out.update(fields)
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})


@contextmanager
def stage_timer(timings: dict, stage: str):
    """Record a pipeline stage's wall-clock latency into `timings` (ms)."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round((time.perf_counter() - t0) * 1000, 2)
