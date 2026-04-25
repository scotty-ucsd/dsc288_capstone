"""
logger.py
Centralised logging factory for the dB/dt forecasting pipeline.

``get_logger`` wires stdout (INFO+) and a rotating file under ``LOGS_DIR`` (from
``swmi.utils.config``). Use ``install_dask_worker_file_logging`` after creating a
Dask ``Client`` so worker processes also append to ``logs/dask_worker.log``.
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_initialized: bool = False


def _logs_dir() -> Path:
    try:
        from swmi.utils import config

        return Path(config.LOGS_DIR)
    except ImportError:
        return Path("logs")


def _pipeline_log_path() -> Path:
    return _logs_dir() / "pipeline.log"


def _initialize_root_logger() -> None:
    global _initialized
    if _initialized:
        return

    log_dir = _logs_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=str(_pipeline_log_path()),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not root.handlers:
        root.addHandler(stream_handler)
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising the root logger on first call."""
    _initialize_root_logger()
    return logging.getLogger(name)


def _dask_log_path() -> Path:
    return _logs_dir() / "dask_worker.log"


def _has_dask_worker_handler(root: logging.Logger) -> bool:
    target = str(_dask_log_path().resolve())
    for h in root.handlers:
        if isinstance(h, logging.FileHandler):
            base = getattr(h, "baseFilename", None)
            if base is not None and str(base) == target:
                return True
    return False


def install_dask_worker_file_logging(client: Any | None = None) -> None:
    """Attach a file handler on the current process and, when possible, on Dask workers.

    LEO feature builds and other distributed tasks should log to ``logs/``; call
    this right after ``Client(...)`` is created. Failures in ``client.run`` are
    logged and do not silence worker-side errors.
    """
    log = get_logger(__name__)

    def _attach_file_handler() -> None:
        root = logging.getLogger()
        if _has_dask_worker_handler(root):
            return
        log_dir = _logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        path = _dask_log_path()
        fmt = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
        fh = logging.FileHandler(path, encoding="utf-8", mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.setLevel(min(root.level, logging.DEBUG))
        root.addHandler(fh)
        for name in (
            "distributed.worker",
            "distributed.worker_state_machine",
        ):
            logging.getLogger(name).setLevel(logging.INFO)

    _attach_file_handler()

    if client is not None:
        try:
            client.run(_attach_file_handler)
        except Exception as exc:
            log.warning("Could not install file logging on Dask workers: %s", exc, exc_info=True)
