"""
logger.py
Centralised logging factory for the dB/dt forecasting pipeline.

Usage:
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Retrieved 44640 rows for 2015-03")
    log.error("OMNI gap exceeds 5 minutes at 2015-03-17 06:22 UTC")

TODO: Add Dask-aware logging; use config.LOGS_DIR consistently

"""

import logging
import os
from logging.handlers import RotatingFileHandler

import sys

# ---- configurable constants ------------------------------------------------
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_FILE = os.path.join("logs", "pipeline.log")
_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_BACKUP_COUNT = 5
# ---------------------------------------------------------------------------

_initialized: bool = False


def _initialize_root_logger() -> None:
    """One-time setup of the root logger with two handlers.

    This is called lazily on the first ``get_logger`` invocation and is
    idempotent — subsequent calls are no-ops.
    """
    global _initialized
    if _initialized:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter independently

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ------------------------------------------------------------------
    # Handler 1: stdout stream (INFO and above)
    # ------------------------------------------------------------------
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # ------------------------------------------------------------------
    # Handler 2: rotating file handler (DEBUG and above → logs/pipeline.log)
    # Ensure the logs/ directory exists before attempting to create the file.
    # ------------------------------------------------------------------
    log_dir = os.path.dirname(_LOG_FILE)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=_LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Only add handlers if none are already registered on the root logger.
    # This prevents duplicate log entries when the pipeline re-imports this
    # module across multiple invocations in the same process.
    if not root.handlers:
        root.addHandler(stream_handler)
        root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising the root logger on first call.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.  Results in log lines
        like ``2015-01-01 00:00:00 | INFO     | retrieve_omni | …``.

    Returns
    -------
    logging.Logger
        A configured Logger that writes to both stdout (INFO+) and the
        rotating log file (DEBUG+).

    Examples
    --------
    >>> from logger import get_logger
    >>> log = get_logger(__name__)
    >>> log.info("Retrieved 44640 rows for 2015-03")
    >>> log.warning("Empty return for Swarm A 2016-07")
    >>> log.error("HTTP 503 from OMNI API at 2018-04-05")
    """
    _initialize_root_logger()
    return logging.getLogger(name)
