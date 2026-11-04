"""
Safe persistence for JSON files shared between GUI and worker processes,
which may be running on different nodes of a shared filesystem.

``atomic_write_json`` means a reader only ever sees a complete file.
``file_lock`` serializes writers, so that a load-change-save cycle can't
overwrite an update another process made in the meantime.
"""
import json
import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # not available on Windows
    fcntl = None

logger = logging.getLogger(__name__)


def atomic_write_json(path: Path, data: Any) -> None:
    """Replace ``path`` with ``data`` as JSON, all at once. Raises on failure."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """
    Hold an exclusive lock, across processes, on ``<path>.lock``.

    Not reentrant: nesting it deadlocks, even in the same thread. If the
    filesystem doesn't support locking, this warns and continues unlocked.
    """
    with open(path.with_name(f"{path.name}.lock"), "a") as f:
        locked = False
        if fcntl is not None:
            try:
                fcntl.flock(f, fcntl.LOCK_EX)
                locked = True
            except OSError as e:
                logger.warning(f"Could not lock {path}, continuing without a lock: {e}")
        try:
            yield
        finally:
            # Unlock explicitly: closing isn't enough if a child process
            # forked meanwhile (e.g. a DataLoader worker) inherited the file.
            if locked:
                fcntl.flock(f, fcntl.LOCK_UN)
