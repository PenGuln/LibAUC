"""
Bridges libauc's per-epoch TrainerCallback hooks to a SessionManager, so the
GUI can show live progress/cancel a run without any direct IPC to the
training-worker subprocess.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
from libauc.trainer import TrainerCallback

from ..session import TrainingCancelled
from ..configuration import AutoMAXConfigration

if TYPE_CHECKING:
    from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class SessionCallback(TrainerCallback):
    """
    Reports per-epoch progress into a SessionManager and raises
    ``TrainingCancelled`` if the user cancels the session mid-epoch.

    Note: ``state.epoch`` is only incremented by ``CLICallback.on_epoch_end``
    (libauc's design, not something this class controls), and
    ``build_pipeline()`` always registers callbacks as
    ``[CLICallback(), *extra_callbacks]``, so by the time this callback's
    ``on_epoch_end`` runs, ``state.epoch`` already reflects the just-completed
    epoch count.

    In a parallel search (``n_workers`` or ``gpus_per_trial`` > 1) it runs in
    each trial's own process, where it reports per trial (see
    ``SessionManager.update_trial``); only rank 0 of a DDP trial reports, but
    every rank stops on a cancel. It is pickled into those processes, so it
    keeps the session files' paths and opens its own SessionManager there.
    """

    def __init__(self, session_manager: "SessionManager", session_id: str, auto_config: AutoMAXConfigration):
        super().__init__()
        self._manager = session_manager
        self._manager_pid = os.getpid()
        self._manager_paths = session_manager.paths
        self.session_id = session_id
        self.n_trials = auto_config.n_trials
        self.parallel = auto_config.is_parallel
        # SessionManager.create_session sets this; trials append "_<trial name>".
        self._base_experiment = f"session_{session_id}"
        self._last_test_true: Optional[List[float]] = None
        self._last_test_pred: Optional[List[float]] = None
        self._status_cache: Optional[str] = None
        self._status_cache_time: float = 0.0
        self._status_cache_ttl = 5.0  # seconds; avoid hammering sessions.json every step

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_manager"] = None
        return state

    @property
    def session_manager(self) -> "SessionManager":
        # A new one per process: a forked copy of the driver's could hold a
        # lock taken by one of the driver's threads, which never get released.
        if self._manager is None or self._manager_pid != os.getpid():
            from .session_manager import SessionManager
            self._manager = SessionManager(**self._manager_paths, monitor=False)
            self._manager_pid = os.getpid()
        return self._manager

    def _trial(self, args) -> Optional[str]:
        """The trial's name (e.g. ``trial3``), from its experiment name."""
        name = getattr(args, "experiment_name", "") or ""
        prefix = self._base_experiment + "_"
        return name[len(prefix):].split(os.sep)[0] if name.startswith(prefix) else None

    @staticmethod
    def _reports() -> bool:
        return int(os.environ.get("RANK", 0)) == 0

    def _status(self) -> Optional[str]:
        now = time.monotonic()
        if now - self._status_cache_time > self._status_cache_ttl:
            session = self.session_manager.get_session(self.session_id)
            self._status_cache = session.status if session else None
            self._status_cache_time = now
        return self._status_cache

    def _raise_if_cancelled(self):
        # No status means the session was deleted. A GUI on another host can't
        # signal this process, so this is how such a delete stops it.
        if self._status() in ("cancelled", None):
            raise TrainingCancelled(f"Session {self.session_id} was cancelled by the user")

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_init_end(self, args, state, **kwargs):
        if self._reports():
            prefix = f"[{self._trial(args)}] " if self.parallel else ""
            self.session_manager.write_session_log(self.session_id, f"{prefix}Trial initialized")

    def on_epoch_begin(self, args, state, **kwargs):
        self._raise_if_cancelled()

    def on_step_begin(self, args, state, **kwargs):
        self._raise_if_cancelled()

    def on_epoch_end(self, args, state, **kwargs):
        self._raise_if_cancelled()
        if not self._reports():
            return
        trial = self._trial(args)

        metrics: List[Dict[str, Any]] = kwargs.get("metrics") or []
        train_loss: float = kwargs.get("train_loss", 0.0)
        test_true = kwargs.get("test_true")
        test_pred = kwargs.get("test_pred")

        if test_true is not None:
            self._last_test_true = np.asarray(test_true).tolist()
            self._last_test_pred = np.asarray(test_pred).tolist()

        primary_metrics = metrics[0] if metrics else {}
        # Note: state.epoch has already been bumped by CLICallback (see class
        # docstring), so it's the just-completed epoch count, not epoch - 1.
        current_epoch = state.epoch
        total_epoch = state.total_epoch or 1

        if self.parallel:
            self.session_manager.update_trial(
                self.session_id, trial, epoch=current_epoch, epochs=total_epoch, loss=train_loss
            )
        else:
            trial_progress = current_epoch / total_epoch
            completed_trials = max(self._current_trial() - 1, 0)
            overall_progress = (completed_trials + trial_progress) / self.n_trials if self.n_trials else trial_progress
            self.session_manager.update_session(
                self.session_id,
                current_epoch=current_epoch,
                epochs=total_epoch,
                progress=overall_progress,
                loss=train_loss,
            )
        self.session_manager.append_epoch_history(
            self.session_id, epoch=current_epoch, train_loss=train_loss, metrics=primary_metrics, trial=trial
        )

        summary = " | ".join(
            f"{k}={v:.4f}" for k, v in primary_metrics.items() if isinstance(v, (int, float))
        )
        self.session_manager.write_session_log(
            self.session_id,
            (f"[{trial}] " if self.parallel else "")
            + f"Epoch {current_epoch}/{total_epoch} | loss={train_loss:.4f}"
            + (f" | {summary}" if summary else ""),
        )

    def on_train_end(self, args, state, **kwargs):
        # A parallel trial's predictions are only shown if it turns out to be
        # the best, which the driver decides; leave them where it can read them.
        if self.parallel and self._reports() and self._last_test_true is not None:
            self.session_manager.save_trial_predictions(
                self.session_id, self._trial(args), self._last_test_true, self._last_test_pred
            )

    def _current_trial(self) -> int:
        session = self.session_manager.get_session(self.session_id)
        return session.trial if session else 1

    def get_last_predictions(self):
        """Return (test_true, test_pred) lists from the most recent epoch, or (None, None)."""
        return self._last_test_true, self._last_test_pred
