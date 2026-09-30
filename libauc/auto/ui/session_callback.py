"""
Bridges libauc's per-epoch TrainerCallback hooks to a SessionManager, so the
GUI can show live progress/cancel a run without any direct IPC to the
training-worker subprocess.
"""
import logging
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
    """

    def __init__(self, session_manager: "SessionManager", session_id: str, auto_config: AutoMAXConfigration):
        super().__init__()
        self.session_manager = session_manager
        self.session_id = session_id
        self.n_trials = auto_config.n_trials
        self._last_test_true: Optional[List[float]] = None
        self._last_test_pred: Optional[List[float]] = None
        self._status_cache: Optional[str] = None
        self._status_cache_time: float = 0.0
        self._status_cache_ttl = 5.0  # seconds; avoid hammering sessions.json every step

    def _status(self) -> Optional[str]:
        now = time.monotonic()
        if now - self._status_cache_time > self._status_cache_ttl:
            session = self.session_manager.get_session(self.session_id)
            self._status_cache = session.status if session else None
            self._status_cache_time = now
        return self._status_cache

    def _raise_if_cancelled(self):
        if self._status() == "cancelled":
            raise TrainingCancelled(f"Session {self.session_id} was cancelled by the user")

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_init_end(self, args, state, **kwargs):
        self.session_manager.write_session_log(self.session_id, "Trial initialized")

    def on_epoch_begin(self, args, state, **kwargs):
        self._raise_if_cancelled()

    def on_step_begin(self, args, state, **kwargs):
        self._raise_if_cancelled()

    def on_epoch_end(self, args, state, **kwargs):
        self._raise_if_cancelled()

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
            self.session_id, epoch=current_epoch, train_loss=train_loss, metrics=primary_metrics
        )

        summary = " | ".join(
            f"{k}={v:.4f}" for k, v in primary_metrics.items() if isinstance(v, (int, float))
        )
        self.session_manager.write_session_log(
            self.session_id,
            f"Epoch {current_epoch}/{total_epoch} | loss={train_loss:.4f}"
            + (f" | {summary}" if summary else ""),
        )

    def _current_trial(self) -> int:
        session = self.session_manager.get_session(self.session_id)
        return session.trial if session else 1

    def get_last_predictions(self):
        """Return (test_true, test_pred) lists from the most recent epoch, or (None, None)."""
        return self._last_test_true, self._last_test_pred
