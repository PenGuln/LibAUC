"""
Session-aware AutoMAX subclass bridging the SMAC search loop with the GUI's
SessionManager.
"""
from typing import List, Optional, TYPE_CHECKING

from .automax import AutoMAX
from .configuration import AutoMAXConfigration

if TYPE_CHECKING:
    from ..ui.session_callback import SessionCallback
    from ..ui.session_manager import SessionManager


class TrainingCancelled(BaseException):
    """Raised when a GUI-driven session has been cancelled by the user.

    Deliberately a ``BaseException`` subclass, NOT ``Exception``: SMAC's
    ``TargetFunctionRunner.run()`` and ``AbstractRunner.run_wrapper()`` both
    wrap the objective call in a bare ``except Exception`` and turn any
    exception into a merely-CRASHED trial, silently continuing the search
    with its remaining trial budget -- an ``Exception`` subclass raised here
    would NOT actually stop the search. Only exception types SMAC doesn't
    catch (like ``KeyboardInterrupt``, or this class) propagate all the way
    out of ``AutoMAX.optimize()`` to the training worker, which is what "cancel
    aborts the whole session" requires. The worker explicitly catches this
    type first (before any ``except Exception``) so it still shuts down
    cleanly rather than surfacing an unhandled traceback.
    """


class SessionAwareAutoMAX(AutoMAX):
    """
    AutoMAX subclass that reports trial-level progress to a SessionManager.

    One trial at a time, SMAC calls ``self.train`` per trial, which this
    subclass wraps with before/after session updates. In a parallel search
    trials run in their own processes instead, so it uses AutoMAX's parallel
    hooks (``_on_parallel_trial_start`` / ``_end``, ``_check_stop``), and each
    trial's ``SessionCallback`` reports its epochs.
    """

    def __init__(
        self,
        trainer,
        config: AutoMAXConfigration,
        target: str,
        session_manager: "SessionManager",
        session_id: str,
        session_callback: Optional["SessionCallback"] = None,
    ):
        super().__init__(trainer, config, target)
        self.session_manager = session_manager
        self.session_id = session_id
        self.session_callback = session_callback
        self.session_manager.update_session(
            session_id,
            n_trials=config.n_trials,
            automax_output_directory=config.output_directory,
            automax_name=config.name,
        )

    # -- parallel search ------------------------------------------------------

    def _check_stop(self):
        session = self.session_manager.get_session(self.session_id)
        if session is None or session.status == "cancelled":
            raise TrainingCancelled(f"Session {self.session_id} was cancelled by the user")

    def _on_parallel_trial_start(self, trial_name: str, gpus: List[str]):
        self.session_manager.update_trial(self.session_id, trial_name, gpus=list(gpus), epoch=0)
        self.session_manager.write_session_log(
            self.session_id, f"Starting {trial_name} on GPU(s) {', '.join(gpus)}"
        )

    def _on_parallel_trial_end(self, trial_name: str, ok: bool, is_best: bool):
        self.session_manager.update_session(self.session_id, trial=self.completed)
        self.session_manager.update_trial(self.session_id, trial_name, finished=True)
        predictions = self.session_manager.load_trial_predictions(self.session_id, trial_name)
        if ok and is_best and predictions:
            self.session_manager.update_session(self.session_id, **predictions)
        self.session_manager.write_session_log(
            self.session_id,
            f"Finished {trial_name} [{self.completed}/{self.config.n_trials}]" if ok else f"{trial_name} crashed",
            level="INFO" if ok else "ERROR",
        )

    # -- one trial at a time ----------------------------------------------------

    def train(self, space, seed: int = 0) -> float:
        trial_num = self.completed + 1
        session = self.session_manager.get_session(self.session_id)
        if session is None or session.status == "cancelled":
            raise TrainingCancelled(f"Cancelled before trial {trial_num} started")

        self.session_manager.write_session_log(
            self.session_id, f"Starting trial {trial_num}/{self.config.n_trials}"
        )
        self.session_manager.update_session(self.session_id, trial=trial_num, current_epoch=0)

        prev_best = self.best_score
        try:
            score = super().train(space, seed=seed)
        except TrainingCancelled:
            self.session_manager.write_session_log(
                self.session_id, f"Trial {trial_num} cancelled", level="WARNING"
            )
            raise
        except Exception as exc:
            self.session_manager.write_session_log(
                self.session_id, f"Trial {trial_num} failed: {exc}", level="ERROR"
            )
            raise

        is_new_best = self.best_score > prev_best
        is_last_trial = self.completed >= self.config.n_trials
        update_kwargs = {
            "trial": self.completed,
            "progress": self.completed / self.config.n_trials if self.config.n_trials else 1.0,
        }
        if (is_new_best or is_last_trial) and self.session_callback is not None:
            test_true, test_pred = self.session_callback.get_last_predictions()
            if test_true is not None:
                update_kwargs["test_true"] = test_true
                update_kwargs["test_pred"] = test_pred

        self.session_manager.update_session(self.session_id, **update_kwargs)
        return score
