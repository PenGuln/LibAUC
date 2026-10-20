"""
Session-aware AutoMAX subclass bridging the SMAC search loop with the GUI's
SessionManager, without modifying :mod:`libauc.auto.automax` itself.
"""
from typing import Optional, TYPE_CHECKING

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

    ``AutoMAX`` has no trial-begin/trial-end hook -- SMAC calls
    ``self.train`` as a bare objective function per trial. This subclass
    wraps ``train()`` with before/after session updates instead of adding a
    trial-level callback abstraction to AutoMAX itself.
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
        if config.is_parallel:
            # Progress/cancel reporting hooks into train(), which only runs
            # in-process; parallel trials run in separate worker processes.
            raise ValueError(
                "GUI sessions run one trial at a time; n_workers and gpus_per_trial must be 1."
            )
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

    def train(self, space, seed: int = 0) -> float:
        trial_num = self.completed + 1
        session = self.session_manager.get_session(self.session_id)
        if session is not None and session.status == "cancelled":
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
