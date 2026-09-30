"""
Thin subprocess entrypoint launched by the GUI to run one AutoMAX search
session. Reuses ``build_pipeline()`` from ``libauc.auto.run_auto_trainer`` /
``libauc.auto.run_auto_gnn`` so dataset-loading / config-resolution logic is never
duplicated between the CLI and the GUI.

Usage::

    python -m libauc.auto.ui.training_worker --config <path/to/config.json> --session-id <id>
"""
import argparse
import json
import logging
import os

from .. import run_auto_gnn as auto_gnn, run_auto_trainer as auto_trainer
from ..session import SessionAwareAutoMAX, TrainingCancelled
from .session_callback import SessionCallback
from .session_manager import get_session_manager

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AutoMAX GUI training worker")
    parser.add_argument("--config", type=str, required=True, help="Path to a JSON config file")
    parser.add_argument("--session-id", type=str, required=True)
    return parser.parse_args()


def run(cfg_dict: dict, session_id: str) -> None:
    session_manager = get_session_manager()
    # Overwrite the PID stamped by the GUI process at session-creation time
    # with this worker's own PID, so cancel_session() terminates the right
    # process. There's a brief window before this line runs during which the
    # session's `pid` is still the GUI's own PID -- the GUI's cancel handler
    # guards against terminating itself during that window (see gui.py).
    session_manager.update_session(session_id, pid=os.getpid())

    backend = cfg_dict.get("backend", "image")
    build_pipeline = auto_gnn.build_pipeline if backend == "gnn" else auto_trainer.build_pipeline

    session_callback_holder = []

    def callback_factory(automax_args):
        callback = SessionCallback(session_manager, session_id, automax_args)
        session_callback_holder.append(callback)
        return [callback]

    trainer, automax_args, metric_names = build_pipeline(cfg_dict, callback_factory=callback_factory)

    tuner = SessionAwareAutoMAX(
        trainer,
        automax_args,
        target=metric_names[0],
        session_manager=session_manager,
        session_id=session_id,
        session_callback=session_callback_holder[0] if session_callback_holder else None,
    )

    try:
        tuner.optimize()
    except TrainingCancelled:
        session_manager.write_session_log(session_id, "Training cancelled by user")
        logger.info(f"Session {session_id} cancelled")
        return
    except Exception as exc:
        session_manager.complete_session(session_id, error_message=str(exc))
        logger.exception(f"Session {session_id} failed")
        raise
    else:
        session_manager.complete_session(session_id)
        logger.info(f"Session {session_id} completed successfully")


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    run(cfg_dict, args.session_id)


if __name__ == "__main__":
    main()
