"""
Session Manager for AutoMAX GUI-driven hyperparameter-search runs.

Handles persistent session tracking, liveness monitoring, and log
management. A "session" is one full AutoMAX search (potentially many SMAC
trials), represented by one ``SessionInfo`` record plus a plaintext log file
and a small JSONL epoch-history file for the live chart.

The GUI process and the training-worker subprocess never talk directly --
they coordinate purely by reading/writing the same ``sessions.json`` file and
per-session log files (the same file-based-polling design as the original
AutoAUC_demo session manager), so closing the GUI never kills training.
"""

import json
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

import logging

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Information about a single AutoMAX search session."""

    session_id: str
    name: str
    status: str  # 'running', 'completed', 'failed', 'cancelled'
    start_time: str
    end_time: Optional[str] = None
    backend: str = "image"  # "image" (auto_trainer) | "gnn" (auto_gnn)
    model: str = ""
    dataset: str = ""
    metrics: List[str] = field(default_factory=list)
    n_trials: int = 0
    trial: int = 0  # 1-based index of the trial currently running / last run
    epochs: int = 0
    current_epoch: int = 0
    progress: float = 0.0  # 0..1 over the WHOLE search (trial/n_trials)
    loss: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    log_file: str = ""
    output_path: str = ""  # training.output_path used for this session
    automax_output_directory: str = ""  # links to the real AutoMAX state.pkl root
    automax_name: str = ""
    pid: Optional[int] = None
    last_heartbeat: Optional[str] = None
    error_message: Optional[str] = None
    test_true: Optional[List[float]] = None  # final epoch of the best/last trial only
    test_pred: Optional[List[float]] = None


class SessionManager:
    """Manages AutoMAX GUI sessions with JSON persistence and liveness monitoring."""

    def __init__(
        self,
        sessions_file: str = "gui_sessions/sessions.json",
        logs_dir: str = "gui_sessions/logs",
        runs_dir: str = "gui_sessions/runs",
        automax_output_dir: str = "gui_sessions/automax_output",
    ):
        self.sessions_file = Path(sessions_file)
        self.logs_dir = Path(logs_dir)
        self.runs_dir = Path(runs_dir)
        self.automax_output_dir = Path(automax_output_dir)

        self.sessions_file.parent.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.automax_output_dir.mkdir(parents=True, exist_ok=True)

        self.sessions: Dict[str, SessionInfo] = {}
        self._lock = threading.RLock()
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5  # seconds

        # GUI callback for progress updates
        self.gui_callback = None
        self.gui_current_session_id = None
        self._last_gui_update = None

        self._load_sessions()
        self.start_monitoring()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_sessions(self):
        if not self.sessions_file.exists():
            return
        try:
            with open(self.sessions_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self.sessions.clear()
                for session_data in data["sessions"]:
                    session = SessionInfo(**session_data)
                    self.sessions[session.session_id] = session
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")

    def _save_sessions(self):
        try:
            with self._lock:
                sessions_data = {"sessions": [asdict(s) for s in self.sessions.values()]}
            with open(self.sessions_file, "w", encoding="utf-8") as f:
                json.dump(sessions_data, f, indent=2)
            logger.debug(f"Saved {len(self.sessions)} sessions to {self.sessions_file}")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")

    # ------------------------------------------------------------------
    # Session config helpers
    # ------------------------------------------------------------------

    def _stamp_output_paths(self, session_id: str, config: Dict[str, Any]) -> None:
        """
        Give this session its own checkpoint/state directories so concurrent
        sessions never collide on a fixed ``output_path``/``experiment_name``
        or a shared AutoMAX ``state.pkl``.
        """
        config.setdefault("training", {})
        config.setdefault("automax", {})
        config["training"]["output_path"] = str(self.runs_dir / session_id)
        config["training"]["experiment_name"] = f"session_{session_id}"
        config["automax"]["output_directory"] = str(self.automax_output_dir)
        config["automax"]["name"] = f"session_{session_id}"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_session(self, name: str, config: Dict[str, Any], backend: str = "image", **kwargs) -> SessionInfo:
        """
        Create a new session. Mutates ``config`` in place, stamping unique
        output/state paths, so the caller can hand the same dict straight to
        the training worker.
        """
        session_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()

        self._stamp_output_paths(session_id, config)
        config["backend"] = backend

        training_cfg = config.get("training", {})
        automax_cfg = config.get("automax", {})

        session = SessionInfo(
            session_id=session_id,
            name=name,
            status="running",
            start_time=current_time,
            backend=backend,
            model=config.get("model", {}).get("name", ""),
            dataset=config.get("dataset", {}).get("name", ""),
            metrics=list(config.get("metrics", [])),
            n_trials=automax_cfg.get("n_trials", 0),
            epochs=training_cfg.get("epochs", 0),
            config=config,
            log_file=str(self.logs_dir / f"session_{session_id}.log"),
            output_path=training_cfg.get("output_path", ""),
            automax_output_directory=automax_cfg.get("output_directory", ""),
            automax_name=automax_cfg.get("name", ""),
            pid=os.getpid(),
            last_heartbeat=current_time,
        )

        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)

        self._load_sessions()
        with self._lock:
            self.sessions[session_id] = session
        self._save_sessions()

        logger.info(f"Created session {session_id}: {name}")
        return session

    def update_session(self, session_id: str, **kwargs) -> bool:
        self._load_sessions()
        with self._lock:
            if session_id not in self.sessions:
                return False
            session = self.sessions[session_id]
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.last_heartbeat = datetime.now().isoformat()
        self._save_sessions()
        return True

    def complete_session(self, session_id: str, error_message: Optional[str] = None) -> bool:
        self._load_sessions()
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        session.status = "failed" if error_message else "completed"
        session.end_time = datetime.now().isoformat()
        if error_message:
            session.error_message = error_message
        self._save_sessions()
        logger.info(f"Session {session_id} {session.status}")
        return True

    def cancel_session(self, session_id: str) -> bool:
        self._load_sessions()
        if session_id not in self.sessions:
            return False
        session = self.sessions[session_id]
        if session.status == "running":
            session.status = "cancelled"
            session.end_time = datetime.now().isoformat()
            self._save_sessions()
            if session.pid:
                try:
                    psutil.Process(session.pid).terminate()
                    logger.info(f"Terminated process {session.pid} for session {session_id}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    logger.warning(f"Could not terminate process {session.pid} for session {session_id}")
        logger.info(f"Cancelled session {session_id}")
        return True

    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        self._load_sessions()
        with self._lock:
            return self.sessions.get(session_id)

    def get_all_sessions(self) -> List[SessionInfo]:
        self._load_sessions()
        with self._lock:
            return list(self.sessions.values())

    def get_running_sessions(self) -> List[SessionInfo]:
        self._load_sessions()
        with self._lock:
            return [s for s in self.sessions.values() if s.status == "running"]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session: registry entry, log/history files, and its checkpoint dir."""
        self._load_sessions()
        if session_id not in self.sessions:
            return False

        session = self.sessions[session_id]
        if session.status == "running":
            self.cancel_session(session_id)

        for path_str in (session.log_file, str(self._history_file(session_id))):
            try:
                path = Path(path_str)
                if path.exists():
                    path.unlink()
            except Exception as e:
                logger.error(f"Error deleting {path_str} for session {session_id}: {e}")

        if session.output_path:
            try:
                shutil.rmtree(session.output_path, ignore_errors=True)
            except Exception as e:
                logger.error(f"Error deleting output dir for session {session_id}: {e}")

        automax_dir = self.automax_output_dir / f"session_{session_id}"
        if automax_dir.exists():
            shutil.rmtree(automax_dir, ignore_errors=True)

        with self._lock:
            del self.sessions[session_id]
        self._save_sessions()

        logger.info(f"Deleted session {session_id}")
        return True

    # ------------------------------------------------------------------
    # Liveness monitoring
    # ------------------------------------------------------------------

    def _check_session_liveness(self, session: SessionInfo) -> bool:
        if session.status != "running":
            return True

        if session.pid:
            try:
                if not psutil.Process(session.pid).is_running():
                    logger.warning(f"Process {session.pid} for session {session.session_id} is not running")
                    return False
            except psutil.NoSuchProcess:
                logger.warning(f"Process {session.pid} for session {session.session_id} not found")
                return False

        if session.last_heartbeat:
            try:
                last_heartbeat = datetime.fromisoformat(session.last_heartbeat)
                if datetime.now().timestamp() - last_heartbeat.timestamp() > 300:
                    logger.warning(f"Session {session.session_id} heartbeat timeout")
                    return False
            except ValueError:
                pass

        return True

    def _monitor_sessions(self):
        while self.monitoring:
            try:
                for session in self.get_running_sessions():
                    if not self._check_session_liveness(session):
                        self.complete_session(session.session_id, error_message="Session timeout or process crashed")
                    else:
                        self._update_gui_progress(session)
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in session monitoring: {e}")
                time.sleep(self.monitor_interval)

    def start_monitoring(self):
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_sessions, daemon=True)
            self.monitor_thread.start()
            logger.info("Started session monitoring")

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        logger.info("Stopped session monitoring")

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------

    def get_session_logs(self, session_id: str, lines: int = 100) -> List[str]:
        session = self.get_session(session_id)
        if not session or not session.log_file:
            return []
        try:
            path = Path(session.log_file)
            if path.exists():
                all_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                return all_lines[-lines:] if lines > 0 else all_lines
        except Exception as e:
            logger.error(f"Error reading logs for session {session_id}: {e}")
        return []

    def write_session_log(self, session_id: str, message: str, level: str = "INFO"):
        session = self.get_session(session_id)
        if not session:
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(session.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{level}] {message}\n")
        except Exception as e:
            logger.error(f"Error writing log for session {session_id}: {e}")

    def heartbeat(self, session_id: str):
        if self.update_session(session_id, last_heartbeat=datetime.now().isoformat()):
            logger.debug(f"Heartbeat for session {session_id}")

    # ------------------------------------------------------------------
    # Epoch history (for the monitor page's live loss/metric chart) --
    # kept in a small side file, NOT inside SessionInfo/sessions.json, so
    # per-epoch points never bloat the (wholesale-rewritten) session registry.
    # ------------------------------------------------------------------

    def _history_file(self, session_id: str) -> Path:
        return self.logs_dir / f"session_{session_id}_history.jsonl"

    def append_epoch_history(
        self, session_id: str, epoch: int, train_loss: float, metrics: Optional[Dict[str, Any]] = None
    ):
        entry = {"epoch": epoch, "train_loss": train_loss}
        for key, value in (metrics or {}).items():
            if isinstance(value, (int, float)):
                entry[key] = value
        try:
            with open(self._history_file(session_id), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Error appending epoch history for session {session_id}: {e}")

    def get_epoch_history(self, session_id: str, max_points: int = 200) -> List[Dict[str, Any]]:
        path = self._history_file(session_id)
        if not path.exists():
            return []
        try:
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error reading epoch history for session {session_id}: {e}")
            return []
        if len(rows) > max_points > 0:
            step = len(rows) / max_points
            rows = [rows[int(i * step)] for i in range(max_points)]
        return rows

    # ------------------------------------------------------------------
    # GUI progress callback (file-polling bridge, no direct IPC)
    # ------------------------------------------------------------------

    def register_gui_callback(self, gui_instance, current_session_id: Optional[str] = None):
        self.gui_callback = gui_instance
        self.gui_current_session_id = current_session_id
        self._last_gui_update = None
        logger.info(f"Registered GUI callback for session: {current_session_id}")

    def unregister_gui_callback(self):
        self.gui_callback = None
        self.gui_current_session_id = None
        self._last_gui_update = None
        logger.info("Unregistered GUI callback")

    def _update_gui_progress(self, session: SessionInfo):
        if not (
            self.gui_callback
            and self.gui_current_session_id
            and session.session_id == self.gui_current_session_id
            and hasattr(self.gui_callback, "set_progressbar")
        ):
            return

        if session.current_epoch == 0 and session.progress == 0.0:
            return

        current_info = (session.trial, session.current_epoch, session.epochs, session.progress, session.loss)
        if self._last_gui_update == current_info:
            return

        try:
            self.gui_callback.set_progressbar(session.progress)
            if hasattr(self.gui_callback, "console_log"):
                self.gui_callback.console_log(
                    f"Trial {session.trial}/{session.n_trials} - "
                    f"Epoch {session.current_epoch}/{session.epochs} - "
                    f"Progress: {session.progress:.1%} - Loss: {session.loss:.4f}",
                    "TRAINING",
                )
            self._last_gui_update = current_info
        except Exception as e:
            logger.error(f"Error updating GUI progress: {e}")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def cleanup_session_manager():
    """Cleanup the global session manager."""
    global _session_manager
    if _session_manager:
        _session_manager.stop_monitoring()
        _session_manager = None
