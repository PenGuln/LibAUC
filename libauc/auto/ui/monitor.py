"""
AutoMAX "Sessions" page: every past/running search, with live progress,
training curves, the best trial's ROC curve, a streaming log tail, and
cancel/delete controls.

Like the rest of the GUI, this only polls ``SessionManager``'s files -- it
never talks to a training worker directly.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from nicegui import ui
from sklearn.metrics import auc, roc_curve

from .gui import reap_finished_workers
from .session_manager import SessionInfo, get_session_manager
from .theme import CHART_COLORS, duration, relative_time, stat_tile, status_chip, status_dot

logger = logging.getLogger(__name__)

_REFRESH_SECONDS = 2.0
_LOG_TAIL_BYTES = 256 * 1024

_FILTERS = {"all": "All", "running": "Running", "completed": "Done", "failed": "Failed"}

# Axis/text styling that reads in both light and dark mode.
_AXIS = {
    "axisLine": {"lineStyle": {"color": "#8b90a5"}},
    "axisLabel": {"color": "#8b90a5"},
    "splitLine": {"lineStyle": {"color": "rgba(139,144,165,.18)"}},
    "nameTextStyle": {"color": "#8b90a5"},
}


def _curve_options(history: List[Dict], metric_names: List[str]) -> dict:
    """ECharts options for the training curves. History rows restart at
    epoch 1 for every trial, so points are laid out sequentially and each
    trial boundary gets a dashed marker line, with loss on the left axis and
    metrics on the right."""
    labels, trial_starts = [], []
    trial, prev_epoch = 0, None
    for i, row in enumerate(history):
        if prev_epoch is None or row["epoch"] <= prev_epoch:
            trial += 1
            trial_starts.append((i, trial))
        prev_epoch = row["epoch"]
        labels.append(f"T{trial} · E{row['epoch']}")

    keys = ["train_loss"] + sorted({k for row in history for k in row} - {"epoch", "train_loss"}, key=lambda k: (k not in metric_names, k))
    series = []
    for i, key in enumerate(keys):
        is_loss = key == "train_loss"
        s = {
            "name": key, "type": "line", "smooth": 0.3, "smoothMonotone": "x", "showSymbol": len(history) < 40, "symbolSize": 5,
            "yAxisIndex": 0 if is_loss else 1,
            "data": [row.get(key) for row in history],
            "lineStyle": {"width": 2.5 if not is_loss else 2, "type": "solid" if not is_loss else "dashed"},
            "color": CHART_COLORS[i % len(CHART_COLORS)],
        }
        if i == 0 and len(trial_starts) > 1:
            s["markLine"] = {
                "silent": True, "symbol": "none",
                "lineStyle": {"color": "#8b90a5", "type": "dotted"},
                "label": {"formatter": "{b}", "color": "#8b90a5", "position": "insideEndTop"},
                "data": [{"xAxis": idx, "name": f"Trial {t}"} for idx, t in trial_starts[1:]],
            }
        series.append(s)

    return {
        "animation": False,
        "backgroundColor": "transparent",
        "grid": {"left": 56, "right": 56, "top": 48, "bottom": 64},
        "tooltip": {"trigger": "axis", "valueFormatter": "(v) => v == null ? '-' : v.toFixed(4)"},
        "legend": {"top": 4, "textStyle": {"color": "#8b90a5"}, "icon": "roundRect"},
        "xAxis": {"type": "category", "data": labels, "boundaryGap": False, **_AXIS},
        "yAxis": [
            {"type": "value", "name": "loss", "scale": True, **_AXIS},
            {"type": "value", "name": "metric", "scale": True, **_AXIS, "splitLine": {"show": False}},
        ],
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 18, "bottom": 8}],
        "series": series,
    }


def _roc_options(test_true, test_pred) -> Optional[dict]:
    y_true = np.asarray(test_true).flatten()
    y_score = np.asarray(test_pred, dtype=float)
    if y_score.min() < 0 or y_score.max() > 1:
        from scipy.special import expit
        y_score = expit(y_score)
    y_score = y_score.flatten()
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_score)
    area = auc(fpr, tpr)
    return {
        "animation": False,
        "backgroundColor": "transparent",
        "grid": {"left": 56, "right": 24, "top": 40, "bottom": 48},
        "title": {"text": f"AUC = {area:.4f}", "left": "center", "top": 4, "textStyle": {"fontSize": 13, "color": "#8b90a5"}},
        "tooltip": {"trigger": "axis", "valueFormatter": "(v) => v.toFixed(3)"},
        "xAxis": {"type": "value", "name": "FPR", "min": 0, "max": 1, "nameLocation": "middle", "nameGap": 28, **_AXIS},
        "yAxis": {"type": "value", "name": "TPR", "min": 0, "max": 1, **_AXIS},
        "series": [
            {
                "type": "line", "name": "ROC", "showSymbol": False, "step": "end",
                "data": [[float(x), float(y)] for x, y in zip(fpr, tpr)],
                "color": CHART_COLORS[0], "lineStyle": {"width": 2.5},
                "areaStyle": {"opacity": 0.12},
            },
            {
                "type": "line", "name": "chance", "showSymbol": False, "silent": True,
                "data": [[0, 0], [1, 1]], "color": "#8b90a5", "lineStyle": {"type": "dashed", "width": 1},
            },
        ],
    }


class SessionsPanel:
    """Two-pane session browser: a filterable list on the left and the
    selected session's live dashboard on the right."""

    def __init__(self):
        self.session_manager = get_session_manager()
        self.sessions: List[SessionInfo] = []
        self.selected_id: Optional[str] = None
        self.filter = "all"
        self.detail_tab = "curves"

        self._list_sig = None
        self._detail_sig = None
        self._live_sig = None
        self._log_pos = 0
        self.curve_chart: Optional[ui.echart] = None
        self.log_view: Optional[ui.log] = None
        self.kpi_box: Optional[ui.element] = None

        with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-6 items-start"):
            with ui.card().props("flat").classes("am-card w-full gap-3"):
                with ui.row().classes("items-center w-full"):
                    ui.label("Sessions").classes("text-base font-semibold")
                    ui.space()
                    ui.button(icon="refresh", on_click=self.refresh).props("flat round dense").tooltip("Refresh")
                ui.toggle(_FILTERS, value="all", on_change=self._filter_changed).props(
                    "dense no-caps unelevated spread toggle-color=primary"
                ).classes("w-full")
                self.list_box = ui.column().classes("w-full gap-1 max-h-[70vh] overflow-y-auto")
            self.detail_box = ui.column().classes("min-w-0 w-full gap-6")

        self.refresh()
        ui.timer(_REFRESH_SECONDS, self.refresh)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(self, session_id: Optional[str]):
        self.selected_id = session_id
        self._detail_sig = None
        self._list_sig = None
        self.refresh()

    def refresh(self):
        reap_finished_workers()
        self.sessions = sorted(self.session_manager.get_all_sessions(), key=lambda s: s.start_time, reverse=True)
        if self.selected_id is None and self.sessions:
            self.selected_id = self.sessions[0].session_id
        if self.selected_id and not any(s.session_id == self.selected_id for s in self.sessions):
            self.selected_id = None
        self._render_list()
        self._render_detail()

    @property
    def selected(self) -> Optional[SessionInfo]:
        return next((s for s in self.sessions if s.session_id == self.selected_id), None)

    # ------------------------------------------------------------------
    # Session list
    # ------------------------------------------------------------------

    def _filter_changed(self, e):
        self.filter = e.value
        self._list_sig = None
        self._render_list()

    def _render_list(self):
        visible = [s for s in self.sessions if self.filter == "all" or s.status == self.filter]
        sig = (self.filter, self.selected_id, tuple((s.session_id, s.status, round(s.progress, 3), s.trial) for s in visible),
               relative_time(visible[0].start_time) if visible else "")
        if sig == self._list_sig:
            return
        self._list_sig = sig

        self.list_box.clear()
        with self.list_box:
            if not visible:
                with ui.column().classes("w-full items-center py-10 gap-2 am-muted"):
                    ui.icon("inbox", size="36px")
                    ui.label("No sessions here yet")
                return
            for s in visible:
                self._session_item(s)

    def _session_item(self, s: SessionInfo):
        classes = "am-session w-full p-3 gap-1" + (" am-selected" if s.session_id == self.selected_id else "")
        with ui.column().classes(classes).on("click", lambda _, sid=s.session_id: self.select(sid)):
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                status_dot(s.status)
                ui.label(s.name).classes("font-medium text-sm truncate grow")
                ui.label(relative_time(s.start_time)).classes("text-xs am-muted shrink-0")
            ui.label(f"{s.model} · {s.dataset} · {', '.join(s.metrics)}").classes("text-xs am-muted truncate w-full")
            if s.status == "running":
                with ui.row().classes("items-center gap-2 w-full no-wrap"):
                    ui.linear_progress(s.progress, show_value=False, size="6px").classes("grow rounded")
                    ui.label(f"{s.trial}/{s.n_trials}").classes("text-xs am-mono am-muted")

    # ------------------------------------------------------------------
    # Session detail
    # ------------------------------------------------------------------

    def _render_detail(self):
        s = self.selected
        # Rebuild the whole dashboard only on selection/status change; live
        # numbers, charts and logs are patched in place by _update_live().
        sig = (s.session_id, s.status, s.test_true is not None) if s else None
        if sig != self._detail_sig:
            self._detail_sig = sig
            self._live_sig = None
            self._build_detail(s)
        if s:
            self._update_live(s)

    def _build_detail(self, s: Optional[SessionInfo]):
        self.detail_box.clear()
        self.curve_chart = self.log_view = self.kpi_box = None
        self._log_pos = 0
        with self.detail_box:
            if s is None:
                with ui.card().props("flat").classes("am-card w-full items-center py-16 gap-3"):
                    ui.icon("insights", size="48px").classes("text-primary")
                    ui.label("No session selected").classes("text-lg font-semibold")
                    ui.label("Launch a search from the New search tab, then follow it here.").classes("am-muted")
                return

            with ui.card().props("flat").classes("am-card w-full gap-3"):
                with ui.row().classes("items-start w-full gap-3"):
                    with ui.column().classes("gap-1 grow min-w-0"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label(s.name).classes("text-xl font-semibold truncate")
                            status_chip(s.status)
                        ui.label(s.session_id).classes("text-xs am-mono am-muted")
                    with ui.row().classes("gap-2"):
                        if s.status == "running":
                            ui.button("Cancel", icon="stop", color="warning", on_click=lambda: self._confirm_cancel(s)).props("unelevated no-caps")
                        ui.button("Delete", icon="delete", color="negative", on_click=lambda: self._confirm_delete(s)).props("flat no-caps")
                training = s.config.get("training", {})
                with ui.row().classes("gap-2"):
                    for icon, text in (
                        ("hub", f"{s.backend} · {s.model}"),
                        ("dataset", s.dataset),
                        ("flag", ", ".join(s.metrics)),
                        ("functions", f"{training.get('loss', '?')} + {training.get('optimizer', '?')}"),
                        ("schedule", f"started {relative_time(s.start_time)}"),
                    ):
                        ui.chip(text, icon=icon, color="grey-7").props("outline dense")
                if s.error_message:
                    with ui.row().classes("w-full items-center gap-2 p-3 rounded-lg bg-red-500/10 no-wrap"):
                        ui.icon("error", color="negative")
                        ui.label(s.error_message).classes("text-negative text-sm am-mono break-all")

            self.kpi_box = ui.element("div").classes("w-full grid grid-cols-2 xl:grid-cols-5 gap-3")

            with ui.card().props("flat").classes("am-card w-full gap-2 p-0"):
                with ui.tabs(value=self.detail_tab, on_change=lambda e: setattr(self, "detail_tab", e.value)).props(
                    "dense no-caps align=left active-color=primary indicator-color=primary"
                ).classes("w-full px-3 pt-2") as tabs:
                    ui.tab("curves", "Training curves", icon="show_chart")
                    ui.tab("roc", "ROC", icon="area_chart")
                    ui.tab("logs", "Logs", icon="terminal")
                    ui.tab("config", "Config", icon="data_object")
                with ui.tab_panels(tabs, value=self.detail_tab).props("animated").classes("w-full bg-transparent"):
                    with ui.tab_panel("curves"):
                        self.curve_box = ui.column().classes("w-full")
                    with ui.tab_panel("roc"):
                        if s.test_true is not None and s.test_pred is not None:
                            try:
                                options = _roc_options(s.test_true, s.test_pred)
                            except Exception as exc:
                                logger.error(f"Error creating ROC chart: {exc}")
                                options = None
                            if options:
                                ui.echart(options).classes("w-full h-[420px]")
                            else:
                                self._empty_state("area_chart", "Could not compute an ROC curve from the saved predictions.")
                        else:
                            self._empty_state("area_chart", "The ROC curve appears once the first trial finishes.")
                    with ui.tab_panel("logs"):
                        with ui.row().classes("w-full justify-end"):
                            ui.button("Copy path", icon="content_copy", on_click=lambda: ui.clipboard.write(s.log_file)).props("flat dense no-caps size=sm")
                        self.log_view = ui.log(max_lines=3000).classes("w-full am-log am-mono")
                    with ui.tab_panel("config"):
                        ui.code(json.dumps(s.config, indent=2), language="json").classes("w-full text-xs")

    @staticmethod
    def _empty_state(icon: str, text: str):
        with ui.column().classes("w-full items-center py-16 gap-2 am-muted"):
            ui.icon(icon, size="40px")
            ui.label(text)

    def _update_live(self, s: SessionInfo):
        history = self.session_manager.get_epoch_history(s.session_id, max_points=0)
        log_size = Path(s.log_file).stat().st_size if s.log_file and os.path.exists(s.log_file) else 0
        sig = (s.trial, s.current_epoch, round(s.progress, 4), s.loss, len(history), log_size, relative_time(s.start_time))
        if sig == self._live_sig:
            return
        self._live_sig = sig

        primary = s.metrics[0] if s.metrics else None
        best = max((row[primary] for row in history if isinstance(row.get(primary), (int, float))), default=None) if primary else None
        self.kpi_box.clear()
        with self.kpi_box:
            stat_tile("Progress", f"{s.progress:.0%}", "donut_large")
            stat_tile("Trial", f"{s.trial}/{s.n_trials}", "science")
            stat_tile("Epoch", f"{s.current_epoch}/{s.epochs}", "repeat")
            stat_tile(f"Best {primary or 'metric'}", f"{best:.4f}" if best is not None else "-", "emoji_events", f"loss {s.loss:.4f}")
            stat_tile("Elapsed", duration(s.start_time, s.end_time), "timer")

        if history:
            options = _curve_options(history, s.metrics)
            if self.curve_chart is None:
                self.curve_box.clear()
                with self.curve_box:
                    self.curve_chart = ui.echart(options).classes("w-full h-[420px]")
            else:
                self.curve_chart.options.clear()
                self.curve_chart.options.update(options)
                self.curve_chart.update()
        elif self.curve_chart is None:
            self.curve_box.clear()
            with self.curve_box:
                self._empty_state("show_chart", "Waiting for the first epoch to finish...")

        self._tail_log(s.log_file, log_size)

    def _tail_log(self, log_file: str, size: int):
        if self.log_view is None or not log_file or size <= self._log_pos:
            return
        first_read = self._log_pos == 0
        start = max(0, size - _LOG_TAIL_BYTES) if first_read else self._log_pos
        try:
            with open(log_file, "rb") as f:
                f.seek(start)
                chunk = f.read(size - start)
        except OSError as exc:
            logger.error(f"Error reading log {log_file}: {exc}")
            return
        # Only consume complete lines; a partially written last line is
        # picked up on the next tick.
        end = chunk.rfind(b"\n") + 1
        if end == 0:
            return
        self._log_pos = start + end
        lines = chunk[:end].decode("utf-8", errors="replace").splitlines()
        if first_read and start > 0:
            lines = lines[1:]  # starts mid-line
        if lines:
            self.log_view.push("\n".join(lines))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def _confirm(self, title: str, text: str, action: str, color: str) -> bool:
        with ui.dialog() as dialog, ui.card().classes("p-5 gap-3 w-[420px] max-w-full"):
            ui.label(title).classes("text-lg font-semibold")
            ui.label(text).classes("text-sm am-muted")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Keep it", on_click=lambda: dialog.submit(False)).props("flat no-caps")
                ui.button(action, color=color, on_click=lambda: dialog.submit(True)).props("unelevated no-caps")
        return bool(await dialog)

    async def _confirm_cancel(self, s: SessionInfo):
        if not await self._confirm("Cancel this search?", "The worker process is terminated; finished trials are kept.", "Cancel search", "warning"):
            return
        session = self.session_manager.get_session(s.session_id)
        # The worker overwrites `pid` with its own once it's alive; until
        # then it's still stamped with the GUI's own PID (see
        # training_worker.run()) -- guard against terminating ourselves.
        if session and session.pid == os.getpid():
            ui.notify("Training is still initializing, try again in a few seconds", type="warning")
            return
        self.session_manager.cancel_session(s.session_id)
        ui.notify("Session cancelled", type="warning")
        self.refresh()

    async def _confirm_delete(self, s: SessionInfo):
        if not await self._confirm(
            "Delete this session?",
            "Removes its record, logs, checkpoints and AutoMAX state from disk. This cannot be undone.",
            "Delete", "negative",
        ):
            return
        session = self.session_manager.get_session(s.session_id)
        if session and session.status == "running" and session.pid == os.getpid():
            ui.notify("Training is still initializing, try again in a few seconds", type="warning")
            return
        self.session_manager.delete_session(s.session_id)
        ui.notify("Session deleted")
        self.select(None)
