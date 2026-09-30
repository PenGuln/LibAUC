"""
Training session monitoring page for AutoMAX.
"""

import logging
import threading
from typing import List, Optional

import flet as ft
import numpy as np
from sklearn.metrics import roc_curve

from .session_manager import SessionInfo, get_session_manager

logger = logging.getLogger(__name__)


def _safe_update(control):
    """``control.update()``, but a no-op if not yet attached to a live page
    (e.g. during headless construction/testing, or a stray background-thread
    refresh tick that fires just before/after the page transition)."""
    try:
        control.update()
    except AssertionError:
        pass


_CHART_COLORS = [ft.colors.BLUE, ft.colors.RED, ft.colors.GREEN, ft.colors.ORANGE, ft.colors.PURPLE]


class MonitorPage(ft.Column):
    """Training session monitoring page."""

    def __init__(self):
        super().__init__()
        self.session_manager = get_session_manager()
        self.sessions: List[SessionInfo] = []
        self.selected_session: Optional[SessionInfo] = None

        self._auto_refresh_active = False
        self._auto_refresh_timer: Optional[threading.Timer] = None

        self._create_header()
        self._create_session_list()
        self._create_session_details()
        self._create_controls()

        self._load_sessions_data()

        self.controls = [
            self.header,
            ft.Container(
                content=ft.Row([self.session_list_container, self.session_details_container]),
                padding=ft.padding.all(20),
            ),
            self.controls_container,
        ]

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _create_header(self):
        self.header = ft.Container(
            content=ft.Row(
                [
                    ft.IconButton(icon=ft.icons.ARROW_BACK, tooltip="Back to Main", on_click=self._go_back, bgcolor=ft.colors.BLUE_100, icon_color=ft.colors.BLUE_700),
                    ft.Text("Training Session Monitor", size=24, weight=ft.FontWeight.BOLD, color=ft.colors.GREY_800),
                    ft.IconButton(icon=ft.icons.REFRESH, tooltip="Refresh Sessions", on_click=self._refresh_sessions, bgcolor=ft.colors.GREEN_100, icon_color=ft.colors.GREEN_700),
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.all(15),
            margin=ft.margin.only(bottom=15),
            bgcolor=ft.colors.GREY_50,
            border_radius=ft.border_radius.all(8),
        )

    def _create_session_list(self):
        self.session_list = ft.ListView(height=400, width=400, spacing=5, padding=ft.padding.all(10))
        self.session_list_container = ft.Container(
            content=ft.Column([ft.Text("Training Sessions", size=18, weight=ft.FontWeight.BOLD), self.session_list]),
            width=400, height=500,
            border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(8), padding=ft.padding.all(10),
        )

    def _create_session_details(self):
        self.session_details = ft.Column(
            [ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD), ft.Text("Select a session to view details", color=ft.colors.GREY_600)],
            spacing=10, scroll=ft.ScrollMode.AUTO, expand=True,
        )
        self.session_details_container = ft.Container(
            content=self.session_details, width=600, height=500,
            border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(8), padding=ft.padding.all(15),
        )

    def _create_controls(self):
        self.controls_container = ft.Container(
            content=ft.Row(
                [
                    ft.ElevatedButton("Cancel Session", icon=ft.icons.STOP, on_click=self._cancel_session, disabled=True, bgcolor=ft.colors.ORANGE_600, color=ft.colors.WHITE),
                    ft.ElevatedButton("View Logs", icon=ft.icons.DESCRIPTION, on_click=self._view_logs, disabled=True, bgcolor=ft.colors.GREEN_600, color=ft.colors.WHITE),
                    ft.ElevatedButton("Delete Session", icon=ft.icons.DELETE, on_click=self._delete_session, disabled=True, bgcolor=ft.colors.RED_600, color=ft.colors.WHITE),
                ],
                alignment=ft.MainAxisAlignment.CENTER, spacing=15,
            ),
            padding=ft.padding.all(15),
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_sessions_data(self):
        self.sessions = self.session_manager.get_all_sessions()

    def _load_sessions(self):
        self._load_sessions_data()
        self._update_session_list()
        self._update_session_details()
        self._update_controls()

    def _update_session_list(self):
        self.session_list.controls = []
        if not self.sessions:
            self.session_list.controls.append(
                ft.Container(content=ft.Text("No training sessions found", color=ft.colors.GREY_600), alignment=ft.alignment.center, height=100)
            )
        else:
            for session in self.sessions:
                self.session_list.controls.append(self._create_session_card(session))
        _safe_update(self.session_list)

    def _create_session_card(self, session: SessionInfo):
        status_color = {
            "running": ft.colors.GREEN_600,
            "completed": ft.colors.BLUE_600,
            "failed": ft.colors.RED_600,
            "cancelled": ft.colors.ORANGE_600,
        }.get(session.status, ft.colors.GREY_600)

        status_display = {
            "running": "\U0001f504 Running",
            "completed": "✅ Completed",
            "failed": "❌ Failed",
            "cancelled": "⏹ Cancelled",
        }.get(session.status, session.status.title())

        return ft.Container(
            content=ft.Column(
                [
                    ft.Row(
                        [
                            ft.Text(f"Session: {session.session_id[:8]}...", weight=ft.FontWeight.BOLD),
                            ft.Container(content=ft.Text(status_display, color=ft.colors.WHITE, size=12), bgcolor=status_color, padding=ft.padding.symmetric(4, 8), border_radius=ft.border_radius.all(4)),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    ft.Text(f"Name: {session.name}", size=12, color=ft.colors.GREY_700),
                    ft.Text(f"Backend: {session.backend}  Model: {session.model}", size=12, color=ft.colors.GREY_700),
                    ft.Text(f"Dataset: {session.dataset}", size=12, color=ft.colors.GREY_700),
                    ft.Text(f"Metrics: {', '.join(session.metrics)}", size=12, color=ft.colors.GREY_700),
                    ft.Text(f"Trial: {session.trial}/{session.n_trials}  Progress: {session.progress:.2f}", size=12, color=ft.colors.GREY_700),
                ],
                spacing=5,
            ),
            padding=ft.padding.all(10),
            border=ft.border.all(1, ft.colors.GREY_300),
            border_radius=ft.border_radius.all(6),
            bgcolor=ft.colors.WHITE,
            width=380, height=190,
            on_click=lambda e, s=session: self._select_session(s),
        )

    def _select_session(self, session: SessionInfo):
        self.selected_session = session
        self._update_session_details()
        self._update_controls()

    def _update_session_details(self):
        if not self.selected_session:
            self.session_details.controls = [
                ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD),
                ft.Text("Select a session to view details", color=ft.colors.GREY_600),
            ]
            _safe_update(self.session_details)
            return

        # Pull the freshest copy in case a background refresh updated it
        session = self.session_manager.get_session(self.selected_session.session_id) or self.selected_session
        self.selected_session = session

        details = [
            ft.Text("Session Details", size=18, weight=ft.FontWeight.BOLD),
            ft.Divider(),
            ft.Text(f"Session ID: {session.session_id}", weight=ft.FontWeight.BOLD),
            ft.Text(f"Name: {session.name}"),
            ft.Text(f"Status: {session.status.title()}"),
            ft.Text(f"Backend: {session.backend}"),
            ft.Text(f"Start Time: {session.start_time}"),
            ft.Text(f"End Time: {session.end_time or 'N/A'}"),
            ft.Text(f"Model: {session.model}"),
            ft.Text(f"Dataset: {session.dataset}"),
            ft.Text(f"Metrics: {', '.join(session.metrics)}"),
            ft.Text(f"Trial: {session.trial}/{session.n_trials}"),
            ft.Text(f"Epochs: {session.current_epoch}/{session.epochs}"),
            ft.Text(f"Progress: {session.progress:.2f}"),
            ft.Text(f"Loss: {session.loss:.4f}"),
        ]

        if session.error_message:
            details.append(ft.Text(f"Error: {session.error_message}", color=ft.colors.RED_600))
        if session.last_heartbeat:
            details.append(ft.Text(f"Last Heartbeat: {session.last_heartbeat}"))

        loss_chart = self._create_loss_chart(session.session_id)
        if loss_chart:
            details.append(ft.Divider())
            details.append(ft.Text("Training Curve", size=16, weight=ft.FontWeight.BOLD))
            details.append(loss_chart)

        if session.test_true is not None and session.test_pred is not None:
            details.append(ft.Divider())
            details.append(ft.Text("ROC Curve (best/final trial)", size=16, weight=ft.FontWeight.BOLD))
            roc_chart = self._create_roc_chart(session.test_true, session.test_pred)
            if roc_chart:
                details.append(roc_chart)

        self.session_details.controls = details
        _safe_update(self.session_details)

    def _create_loss_chart(self, session_id: str):
        """Live loss/metric line chart from the per-session epoch-history
        JSONL file -- replaces the old GUI's wandb-only `update_chart()`,
        since wandb isn't installed in this environment."""
        history = self.session_manager.get_epoch_history(session_id, max_points=200)
        if not history:
            return None
        try:
            keys = sorted({k for row in history for k in row if k != "epoch"})
            series = []
            legend_items = []
            for i, key in enumerate(keys):
                points = [ft.LineChartDataPoint(row["epoch"], row[key]) for row in history if key in row]
                if not points:
                    continue
                color = _CHART_COLORS[i % len(_CHART_COLORS)]
                series.append(ft.LineChartData(data_points=points, stroke_width=2, color=color, curved=True, stroke_cap_round=True))
                legend_items.append(
                    ft.Row([ft.Container(width=12, height=12, bgcolor=color, border_radius=3), ft.Text(key, size=11)], spacing=4)
                )
            if not series:
                return None

            max_epoch = max(row["epoch"] for row in history)
            chart = ft.Container(
                content=ft.LineChart(
                    data_series=series,
                    border=ft.border.all(1, ft.colors.GREY_400),
                    left_axis=ft.ChartAxis(labels_size=30, title=ft.Text("value")),
                    bottom_axis=ft.ChartAxis(labels_size=30, title=ft.Text("epoch")),
                    tooltip_bgcolor=ft.colors.with_opacity(0.8, ft.colors.BLUE_GREY),
                    min_x=1, max_x=max(max_epoch, 1),
                    width=450, height=300,
                ),
            )
            return ft.Column([ft.Row(legend_items, spacing=15), chart], spacing=5)
        except Exception as e:
            logger.error(f"Error creating loss chart for session {session_id}: {e}")
            return ft.Text(f"Error creating training curve: {e}", color=ft.colors.RED_600)

    def _create_roc_chart(self, test_true: List[float], test_pred: List[float]):
        try:
            test_true = np.asarray(test_true)
            test_pred = np.asarray(test_pred)
            if test_pred.min() < 0 or test_pred.max() > 1:
                from scipy.special import expit
                test_pred = expit(test_pred)
            test_true = test_true.flatten()
            test_pred = test_pred.flatten()

            fpr, tpr, _ = roc_curve(test_true, test_pred)
            data_points = [ft.LineChartDataPoint(x, y) for x, y in zip(fpr.tolist(), tpr.tolist())]
            roc_data = ft.LineChartData(data_points=data_points, stroke_width=2, color=ft.colors.BLUE, curved=True, stroke_cap_round=True)

            return ft.Container(
                content=ft.LineChart(
                    data_series=[roc_data],
                    border=ft.border.all(1, ft.colors.GREY_400),
                    left_axis=ft.ChartAxis(labels_size=30, title=ft.Text("TPR")),
                    bottom_axis=ft.ChartAxis(labels_size=30, title=ft.Text("FPR")),
                    tooltip_bgcolor=ft.colors.with_opacity(0.8, ft.colors.BLUE_GREY),
                    min_x=0, max_x=1, min_y=0, max_y=1,
                    width=450, height=400,
                ),
            )
        except Exception as e:
            logger.error(f"Error creating ROC chart: {e}")
            return ft.Text(f"Error creating ROC curve: {e}", color=ft.colors.RED_600)

    def _update_controls(self):
        has_selection = self.selected_session is not None
        is_running = has_selection and self.selected_session.status == "running"
        buttons = self.controls_container.content.controls
        for i, button in enumerate(buttons):
            button.disabled = (not has_selection or not is_running) if i == 0 else not has_selection
        _safe_update(self.controls_container)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _cancel_session(self, e):
        if self.selected_session and self.selected_session.status == "running":
            session_id = self.selected_session.session_id
            if self.session_manager.cancel_session(session_id):
                logger.info(f"Cancelled session {session_id}")
                self._load_sessions()
            else:
                logger.error(f"Failed to cancel session {session_id}")

    def _refresh_sessions(self, e):
        self.selected_session = None
        self._load_sessions()

    def _delete_session(self, e):
        if self.selected_session:
            session_id = self.selected_session.session_id
            if self.session_manager.delete_session(session_id):
                logger.info(f"Deleted session {session_id}")
                self.selected_session = None
                self._load_sessions()
            else:
                logger.error(f"Failed to delete session {session_id}")

    def _view_logs(self, e):
        if not self.selected_session:
            return
        session = self.selected_session
        logs = self.session_manager.get_session_logs(session.session_id, lines=1000)
        if not logs:
            logger.warning(f"No logs available for session {session.session_id}")
            return
        log_dialog = ft.AlertDialog(
            title=ft.Text(f"Logs for {session.name}"),
            content=ft.Container(content=ft.Text("".join(logs), selectable=True, size=10), width=600, height=400),
            actions=[ft.TextButton("Close", on_click=lambda e: self.page.close(log_dialog))],
        )
        self.page.dialog = log_dialog
        log_dialog.open = True
        self.page.update()

    def _go_back(self, e):
        self._stop_auto_refresh()
        self.page.go("/")

    # ------------------------------------------------------------------
    # Auto-refresh (old GUI's monitor page required a manual click; this
    # periodically reloads while any session is running)
    # ------------------------------------------------------------------

    def _start_auto_refresh(self):
        self._auto_refresh_active = True
        self._schedule_auto_refresh()

    def _schedule_auto_refresh(self):
        if not self._auto_refresh_active:
            return
        self._auto_refresh_timer = threading.Timer(4.0, self._auto_refresh_tick)
        self._auto_refresh_timer.daemon = True
        self._auto_refresh_timer.start()

    def _auto_refresh_tick(self):
        if not self._auto_refresh_active:
            return
        try:
            if any(s.status == "running" for s in self.session_manager.get_all_sessions()):
                self._load_sessions()
        except Exception as exc:
            logger.error(f"Auto-refresh error: {exc}")
        finally:
            self._schedule_auto_refresh()

    def _stop_auto_refresh(self):
        self._auto_refresh_active = False
        if self._auto_refresh_timer:
            self._auto_refresh_timer.cancel()
            self._auto_refresh_timer = None

    def initialize(self):
        """Initialize the monitor page after it's added to the page."""
        self._load_sessions()
        self._start_auto_refresh()
