"""
GUI main entry point for AutoMAX: a NiceGUI web app.

Usage::

    python -m libauc.auto.ui.gui_main [--host 127.0.0.1] [--port 8080] [--show]

On a remote/cluster machine, forward the port (VS Code does this
automatically; otherwise ``ssh -L 8080:localhost:8080 <host>``) and open
http://localhost:8080 in a local browser.
"""

import argparse
import logging
import secrets
import socket

from fastapi import Request
from fastapi.responses import PlainTextResponse
from nicegui import app, ui

from .gui import ConfigurePanel
from .monitor import SessionsPanel
from .session_manager import get_session_manager
from .theme import apply_theme

# Configure logging BEFORE any session manager is created
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("automax_gui.log"),
    ],
)


_TOKEN_COOKIE = "automax_token"


@ui.page("/")
def index():
    apply_theme()
    dark = ui.dark_mode(None)  # follow the OS until toggled

    with ui.header().classes("am-header items-center px-4 md:px-8 py-2 gap-4"):
        with ui.row().classes("items-center gap-3 no-wrap"):
            with ui.element("div").classes("am-brand"):
                ui.icon("auto_graph", size="22px").classes("text-white")
            with ui.column().classes("gap-0"):
                ui.label("AutoMAX").classes("text-lg font-bold leading-tight text-white")
                ui.label("LibAUC hyperparameter search").classes("text-xs text-white/70 leading-tight gt-xs")
        ui.space()
        with ui.tabs(value="configure").props("dense no-caps indicator-color=white active-color=white inline-label").classes("text-white/75") as tabs:
            ui.tab("configure", "New search", icon="tune")
            with ui.tab("sessions", "Sessions", icon="monitor_heart"):
                running_badge = ui.badge("", color="pink-5").props("floating rounded")
        ui.space()
        ui.button(icon="dark_mode", on_click=dark.toggle).props("flat round color=white").tooltip("Toggle dark mode")

    def on_launched(session_id: str):
        sessions.select(session_id)
        tabs.set_value("sessions")

    with ui.column().classes("w-full max-w-[1400px] mx-auto px-4 md:px-8 py-6"):
        with ui.tab_panels(tabs, value="configure").props("animated").classes("w-full bg-transparent"):
            with ui.tab_panel("configure").classes("p-0"):
                ConfigurePanel(on_launched=on_launched)
            with ui.tab_panel("sessions").classes("p-0"):
                sessions = SessionsPanel()

    def update_badge():
        n = sum(s.status == "running" for s in sessions.sessions)
        running_badge.text = str(n)
        running_badge.visible = n > 0

    update_badge()
    ui.timer(2.0, update_badge)


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def _first_free_port(host: str, start: int, attempts: int = 50) -> int:
    for port in range(start, start + attempts):
        if _port_is_free(host, port):
            return port
    raise SystemExit(f"No free port in {start}-{start + attempts - 1}; pass one with --port")


def _require_token(token: str) -> None:
    """Gate every HTTP request behind ``token``: accepted once as a
    ``?token=`` query parameter, then remembered in a cookie."""

    @app.middleware("http")
    async def check_token(request: Request, call_next):
        if request.cookies.get(_TOKEN_COOKIE) == token:
            return await call_next(request)
        if request.query_params.get("token") == token:
            response = await call_next(request)
            response.set_cookie(_TOKEN_COOKIE, token, httponly=True, samesite="strict")
            return response
        return PlainTextResponse("AutoMAX: open the URL with ?token=... printed in the server terminal.", status_code=403)


def main():
    parser = argparse.ArgumentParser(description="AutoMAX GUI")
    # Loopback by default: the GUI can launch processes and browse the
    # filesystem, so don't expose it on a shared cluster network.
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="interface to bind (default: 127.0.0.1). Use 0.0.0.0 to reach it from another node, "
             "e.g. a GUI on a compute node viewed through the login node; access then needs a token.",
    )
    parser.add_argument("--port", type=int, default=None, help="port to serve on (default: first free port from 8080)")
    parser.add_argument("--show", action="store_true", help="open a browser tab on start")
    args = parser.parse_args()

    if args.port is None:
        # Shared cluster nodes often have 8080 taken already.
        port = _first_free_port(args.host, 8080)
    elif _port_is_free(args.host, args.port):
        port = args.port
    else:
        raise SystemExit(f"Port {args.port} is already in use; pick another with --port")

    is_loopback = args.host in ("127.0.0.1", "localhost", "::1")
    if is_loopback:
        url = f"http://127.0.0.1:{port}"
    else:
        token = secrets.token_urlsafe(16)
        _require_token(token)
        url = f"http://{socket.gethostname()}:{port}/?token={token}"

    get_session_manager()  # start liveness monitoring before the first page load
    app.on_startup(lambda: print(f"\nAutoMAX GUI: {url}\n", flush=True))
    ui.run(
        host=args.host, port=port, show=args.show,
        title="AutoMAX", favicon="📈", reload=False, show_welcome_message=False,
    )


if __name__ == "__main__":
    main()
