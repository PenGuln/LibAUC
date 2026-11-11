"""
Shared look-and-feel for the AutoMAX NiceGUI app: palette, session-status
metadata, page CSS, and a few small reusable widgets (section cards, status
chips, a server-side path picker).
"""
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from nicegui import ui

PRIMARY = "#4f46e5"

# Quasar prop string shared by every form field, so inputs look uniform.
FIELD = "outlined dense"

# status -> (label, material icon, quasar color)
STATUS = {
    "running": ("Running", "autorenew", "primary"),
    "completed": ("Completed", "check_circle", "positive"),
    "failed": ("Failed", "error", "negative"),
    "cancelled": ("Cancelled", "stop_circle", "warning"),
}

# modality -> (material icon, color), in the order datasets are grouped.
MODALITY_STYLE = {
    "X-Ray": ("personal_injury", "#0284c7"),
    "CT": ("donut_large", "#4f46e5"),
    "MRI": ("psychology", "#9333ea"),
    "Ultrasound": ("graphic_eq", "#0d9488"),
    "OCT": ("layers", "#d97706"),
    "Fundus": ("visibility", "#dc2626"),
    "Dermoscopy": ("back_hand", "#db2777"),
    "Endoscopy": ("gesture", "#65a30d"),
    "Microscopy": ("biotech", "#059669"),
}
_DEFAULT_MODALITY_STYLE = ("category", "#64748b")

CHART_COLORS = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#06b6d4", "#a855f7"]

_CSS = """
body { font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.body--light { background: #f5f6fb; }
.body--dark  { background: #0d0f16; }

.am-header { background: linear-gradient(100deg, #1e1b4b 0%, #4338ca 55%, #7c3aed 100%); }
.am-brand  { width: 38px; height: 38px; border-radius: 11px; background: rgba(255,255,255,.16);
             display: flex; align-items: center; justify-content: center; }

.am-card { border-radius: 16px; padding: 20px; border: 1px solid rgba(127,127,127,.16);
           box-shadow: 0 1px 2px rgba(16,24,40,.04); }
.body--light .am-card { background: #ffffff; }
.body--dark  .am-card { background: #161925; }
.am-card-icon { font-size: 20px; color: var(--q-primary); background: rgba(99,102,241,.12);
                border-radius: 10px; padding: 8px; }

.am-stat { border-radius: 14px; padding: 14px 16px; border: 1px solid rgba(127,127,127,.16); min-width: 150px; }
.body--light .am-stat { background: #fafaff; }
.body--dark  .am-stat { background: #1b1f2e; }

.am-mono { font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace; }
.am-muted { opacity: .62; }

.am-session { cursor: pointer; border-radius: 12px; border: 1px solid transparent; transition: background .15s, border-color .15s; }
.am-session:hover { background: rgba(99,102,241,.07); }
.am-session.am-selected { border-color: var(--q-primary); background: rgba(99,102,241,.11); }

.am-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.am-pulse { animation: am-pulse 1.4s ease-in-out infinite; }
@keyframes am-pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(99,102,241,.55); } 50% { box-shadow: 0 0 0 6px rgba(99,102,241,0); } }

.am-hp { border-radius: 10px; padding: 6px 8px; }
.am-hp:hover { background: rgba(127,127,127,.06); }

.am-launch { border-radius: 12px; font-weight: 600; letter-spacing: .2px;
             background: linear-gradient(100deg, #4f46e5, #7c3aed) !important; }

.am-group { display: flex; align-items: center; gap: 8px; padding: 12px 16px 4px;
            font-size: 11px; font-weight: 700; letter-spacing: .6px; text-transform: uppercase; }
.am-group-count { margin-left: auto; opacity: .7; font-weight: 500; }
.am-modality-chip { display: inline-flex; align-items: center; gap: 4px; margin-left: 8px; padding: 1px 8px;
                    border-radius: 999px; font-size: 11px; font-weight: 600; white-space: nowrap; }

/* The modality colors are mid-tones, too dim on dark surfaces at this size. */
.body--dark .am-group, .body--dark .am-modality-chip { filter: brightness(1.45); }

.am-gpu { cursor: pointer; border-radius: 12px; padding: 10px 12px; border: 1px solid rgba(127,127,127,.18);
          transition: background .15s, border-color .15s; }
.am-gpu:hover { background: rgba(99,102,241,.06); }
.am-gpu.am-selected { border-color: var(--q-primary); background: rgba(99,102,241,.10); }
.am-gpu.am-gpu-off { opacity: .45; cursor: not-allowed; }

.am-log { height: 420px; font-size: 12px; border-radius: 12px; }
.body--light .am-log { background: #0f1220; color: #d7dbea; }
.body--dark  .am-log { background: #0a0c13; color: #d7dbea; }
"""

_FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700'
    '&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'
)


def apply_theme() -> None:
    """Install the palette, fonts and page CSS for the current client."""
    ui.colors(
        primary=PRIMARY, secondary="#0ea5e9", accent="#a855f7",
        positive="#10b981", negative="#ef4444", warning="#f59e0b", info="#0ea5e9",
    )
    ui.add_head_html(_FONTS)
    ui.add_css(_CSS)


def section_card(title: str, icon: str, subtitle: Optional[str] = None) -> ui.card:
    """A rounded card with an icon/title header; use as a context manager."""
    card = ui.card().props("flat").classes("am-card w-full gap-4")
    with card:
        with ui.row().classes("items-center gap-3 no-wrap"):
            ui.icon(icon).classes("am-card-icon")
            with ui.column().classes("gap-0"):
                ui.label(title).classes("text-base font-semibold")
                if subtitle:
                    ui.label(subtitle).classes("text-xs am-muted")
    return card


def modality_sort_key(modality: str) -> int:
    """Position of ``modality`` in the display order; unknown ones go last."""
    order = list(MODALITY_STYLE)
    return order.index(modality) if modality in order else len(order)


def add_modality_groups(select: ui.select, modalities: Dict[str, str]) -> None:
    """
    Show ``select``'s options under a header per modality, and the selected one
    with a modality chip. ``modalities`` maps option labels to modalities, in
    display order, so each group must be contiguous. Other options show plainly.
    """
    counts = Counter(modalities.values())
    meta, seen = {}, set()
    for label, modality in modalities.items():
        icon, color = MODALITY_STYLE.get(modality, _DEFAULT_MODALITY_STYLE)
        meta[label] = {"modality": modality, "icon": icon, "color": color,
                       "count": 0 if modality in seen else counts[modality]}
        seen.add(modality)
    # Embedded in a double-quoted Vue attribute, so it may only use single quotes.
    lookup = json.dumps(meta).replace('"', "'") + "[props.opt.label]"
    select.add_slot("option", f"""
        <div v-for="m in [{lookup}]" :key="props.opt.value">
            <div v-if="m && m.count" class="am-group" :style="{{color: m.color}}">
                <q-icon :name="m.icon" size="16px" /><span>{{{{ m.modality }}}}</span>
                <span class="am-group-count">{{{{ m.count }}}}</span>
            </div>
            <q-item v-bind="props.itemProps" dense>
                <q-item-section :class="m ? 'q-pl-lg' : ''">{{{{ props.opt.label }}}}</q-item-section>
            </q-item>
        </div>
    """)
    select.add_slot("selected-item", f"""
        <div v-for="m in [{lookup}]" :key="props.opt.value" class="row items-center no-wrap">
            <span class="ellipsis">{{{{ props.opt.label }}}}</span>
            <span v-if="m" class="am-modality-chip" :style="{{color: m.color, background: m.color + '1f'}}">
                <q-icon :name="m.icon" size="13px" />{{{{ m.modality }}}}
            </span>
        </div>
    """)


def status_chip(status: str) -> ui.chip:
    label, icon, color = STATUS.get(status, (status.title(), "help", "grey"))
    return ui.chip(label, icon=icon, color=color, text_color="white").props("dense square")


def status_dot(status: str) -> ui.element:
    color = STATUS.get(status, ("", "", "grey"))[2]
    dot = ui.element("div").classes(f"am-dot bg-{color}")
    if status == "running":
        dot.classes("am-pulse")
    return dot


def stat_tile(label: str, value: str, icon: str, caption: Optional[str] = None) -> None:
    with ui.column().classes("am-stat gap-1 grow"):
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon(icon, size="16px").classes("text-primary")
            ui.label(label).classes("text-xs uppercase tracking-wide am-muted")
        ui.label(value).classes("text-xl font-semibold am-mono")
        if caption:
            ui.label(caption).classes("text-xs am-muted")


def relative_time(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        seconds = (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except ValueError:
        return ""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit} ago"
    return "just now"


def duration(start: Optional[str], end: Optional[str]) -> str:
    if not start:
        return "-"
    try:
        stop = datetime.fromisoformat(end) if end else datetime.now()
        seconds = int((stop - datetime.fromisoformat(start)).total_seconds())
    except ValueError:
        return "-"
    h, rem = divmod(max(seconds, 0), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


class PathPicker(ui.dialog):
    """
    Server-side file/directory browser. The GUI is usually opened through a
    forwarded port from a laptop while data lives on the training host, so a
    browser-native picker (which browses the *client* machine) is useless --
    this lists the server's filesystem instead.

    Usage::

        path = await PathPicker(start, pick_files=False)
    """

    _MAX_ENTRIES = 500

    def __init__(self, start: str = "", pick_files: bool = False):
        super().__init__()
        self.pick_files = pick_files
        start = os.path.expanduser(start or "")
        self.path = Path(start if os.path.isdir(start) else os.path.dirname(start) or os.getcwd()).resolve()
        if not self.path.is_dir():
            self.path = Path.cwd()

        with self, ui.card().classes("w-[560px] max-w-full gap-3 p-5"):
            with ui.row().classes("items-center gap-2 w-full no-wrap"):
                ui.icon("folder_open", size="22px").classes("text-primary")
                ui.label("Choose a file" if pick_files else "Choose a folder").classes("text-lg font-semibold")
            with ui.row().classes("items-center gap-1 w-full no-wrap"):
                ui.button(icon="arrow_upward", on_click=self._up).props("flat round dense").tooltip("Parent folder")
                ui.button(icon="home", on_click=lambda: self._go(Path.home())).props("flat round dense").tooltip("Home")
                self.path_input = ui.input(value=str(self.path)).props(FIELD).classes("grow am-mono text-xs")
                self.path_input.on("keydown.enter", lambda: self._go(Path(os.path.expanduser(self.path_input.value))))
            self.listing = ui.scroll_area().classes("w-full h-80 border rounded-lg")
            with ui.row().classes("w-full justify-end gap-2"):
                ui.button("Cancel", on_click=lambda: self.submit(None)).props("flat no-caps")
                if not pick_files:
                    ui.button("Select this folder", icon="check", on_click=lambda: self.submit(str(self.path))).props("unelevated no-caps")
        self._render()
        self.open()

    def _up(self):
        self._go(self.path.parent)

    def _go(self, path: Path):
        if path.is_dir():
            self.path = path.resolve()
            self._render()
        elif self.pick_files and path.is_file():
            self.submit(str(path))
        else:
            ui.notify(f"Not a folder: {path}", type="warning")

    def _render(self):
        self.path_input.value = str(self.path)
        self.listing.clear()
        try:
            entries = sorted(
                (p for p in self.path.iterdir() if not p.name.startswith(".")),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except OSError as exc:
            with self.listing:
                ui.label(f"Cannot read folder: {exc}").classes("text-negative text-sm p-3")
            return
        with self.listing, ui.list().props("dense separator").classes("w-full"):
            shown = 0
            for p in entries:
                is_dir = p.is_dir()
                if not is_dir and not self.pick_files:
                    continue
                if shown >= self._MAX_ENTRIES:
                    ui.item_label(f"... more than {self._MAX_ENTRIES} entries, type a path above").classes("p-3 am-muted")
                    break
                shown += 1
                with ui.item(on_click=lambda p=p: self._go(p)):
                    with ui.item_section().props("avatar"):
                        ui.icon("folder" if is_dir else "description", color="amber-7" if is_dir else "grey-6")
                    with ui.item_section():
                        ui.item_label(p.name).classes("am-mono text-sm")
            if not shown:
                ui.item_label("Empty folder").classes("p-3 am-muted")
