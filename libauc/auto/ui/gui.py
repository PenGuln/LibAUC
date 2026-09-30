"""
AutoMAX GUI form: configure a dataset/model/loss/optimizer/search-space and
launch a training session, ported from AutoAUC_demo's Flet GUI and adapted
to AutoMAX's search stack (libauc's ``parse_defaultconfig``/spaces, and the
``"range"``-key hyperparameter contract in ``libauc.auto.helpers``).
"""
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import flet as ft
import numpy as np

from libauc.trainer import Trainer, load_dataset, parse_defaultconfig

from ..custom_datasets import load_custom_csv_dataset
from .dataset_config import get_dataset_config_store
from .session_manager import get_session_manager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed choices (v1: no free-text override -- these are the names actually
# supported by the installed libauc.trainer.data.datasets.load_dataset() /
# image_trainer.py's / graph_trainer.py's model registries)
# ---------------------------------------------------------------------------

_IMAGE_MODELS = ["resnet18", "resnet20", "densenet121"]
_IMAGE_DATASETS = ["cifar10", "pneumoniamnist", "breastmnist", "chestmnist", "melanoma", "ddsm", "chexpert"]
_GNN_MODELS = ["gcn", "gin", "gine", "graphsage", "gat", "mpnn", "deepergcn", "pna"]
_GNN_DATASETS = ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molmuv", "ogbg-molpcba"]

# Sentinel dropdown option that switches the model field over to a free-text
# HuggingFace repo ID. Both image_trainer.py and graph_trainer.py already
# recognize any model name containing "/" as an HF Hub repo ID (loaded via
# AutoModelForImageClassification / AutoModel respectively) -- this option
# just exposes that existing capability in the GUI. Never collides with a
# real repo ID, since those always contain "/".
_HF_MODEL_OPTION = "HuggingFace Model..."

# Sentinel dropdown option for a user-defined dataset.
#
# Image backend: reveals a small form (CSV paths, image root, column names)
# and loads via libauc.auto.custom_datasets.load_custom_csv_dataset()
# (libauc's MedicalImageCSVDataset) instead of the fixed name registry in
# libauc.trainer.load_dataset(). The chosen name is just an identifier (also
# used as the persisted registry key, see ui.dataset_config), not passed to
# load_dataset().
#
# GNN backend: not implemented yet -- selecting it just shows a message and
# blocks Inspect/Start.
_CUSTOM_DATASET_OPTION = "Custom Dataset..."

# The 10 canonical loss names libauc.trainer.parse_defaultconfig understands,
# each of which deterministically fixes its paired optimizer -- replaces the
# old GUI's two independently-editable optimizer/loss dropdowns (which could
# silently end up mismatched) with a single driver.
#
# Note: "tpAUC_CVaR_loss" is listed here because parse_defaultconfig() (and
# thus the search-space UI below) supports it, but the installed libauc
# package cannot currently resolve this loss name at training time
# (Trainer._get_loss("tpAUC_CVaR_loss") raises -- a pre-existing upstream
# issue, not something introduced by this GUI).
_LOSS_NAMES = [
    "AUCMLoss", "CompositionalAUCLoss", "APLoss", "pAUC_CVaR_Loss",
    "pAUC_DRO_Loss", "tpAUC_KL_Loss", "tpAUC_CVaR_loss", "NDCGLoss",
    "CrossEntropyLoss", "BCELoss",
]

# Constructor parameter names to never expose as an extra tunable field
# (injected by the Trainer itself, not user-facing hyperparameters).
_SKIP_PARAM_NAMES = {"self", "params", "loss_fn", "model", "args", "kwargs"}

# Filenames looked for directly under a custom dataset's Root Path to
# auto-detect its train/val/test CSVs (first match wins).
_TRAIN_CSV_CANDIDATES = ["train.csv"]
_VAL_CSV_CANDIDATES = ["val.csv", "valid.csv", "validation.csv"]
_TEST_CSV_CANDIDATES = ["test.csv"]


class Parameter(ft.Column):
    """One hyperparameter row: interval (min/max + optional log scale),
    categorical (Python-literal list), or a single fixed value."""

    def __init__(self, name, default, val, log=False, cat=False):
        super().__init__()
        self.name = name
        if val is None:
            val = "'None'"
        elif isinstance(val, str):
            val = "'" + val + "'"
        elif val is inspect._empty:
            val = "'None'"

        if default is None:
            default = "'None'"
        elif isinstance(default, str):
            default = "'" + default + "'"
        elif default is inspect._empty:
            default = "'None'"

        self.edit = ft.TextField(value=str(val), label=self.name, width=220)

        self.low = ft.TextField(label=self.name + " min", width=105, visible=False)
        self.high = ft.TextField(label="max", width=105, visible=False)
        if not cat:
            self.low.value = str(val[0])
            self.high.value = str(val[1])
            self.edit.visible = False
            self.low.visible = True
            self.high.visible = True

        self.default = ft.TextField(value=str(default), label="default value", width=100)

        if cat and log:
            log = False

        self.log = ft.Row(
            [
                ft.IconButton(
                    icon=ft.icons.MOVING,
                    tooltip="Log scale",
                    on_click=self.log_toggle,
                    style=ft.ButtonStyle(
                        bgcolor=ft.colors.BLUE_100 if log else ft.colors.GREY_100,
                        color=ft.colors.BLUE_700 if log else ft.colors.GREY_600,
                    ),
                )
            ],
            visible=False if cat else True,
        )
        self.switch = ft.CupertinoSlidingSegmentedButton(
            selected_index=1 if cat else 0,
            disabled=log,
            thumb_color=ft.colors.BLUE_400,
            on_change=self.categorical_changed,
            padding=ft.padding.symmetric(0, 10),
            controls=[
                ft.Icon(ft.icons.TRENDING_FLAT, tooltip="Interval"),
                ft.Icon(ft.icons.CATEGORY_OUTLINED, tooltip="Categorical"),
            ],
        )

        self.controls = [
            ft.Row(controls=[self.edit, self.low, self.high, self.default, self.switch, self.log])
        ]

    def log_toggle(self, e):
        current_state = self.log.controls[0].style.bgcolor == ft.colors.BLUE_100
        new_state = not current_state
        self.log.controls[0].style.bgcolor = ft.colors.BLUE_100 if new_state else ft.colors.GREY_100
        self.log.controls[0].style.color = ft.colors.BLUE_700 if new_state else ft.colors.GREY_600
        self.switch.disabled = new_state
        self.update()

    def categorical_changed(self, e):
        if self.switch.selected_index == 1:
            self.log.visible = False
            self.edit.visible = True
            self.low.visible = False
            self.high.visible = False
        else:
            self.log.visible = True
            self.edit.visible = False
            self.low.visible = True
            self.high.visible = True
        self.update()

    def export_setting(self):
        data = {}
        if self.switch.selected_index == 0:
            # Continuous range -- use the explicit "range" key (see
            # libauc.auto.helpers.parse_hyperparameters_from_dict) so this survives
            # the GUI -> JSON -> subprocess round-trip as a bound, never
            # misread as a 2-choice categorical (JSON/YAML have no tuple type).
            data["range"] = [eval(self.low.value), eval(self.high.value)]
        else:
            data["val"] = eval(self.edit.value)

        log_enabled = self.log.controls[0].style.bgcolor == ft.colors.BLUE_100
        if log_enabled:
            data["log"] = True
        if self.default.value != "'None'":
            data["default"] = eval(self.default.value)
        return data


class Space(ft.Column):
    """
    Displays (and lets the user extend) the resolved hyperparameter search
    space for one optimizer or loss. Populated externally via ``set_paras()``
    -- ``AutoMAXApp``'s single canonical Loss dropdown is the source of
    truth for which optimizer/loss pair is active.
    """

    def __init__(self, kind: str, resolver):
        super().__init__()
        self.width = 600
        self.alignment = ft.MainAxisAlignment.START
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER
        self.kind = kind  # "optimizer" | "loss"
        self.resolver = resolver  # e.g. Trainer._get_optimizer / Trainer._get_loss
        self.type_name = None

        self.title = ft.Text(f"{kind.capitalize()}: (none)", size=14, weight=ft.FontWeight.BOLD)
        self.export_btn = ft.IconButton(icon=ft.icons.CONTENT_COPY, tooltip="copy", on_click=self.copy_setting)
        self.expand_btn = ft.IconButton(icon=ft.icons.ADD, tooltip="add extra kwarg", on_click=self.expand_setting)

        self.paras = ft.Column()
        self.controls = [
            self.title,
            self.paras,
            ft.Row(
                alignment=ft.MainAxisAlignment.CENTER,
                controls=[self.export_btn, self.expand_btn],
            ),
        ]

    def set_paras(self, data):
        self.type_name = data["type"]
        self.title.value = f"{self.kind.capitalize()}: {self.type_name}"
        self.paras.controls = []
        for name, v in data["space"].items():
            if isinstance(v["val"], (int, str, float)):
                para = Parameter(name, v["val"], [v["val"]], False, True)
            elif isinstance(v["val"], list):
                para = Parameter(name, v.get("default", None), v["val"], False, True)
            else:
                para = Parameter(name, v.get("default", None), v["val"], v.get("log", False), False)
            self.paras.controls.append(para)
        if self.page:
            self.update()

    def expand_setting(self, e):
        if self.type_name is None or self.resolver is None:
            return
        cls = self.resolver(self.type_name)
        if cls is None:
            return
        existing = {para.name for para in self.paras.controls}
        for p in inspect.signature(cls).parameters.values():
            if p.name in _SKIP_PARAM_NAMES or p.name in existing:
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            self.paras.controls.append(Parameter(p.name, p.default, [p.default], False, True))
        self.update()

    def copy_setting(self, e):
        data = self.export_setting()
        self.page.set_clipboard(str(data))
        self.page.open(ft.SnackBar(ft.Text("Setting copied"), open=True))

    def export_setting(self):
        return {
            "type": self.type_name,
            "space": {para.name: para.export_setting() for para in self.paras.controls},
        }


class AutoMAXApp(ft.Column):
    def __init__(self):
        super().__init__()
        self.session_manager = get_session_manager()
        self.current_session = None
        self.is_training = False
        self.training_process = None
        self.multilabel = False

        # ---- Backend-specific model/dataset controls ----------------------
        self.image_model = ft.Dropdown(
            label="Model Architecture", width=260,
            on_change=self.image_model_changed,
            options=[ft.dropdown.Option(m) for m in _IMAGE_MODELS] + [ft.dropdown.Option(_HF_MODEL_OPTION)],
        )
        self.image_model_hf_id = ft.TextField(
            label="HuggingFace Model ID", hint_text="e.g. google/vit-base-patch16-224",
            width=300, visible=False,
        )
        self.pretrained_ck = ft.Checkbox(value=False, label="pretrained", on_change=self.pretrained_changed)
        self.model_path = ft.TextField(label="Model Path", visible=False, width=260)
        # Single shared FilePicker for every browse button in this form --
        # `_file_picker_target` records which field the next result should
        # populate (set right before invoking pick_files()/get_directory_path()).
        self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
        self._file_picker_target: str = ""
        self.browse_btn = ft.ElevatedButton(
            "Browse", visible=False,
            on_click=lambda _: self._browse_file("model_path"),
            bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE,
        )
        self.image_dataset = ft.Dropdown(
            label="Dataset", width=260, on_change=self.image_dataset_changed,
            options=[ft.dropdown.Option(d) for d in _IMAGE_DATASETS] + [ft.dropdown.Option(_CUSTOM_DATASET_OPTION)],
        )
        self.image_dataset_custom_name = ft.TextField(
            label="Custom Dataset Name", hint_text="a unique name to identify this dataset",
            width=260, visible=False, on_change=self.custom_dataset_name_changed,
        )

        self.gnn_model = ft.Dropdown(
            label="GNN Architecture", width=260,
            on_change=self.gnn_model_changed,
            options=[ft.dropdown.Option(m) for m in _GNN_MODELS] + [ft.dropdown.Option(_HF_MODEL_OPTION)],
        )
        self.gnn_model_hf_id = ft.TextField(
            label="HuggingFace Model ID", hint_text="e.g. user/my-gcn",
            width=300, visible=False,
        )
        self.gnn_emb_dim = ft.TextField(value="256", label="Embedding Dim", width=140)
        self.gnn_num_layers = ft.TextField(value="5", label="Num Layers", width=120)
        self.gnn_decay_factor = ft.TextField(value="10.0", label="Decay Factor", width=140)
        self.gnn_dataset = ft.Dropdown(
            label="Dataset", width=260, on_change=self.gnn_dataset_changed, visible=False,
            options=[ft.dropdown.Option(d) for d in _GNN_DATASETS] + [ft.dropdown.Option(_CUSTOM_DATASET_OPTION)],
        )
        # Custom GNN datasets are not implemented yet -- no fields to fill in,
        # just a message shown when the option is selected.
        self.gnn_custom_dataset_message = ft.Text(
            "Custom graph datasets are not yet supported.",
            color=ft.colors.ORANGE_700, visible=False,
        )

        self.inspect_btn = ft.IconButton(icon=ft.icons.ANALYTICS, tooltip="Inspect Dataset", on_click=self.datasets_inspected)
        self.dataset_stat = ft.Text("")
        self.progress_ring = ft.ProgressRing(visible=False)

        # Root Path starts blank on purpose: a predefined dataset can't be
        # loaded without one (see _dataset_kwargs()/the validation in
        # datasets_inspected()/train()), rather than silently defaulting to
        # some "./data" that likely doesn't contain the user's data.
        self.dataset_root_path = ft.TextField(value="", label="Root Path", width=260)
        self.dataset_root_path_browse_btn = ft.IconButton(
            icon=ft.icons.FOLDER_OPEN, tooltip="Browse for Root Path",
            on_click=lambda _: self._browse_directory("dataset_root_path"),
        )

        # ---- Custom image dataset (CSV-backed, via MedicalImageCSVDataset) --
        # Train/val/test CSVs are auto-detected from Root Path (looking for
        # train.csv / val.csv|valid.csv / test.csv directly under it) rather
        # than typed in by hand -- see _detect_custom_csvs().
        self._custom_train_csv = ""
        self._custom_val_csv = ""
        self._custom_test_csv = ""
        self.custom_root_path = ft.TextField(
            label="Dataset Root Path", width=280, on_change=self._custom_root_path_text_changed,
        )
        self.custom_root_path_browse_btn = ft.IconButton(
            icon=ft.icons.FOLDER_OPEN, tooltip="Browse for Root Path",
            on_click=lambda _: self._browse_directory("custom_root_path"),
        )
        # Image paths in the CSV are resolved relative to this same Root
        # Path (image_root == the folder the CSVs were auto-detected from) --
        # no separate field to set.
        self.custom_image_col = ft.TextField(value="image_path", label="Image Column", width=160)
        self.custom_label_col = ft.TextField(value="label", label="Label Column", width=160)
        self.custom_image_size = ft.TextField(value="224", label="Image Size", width=120)
        self.save_custom_dataset_btn = ft.ElevatedButton(
            "Save Custom Dataset", on_click=self.save_custom_dataset,
            bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE,
        )
        self.custom_dataset_stat = ft.Text("", color=ft.colors.GREY_700)
        self.custom_image_dataset_column = ft.Column(
            controls=[
                ft.Row(controls=[self.custom_root_path, self.custom_root_path_browse_btn]),
                ft.Row(controls=[self.custom_image_col, self.custom_label_col, self.custom_image_size]),
                ft.Row(controls=[self.save_custom_dataset_btn, self.custom_dataset_stat]),
            ],
            visible=False,
        )

        # Both backends' dataset dropdown + custom-name field are always kept
        # mounted here (visibility toggled by backend_changed) rather than
        # swapped in/out of this Row's controls list -- Flet 0.24.1 raises
        # `assert self.__uid is not None` when a not-yet-mounted control is
        # swapped into an already-mounted Row's controls and .update() is
        # called (reproduced independently of this app), so control identity
        # in a live control list must stay stable after the first mount.
        self.dataset_row = ft.Row(
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[
                self.image_dataset, self.image_dataset_custom_name,
                self.gnn_dataset, self.gnn_custom_dataset_message,
                self.progress_ring, self.inspect_btn, self.dataset_stat,
            ],
        )
        # Shown only for a predefined dataset selection (either backend) --
        # hidden for "Custom Dataset..." (image custom uses its own fields
        # below; GNN custom isn't implemented).
        self.dataset_kwargs_row = ft.Row(
            alignment=ft.MainAxisAlignment.START,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            controls=[self.dataset_root_path, self.dataset_root_path_browse_btn],
        )

        self.backend_tabs = ft.Tabs(
            selected_index=0,
            on_change=self.backend_changed,
            height=110,  # Tabs/TabBarView collapses to zero height without an explicit bound
            tabs=[
                ft.Tab(
                    text="Image Classification",
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row(controls=[
                                self.image_model, self.image_model_hf_id,
                                self.pretrained_ck, self.model_path, self.browse_btn,
                            ]),
                        ]),
                        padding=ft.padding.only(top=10),
                    ),
                ),
                ft.Tab(
                    text="Graph Classification",
                    content=ft.Container(
                        content=ft.Column([
                            ft.Row(controls=[
                                self.gnn_model, self.gnn_model_hf_id, self.gnn_emb_dim,
                                self.gnn_num_layers, self.gnn_decay_factor,
                            ]),
                        ]),
                        padding=ft.padding.only(top=10),
                    ),
                ),
            ],
        )

        # ---- Target / metric configuration ---------------------------------
        self.metric_dropdown = ft.Dropdown(
            label="Target Metric", width=200, value="AUROC",
            options=[ft.dropdown.Option(m) for m in ("AUROC", "AUPRC", "ACC")],
            on_change=self.metric_changed,
        )
        self.partial_auroc_ck = ft.Checkbox(
            label="Partial AUROC (pAUC)", value=False, on_change=self.partial_auroc_changed,
        )
        self.max_fpr = ft.TextField(value="0.3", label="Max FPR", width=120, visible=False)
        self.min_tpr = ft.TextField(value="0.7", label="Min TPR", width=120, visible=False)

        # ---- Loss / optimizer (canonical single driver) --------------------
        self.loss_dropdown = ft.Dropdown(
            label="Loss Function", width=260,
            options=[ft.dropdown.Option(name) for name in _LOSS_NAMES],
            on_change=self.loss_changed,
        )
        self.optimizers = Space("optimizer", Trainer._get_optimizer)
        self.losses = Space("loss", Trainer._get_loss)

        # ---- Training hyperparameters ---------------------------------------
        self.seed = ft.TextField(value="42", label="SEED")
        self.batch_size = ft.TextField(value="128", label="Batch Size")
        self.eval_batch_size = ft.TextField(value="128", label="Eval Batch Size")
        self.sampling_rate = ft.TextField(value="0.2", label="Sampling Rate")
        self.epochs = ft.TextField(value="50", label="Epochs")
        self.decay_epochs = ft.TextField(value="[0.5, 0.75]", label="Decay Epochs")
        self.num_workers = ft.TextField(value="0", label="Num Workers")
        self.output_path = ft.TextField(value="./output", label="Output Path (base)")
        self.n_trials = ft.TextField(value="5", label="N Trials")

        self.advanced = ft.ExpansionTile(
            title=ft.Text("Advanced Settings (Optimizer / Loss Search Space)", size=16, weight=ft.FontWeight.BOLD),
            affinity=ft.TileAffinity.PLATFORM,
            maintain_state=True,
            controls=[
                ft.Container(alignment=ft.alignment.center, height=20),
                ft.Container(
                    content=ft.Row(
                        alignment=ft.MainAxisAlignment.START,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                        spacing=30,
                        controls=[self.optimizers, self.losses],
                    ),
                    padding=ft.padding.only(left=20),
                ),
                ft.Container(alignment=ft.alignment.center, height=20),
            ],
        )

        self.others = ft.ExpansionTile(
            title=ft.Text("Training Parameters", size=16, weight=ft.FontWeight.BOLD),
            affinity=ft.TileAffinity.PLATFORM,
            maintain_state=True,
            controls=[
                ft.Container(alignment=ft.alignment.center, height=20),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.seed]),
                        ft.Column(col={"sm": 3}, controls=[self.batch_size]),
                        ft.Column(col={"sm": 3}, controls=[self.eval_batch_size]),
                        ft.Column(col={"sm": 3}, controls=[self.sampling_rate]),
                    ]),
                    padding=ft.padding.only(left=20),
                ),
                ft.Container(alignment=ft.alignment.center, height=20),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.epochs]),
                        ft.Column(col={"sm": 3}, controls=[self.decay_epochs]),
                        ft.Column(col={"sm": 3}, controls=[self.num_workers]),
                        ft.Column(col={"sm": 3}, controls=[self.output_path]),
                    ]),
                    padding=ft.padding.only(left=20),
                ),
                ft.Container(alignment=ft.alignment.center, height=20),
                ft.Container(
                    content=ft.ResponsiveRow([
                        ft.Column(col={"sm": 3}, controls=[self.n_trials]),
                    ]),
                    padding=ft.padding.only(left=20),
                ),
                ft.Container(alignment=ft.alignment.center, height=20),
            ],
        )

        self.train_btn = ft.ElevatedButton("Start Training", on_click=self.train, bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE)
        self.cancel_btn = ft.ElevatedButton("Cancel Training", on_click=self.cancel_training, visible=False, bgcolor=ft.colors.RED_600, color=ft.colors.WHITE)
        self.monitor_btn = ft.ElevatedButton("Monitor Sessions", on_click=self.go_to_monitor, bgcolor=ft.colors.GREEN_600, color=ft.colors.WHITE, icon=ft.icons.MONITOR_HEART)
        self.train_pb = ft.ProgressBar(visible=False, width=400)

        self.console_text = ft.Text("", selectable=True, size=12, font_family="Consolas, 'Courier New', monospace")
        self.console_scroll = ft.ListView([self.console_text], height=300, width=1200, spacing=2, padding=ft.padding.all(10), auto_scroll=True)
        self.clear_console_btn = ft.IconButton(icon=ft.icons.CLEAR, tooltip="Clear Console", on_click=self.clear_console, bgcolor=ft.colors.RED_100, icon_color=ft.colors.RED_700)
        self.console_header = ft.Row(
            [ft.Text("Training Console", theme_style=ft.TextThemeStyle.TITLE_MEDIUM, weight=ft.FontWeight.BOLD), self.clear_console_btn],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        )
        self.console = ft.Container(
            content=ft.Column([
                self.console_header,
                ft.Container(content=self.console_scroll, width=1200, height=300, bgcolor=ft.colors.GREY_50, border=ft.border.all(2, ft.colors.GREY_300), border_radius=8, padding=ft.padding.all(0)),
            ]),
            width=1200, height=350,
        )

        self.header = ft.Container(
            content=ft.Text("AutoMAX", size=24, weight=ft.FontWeight.BOLD, color=ft.colors.GREY_800),
            alignment=ft.alignment.center, padding=ft.padding.all(15), margin=ft.margin.only(bottom=15),
        )

        self.controls = [
            self.header,
            ft.Container(
                content=ft.Column([
                    ft.Text("Model / Backend Configuration", size=16, weight=ft.FontWeight.BOLD),
                    self.backend_tabs,
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15), margin=ft.margin.only(bottom=15),
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Dataset Configuration", size=16, weight=ft.FontWeight.BOLD),
                    self.dataset_row,
                    self.dataset_kwargs_row,
                    self.custom_image_dataset_column,
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15), margin=ft.margin.only(bottom=15),
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Target Configuration", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        alignment=ft.MainAxisAlignment.START, vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        controls=[self.metric_dropdown, self.partial_auroc_ck, self.max_fpr, self.min_tpr],
                    ),
                    ft.Row(alignment=ft.MainAxisAlignment.START, controls=[self.loss_dropdown]),
                ], spacing=10),
                border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15), margin=ft.margin.only(bottom=15),
            ),
            self.advanced,
            self.others,
            ft.Container(
                content=ft.Column([
                    ft.Text("Training Control", size=16, weight=ft.FontWeight.BOLD),
                    ft.Row(alignment=ft.MainAxisAlignment.CENTER, controls=[self.train_btn, self.cancel_btn, self.monitor_btn]),
                    self.train_pb,
                ], spacing=15),
                border=ft.border.all(1, ft.colors.GREY_300), border_radius=ft.border_radius.all(4),
                padding=ft.padding.all(15), margin=ft.margin.only(bottom=15),
            ),
            self.console,
            self.pick_files_dialog,
        ]

    # ------------------------------------------------------------------
    # Backend / dataset / model helpers
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return "gnn" if self.backend_tabs.selected_index == 1 else "image"

    @property
    def model_name(self):
        if self.backend == "gnn":
            if self.gnn_model.value == _HF_MODEL_OPTION:
                return self.gnn_model_hf_id.value
            return self.gnn_model.value
        if self.image_model.value == _HF_MODEL_OPTION:
            return self.image_model_hf_id.value
        return self.image_model.value

    @property
    def dataset_name(self):
        if self.backend == "gnn":
            # Custom GNN datasets aren't implemented -- no name to report.
            return None if self.gnn_dataset.value == _CUSTOM_DATASET_OPTION else self.gnn_dataset.value
        if self.image_dataset.value == _CUSTOM_DATASET_OPTION:
            return self.image_dataset_custom_name.value
        return self.image_dataset.value

    @property
    def _is_custom_image_dataset(self) -> bool:
        return self.backend == "image" and self.image_dataset.value == _CUSTOM_DATASET_OPTION

    @property
    def _is_custom_gnn_dataset(self) -> bool:
        return self.backend == "gnn" and self.gnn_dataset.value == _CUSTOM_DATASET_OPTION

    def _refresh_dataset_section_visibility(self):
        is_custom_image = self._is_custom_image_dataset
        is_custom_gnn = self._is_custom_gnn_dataset
        self.image_dataset_custom_name.visible = is_custom_image
        self.gnn_custom_dataset_message.visible = is_custom_gnn
        self.custom_image_dataset_column.visible = is_custom_image
        self.dataset_kwargs_row.visible = not (is_custom_image or is_custom_gnn)

    def _load_root_path_for_current_dataset(self):
        """Recall the last-used Root Path for the newly-selected predefined dataset."""
        name = self.dataset_name
        self.dataset_root_path.value = get_dataset_config_store().get_root_path(name) if name else ""

    def backend_changed(self, e):
        is_gnn = self.backend == "gnn"
        self.image_dataset.visible = not is_gnn
        self.gnn_dataset.visible = is_gnn
        self.dataset_stat.value = ""
        if not (self._is_custom_image_dataset or self._is_custom_gnn_dataset):
            self._load_root_path_for_current_dataset()
        self._refresh_dataset_section_visibility()
        self.update()

    def image_dataset_changed(self, e):
        self.dataset_stat.value = ""
        if self.image_dataset.value == _CUSTOM_DATASET_OPTION:
            self.custom_dataset_name_changed(e)
        else:
            self._load_root_path_for_current_dataset()
        self._refresh_dataset_section_visibility()
        self.update()

    def gnn_dataset_changed(self, e):
        self.dataset_stat.value = ""
        if self.gnn_dataset.value != _CUSTOM_DATASET_OPTION:
            self._load_root_path_for_current_dataset()
        self._refresh_dataset_section_visibility()
        self.update()

    def custom_dataset_name_changed(self, e):
        """Auto-fill the custom-dataset form from a previously saved entry
        when the typed name matches one already in the registry."""
        name = self.image_dataset_custom_name.value.strip()
        existing = get_dataset_config_store().get_custom_image_dataset(name) if name else None
        if existing:
            self.custom_root_path.value = existing.get("root_path", "")
            self.custom_image_col.value = existing.get("image_col", "image_path")
            self.custom_label_col.value = existing.get("label_col", "label")
            self.custom_image_size.value = str(existing.get("image_size", 224))
            self.custom_dataset_stat.value = f"Loaded saved config for '{name}'."
            self.custom_dataset_stat.color = ft.colors.GREY_700
            self._detect_custom_csvs()  # re-scan Root Path fresh; also refreshes the status text
        else:
            self.custom_dataset_stat.value = ""
        self.update()

    @staticmethod
    def _find_csv(root_path: str, candidates) -> str:
        for name in candidates:
            path = os.path.join(root_path, name)
            if os.path.isfile(path):
                return path
        return ""

    def _custom_root_path_text_changed(self, e):
        self._detect_custom_csvs()
        self.update()

    def _detect_custom_csvs(self):
        """Look for train.csv / val.csv|valid.csv / test.csv directly under
        the custom dataset's Root Path and populate _custom_{train,val,test}_csv.
        Images referenced by the CSVs are also resolved relative to this same
        Root Path (it's passed as MedicalImageCSVDataset's image_root)."""
        root_path = self.custom_root_path.value.strip()
        if not root_path or not os.path.isdir(root_path):
            self._custom_train_csv = ""
            self._custom_val_csv = ""
            self._custom_test_csv = ""
            if root_path:
                self.custom_dataset_stat.value = f"'{root_path}' is not a valid directory."
                self.custom_dataset_stat.color = ft.colors.RED_600
            else:
                self.custom_dataset_stat.value = ""
            return

        self._custom_train_csv = self._find_csv(root_path, _TRAIN_CSV_CANDIDATES)
        self._custom_val_csv = self._find_csv(root_path, _VAL_CSV_CANDIDATES)
        self._custom_test_csv = self._find_csv(root_path, _TEST_CSV_CANDIDATES)

        found = [label for label, path in (("train", self._custom_train_csv), ("val", self._custom_val_csv), ("test", self._custom_test_csv)) if path]
        missing = []
        if not self._custom_train_csv:
            missing.append("train.csv")
        if not self._custom_val_csv:
            missing.append("val.csv/valid.csv")

        if missing:
            self.custom_dataset_stat.value = f"Detected: {', '.join(found) or 'none'}. Missing: {', '.join(missing)}."
            self.custom_dataset_stat.color = ft.colors.ORANGE_700
        else:
            self.custom_dataset_stat.value = f"Detected CSVs: {', '.join(found)}."
            self.custom_dataset_stat.color = ft.colors.GREEN_700

    def save_custom_dataset(self, e):
        name = self.image_dataset_custom_name.value.strip()
        if not self._custom_train_csv or not self._custom_val_csv:
            self.custom_dataset_stat.value = "Could not detect train.csv/val.csv under this Root Path."
            self.custom_dataset_stat.color = ft.colors.RED_600
            self.custom_dataset_stat.update()
            return

        definition = {
            "root_path": self.custom_root_path.value.strip(),
            "image_col": self.custom_image_col.value.strip() or "image_path",
            "label_col": self.custom_label_col.value.strip() or "label",
            "image_size": int(self.custom_image_size.value or 224),
        }
        try:
            get_dataset_config_store().add_custom_image_dataset(name, definition, reserved_names=_IMAGE_DATASETS)
        except ValueError as exc:
            self.custom_dataset_stat.value = str(exc)
            self.custom_dataset_stat.color = ft.colors.RED_600
        else:
            self.custom_dataset_stat.value = f"Saved custom dataset '{name}'."
            self.custom_dataset_stat.color = ft.colors.GREEN_700
        self.custom_dataset_stat.update()

    def _dataset_kwargs(self) -> dict:
        """kwargs for a predefined dataset (just root_path)."""
        root_path = self.dataset_root_path.value.strip()
        return {"root_path": root_path} if root_path else {}

    def _custom_image_dataset_kwargs(self) -> dict:
        """kwargs for the custom CSV-backed image dataset (see libauc.auto.custom_datasets)."""
        kwargs = {
            "train_csv": self._custom_train_csv,
            "image_root": self.custom_root_path.value.strip(),
            "image_col": self.custom_image_col.value.strip() or "image_path",
            "label_col": self.custom_label_col.value.strip() or "label",
            "image_size": int(self.custom_image_size.value or 224),
        }
        if self._custom_val_csv:
            kwargs["val_csv"] = self._custom_val_csv
        if self._custom_test_csv:
            kwargs["test_csv"] = self._custom_test_csv
        return kwargs

    def image_model_changed(self, e):
        self.image_model_hf_id.visible = self.image_model.value == _HF_MODEL_OPTION
        self.update()

    def gnn_model_changed(self, e):
        self.gnn_model_hf_id.visible = self.gnn_model.value == _HF_MODEL_OPTION
        self.update()

    def _browse_file(self, target: str):
        self._file_picker_target = target
        self.pick_files_dialog.pick_files(allow_multiple=False)

    def _browse_directory(self, target: str):
        self._file_picker_target = target
        self.pick_files_dialog.get_directory_path()

    def pick_files_result(self, e: ft.FilePickerResultEvent):
        target, self._file_picker_target = self._file_picker_target, ""
        if target == "model_path":
            self.model_path.value = ", ".join(f.path for f in e.files) if e.files else ""
        elif target == "dataset_root_path" and e.path:
            self.dataset_root_path.value = e.path
        elif target == "custom_root_path" and e.path:
            self.custom_root_path.value = e.path
            self._detect_custom_csvs()
        self.update()

    def pretrained_changed(self, e):
        self.model_path.visible = self.pretrained_ck.value
        self.browse_btn.visible = self.pretrained_ck.value
        self.update()

    def metric_changed(self, e):
        is_auroc = self.metric_dropdown.value == "AUROC"
        self.partial_auroc_ck.visible = is_auroc
        self.max_fpr.visible = is_auroc and self.partial_auroc_ck.value
        self.min_tpr.visible = is_auroc and self.partial_auroc_ck.value
        self.update()

    def partial_auroc_changed(self, e):
        self.max_fpr.visible = self.partial_auroc_ck.value
        self.min_tpr.visible = self.partial_auroc_ck.value
        self.update()

    def loss_changed(self, e):
        if not self.loss_dropdown.value:
            return
        conf = parse_defaultconfig(self.loss_dropdown.value, self.multilabel, {})
        self.optimizers.set_paras(conf["optimizer"])
        self.losses.set_paras(conf["loss"])

    def _load_preview_dataset(self):
        """(train_dataset, eval_datasets) for the current selection -- used
        by both 'Inspect Dataset' and (indirectly, via the same code path in
        build_pipeline()) an actual training run."""
        if self._is_custom_image_dataset:
            return load_custom_csv_dataset(self._custom_image_dataset_kwargs(), ["val"])
        root_path = self.dataset_root_path.value.strip()
        if not root_path:
            raise ValueError("Root Path is required to load this dataset.")
        return load_dataset(self.dataset_name, splits=["val"], **self._dataset_kwargs())

    def datasets_inspected(self, e):
        if self._is_custom_gnn_dataset:
            self.dataset_stat.value = "Custom graph datasets are not yet supported."
            self.dataset_stat.update()
            return

        name = self.dataset_name
        if not name:
            self.dataset_stat.value = "Please choose a dataset"
            self.dataset_stat.update()
            return

        self.inspect_btn.visible = False
        self.progress_ring.visible = True
        self.update()
        try:
            train_dataset, eval_datasets = self._load_preview_dataset()
            n_train = len(train_dataset)
            n_eval = len(eval_datasets[0]) if eval_datasets else 0
            if self.backend == "image":
                labels = np.array(train_dataset.targets).squeeze()
                num_tasks = len(np.unique(labels)) if labels.ndim == 1 else labels.shape[-1]
                self.multilabel = num_tasks >= 3
                self.dataset_stat.value = (
                    f"Train size: {n_train}  Eval size: {n_eval}  "
                    f"Tasks: {num_tasks} ({'multi-label' if self.multilabel else 'binary'})"
                )
            else:
                self.multilabel = False
                self.dataset_stat.value = f"Train size: {n_train}  Eval size: {n_eval}"
            if not self._is_custom_image_dataset:
                get_dataset_config_store().set_root_path(name, self.dataset_root_path.value.strip())
        except Exception as exc:
            self.dataset_stat.value = f"Error: {exc}"
        finally:
            self.inspect_btn.visible = True
            self.progress_ring.visible = False
        self.update()

    # ------------------------------------------------------------------
    # Console / progress
    # ------------------------------------------------------------------

    def set_progressbar(self, value):
        self.train_pb.value = value
        try:
            self.train_pb.update()
        except Exception:
            pass  # not yet connected to a page

    def console_log(self, s, level="INFO"):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        level_prefix = {"WARNING": "⚠ ", "ERROR": "❌ ", "SUCCESS": "✅ ", "TRAINING": "\U0001f680 "}.get(level, "")
        log_message = f"[{timestamp}] {level_prefix}{s}"
        self.console_text.value = f"{self.console_text.value}\n{log_message}" if self.console_text.value else log_message
        try:
            self.console_text.update()
            self.console_scroll.scroll_to(offset=-1, duration=100)
        except Exception:
            pass

    def clear_console(self, e):
        self.console_text.value = ""
        self.console_text.update()
        self.console_log("Console cleared")

    def initialize(self):
        self.console_log("AutoMAX Training Console Ready", "SUCCESS")
        self.console_log("Configure your training parameters and click 'Start Training' to begin", "INFO")

    # ------------------------------------------------------------------
    # Config building + training control
    # ------------------------------------------------------------------

    def _build_config(self) -> dict:
        metrics = [self.metric_dropdown.value]
        metric_kwargs_entry = {}
        if self.metric_dropdown.value == "AUROC" and self.partial_auroc_ck.value:
            metric_kwargs_entry["max_fpr"] = float(self.max_fpr.value)
            metric_kwargs_entry["min_tpr"] = float(self.min_tpr.value)

        training = {
            "optimizer": self.optimizers.export_setting()["type"],
            "optimizer_kwargs": self.optimizers.export_setting()["space"],
            "loss": self.losses.export_setting()["type"],
            "loss_kwargs": self.losses.export_setting()["space"],
            "SEED": int(self.seed.value),
            "batch_size": int(self.batch_size.value),
            "eval_batch_size": int(self.eval_batch_size.value),
            "sampling_rate": float(self.sampling_rate.value),
            "epochs": int(self.epochs.value),
            "decay_epochs": eval(self.decay_epochs.value),
            "num_workers": int(self.num_workers.value),
            "output_path": self.output_path.value,
            "resume_from_checkpoint": False,
            "save_checkpoint_every": 5,
            "project_name": "automax_gui",
            "experiment_name": "placeholder",  # overwritten by SessionManager.create_session
            "verbose": 1,
        }

        if self.backend == "gnn":
            training["decay_factor"] = float(self.gnn_decay_factor.value)
            model = {
                "name": self.model_name,
                "num_tasks": 1,
                "emb_dim": int(self.gnn_emb_dim.value),
                "num_layers": int(self.gnn_num_layers.value),
            }
        else:
            model = {
                "name": self.model_name,
                "pretrained": self.pretrained_ck.value,
            }
            if self.pretrained_ck.value and self.model_path.value:
                model["pretrained_path"] = self.model_path.value

        automax = {
            "deterministic": True,
            "n_trials": int(self.n_trials.value),
            "n_configs": 1,
            "SEED": int(self.seed.value),
            "name": "placeholder",  # overwritten by SessionManager.create_session
            "overwrite": True,
        }

        if self._is_custom_image_dataset:
            dataset_cfg = {
                "name": self.dataset_name,
                "kind": "custom_csv",
                "kwargs": self._custom_image_dataset_kwargs(),
                "eval_splits": ["val"],
            }
        else:
            dataset_cfg = {
                "name": self.dataset_name,
                "kwargs": self._dataset_kwargs(),
                "eval_splits": ["val"],
            }

        return {
            "dataset": dataset_cfg,
            "model": model,
            "metrics": metrics,
            "metric_kwargs": [metric_kwargs_entry],
            "training": training,
            "automax": automax,
        }

    def train(self, e):
        if not self.loss_dropdown.value:
            self.console_log("Please select a loss function first", "WARNING")
            return
        if self._is_custom_gnn_dataset:
            self.console_log("Custom graph datasets are not yet supported", "WARNING")
            return
        if not self.model_name or not self.dataset_name:
            self.console_log("Please choose a model and a dataset first", "WARNING")
            return
        if self._is_custom_image_dataset:
            custom_kwargs = self._custom_image_dataset_kwargs()
            if not custom_kwargs.get("train_csv") or not custom_kwargs.get("val_csv") or not custom_kwargs.get("image_root"):
                self.console_log("Custom dataset requires Train CSV, Val CSV, and Image Root", "WARNING")
                return
        elif not self.dataset_root_path.value.strip():
            self.console_log(f"Root Path is required to load '{self.dataset_name}'", "WARNING")
            return
        else:
            get_dataset_config_store().set_root_path(self.dataset_name, self.dataset_root_path.value.strip())

        self.train_btn.disabled = True
        self.cancel_btn.visible = True
        self.train_pb.visible = True
        self.train_pb.value = 0
        self.update()

        config = self._build_config()
        session_name = f"{self.model_name}_{self.loss_dropdown.value}_{self.dataset_name}"
        session = self.session_manager.create_session(session_name, config, backend=self.backend)
        self.current_session = session.session_id
        self.console_log(f"Created training session: {self.current_session}", "SUCCESS")

        self.session_manager.register_gui_callback(self, self.current_session)
        self._start_training_process(config)
        self.is_training = True

    def _start_training_process(self, config: dict):
        tmp_dir = Path(tempfile.gettempdir())
        config_file = tmp_dir / f"automax_gui_session_{self.current_session}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)

        log_path = self.session_manager.get_session(self.current_session).log_file

        try:
            with open(log_path, "a", encoding="utf-8") as log_fh:
                process = subprocess.Popen(
                    [
                        sys.executable, "-m", "libauc.auto.ui.training_worker",
                        "--config", str(config_file),
                        "--session-id", self.current_session,
                    ],
                    # Same cwd as the GUI: gui_sessions/ paths are relative.
                    cwd=os.getcwd(),
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
            self.console_log(f"Started training process (PID: {process.pid})", "SUCCESS")
            self.console_log("Training is running independently. You can close the GUI if needed.", "INFO")
            self.training_process = process
        except Exception as exc:
            self.console_log(f"Failed to start training process: {exc}", "ERROR")
            self._training_finished()

    def cancel_training(self, e):
        if not self.current_session:
            return
        session = self.session_manager.get_session(self.current_session)
        # The worker overwrites `pid` with its own once it's alive; until
        # then it's still stamped with the GUI's own PID (see
        # training_worker.run()) -- guard against terminating ourselves.
        if session and session.pid == os.getpid():
            self.console_log("Training is initializing, cannot be cancelled yet", "WARNING")
            return

        if self.is_training:
            self.console_log("Cancelling training...", "WARNING")
            self.session_manager.cancel_session(self.current_session)
            self._training_finished()

    def _training_finished(self):
        self.train_btn.disabled = False
        self.cancel_btn.visible = False
        self.train_pb.visible = False
        self.train_pb.value = 0
        self.is_training = False
        if self.current_session:
            self.session_manager.unregister_gui_callback()
        self.current_session = None
        self.training_process = None
        self.update()

    def go_to_monitor(self, e):
        self.page.go("/monitor")
