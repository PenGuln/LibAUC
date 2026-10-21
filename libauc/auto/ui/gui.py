"""
AutoMAX "New search" page: configure a dataset/model/loss/optimizer/search
space and launch a training session, adapted to AutoMAX's search stack
(libauc's ``parse_defaultconfig``/spaces, and the ``"range"``-key
hyperparameter contract in ``libauc.auto.helpers``).
"""
import ast
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from nicegui import run, ui

from libauc.trainer import Trainer, load_dataset, parse_defaultconfig

from ..custom_datasets import load_custom_csv_dataset
from .dataset_config import get_dataset_config_store
from .session_manager import get_session_manager
from .theme import FIELD, PathPicker, section_card

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

# Sentinel model option that switches the model field over to a free-text
# HuggingFace repo ID. Both image_trainer.py and graph_trainer.py already
# recognize any model name containing "/" as an HF Hub repo ID (loaded via
# AutoModelForImageClassification / AutoModel respectively) -- this option
# just exposes that existing capability in the GUI. Never collides with a
# real repo ID, since those always contain "/".
_HF_MODEL_OPTION = "HuggingFace Model..."

# Sentinel dataset option for defining a new user-defined dataset.
#
# Image backend: reveals a small form (root path, column names) and loads via
# libauc.auto.custom_datasets.load_custom_csv_dataset() (libauc's
# MedicalImageCSVDataset) instead of the fixed name registry in
# libauc.trainer.load_dataset(). The chosen name is just an identifier (also
# used as the persisted registry key, see ui.dataset_config), not passed to
# load_dataset(). Previously saved custom datasets are listed as their own
# options, prefixed with _SAVED_CUSTOM_PREFIX.
#
# GNN backend: not implemented yet -- selecting it just shows a message and
# blocks Inspect/Launch.
_CUSTOM_DATASET_OPTION = "Custom Dataset..."
_SAVED_CUSTOM_PREFIX = "custom:"

# The 10 canonical loss names libauc.trainer.parse_defaultconfig understands,
# each of which deterministically fixes its paired optimizer, so a single
# Loss selector is the one driver for both search spaces.
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

# Some losses (BCELoss) have no paired optimizer in libauc.trainer's spaces,
# leaving the choice to the recipe; the GUI tunes this one for them, as the
# BCELoss recipes under examples/auto do.
_FALLBACK_OPTIMIZER = "Adam"

# Constructor parameter names to never expose as an extra tunable field
# (injected by the Trainer itself, not user-facing hyperparameters).
_SKIP_PARAM_NAMES = {"self", "params", "loss_fn", "model", "args", "kwargs"}

# Filenames looked for directly under a custom dataset's Root Path to
# auto-detect its train/val/test CSVs (first match wins).
_TRAIN_CSV_CANDIDATES = ["train.csv"]
_VAL_CSV_CANDIDATES = ["val.csv", "valid.csv", "validation.csv"]
_TEST_CSV_CANDIDATES = ["test.csv"]

# Worker subprocesses launched by this GUI process. Polled by the sessions
# page so finished workers are reaped instead of lingering as zombies (which
# psutil would still report as "running").
_worker_processes: List[subprocess.Popen] = []


def reap_finished_workers() -> None:
    _worker_processes[:] = [p for p in _worker_processes if p.poll() is None]


def _literal(text: str, what: str):
    try:
        return ast.literal_eval(text.strip())
    except (ValueError, SyntaxError):
        raise ValueError(f"{what}: '{text}' is not a valid Python literal")


def _fmt(value) -> str:
    return "" if value is None or value is inspect.Parameter.empty else repr(value)


class HyperParameter:
    """One search-space row: a continuous range (min/max + optional log
    scale) or a categorical list of choices (a single value pins it).
    Leaving the value blank drops the row from the exported space."""

    def __init__(self, name: str, spec: dict, on_remove: Callable[["HyperParameter"], None]):
        self.name = name
        val = spec.get("val")
        is_range = isinstance(val, tuple)
        if is_range:
            choices = ""
        elif isinstance(val, list):
            choices = repr(val)
        else:
            choices = "" if val is None or val is inspect.Parameter.empty else repr([val])

        with ui.row().classes("am-hp items-center gap-2 w-full no-wrap") as self.row:
            ui.label(name).classes("am-mono text-sm w-28 shrink-0 truncate").tooltip(name)
            self.mode = ui.toggle(
                {"range": "Range", "choice": "Choices"}, value="range" if is_range else "choice",
                on_change=self._sync,
            ).props("dense no-caps unelevated toggle-color=primary size=sm")
            with ui.row().classes("gap-2 grow no-wrap") as self.range_box:
                self.low = ui.input("min", value=_fmt(val[0]) if is_range else "").props(FIELD).classes("grow min-w-0")
                self.high = ui.input("max", value=_fmt(val[1]) if is_range else "").props(FIELD).classes("grow min-w-0")
            self.choices = ui.input("choices", value=choices, placeholder="[a, b, c]").props(FIELD).classes("grow min-w-0")
            self.default = ui.input("default", value=_fmt(spec.get("default"))).props(FIELD).classes("w-24 shrink-0")
            self.log = ui.checkbox("log", value=bool(spec.get("log"))).props("dense")
            self.log.tooltip("Sample on a log scale")
            ui.button(icon="close", on_click=lambda: on_remove(self)).props("flat round dense size=sm color=grey")
        self._sync()

    def _sync(self):
        is_range = self.mode.value == "range"
        self.range_box.visible = is_range
        self.log.visible = is_range
        self.choices.visible = not is_range

    def export_setting(self) -> Optional[dict]:
        if self.mode.value == "range":
            if not self.low.value.strip() or not self.high.value.strip():
                return None
            # Continuous range -- use the explicit "range" key (see
            # libauc.auto.helpers.parse_hyperparameters_from_dict) so this
            # survives the GUI -> JSON -> subprocess round-trip as a bound,
            # never misread as a 2-choice categorical (JSON has no tuples).
            data = {"range": [_literal(self.low.value, f"{self.name} min"), _literal(self.high.value, f"{self.name} max")]}
            if self.log.value:
                data["log"] = True
        else:
            if not self.choices.value.strip():
                return None
            val = _literal(self.choices.value, f"{self.name} choices")
            data = {"val": list(val) if isinstance(val, tuple) else val}
        if self.default.value.strip():
            data["default"] = _literal(self.default.value, f"{self.name} default")
        return data


class SearchSpace:
    """
    The resolved hyperparameter search space for one optimizer or loss.
    Populated via ``set_space()`` -- the Loss selector is the single source
    of truth for which optimizer/loss pair is active.
    """

    def __init__(self, kind: str, resolver):
        self.kind = kind  # "optimizer" | "loss"
        self.resolver = resolver  # e.g. Trainer._get_optimizer / Trainer._get_loss
        self.type_name: Optional[str] = None
        self.params: List[HyperParameter] = []

        with ui.column().classes("w-full gap-1"):
            with ui.row().classes("items-center w-full"):
                ui.label(kind.capitalize()).classes("text-xs uppercase tracking-wide am-muted")
                self.title = ui.label("-").classes("am-mono font-semibold")
                ui.space()
                ui.button("Add kwargs", icon="add", on_click=self.add_constructor_kwargs).props("flat dense no-caps size=sm").tooltip(
                    "Add the remaining constructor arguments as tunable rows"
                )
            self.rows = ui.column().classes("w-full gap-0")
            self.empty = ui.label("Choose a loss function to populate.").classes("text-sm am-muted px-2")

    def set_space(self, data: dict):
        self.type_name = data["type"]
        self.title.text = self.type_name
        self.rows.clear()
        self.params = []
        with self.rows:
            for name, spec in data["space"].items():
                self.params.append(HyperParameter(name, spec, self._remove))
        self.empty.text = "No tunable hyperparameters."
        self.empty.visible = not self.params

    def _remove(self, param: HyperParameter):
        self.params.remove(param)
        self.rows.remove(param.row)
        self.empty.visible = not self.params

    def add_constructor_kwargs(self):
        if self.type_name is None:
            ui.notify("Choose a loss function first", type="warning")
            return
        cls = self.resolver(self.type_name)
        if cls is None:
            return
        existing = {p.name for p in self.params}
        added = 0
        with self.rows:
            for p in inspect.signature(cls).parameters.values():
                if p.name in _SKIP_PARAM_NAMES or p.name in existing:
                    continue
                if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue
                self.params.append(HyperParameter(p.name, {"val": p.default, "default": p.default}, self._remove))
                added += 1
        self.empty.visible = not self.params
        ui.notify(f"Added {added} argument(s) from {self.type_name}" if added else "No more arguments to add")

    def export_setting(self) -> dict:
        space = {}
        for p in self.params:
            spec = p.export_setting()
            if spec is not None:
                space[p.name] = spec
        return {"type": self.type_name, "space": space}


class ConfigurePanel:
    """The whole "New search" page. ``on_launched(session_id)`` is called
    after a worker has been started for a new session."""

    def __init__(self, on_launched: Callable[[str], None]):
        self.session_manager = get_session_manager()
        self.store = get_dataset_config_store()
        self.on_launched = on_launched
        self.multilabel = False
        self._custom_train_csv = ""
        self._custom_val_csv = ""
        self._custom_test_csv = ""

        with ui.element("div").classes("w-full grid grid-cols-1 lg:grid-cols-12 gap-6 items-start"):
            with ui.column().classes("lg:col-span-8 gap-6 min-w-0"):
                self._build_model_card()
                self._build_dataset_card()
                self._build_objective_card()
                self._build_space_card()
                self._build_training_card()
            with ui.column().classes("lg:col-span-4 gap-6"):
                self._build_summary_card()

        self._backend_changed()
        self._metric_changed()
        self._loss_changed()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_model_card(self):
        with section_card("Task & model", "hub", "What kind of data, and which network to tune"):
            self.backend_toggle = ui.toggle(
                {"image": "Image classification", "gnn": "Graph classification"}, value="image",
                on_change=self._backend_changed,
            ).props("no-caps unelevated spread toggle-color=primary").classes("w-full")

            with ui.row().classes("w-full items-start gap-3") as self.image_model_row:
                self.image_model = ui.select(
                    _IMAGE_MODELS + [_HF_MODEL_OPTION], label="Architecture", on_change=self._model_changed,
                ).props(FIELD).classes("w-56")
                self.image_model_hf_id = ui.input("HuggingFace model ID", placeholder="google/vit-base-patch16-224").props(FIELD).classes("grow")
                with ui.column().classes("gap-1 grow"):
                    self.pretrained_ck = ui.switch("Start from a checkpoint", on_change=self._model_changed)
                    with ui.row().classes("w-full items-center gap-1 no-wrap") as self.model_path_row:
                        self.model_path = ui.input("Checkpoint path").props(FIELD).classes("grow am-mono")
                        ui.button(icon="folder_open", on_click=lambda: self._browse(self.model_path, files=True)).props("flat round dense")

            with ui.row().classes("w-full items-start gap-3") as self.gnn_model_row:
                self.gnn_model = ui.select(
                    _GNN_MODELS + [_HF_MODEL_OPTION], label="GNN architecture", on_change=self._model_changed,
                ).props(FIELD).classes("w-56")
                self.gnn_model_hf_id = ui.input("HuggingFace model ID", placeholder="user/my-gcn").props(FIELD).classes("grow")
                self.gnn_emb_dim = ui.number("Embedding dim", value=256, min=1, precision=0).props(FIELD).classes("w-32")
                self.gnn_num_layers = ui.number("Layers", value=5, min=1, precision=0).props(FIELD).classes("w-24")
                self.gnn_decay_factor = ui.number("Decay factor", value=10.0, min=0).props(FIELD).classes("w-32")

    def _build_dataset_card(self):
        with section_card("Dataset", "dataset", "A built-in benchmark or your own CSV-indexed images"):
            with ui.row().classes("w-full items-start gap-3"):
                self.dataset_select = ui.select({}, label="Dataset", on_change=self._dataset_changed).props(FIELD).classes("w-64")
                # Predefined datasets: Root Path starts blank on purpose -- a
                # predefined dataset can't be loaded without one, rather than
                # silently defaulting to some "./data" that likely doesn't
                # contain the user's data. Recalled per dataset name.
                with ui.row().classes("grow items-center gap-1 no-wrap") as self.root_path_row:
                    self.dataset_root_path = ui.input("Root path", on_change=self.refresh_summary).props(FIELD).classes("grow am-mono")
                    ui.button(icon="folder_open", on_click=lambda: self._browse(self.dataset_root_path)).props("flat round dense")
                self.custom_name = ui.input(
                    "Name for this dataset", placeholder="a unique name", on_change=self.refresh_summary,
                ).props(FIELD).classes("grow")

            # Custom image dataset (CSV-backed, via MedicalImageCSVDataset).
            # Train/val/test CSVs are auto-detected from Root Path; image
            # paths in the CSV resolve relative to that same Root Path.
            with ui.column().classes("w-full gap-3") as self.custom_box:
                with ui.row().classes("w-full items-center gap-1 no-wrap"):
                    self.custom_root_path = ui.input(
                        "Dataset root (contains train.csv, val.csv)", on_change=self._detect_custom_csvs,
                    ).props(FIELD).classes("grow am-mono")
                    ui.button(icon="folder_open", on_click=lambda: self._browse(self.custom_root_path)).props("flat round dense")
                with ui.row().classes("w-full gap-3"):
                    self.custom_image_col = ui.input("Image column", value="image_path").props(FIELD).classes("w-40")
                    self.custom_label_col = ui.input("Label column", value="label").props(FIELD).classes("w-40")
                    self.custom_image_size = ui.number("Image size", value=224, min=8, precision=0).props(FIELD).classes("w-32")
                    ui.space()
                    self.save_custom_btn = ui.button("Save dataset", icon="bookmark_add", on_click=self._save_custom_dataset).props("outline no-caps")
                self.custom_csv_chips = ui.row().classes("gap-2 items-center")

            self.gnn_custom_message = ui.label("Custom graph datasets are not yet supported.").classes("text-warning text-sm")

            with ui.row().classes("w-full items-center gap-3"):
                self.inspect_btn = ui.button("Inspect dataset", icon="query_stats", on_click=self._inspect_dataset).props("flat no-caps")
                self.dataset_stats = ui.row().classes("gap-2 items-center")

    def _build_objective_card(self):
        with section_card("Objective", "flag", "The metric to maximize, and the LibAUC loss used to get there"):
            with ui.row().classes("w-full items-center gap-4"):
                self.metric = ui.toggle(["AUROC", "AUPRC", "ACC"], value="AUROC", on_change=self._metric_changed).props(
                    "no-caps unelevated toggle-color=primary"
                )
                self.partial_auroc = ui.switch("Partial AUROC", on_change=self._metric_changed)
                self.max_fpr = ui.number("Max FPR", value=0.3, min=0, max=1, step=0.05).props(FIELD).classes("w-28")
                self.min_tpr = ui.number("Min TPR", value=0.7, min=0, max=1, step=0.05).props(FIELD).classes("w-28")
            with ui.row().classes("w-full items-center gap-3"):
                self.loss_select = ui.select(_LOSS_NAMES, label="Loss function", on_change=self._loss_changed).props(FIELD).classes("w-64")
                self.optimizer_chip = ui.chip("", icon="bolt", color="primary").props("outline")
                self.optimizer_chip.tooltip("Optimizer paired with this loss")

    def _build_space_card(self):
        card = section_card("Search space", "tune", "Ranges SMAC samples from; defaults seed the first trial")
        with card:
            self.optimizers = SearchSpace("optimizer", Trainer._get_optimizer)
            ui.separator()
            self.losses = SearchSpace("loss", Trainer._get_loss)
            with ui.row().classes("w-full justify-end"):
                ui.button("Copy as JSON", icon="content_copy", on_click=self._copy_space).props("flat dense no-caps size=sm")

    def _build_training_card(self):
        with section_card("Training & search budget", "speed", "Applied to every trial"):
            with ui.element("div").classes("w-full grid grid-cols-2 md:grid-cols-4 gap-3"):
                self.n_trials = ui.number("Trials", value=5, min=1, precision=0, on_change=self.refresh_summary).props(FIELD)
                self.epochs = ui.number("Epochs / trial", value=50, min=1, precision=0, on_change=self.refresh_summary).props(FIELD)
                self.batch_size = ui.number("Batch size", value=128, min=1, precision=0).props(FIELD)
                self.eval_batch_size = ui.number("Eval batch size", value=128, min=1, precision=0).props(FIELD)
                self.sampling_rate = ui.number("Sampling rate", value=0.2, min=0, max=1, step=0.05).props(FIELD)
                self.decay_epochs = ui.input("Decay epochs", value="[0.5, 0.75]").props(FIELD)
                self.num_workers = ui.number("Data workers", value=0, min=0, precision=0).props(FIELD)
                self.seed = ui.number("Seed", value=42, precision=0).props(FIELD)
            self.output_path = ui.input("Output path (base)", value="./output").props(FIELD).classes("w-full am-mono")

    def _build_summary_card(self):
        with section_card("Run summary", "rocket_launch"):
            self.summary_box = ui.column().classes("w-full gap-2")
            ui.separator()
            self.checklist = ui.column().classes("w-full gap-1")
            self.launch_btn = ui.button("Launch search", icon="play_arrow", on_click=self._launch).props(
                "unelevated no-caps size=lg"
            ).classes("am-launch w-full text-white")
            ui.label(
                "Training runs in its own process. Closing this page or the server does not stop it."
            ).classes("text-xs am-muted")

    # ------------------------------------------------------------------
    # Derived state
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return self.backend_toggle.value

    @property
    def model_name(self) -> Optional[str]:
        select, hf_id = (self.gnn_model, self.gnn_model_hf_id) if self.backend == "gnn" else (self.image_model, self.image_model_hf_id)
        if select.value == _HF_MODEL_OPTION:
            return hf_id.value.strip() or None
        return select.value

    @property
    def _is_custom_image_dataset(self) -> bool:
        value = self.dataset_select.value or ""
        return self.backend == "image" and (value == _CUSTOM_DATASET_OPTION or value.startswith(_SAVED_CUSTOM_PREFIX))

    @property
    def _is_saved_custom_dataset(self) -> bool:
        return (self.dataset_select.value or "").startswith(_SAVED_CUSTOM_PREFIX)

    @property
    def _is_custom_gnn_dataset(self) -> bool:
        return self.backend == "gnn" and self.dataset_select.value == _CUSTOM_DATASET_OPTION

    @property
    def dataset_name(self) -> Optional[str]:
        value = self.dataset_select.value
        if self._is_custom_gnn_dataset:
            return None
        if self._is_saved_custom_dataset:
            return value[len(_SAVED_CUSTOM_PREFIX):]
        if value == _CUSTOM_DATASET_OPTION:
            return self.custom_name.value.strip() or None
        return value

    def _dataset_options(self) -> dict:
        if self.backend == "gnn":
            names = _GNN_DATASETS
            saved = []
        else:
            names = _IMAGE_DATASETS
            saved = self.store.custom_image_dataset_names()
        options = {n: n for n in names}
        options.update({f"{_SAVED_CUSTOM_PREFIX}{n}": f"{n}  (custom)" for n in saved})
        options[_CUSTOM_DATASET_OPTION] = "+ New custom dataset..." if self.backend == "image" else _CUSTOM_DATASET_OPTION
        return options

    def _issues(self) -> List[str]:
        issues = []
        if not self.model_name:
            issues.append("Choose a model")
        if self._is_custom_gnn_dataset:
            issues.append("Custom graph datasets are not supported yet")
        elif not self.dataset_name:
            issues.append("Choose a dataset")
        elif self._is_custom_image_dataset:
            if not self._custom_train_csv or not self._custom_val_csv:
                issues.append("Dataset root needs train.csv and val.csv")
        elif not self.dataset_root_path.value.strip():
            issues.append("Set the dataset root path")
        if not self.loss_select.value:
            issues.append("Choose a loss function")
        return issues

    def refresh_summary(self, *_):
        n_trials = int(self.n_trials.value or 0)
        epochs = int(self.epochs.value or 0)
        rows = [
            ("Task", "Graph" if self.backend == "gnn" else "Image"),
            ("Model", self.model_name or "-"),
            ("Dataset", self.dataset_name or "-"),
            ("Target", self.metric.value + (" (partial)" if self._is_partial_auroc else "")),
            ("Loss", self.loss_select.value or "-"),
            ("Optimizer", self.optimizers.type_name or "-"),
            ("Budget", f"{n_trials} trials x {epochs} epochs"),
        ]
        self.summary_box.clear()
        with self.summary_box:
            for key, value in rows:
                with ui.row().classes("w-full justify-between no-wrap gap-3"):
                    ui.label(key).classes("text-sm am-muted")
                    ui.label(value).classes("text-sm font-medium am-mono truncate text-right")

        issues = self._issues()
        self.checklist.clear()
        with self.checklist:
            if issues:
                for issue in issues:
                    with ui.row().classes("items-center gap-2 no-wrap"):
                        ui.icon("radio_button_unchecked", size="16px").classes("text-warning")
                        ui.label(issue).classes("text-sm")
            else:
                with ui.row().classes("items-center gap-2 no-wrap"):
                    ui.icon("check_circle", size="16px").classes("text-positive")
                    ui.label("Ready to launch").classes("text-sm")
        self.launch_btn.set_enabled(not issues)

    @property
    def _is_partial_auroc(self) -> bool:
        return self.metric.value == "AUROC" and self.partial_auroc.value

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _backend_changed(self, *_):
        is_gnn = self.backend == "gnn"
        self.image_model_row.visible = not is_gnn
        self.gnn_model_row.visible = is_gnn
        self.dataset_select.options = self._dataset_options()
        self.dataset_select.value = None
        self.dataset_select.update()
        self._model_changed()
        self._dataset_changed()

    def _model_changed(self, *_):
        self.image_model_hf_id.visible = self.image_model.value == _HF_MODEL_OPTION
        self.gnn_model_hf_id.visible = self.gnn_model.value == _HF_MODEL_OPTION
        self.model_path_row.visible = self.pretrained_ck.value
        self.refresh_summary()

    def _dataset_changed(self, *_):
        self.dataset_stats.clear()
        is_custom_image = self._is_custom_image_dataset
        self.root_path_row.visible = not (is_custom_image or self._is_custom_gnn_dataset) and bool(self.dataset_select.value)
        self.custom_name.visible = self.backend == "image" and self.dataset_select.value == _CUSTOM_DATASET_OPTION
        self.custom_box.visible = is_custom_image
        self.save_custom_btn.visible = is_custom_image and not self._is_saved_custom_dataset
        self.gnn_custom_message.visible = self._is_custom_gnn_dataset
        self.inspect_btn.visible = not self._is_custom_gnn_dataset

        if self._is_saved_custom_dataset:
            saved = self.store.get_custom_image_dataset(self.dataset_name) or {}
            self.custom_root_path.value = saved.get("root_path", "")
            self.custom_image_col.value = saved.get("image_col", "image_path")
            self.custom_label_col.value = saved.get("label_col", "label")
            self.custom_image_size.value = saved.get("image_size", 224)
        elif self.dataset_select.value and not is_custom_image and not self._is_custom_gnn_dataset:
            # Recall the last-used Root Path for this predefined dataset.
            self.dataset_root_path.value = self.store.get_root_path(self.dataset_name)
        self._detect_custom_csvs()

    def _metric_changed(self, *_):
        is_auroc = self.metric.value == "AUROC"
        self.partial_auroc.visible = is_auroc
        self.max_fpr.visible = self.min_tpr.visible = self._is_partial_auroc
        self.refresh_summary()

    def _loss_changed(self, *_):
        if self.loss_select.value:
            conf = parse_defaultconfig(self.loss_select.value, self.multilabel, {})
            if not conf["optimizer"]:
                conf["optimizer"] = parse_defaultconfig(_FALLBACK_OPTIMIZER, self.multilabel, {})["optimizer"]
            self.optimizers.set_space(conf["optimizer"])
            self.losses.set_space(conf["loss"])
            self.optimizer_chip.text = conf["optimizer"]["type"]
        self.optimizer_chip.visible = bool(self.loss_select.value)
        self.refresh_summary()

    async def _browse(self, field: ui.input, files: bool = False):
        path = await PathPicker(field.value, pick_files=files)
        if path:
            field.value = path

    def _copy_space(self):
        try:
            data = {"optimizer": self.optimizers.export_setting(), "loss": self.losses.export_setting()}
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return
        ui.clipboard.write(json.dumps(data, indent=2))
        ui.notify("Search space copied", type="positive")

    # ------------------------------------------------------------------
    # Custom image dataset
    # ------------------------------------------------------------------

    @staticmethod
    def _find_csv(root_path: str, candidates) -> str:
        for name in candidates:
            path = os.path.join(root_path, name)
            if os.path.isfile(path):
                return path
        return ""

    def _detect_custom_csvs(self, *_):
        """Look for train.csv / val.csv|valid.csv / test.csv directly under
        the custom dataset's Root Path and populate _custom_{train,val,test}_csv.
        Images referenced by the CSVs are also resolved relative to this same
        Root Path (it's passed as MedicalImageCSVDataset's image_root)."""
        root_path = self.custom_root_path.value.strip()
        is_dir = bool(root_path) and os.path.isdir(root_path)
        self._custom_train_csv = self._find_csv(root_path, _TRAIN_CSV_CANDIDATES) if is_dir else ""
        self._custom_val_csv = self._find_csv(root_path, _VAL_CSV_CANDIDATES) if is_dir else ""
        self._custom_test_csv = self._find_csv(root_path, _TEST_CSV_CANDIDATES) if is_dir else ""

        self.custom_csv_chips.clear()
        with self.custom_csv_chips:
            if root_path and not is_dir:
                ui.label("Not a valid directory").classes("text-negative text-sm")
            elif is_dir:
                for label, path, required in (
                    ("train", self._custom_train_csv, True),
                    ("val", self._custom_val_csv, True),
                    ("test", self._custom_test_csv, False),
                ):
                    color = "positive" if path else ("negative" if required else "grey")
                    icon = "check" if path else ("close" if required else "remove")
                    ui.chip(f"{label}.csv", icon=icon, color=color).props("outline dense").tooltip(path or "not found")
        self.refresh_summary()

    def _save_custom_dataset(self):
        name = self.custom_name.value.strip()
        if not self._custom_train_csv or not self._custom_val_csv:
            ui.notify("Could not detect train.csv / val.csv under this root path", type="negative")
            return
        definition = {
            "root_path": self.custom_root_path.value.strip(),
            "image_col": self.custom_image_col.value.strip() or "image_path",
            "label_col": self.custom_label_col.value.strip() or "label",
            "image_size": int(self.custom_image_size.value or 224),
        }
        try:
            self.store.add_custom_image_dataset(name, definition, reserved_names=_IMAGE_DATASETS)
        except ValueError as exc:
            ui.notify(str(exc), type="negative")
            return
        ui.notify(f"Saved custom dataset '{name}'", type="positive")
        self.dataset_select.options = self._dataset_options()
        self.dataset_select.value = f"{_SAVED_CUSTOM_PREFIX}{name}"
        self.dataset_select.update()

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

    # ------------------------------------------------------------------
    # Dataset inspection
    # ------------------------------------------------------------------

    def _load_preview_dataset(self):
        """(train_dataset, eval_datasets) for the current selection -- the
        same code path build_pipeline() uses for an actual training run."""
        if self._is_custom_image_dataset:
            return load_custom_csv_dataset(self._custom_image_dataset_kwargs(), ["val"])
        if not self.dataset_root_path.value.strip():
            raise ValueError("Root path is required to load this dataset.")
        return load_dataset(self.dataset_name, splits=["val"], **self._dataset_kwargs())

    async def _inspect_dataset(self):
        name = self.dataset_name
        if not name:
            ui.notify("Choose a dataset first", type="warning")
            return
        backend = self.backend
        self.inspect_btn.props("loading")
        self.dataset_stats.clear()
        try:
            train_dataset, eval_datasets = await run.io_bound(self._load_preview_dataset)
        except Exception as exc:
            with self.dataset_stats:
                ui.label(f"Error: {exc}").classes("text-negative text-sm")
            return
        finally:
            self.inspect_btn.props(remove="loading")

        stats = [("Train", f"{len(train_dataset):,}"), ("Eval", f"{len(eval_datasets[0]) if eval_datasets else 0:,}")]
        if backend == "image":
            labels = np.array(train_dataset.targets).squeeze()
            num_tasks = len(np.unique(labels)) if labels.ndim == 1 else labels.shape[-1]
            self.multilabel = num_tasks >= 3
            stats += [("Tasks", str(num_tasks)), ("Type", "multi-label" if self.multilabel else "binary")]
            # Multi-label changes the default search space; re-derive it.
            self._loss_changed()
        else:
            self.multilabel = False
        with self.dataset_stats:
            for label, value in stats:
                ui.chip(f"{label}: {value}", color="primary").props("outline dense")
        if not self._is_custom_image_dataset:
            self.store.set_root_path(name, self.dataset_root_path.value.strip())

    # ------------------------------------------------------------------
    # Config building + launch
    # ------------------------------------------------------------------

    def _build_config(self) -> dict:
        metric_kwargs_entry = {}
        if self._is_partial_auroc:
            metric_kwargs_entry["max_fpr"] = float(self.max_fpr.value)
            metric_kwargs_entry["min_tpr"] = float(self.min_tpr.value)

        optimizer = self.optimizers.export_setting()
        loss = self.losses.export_setting()
        training = {
            "optimizer": optimizer["type"],
            "optimizer_kwargs": optimizer["space"],
            "loss": loss["type"],
            "loss_kwargs": loss["space"],
            "SEED": int(self.seed.value),
            "batch_size": int(self.batch_size.value),
            "eval_batch_size": int(self.eval_batch_size.value),
            "sampling_rate": float(self.sampling_rate.value),
            "epochs": int(self.epochs.value),
            "decay_epochs": _literal(self.decay_epochs.value, "Decay epochs"),
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
            model = {"name": self.model_name, "pretrained": self.pretrained_ck.value}
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
            dataset_cfg = {"name": self.dataset_name, "kwargs": self._dataset_kwargs(), "eval_splits": ["val"]}

        return {
            "dataset": dataset_cfg,
            "model": model,
            "metrics": [self.metric.value],
            "metric_kwargs": [metric_kwargs_entry],
            "training": training,
            "automax": automax,
        }

    def _launch(self):
        issues = self._issues()
        if issues:
            ui.notify(issues[0], type="warning")
            return
        try:
            config = self._build_config()
        except (TypeError, ValueError) as exc:
            ui.notify(f"Invalid setting: {exc}", type="negative")
            return
        if not self._is_custom_image_dataset:
            self.store.set_root_path(self.dataset_name, self.dataset_root_path.value.strip())

        session_name = f"{self.model_name}_{self.loss_select.value}_{self.dataset_name}"
        session = self.session_manager.create_session(session_name, config, backend=self.backend)
        try:
            process = self._start_training_process(session.session_id, session.log_file, config)
        except Exception as exc:
            self.session_manager.complete_session(session.session_id, error_message=f"Failed to start worker: {exc}")
            ui.notify(f"Failed to start training process: {exc}", type="negative")
            return
        ui.notify(f"Launched {session_name} (PID {process.pid})", type="positive", icon="rocket_launch")
        self.on_launched(session.session_id)

    def _start_training_process(self, session_id: str, log_path: str, config: dict) -> subprocess.Popen:
        config_file = Path(tempfile.gettempdir()) / f"automax_gui_session_{session_id}.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f)
        with open(log_path, "a", encoding="utf-8") as log_fh:
            process = subprocess.Popen(
                [sys.executable, "-m", "libauc.auto.ui.training_worker", "--config", str(config_file), "--session-id", session_id],
                # Same cwd as the GUI: gui_sessions/ paths are relative.
                cwd=os.getcwd(),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
        _worker_processes.append(process)
        return process
