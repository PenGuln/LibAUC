"""
Entry point for AutoMAX hyperparameter search (image / standard trainer).

Usage::

    python -m libauc.auto.run_auto_trainer --config_file config.yaml

Parallel search (4 trials at once, 1 GPU each; or 2 at once with 2-GPU DDP)::

    python -m libauc.auto.run_auto_trainer --config_file config.yaml --n_workers 4
    python -m libauc.auto.run_auto_trainer --config_file config.yaml --n_workers 2 --gpus_per_trial 2

Multi-node search (trials run one after another, each with DDP on every
rank of the ``torchrun`` job)::

    torchrun --nnodes 8 --nproc_per_node 2 ... -m libauc.auto.run_auto_trainer --config_file config.yaml
"""

import argparse
import logging
import sys
from functools import partial
from typing import Callable, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf

from libauc.trainer import TrainingArguments, CLICallback, Trainer, MILTrainer
from libauc.trainer import init_distributed, cleanup_distributed
from libauc.utils import is_distributed
from libauc.trainer import load_dataset, parse_defaultconfig
from libauc.trainer.helpers import build_megaconf, build_metric, set_seed
from libauc.trainer.helpers import apply_cli_overrides as _apply_training_cli_overrides
from .automax import AutoMAX
from .autopartial import autopartial
from .configuration import AutoMAXConfigration
from .helpers import parse_hyperparameters_from_dict
from .custom_datasets import load_custom_csv_dataset

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default config (megaconf)
# ---------------------------------------------------------------------------

_MEGACONF_DEFAULTS = {
    "training": {
        "optimizer":              "PESG",
        "optimizer_kwargs":       {},
        "loss":                   "AUCMLoss",
        "loss_kwargs":            {},
        "SEED":                   42,
        "batch_size":             128,
        "eval_batch_size":        128,
        "sampling_rate":          0.5,
        "epochs":                 50,
        "early_stopping_patience":  None,  # epochs without val improvement; None: off
        "early_stopping_min_delta": 0.0,
        "decay_epochs":           [],
        "num_workers":            2,
        "output_path":            "./output",
        "resume_from_checkpoint": True,
        "save_checkpoint_every":  5,
        "project_name":           "libauc",
        "experiment_name":        "run_auto",
        "verbose":                1,
        "instance_batch_size":    4,   # MIDAMLoss only: instances sampled per bag per step
    },
    "automax": {
        "deterministic":    True,
        "n_trials":         5,
        "n_configs":        1,
        "SEED":             42,
        "name":             "automax_search",
        "output_directory": None,  # falls back to automax.output_path, then ./automax_output
        "overwrite":        True,
        "n_workers":        1,     # trials evaluated concurrently
        "gpus_per_trial":   1,     # >1: each trial trains with DDP
        "devices":          None,  # GPU ids to use (default: all visible)
        "start_method":     "spawn",
        "seeds_per_trial":  1,     # seeds per configuration; SMAC sees their mean score
        "keep_top_k":       1,     # checkpoints of the k best trials are kept (+ <name>_top<k>.json)
        "trial_retries":    1,     # restarts of a crashed trial (from its checkpoints) before it counts as crashed
    },
    "dataset": {
        "name":        "",
        "kwargs":      {},
        "eval_splits": ["val"],
    },
    # type: image | mil (default: mil for MIDAM losses, image otherwise).
    # num_slices (mil, study-level datasets): slices sampled per study per step.
    "trainer":       {},
    # Per-split transform lists for custom_csv / medical_study_csv datasets.
    "preprocessing": {},
    "model": {
        "name":        "resnet18",
        "pretrained":  False,
    },
    "metrics":       ["AUROC"],
    "metric_kwargs": [],
}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="AutoMAX training script")

    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to a YAML configuration file.",
    )

    parser.add_argument("--epochs",                type=int,   default=None)
    parser.add_argument("--batch_size",            type=int,   default=None)
    parser.add_argument("--eval_batch_size",       type=int,   default=None)
    parser.add_argument("--sampling_rate",         type=float, default=None)
    parser.add_argument("--num_workers",           type=int,   default=None)
    parser.add_argument("--output_path",           type=str,   default=None)
    parser.add_argument("--seed",                  type=int,   default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--save_checkpoint_every", type=int,   default=None)

    # Parallel / distributed search
    parser.add_argument("--n_workers",      type=int, default=None,
                        help="Number of trials to run concurrently.")
    parser.add_argument("--gpus_per_trial", type=int, default=None,
                        help="GPUs per trial; >1 trains each trial with DDP.")
    parser.add_argument("--devices",        type=int, nargs="+", default=None,
                        help="GPU ids to use, e.g. --devices 0 1 2 3 (default: all visible).")
    parser.add_argument("--seeds_per_trial", type=int, default=None,
                        help="Train every configuration with this many seeds; SMAC sees the mean score.")
    parser.add_argument("--keep_top_k",     type=int, default=None,
                        help="Keep the checkpoints of the k best trials (ranked in <name>_top<k>.json).")
    parser.add_argument("--trial_retries",  type=int, default=None,
                        help="Restart a crashed trial this many times, resuming from its checkpoints (default 1).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# CLI override helper
# ---------------------------------------------------------------------------

# (section, cfg_key, args_attr) for AutoMAX-specific flags; the training
# flags are handled by libauc.trainer.helpers.apply_cli_overrides.
_AUTOMAX_CLI_MAP: list[tuple[str, str, str]] = [
    ("automax", "n_workers",      "n_workers"),
    ("automax", "gpus_per_trial", "gpus_per_trial"),
    ("automax", "devices",        "devices"),
    ("automax", "seeds_per_trial", "seeds_per_trial"),
    ("automax", "keep_top_k",     "keep_top_k"),
    ("automax", "trial_retries",  "trial_retries"),
]


def apply_cli_overrides(cfg: OmegaConf, args) -> OmegaConf:
    """Merge CLI-supplied training and AutoMAX values into the megaconf (in-place)."""
    cfg = _apply_training_cli_overrides(cfg, args)
    for section, cfg_key, arg_attr in _AUTOMAX_CLI_MAP:
        value = getattr(args, arg_attr, None)
        if value is not None:
            logger.info(f"CLI override: {section}.{cfg_key} = {value}")
            OmegaConf.update(cfg, f"{section}.{cfg_key}", value)
    return cfg


# ---------------------------------------------------------------------------
# Pipeline construction (shared by the CLI `main()` and the GUI training
# worker, so the two never re-implement dataset loading / config resolution
# independently)
# ---------------------------------------------------------------------------

def build_pipeline(
    cfg_dict: dict,
    callback_factory: Optional[Callable[[AutoMAXConfigration], List]] = None,
):
    """
    Build the autopartial Trainer, AutoMAXConfigration, and metric name list
    from a fully-resolved config dict.

    Args:
        cfg_dict (dict): Dict with the same top-level shape as
            ``_MEGACONF_DEFAULTS`` (already ``OmegaConf.to_container()``'d,
            with any CLI overrides applied by the caller).
        callback_factory: Optional ``callable(automax_args) -> list`` of
            extra ``TrainerCallback`` instances, appended after
            ``CLICallback()`` in the Trainer's callback list. Lets a caller
            (e.g. the GUI's training worker) observe/report training
            progress without re-implementing this pipeline.

    Returns:
        tuple: ``(trainer, automax_args, metric_names)``.
    """
    training_cfg  = cfg_dict["training"]
    automax_cfg   = cfg_dict["automax"]
    dataset_cfg   = cfg_dict["dataset"]
    model_cfg     = cfg_dict["model"]
    metric_names  = cfg_dict.get("metrics",       ["AUROC"])
    metric_kwargs = cfg_dict.get("metric_kwargs", [])
    trainer_cfg   = cfg_dict.get("trainer") or {}
    preprocessing = cfg_dict.get("preprocessing") or {}

    # 1. Reproducibility
    set_seed(training_cfg.get("SEED", 42))

    # 2. Load datasets
    dataset_name   = dataset_cfg["name"]
    dataset_kwargs = dataset_cfg.get("kwargs", {})
    eval_splits    = dataset_cfg.get("eval_splits", ["val"])

    dataset_kind   = dataset_cfg.get("kind")

    if dataset_kind == "custom_csv":
        # GUI "Custom Dataset..." option: a user-defined CSV/image-folder
        # layout, loaded via MedicalImageCSVDataset instead of the fixed
        # name registry in libauc.trainer.load_dataset().
        logger.info(f"Loading custom CSV dataset '{dataset_name}' (splits: {eval_splits})")
        train_dataset, eval_datasets = load_custom_csv_dataset(dataset_kwargs, eval_splits, preprocessing)
    elif dataset_kind == "medical_study_csv":
        # Study-level image bags (one or more series per study) for MILTrainer.
        from libauc.trainer.data import load_medical_study_csv_datasets
        logger.info(f"Loading study-level medical dataset '{dataset_name}' (splits: {eval_splits})")
        train_dataset, eval_datasets = load_medical_study_csv_datasets(dataset_kwargs, eval_splits, preprocessing)
    else:
        logger.info(f"Loading train and {eval_splits} splits of dataset: {dataset_name}")
        train_dataset, eval_datasets = load_dataset(
            dataset_name, splits=eval_splits, **dataset_kwargs
        )

    # 3. Infer task count
    labels = np.array(train_dataset.targets).squeeze()
    num_tasks = len(np.unique(labels)) if labels.ndim == 1 else labels.shape[-1]
    logger.info(f"Number of tasks: {num_tasks}")
    multilabel = num_tasks >= 3

    # 4. Build metric function
    metric_fn = build_metric(metric_names, metric_kwargs)

    # 5. Resolve default optimizer / loss configs, then merge search-space
    #    defaults with any user-supplied overrides from the YAML
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    default_optimizer_config = parse_defaultconfig(
        training_cfg["optimizer"], multilabel, optimizer_kwargs
    )["optimizer"]
    optimizer_kwargs = {**default_optimizer_config["space"], **optimizer_kwargs}

    loss_kwargs = training_cfg.get("loss_kwargs", {})
    if multilabel:
        loss_kwargs["num_labels"] = num_tasks
    default_loss_config = parse_defaultconfig(
        training_cfg["loss"], multilabel, loss_kwargs
    )["loss"]
    loss_kwargs = {**default_loss_config["space"], **loss_kwargs}

    # 6. Build autopartial TrainingArguments (carries ConfigSpace distributions)
    train_args = autopartial(
        partial(
            TrainingArguments,
            optimizer              = default_optimizer_config["type"],
            loss                   = default_loss_config["type"],
            SEED                   = training_cfg.get("SEED",                42),
            eval_batch_size        = training_cfg.get("eval_batch_size",     128),
            sampling_rate          = training_cfg.get("sampling_rate",       0.5),
            epochs                 = training_cfg.get("epochs",              50),
            early_stopping_patience  = training_cfg.get("early_stopping_patience"),
            early_stopping_min_delta = training_cfg.get("early_stopping_min_delta", 0.0),
            decay_epochs           = training_cfg.get("decay_epochs",        []),
            num_workers            = training_cfg.get("num_workers",         2),
            output_path            = training_cfg.get("output_path",         "./output"),
            num_tasks              = num_tasks,
            resume_from_checkpoint = training_cfg.get("resume_from_checkpoint", True),
            save_checkpoint_every  = training_cfg.get("save_checkpoint_every",  5),
            project_name           = training_cfg.get("project_name",        "libauc"),
            experiment_name        = training_cfg["experiment_name"],
            verbose                = training_cfg.get("verbose",             1),
        ),
        # batch_size may be a search dimension ({"val": [...]}) like any kwarg.
        **parse_hyperparameters_from_dict({"batch_size": training_cfg.get("batch_size", 128)}),
        optimizer_kwargs = autopartial(dict, **parse_hyperparameters_from_dict(optimizer_kwargs)),
        loss_kwargs      = autopartial(dict, **parse_hyperparameters_from_dict(loss_kwargs)),
    )

    # 7. AutoMAX configuration
    automax_args = AutoMAXConfigration(
        deterministic    = automax_cfg.get("deterministic",    True),
        n_trials         = automax_cfg.get("n_trials",         5),
        n_configs        = automax_cfg.get("n_configs",        1),
        SEED             = automax_cfg.get("SEED",             42),
        name             = automax_cfg.get("name"),
        output_directory = automax_cfg.get("output_directory"),
        output_path      = automax_cfg.get("output_path"),
        overwrite        = automax_cfg.get("overwrite",        True),
        n_workers        = automax_cfg.get("n_workers",        1),
        gpus_per_trial   = automax_cfg.get("gpus_per_trial",   1),
        devices          = automax_cfg.get("devices"),
        start_method     = automax_cfg.get("start_method",     "spawn"),
        seeds_per_trial  = automax_cfg.get("seeds_per_trial",  1),
        keep_top_k       = automax_cfg.get("keep_top_k",       1),
        trial_retries    = automax_cfg.get("trial_retries",    1),
    )

    extra_callbacks = callback_factory(automax_args) if callback_factory else []

    # 8. Build autopartial Trainer (MILTrainer for multiple instance learning)
    trainer_type = trainer_cfg.get("type")
    if trainer_type is None:
        trainer_type = "mil" if default_loss_config["type"] in ("MIDAMLoss", "MultiLabelMIDAMLoss") else "image"
    trainer_type = str(trainer_type).lower()

    # Hyperparameters of the trainer itself and of the model config (any value
    # may be a search spec, e.g. model.query_heads: {val: [2, 4, 8]}).
    trainer_hps = {}
    if trainer_type == "mil":
        trainer_cls = MILTrainer
        if dataset_kind == "medical_study_csv":
            trainer_hps["num_slices"] = trainer_cfg.get("num_slices", 16)
        else:
            trainer_hps["instance_batch_size"] = trainer_cfg.get(
                "instance_batch_size", training_cfg.get("instance_batch_size", 4))
    elif trainer_type == "image":
        trainer_cls = Trainer
    else:
        raise ValueError(f"Unknown trainer.type '{trainer_type}'. Choose 'image' or 'mil'.")

    trainer_kwargs = {}
    if any(isinstance(v, dict) for v in model_cfg.values()):
        trainer_hps["model_cfg"] = autopartial(dict, **parse_hyperparameters_from_dict(
            {k: v for k, v in model_cfg.items() if v is not None}))
    else:
        trainer_kwargs["model_cfg"] = model_cfg

    logger.info(f"Initialising {trainer_cls.__name__} (autopartial)...")
    trainer = autopartial(
        partial(
            trainer_cls,
            train_dataset = train_dataset,
            eval_dataset  = eval_datasets if eval_datasets else None,
            metric        = metric_fn,
            callbacks     = [CLICallback(), *extra_callbacks],
            **trainer_kwargs,
        ),
        train_args = train_args,
        **{k: v if hasattr(v, "autofunc") else parse_hyperparameters_from_dict({k: v})[k]
           for k, v in trainer_hps.items()},
    )

    return trainer, automax_args, metric_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load config (megaconf) and apply CLI overrides
    logger.info(f"Loading config from: {args.config_file}")
    cfg = build_megaconf(args.config_file, _MEGACONF_DEFAULTS)
    cfg = apply_cli_overrides(cfg, args)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    # Under torchrun every rank runs the same search in lockstep and each
    # trial trains with DDP over all ranks (no-op for a plain launch).
    init_distributed()
    try:
        # 2. Build the search pipeline and run it
        trainer, automax_args, metric_names = build_pipeline(cfg_dict)

        if not is_distributed() or torch.distributed.get_rank() == 0:
            print("\n[Hyperparameter Space]")
            for value in trainer.cs.values():
                print(f"  {value}")

        tuner = AutoMAX(trainer, automax_args, target=metric_names[0])
        tuner.optimize()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()