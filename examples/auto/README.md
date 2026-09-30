# AutoMAX

AutoMAX (`libauc.auto`) is an automated hyperparameter optimization framework for AUC-focused objectives, built on LibAUC and [SMAC3](https://github.com/automl/SMAC3). It used to be a separate library ([Optimization-AI/AutoMAX](https://github.com/Optimization-AI/AutoMAX)) and is now part of LibAUC.

Instead of manually sweeping learning rates, margins, and decay schedules, AutoMAX performs Bayesian search over predefined hyperparameter spaces for each supported loss/optimizer pair, then returns the best configuration for your dataset. Trials can run several at a time, and each trial can train on several GPUs with DistributedDataParallel.

Full documentation: [AutoMAX Documentation](https://optimization-ai.github.io/AutoMAX_doc/)

## Installation

```bash
pip install "libauc[auto]"          # search (SMAC3)
pip install "libauc[auto,gui]"      # + desktop GUI (Flet)
```

SMAC3 needs `swig` and a C++ compiler to build. With conda: `conda install gxx_linux-64 gcc_linux-64 swig`.

## Quick Start

Run a search for AUROC on CIFAR-10 with ResNet-18:

```bash
python -m libauc.auto.run_auto_trainer --config_file examples/auto/recipes/config_auc.yaml
```

Any `training` field can be overridden from the command line:

```bash
python -m libauc.auto.run_auto_trainer --config_file examples/auto/recipes/config_auc.yaml --epochs 50 --output_path ./runs
```

For graph-classification datasets (OGB `ogbg-*`) with a GNN model, use `libauc.auto.run_auto_gnn` instead. It accepts the same config sections (`dataset`/`model`/`metrics`/`training`/`automax`) and CLI overrides, with a GNN `model` (e.g. `gcn`) in place of an image backbone.

## Parallel and distributed search

By default trials run one after another on one GPU. Two settings in the `automax` section (or on the command line) change that:

| Setting | CLI flag | Meaning |
|---|---|---|
| `n_workers` | `--n_workers` | Number of trials evaluated at the same time (default `1`). |
| `gpus_per_trial` | `--gpus_per_trial` | GPUs per trial (default `1`). Above 1, each trial trains with DistributedDataParallel and `training.batch_size` is the **per-GPU** batch size. |
| `devices` | `--devices` | GPU ids to use, e.g. `[0, 1, 2, 3]` (default: all visible GPUs). |
| `start_method` | – | `spawn` (default), `forkserver` or `fork`; see below. |

```bash
# 8 GPUs: 8 trials at a time, one GPU each
python -m libauc.auto.run_auto_trainer --config_file cfg.yaml --n_workers 8

# 8 GPUs: 4 trials at a time, each trained with DDP on 2 GPUs
python -m libauc.auto.run_auto_trainer --config_file cfg.yaml --n_workers 4 --gpus_per_trial 2

# Only GPUs 2 and 3
python -m libauc.auto.run_auto_trainer --config_file cfg.yaml --n_workers 2 --devices 2 3
```

```yaml
automax:
  n_trials: 40
  n_workers: 4
  gpus_per_trial: 2
  devices: [0, 1, 2, 3, 4, 5, 6, 7]
```

How it works: AutoMAX drives SMAC through its ask/tell interface. Whenever a GPU slot is free it asks SMAC for a new configuration and starts that trial in its own process group pinned to the slot's GPUs (`CUDA_VISIBLE_DEVICES`). When a trial finishes, its validation score goes back to SMAC and the slot takes the next configuration. SMAC keeps unfinished trials as "running" and proposes a different configuration for each concurrent request. SMAC's built-in `n_workers` (Dask) is not used because it cannot assign GPUs or host multi-process DDP trials.

- Each trial logs to `<output_directory>/<name>/trial_logs/trial<N>.log`. The console shows one line per trial start and finish.
- Each trial checkpoints to `<output_path>/<experiment_name>_trial<N>`. When a trial finishes, the best trial so far is moved to `<experiment_name>_best` and the other trials' checkpoints are deleted.
- A crashed trial (exception, out of memory, a failing DDP rank) is reported to SMAC as crashed, and the search continues.
- An interrupted search resumes with `overwrite: false` (keep the other settings unchanged). Trials that were running are re-run first. With `resume_from_checkpoint: true`, they continue from their own last checkpoint.
- If `n_workers × gpus_per_trial` exceeds the number of devices, GPUs are shared between trials. This is useful for small models.

### Python API

```python
from libauc.auto import AutoMAX, AutoMAXConfiguration
from libauc.auto.run_auto_trainer import build_pipeline

def main():
    trainer, automax_cfg, metrics = build_pipeline(cfg_dict)  # or build your own autopartial Trainer
    automax_cfg.n_workers, automax_cfg.gpus_per_trial = 4, 2
    AutoMAX(trainer, automax_cfg, target=metrics[0]).optimize()

if __name__ == "__main__":   # required by the default "spawn" start method
    main()
```

With the default `spawn` start method, the trainer (its datasets, metric and callbacks) is pickled once and loaded by each trial process. It must therefore be picklable: use module-level classes and functions rather than lambdas. `start_method: fork` avoids pickling, but it only works if the calling process has not touched CUDA, and even `torch.cuda.is_available()` counts.

### Supported loss/optimizer pairs for multi-GPU trials

DDP averages model gradients, but losses with auxiliary state must also be synchronized by LibAUC's distributed-aware losses and optimizers (see `examples/20_Distributed_Training_*`). Supported pairs: `AUCMLoss`/`PESG`, `APLoss`/`SOAP`, `pAUC_CVaR_Loss`/`SOPA`, `pAUC_DRO_Loss`/`SOPAs`, `tpAUC_KL_Loss`/`SOTAs`, `tpAUC_CVaR_loss`/`STACO`, `NDCGLoss`/`SONG`, plus `BCELoss`/`CrossEntropyLoss`. Other pairs still run with a warning, but their loss state is not synchronized across GPUs. Single-GPU trials (`gpus_per_trial: 1`) support every pair, at any `n_workers`.

The same DDP support is available to plain LibAUC training without a search:

```bash
torchrun --nproc_per_node 4 -m libauc.trainer.run_image_trainer --config_file cfg.yaml
```

## GUI

AutoMAX also ships a desktop GUI (built with [Flet](https://flet.dev)) for configuring and launching a search without writing a YAML recipe by hand, plus a session manager for tracking, monitoring, and cancelling runs.

```bash
python -m libauc.auto.ui.gui_main
```

The main window lets you:

- Pick a backend (Image Classification or Graph/GNN), a model, and a dataset from the supported choices — or select **HuggingFace Model...** from the model dropdown and type any HF Hub repo ID (e.g. `google/vit-base-patch16-224`) to load a custom model instead.
- For predefined datasets, set a **Root Path** (type it or use the folder-browse button) pointing at your local copy of the chosen dataset — it starts blank on purpose, so loading fails with a clear message if it's not set; the path is remembered per dataset across restarts.
- Select **Custom Dataset...** to load your own CSV-backed image dataset (image backend only): point **Dataset Root Path** (type it or browse) at a folder containing `train.csv` + `val.csv`/`valid.csv` (and optionally `test.csv`) — they're auto-detected by filename, no need to type each path — and set image/label column names. Image paths in the CSV are resolved relative to that same Root Path. Loaded via LibAUC's `MedicalImageCSVDataset`. Click **Save Custom Dataset** to register it under a name of your choice (must not collide with a predefined or already-added custom dataset name) so it can be reused later just by retyping that name. Custom graph datasets are not yet supported.
- Pick a target metric (AUROC/AUPRC/ACC, with optional partial-AUROC `max_fpr`/`min_tpr`) and a loss function — the paired optimizer and both hyperparameter search spaces are populated automatically.
- Tune the search space per hyperparameter (continuous range with optional log scale, categorical choices, or a fixed value) and set training/search parameters (epochs, batch size, number of trials, etc.).
- Start training, which runs in an independent subprocess — closing the GUI does not stop it.

Click **Monitor Sessions** to see all past and running sessions, live training-curve/metric charts, an ROC curve for the best trial, per-session logs, and controls to cancel or delete a session.

Sessions, logs, and per-session outputs are stored under `gui_sessions/` (created on first run; not needed for the plain CLI workflow above). Remembered dataset root paths and saved custom datasets are stored in `gui_sessions/dataset_config.json`.

The GUI runs one trial at a time. For parallel or multi-GPU search, use the CLI.

## Supported Loss / Optimizer Pairs

The supported loss and optimizer pairs can be found in the [LibAUC documentation](https://docs.libauc.org/api/libauc.optimizers.html).
