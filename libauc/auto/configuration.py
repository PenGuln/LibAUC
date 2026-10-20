from typing import List, Optional


class AutoMAXConfigration:
    r"""
    Configuration container for an AutoMAX hyperparameter-optimisation run.

    Args:
        deterministic (bool): Whether to enforce deterministic behaviour.
        n_trials (int): Number of trials that must finish successfully.
            Crashed trials don't count, so resuming an interrupted search
            never needs a larger ``n_trials``.
        n_configs (int): Number of configurations evaluated per trial.
        SEED (int): Random seed used by the search algorithm (default: ``0``).
        name (str): Human-readable name for the search run.
        seeds_per_trial (int): Seeds each configuration is trained with; SMAC
            sees the mean validation score (default: ``1``).
        keep_top_k (int): Number of best trials whose checkpoints are kept,
            listed in ``<experiment>_top<k>.json`` (default: ``1``).
        trial_retries (int): How many times a crashed trial is restarted,
            resuming from its own checkpoints, before it is recorded as
            crashed (default: ``1``; ``0`` disables retries).
        output_directory (str): Directory where AutoMAX results are written
            (default: ``"./automax_output"``). ``output_path`` is accepted as
            a fallback alias, since shipped recipes historically wrote
            ``automax.output_path`` in YAML even though the code only ever
            read ``output_directory``.
        overwrite (bool): Whether to overwrite an existing output directory.
        n_workers (int): Number of trials evaluated concurrently
            (default: ``1``). When greater than the number of GPU groups
            available, groups are shared round-robin (e.g. two small trials
            on one GPU).
        gpus_per_trial (int): Number of GPUs each trial trains on with
            ``DistributedDataParallel`` (default: ``1``). ``training.batch_size``
            is then the per-GPU batch size.
        devices (list[int], optional): GPU ids the search may use. Defaults to
            every GPU visible to the process.
        start_method (str): ``multiprocessing`` start method for trial
            processes: ``"spawn"`` (default), ``"forkserver"`` or ``"fork"``.
            With ``"spawn"`` the trainer (datasets, metric, callbacks) must be
            picklable and a script that starts a parallel search must be
            guarded by ``if __name__ == "__main__":``. ``"fork"`` needs
            neither, but only works if the calling process has not touched
            CUDA at all -- even ``torch.cuda.is_available()`` initialises the
            driver and makes CUDA unusable in forked children.

    With the defaults (``n_workers=1``, ``gpus_per_trial=1``) every trial runs
    in the calling process, exactly as before. Otherwise each trial runs in
    its own process group pinned to its GPUs (see :mod:`libauc.auto.parallel`).

    Example::

        >>> automax_cfg = AutoMAXConfigration(
        ...     deterministic=True,
        ...     n_trials=50,
        ...     n_configs=10,
        ...     SEED=0,
        ...     name="my_search",
        ...     output_directory="./automax_output",
        ...     overwrite=False,
        ...     n_workers=4,        # 4 trials at a time ...
        ...     gpus_per_trial=2,   # ... each on 2 GPUs (DDP) -> 8 GPUs
        ... )
    """

    def __init__(self, **kwargs):
        self.deterministic     = kwargs.pop("deterministic")
        self.n_trials          = kwargs.pop("n_trials")
        self.n_configs         = kwargs.pop("n_configs")
        self.SEED              = kwargs.pop("SEED", 0)
        self.name              = kwargs.pop("name")
        self.output_directory  = (
            kwargs.pop("output_directory", None)
            or kwargs.pop("output_path", None)
            or "./automax_output"
        )
        self.overwrite         = kwargs.pop("overwrite")

        # ── Parallel / distributed search ────────────────────────────────────
        self.n_workers         = int(kwargs.pop("n_workers", None) or 1)
        self.gpus_per_trial    = int(kwargs.pop("gpus_per_trial", None) or 1)
        self.devices: Optional[List[int]] = kwargs.pop("devices", None)
        self.start_method      = kwargs.pop("start_method", None) or "spawn"

        # ── Robust selection ─────────────────────────────────────────────────
        # seeds_per_trial: train each configuration with this many seeds and
        # report their mean to SMAC. keep_top_k: keep the checkpoints of the
        # k best trials (and a ranked <experiment>_top<k>.json) for ensembling.
        self.seeds_per_trial   = int(kwargs.pop("seeds_per_trial", None) or 1)
        self.keep_top_k        = int(kwargs.pop("keep_top_k", None) or 1)
        if self.seeds_per_trial < 1:
            raise ValueError(f"seeds_per_trial must be >= 1, got {self.seeds_per_trial}")
        # ── Crash handling ───────────────────────────────────────────────────
        retries = kwargs.pop("trial_retries", None)
        self.trial_retries     = 1 if retries is None else int(retries)
        if self.trial_retries < 0:
            raise ValueError(f"trial_retries must be >= 0, got {self.trial_retries}")
        if self.keep_top_k < 1:
            raise ValueError(f"keep_top_k must be >= 1, got {self.keep_top_k}")

        if self.n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {self.n_workers}")
        if self.gpus_per_trial < 1:
            raise ValueError(f"gpus_per_trial must be >= 1, got {self.gpus_per_trial}")
        if self.start_method not in ("spawn", "fork", "forkserver"):
            raise ValueError(f"Unknown start_method '{self.start_method}'")

    @property
    def is_parallel(self) -> bool:
        """``True`` when trials run in separate worker processes."""
        return self.n_workers > 1 or self.gpus_per_trial > 1


# Correctly spelled alias; ``AutoMAXConfigration`` is kept for compatibility.
AutoMAXConfiguration = AutoMAXConfigration
