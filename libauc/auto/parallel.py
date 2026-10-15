"""
Parallel and distributed trial execution for AutoMAX.

SMAC's own parallelism (``Scenario(n_workers=N)``) runs trials on a Dask
cluster, which has no notion of GPUs and cannot host multi-process DDP
trials. AutoMAX instead drives SMAC through its ask/tell interface and runs
each trial in a dedicated process group:

* the GPUs in ``AutoMAXConfigration.devices`` are split into ``n_workers``
  *slots* of ``gpus_per_trial`` GPUs each;
* for every trial SMAC hands out (``smac.ask()``), :class:`TrialPool` starts
  ``gpus_per_trial`` processes pinned to a free slot via
  ``CUDA_VISIBLE_DEVICES``; with more than one GPU they form a
  ``torch.distributed`` group and the Trainer trains with DDP;
* when a trial ends its score is returned to SMAC (``smac.tell()``) and the
  slot is reused for the next configuration.

SMAC keeps asked-but-not-told trials as "running" and proposes distinct
configurations for concurrent asks, so the search stays sequential-model-based
while up to ``n_workers`` trials train at once.
"""
import logging
import os
import pickle
import queue
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional, Tuple

import multiprocessing as mp

logger = logging.getLogger(__name__)

# Seconds to wait for the other ranks of a finished trial to exit before
# terminating them, and for terminated processes to die before SIGKILL.
_EXIT_GRACE = 120.0
_TERMINATE_GRACE = 10.0


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(trainer_factory, space, trial_name: str, seeds_per_trial: int = 1) -> Dict[str, Any]:
    """Build a Trainer for *space*, train it, and report where it saved.

    Shared by the in-process (serial) path and the trial worker processes.
    Each trial writes checkpoints to ``<output_path>/<experiment_name>_<trial_name>``
    so concurrent trials never collide and an interrupted trial can resume
    from its own checkpoints.

    With ``seeds_per_trial`` k > 1 the configuration is trained k times with
    seeds ``SEED .. SEED + k - 1`` (into ``.../seed<S>``) and the summary is
    the mean over seeds, a less noisy objective for the search. A seed that
    already finished (``summary.json`` present) is not retrained on resume.

    Returns:
        dict: ``summary`` (``state.train_summary``: ``val`` and optionally
        ``test``; the seed mean with k > 1), ``trial_dir`` and ``best_dir``
        (where AutoMAX keeps the best trial's checkpoints), and with k > 1
        ``seed_summaries`` ({seed: summary}).
    """
    if seeds_per_trial > 1:
        return _run_seeds(trainer_factory, space, trial_name, seeds_per_trial)
    trainer = trainer_factory(space)
    base_name = trainer.args.experiment_name
    output_path = trainer.args.output_path
    trainer.args.experiment_name = f"{base_name}_{trial_name}"
    trainer.train()
    return {
        "summary":   dict(trainer.state.train_summary),
        "trial_dir": os.path.join(output_path, trainer.args.experiment_name),
        "best_dir":  os.path.join(output_path, base_name + "_best"),
    }


def _run_seeds(trainer_factory, space, trial_name: str, k: int) -> Dict[str, Any]:
    import json
    if not hasattr(trainer_factory, "resolve"):
        raise ValueError("seeds_per_trial > 1 needs an autopartial trainer factory")
    seed_summaries, base_name, output_path = {}, None, None
    for i in range(k):
        args, kwds = trainer_factory.resolve(space)          # fresh arguments per seed
        train_args = kwds.get("train_args")
        if train_args is None or not hasattr(train_args, "SEED"):
            raise ValueError("seeds_per_trial > 1 needs a Trainer built with train_args (TrainingArguments)")
        base_name, output_path = train_args.experiment_name, train_args.output_path
        seed = int(train_args.SEED) + i
        train_args.SEED = seed
        train_args.experiment_name = os.path.join(f"{base_name}_{trial_name}", f"seed{seed}")
        done = os.path.join(output_path, train_args.experiment_name, "summary.json")
        if os.path.exists(done):                            # finished before an interruption
            with open(done) as f:
                seed_summaries[seed] = json.load(f)
            logger.info(f"{trial_name}: seed {seed} already finished, reusing {done}")
            continue
        trainer = trainer_factory.autofunc(*args, **kwds)
        trainer.train()
        summary = {k_: float(v) for k_, v in dict(trainer.state.train_summary).items()}
        seed_summaries[seed] = summary
        if getattr(trainer, "is_world_process_zero", True):
            os.makedirs(os.path.dirname(done), exist_ok=True)
            with open(done, "w") as f:
                json.dump(summary, f)
        del trainer
    keys = set.intersection(*(set(s) for s in seed_summaries.values()))
    mean = {k_: sum(s[k_] for s in seed_summaries.values()) / len(seed_summaries) for k_ in sorted(keys)}
    return {
        "summary":        mean,
        "seed_summaries": seed_summaries,
        "trial_dir":      os.path.join(output_path, f"{base_name}_{trial_name}"),
        "best_dir":       os.path.join(output_path, base_name + "_best"),
    }


def _trial_worker(trainer_factory, space, trial_name, key, result_queue, log_path, seeds_per_trial=1):
    """Entry point of one rank of one trial (runs in a child process).

    *trainer_factory* is the trainer itself (``fork``) or the path of its
    pickle (``spawn``/``forkserver``; see :class:`TrialPool`).
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Route this process's output (including C-level writes) to the trial log
    # so concurrent trials don't interleave progress bars on the console.
    log_file = open(log_path, "a", buffering=1)
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    sys.stdout = sys.stderr = log_file
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [rank {rank}] %(name)s - %(message)s",
        stream=log_file,
        force=True,
    )

    # A spawned child inherits "spawn" as its default start method, which
    # would make every DataLoader re-spawn its workers (re-importing torch
    # and re-pickling the dataset) each epoch -- ~7x slower epochs observed.
    # Restore the platform default that DataLoader workers normally use.
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)

    import torch
    from libauc.trainer import init_distributed, cleanup_distributed

    try:
        if isinstance(trainer_factory, str):
            with open(trainer_factory, "rb") as f:
                trainer_factory = pickle.load(f)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        if world_size > 1:
            init_distributed()
        result = run_trial(trainer_factory, space, trial_name, seeds_per_trial)
        if rank == 0:
            result_queue.put((key, rank, True, result))
    except BaseException:
        tb = traceback.format_exc()
        print(tb, flush=True)
        result_queue.put((key, rank, False, tb))
        raise SystemExit(1)
    finally:
        cleanup_distributed()


# ---------------------------------------------------------------------------
# Device slots
# ---------------------------------------------------------------------------

def _visible_devices() -> List[str]:
    """GPU identifiers visible to this process, in torch's logical order."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [d.strip() for d in env.split(",") if d.strip()]
    import torch
    # device_count() queries NVML and does not initialise CUDA, so it is
    # safe before forking trial processes.
    return [str(i) for i in range(torch.cuda.device_count())]


def make_slots(devices: Optional[List[int]], n_workers: int, gpus_per_trial: int) -> List[List[str]]:
    """Assign ``gpus_per_trial`` GPUs to each of ``n_workers`` concurrent trials.

    *devices* are logical GPU indices (as numbered by torch in this process,
    i.e. after any ``CUDA_VISIBLE_DEVICES`` of the parent); they are
    translated back to the parent's identifiers so that the child's
    ``CUDA_VISIBLE_DEVICES`` selects the right physical GPUs. When
    ``n_workers * gpus_per_trial`` exceeds the number of devices, slots wrap
    around and share GPUs.
    """
    visible = _visible_devices()
    if not visible:
        raise RuntimeError("AutoMAX parallel search requires at least one visible CUDA device.")
    if devices is None:
        pool = visible
    else:
        bad = [d for d in devices if not 0 <= int(d) < len(visible)]
        if bad:
            raise ValueError(f"devices {bad} out of range; {len(visible)} GPU(s) visible.")
        pool = [visible[int(d)] for d in devices]

    if gpus_per_trial > len(pool):
        raise ValueError(
            f"gpus_per_trial={gpus_per_trial} but only {len(pool)} device(s) available: {pool}"
        )
    if n_workers * gpus_per_trial > len(pool):
        logger.warning(
            f"{n_workers} concurrent trials x {gpus_per_trial} GPU(s) exceeds the "
            f"{len(pool)} available device(s); some GPUs will be shared between trials."
        )
    return [
        [pool[(i * gpus_per_trial + j) % len(pool)] for j in range(gpus_per_trial)]
        for i in range(n_workers)
    ]


# ---------------------------------------------------------------------------
# Trial pool
# ---------------------------------------------------------------------------

@dataclass
class TrialOutcome:
    """Result of one finished trial, returned by :meth:`TrialPool.wait`."""
    key: Hashable
    ok: bool
    result: Optional[Dict[str, Any]]   # run_trial() output when ok
    error: Optional[str]               # traceback / reason when not ok
    elapsed: float
    log_path: str


@dataclass
class _Running:
    slot: int
    procs: List[Any]
    log_path: str
    start: float
    result: Optional[Dict[str, Any]] = None
    result_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _patched_environ(env: Dict[str, str]):
    """Temporarily set env vars so a starting child process inherits them."""
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TrialPool:
    """Runs trials concurrently, one process group per trial, one trial per slot.

    Args:
        trainer_factory: ``autopartial`` Trainer; called with a configuration
            inside each trial process. Must be picklable for the ``spawn``
            start method.
        slots: GPU identifiers for each concurrent trial (see :func:`make_slots`).
        log_dir: Directory for per-trial log files.
        start_method: ``multiprocessing`` start method.

    With ``fork`` the trial processes inherit the trainer (datasets are
    shared copy-on-write, nothing is pickled). With ``spawn`` the trainer is
    pickled once to ``<log_dir>/trainer.pkl`` and each process loads it from
    there, keeping the payload sent through the spawn pipe small (CPython
    blocks forever writing a large payload to a child that dies early).
    """

    def __init__(self, trainer_factory, slots: List[List[str]], log_dir: str, start_method: str = "spawn",
                 seeds_per_trial: int = 1):
        self._slots = slots
        self._free = list(range(len(slots)))
        self._running: Dict[Hashable, _Running] = {}
        self._log_dir = log_dir
        self._seeds_per_trial = int(seeds_per_trial)
        os.makedirs(log_dir, exist_ok=True)

        self.start_method = start_method
        self._pickle_path = None
        if self.start_method == "fork":
            self._trainer_factory = trainer_factory
        else:
            self._pickle_path = os.path.join(log_dir, "trainer.pkl")
            self._dump_trainer(trainer_factory, self._pickle_path, self.start_method)
            self._trainer_factory = self._pickle_path
        self._ctx = mp.get_context(self.start_method)
        self._queue = self._ctx.Queue()

    @staticmethod
    def _dump_trainer(trainer_factory, path, start_method):
        try:
            with open(path, "wb") as f:
                pickle.dump(trainer_factory, f)
        except Exception as exc:
            raise TypeError(
                f"The trainer (its datasets, metric or callbacks) cannot be pickled for the "
                f"'{start_method}' start method: {exc}. Make them picklable (e.g. module-level "
                f"functions instead of lambdas) or use start_method='fork'."
            ) from exc

    # -- queries -------------------------------------------------------------

    @property
    def n_running(self) -> int:
        return len(self._running)

    def has_free_slot(self) -> bool:
        return bool(self._free)

    # -- submission ------------------------------------------------------------

    def submit(self, key: Hashable, space: Dict[str, Any], trial_name: str) -> Tuple[List[str], str]:
        """Start *space* on a free slot. Returns ``(gpus, log_path)``."""
        if not self._free:
            raise RuntimeError("No free slot; call wait() first.")
        slot = self._free.pop(0)
        gpus = self._slots[slot]
        world_size = len(gpus)
        log_path = os.path.join(self._log_dir, f"{trial_name}.log")
        port = _free_port()

        # Like torchrun, cap intra-op threads: every trial process would
        # otherwise start one OpenMP thread per core, and a few of them
        # oversubscribe the CPU badly (10x slower steps observed).
        n_procs = sum(len(s) for s in self._slots)
        omp_threads = os.environ.get("OMP_NUM_THREADS") or str(max(1, (os.cpu_count() or 1) // n_procs))

        procs = []
        for rank in range(world_size):
            env = {
                "OMP_NUM_THREADS":      omp_threads,
                "CUDA_VISIBLE_DEVICES": ",".join(gpus),
                "MASTER_ADDR":          "127.0.0.1",
                "MASTER_PORT":          str(port),
                "RANK":                 str(rank),
                "LOCAL_RANK":           str(rank),
                "WORLD_SIZE":           str(world_size),
                "LOCAL_WORLD_SIZE":     str(world_size),
            }
            proc = self._ctx.Process(
                target=_trial_worker,
                args=(self._trainer_factory, space, trial_name, key, self._queue, log_path, self._seeds_per_trial),
                name=f"automax-{trial_name}-rank{rank}",
                # Not a daemon: trial processes need their own DataLoader workers.
                daemon=False,
            )
            with _patched_environ(env):
                proc.start()
            procs.append(proc)

        self._running[key] = _Running(slot=slot, procs=procs, log_path=log_path, start=time.time())
        return gpus, log_path

    # -- completion ------------------------------------------------------------

    def wait(self, poll_interval: float = 1.0) -> TrialOutcome:
        """Block until any running trial finishes (or crashes) and return it."""
        if not self._running:
            raise RuntimeError("No running trials to wait for.")
        while True:
            self._drain_queue(timeout=poll_interval)
            for key, run in list(self._running.items()):
                outcome = self._check(key, run)
                if outcome is not None:
                    return outcome

    def _drain_queue(self, timeout: float):
        try:
            msg = self._queue.get(timeout=timeout)
        except queue.Empty:
            return
        while True:
            key, rank, ok, payload = msg
            run = self._running.get(key)
            if run is not None:
                if ok:
                    run.result, run.result_time = payload, time.time()
                else:
                    run.errors.append(f"[rank {rank}]\n{payload}")
            try:
                msg = self._queue.get_nowait()
            except queue.Empty:
                return

    def _check(self, key, run: _Running) -> Optional[TrialOutcome]:
        exitcodes = [p.exitcode for p in run.procs]
        all_exited = all(c is not None for c in exitcodes)
        failed = [c for c in exitcodes if c not in (None, 0)]

        if run.errors or failed:
            # One rank failing leaves the others blocked in collectives.
            self._terminate(run.procs)
            error = "\n".join(run.errors) or (
                f"trial process exited with code(s) {failed} before running the trial"
                + (" (with the spawn start method, the script that starts AutoMAX must be "
                   "guarded by `if __name__ == '__main__':`)" if self.start_method != "fork" else "")
            )
            return self._finish(key, run, ok=False, error=error)

        if run.result is not None:
            if not all_exited and time.time() - run.result_time > _EXIT_GRACE:
                logger.warning(f"Trial {key}: ranks did not exit after finishing; terminating.")
                self._terminate(run.procs)
                all_exited = True
            if all_exited:
                return self._finish(key, run, ok=True)
            return None

        if all_exited:
            # Clean exit but no result read yet: the message may still be in flight.
            self._drain_queue(timeout=0.5)
            if run.result is not None:
                return self._finish(key, run, ok=True)
            return self._finish(key, run, ok=False, error="trial processes exited without reporting a result")
        return None

    def _finish(self, key, run: _Running, ok: bool, error: Optional[str] = None) -> TrialOutcome:
        for p in run.procs:
            p.join(timeout=_TERMINATE_GRACE)
        del self._running[key]
        self._free.append(run.slot)
        return TrialOutcome(
            key=key,
            ok=ok,
            result=run.result if ok else None,
            error=error,
            elapsed=time.time() - run.start,
            log_path=run.log_path,
        )

    @staticmethod
    def _terminate(procs):
        for p in procs:
            if p.is_alive():
                p.terminate()
        deadline = time.time() + _TERMINATE_GRACE
        for p in procs:
            p.join(timeout=max(0.0, deadline - time.time()))
            if p.is_alive():
                p.kill()
                p.join()

    def shutdown(self):
        """Terminate every running trial (e.g. on Ctrl-C) and clean up."""
        for run in self._running.values():
            self._terminate(run.procs)
        self._running.clear()
        self._free = list(range(len(self._slots)))
        if self._pickle_path and os.path.exists(self._pickle_path):
            os.remove(self._pickle_path)

