"""
AutoMAX optimization framework.
"""
import glob
import json
import os
import sys
import time
import traceback
import logging
from typing import Dict, Any, Optional
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign
from .configuration import AutoMAXConfigration
import shutil
from smac.runhistory import StatusType, TrialInfo, TrialValue
import pickle

import torch.distributed as dist

from .parallel import TrialPool, make_slots, run_trial

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

__all__ = ["AutoMAX"]


class AutoMAX:
    """
    Automated AUC optimization using SMAC3.

    This class handles hyperparameter optimization for AUC-based metrics
    using the SMAC3 framework.

    Trials run one at a time in the calling process by default. Set
    ``config.n_workers`` to evaluate several configurations concurrently and
    ``config.gpus_per_trial`` to train each one with DDP on several GPUs; see
    :mod:`libauc.auto.parallel`.

    Launched under ``torchrun`` (a process group exists when AutoMAX is
    built), every rank runs the same seeded search in lockstep and each trial
    trains with DDP across all ranks; rank 0 owns the checkpoints and logs.

    ``config.n_trials`` counts successful trials. A crashed trial is restarted
    up to ``config.trial_retries`` times (resuming from its checkpoints when
    the recipe sets ``resume_from_checkpoint``) and then recorded as crashed,
    which doesn't use up the budget; so a search resumed after crashes keeps
    its ``n_trials`` (changing it would make SMAC stop and ask whether to
    overwrite the run). The search gives up once ``n_trials`` trials crashed.
    """

    def __init__(self, trainer, config: AutoMAXConfigration, target: str):
        """
        Initialize AutoAUC optimizer.

        Args:
            trainer: Trainer function/class for training models
            config: AutoAUC configuration
            data: Training data configuration
            model: Model to optimize
            target: Target metric to optimize
        """
        # Setup configuration space
        self.configspace = trainer.cs
        self.trainer = trainer
        self.target = target
        self.config = config
        self.log = []
        self.top = []        # kept trials, best first (keep_top_k > 1)

        self.distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        if self.distributed and config.is_parallel:
            raise ValueError("n_workers / gpus_per_trial > 1 cannot be combined with a torchrun launch; "
                             "under torchrun each trial already trains on every rank.")
        # Each rank keeps its own SMAC state so they don't write the same files.
        self.smac_directory = config.output_directory
        if self.rank != 0:
            self.smac_directory = os.path.join(config.output_directory, f".rank_{self.rank}")

        # Setup SMAC scenario. Parallelism is handled by AutoMAX itself
        # (ask/tell + TrialPool), so SMAC always sees a single worker.
        scenario = Scenario(
            self.configspace,
            name=config.name,
            output_directory=self.smac_directory,
            n_trials=config.n_trials,
            deterministic=config.deterministic,
            seed=config.SEED
        )

        initial_design = DefaultInitialDesign(scenario)

        # Initialize SMAC
        self.smac = HyperparameterOptimizationFacade(
            scenario,
            self.train,
            initial_design=initial_design,
            overwrite=config.overwrite,
        )

        self.finished = self.smac.runhistory.finished

        self.best_score = float('-inf')
        if not config.overwrite and self.finished > 0:
            filepath = self._state_path()
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    self.log = pickle.load(f)
            self.best_score = max(
                (entry["val"] for entry in self.log if entry.get("val") == entry.get("val")),
                default=float('-inf'),
            )
            self.top = sorted(
                ({"trial": e["trial"], "score": float(e["val"]), "dir": e["trial_dir"], "space": dict(e["space"])}
                 for e in self.log if e.get("trial_dir") and e.get("val") == e.get("val") and os.path.exists(e["trial_dir"])),
                key=lambda e: -e["score"])[:getattr(config, "keep_top_k", 1)]

        assert len(self.log) == self.finished
        self._crashed_before = self.crashed
        self._reported_give_up = False

    @property
    def completed(self) -> int:
        """Trials that finished with a score; only these count toward ``n_trials``."""
        return sum(1 for entry in self.log if entry.get("val") == entry.get("val"))

    @property
    def crashed(self) -> int:
        return len(self.log) - self.completed

    @property
    def _gave_up(self) -> bool:
        # Only this run's crashes count: a search resumed after the environment was fixed should go on.
        return self.crashed - self._crashed_before >= self.config.n_trials

    def _budget_left(self, pending: int = 0) -> bool:
        if self._gave_up:
            if self._is_main and not self._reported_give_up:
                self._reported_give_up = True
                logger.error(f"{self.crashed - self._crashed_before} trials crashed in this run; stopping the search "
                             f"(see the crash logs above).")
            return False
        return self.completed + pending < self.config.n_trials

    def _state_path(self) -> str:
        return os.path.join(self.smac_directory, self.config.name, "state.pkl")

    @property
    def _is_main(self) -> bool:
        return self.rank == 0

    def _trial_name(self, config) -> str:
        # SMAC assigns the config id when the trial is asked, and it is stable
        # across resumes, so an interrupted trial finds its own checkpoints.
        return f"trial{self.smac.runhistory.get_config_id(config)}"

    def train(self, space, seed: int = 0) -> float:
        """
        Train a model with given hyperparameters.

        Args:
            space: Hyperparameter configuration
            seed: Random seed for reproducibility

        Returns:
            Negative score (for minimization)
        """
        trial_name = self._trial_name(space)
        if self.distributed:
            configs = [None] * dist.get_world_size()
            dist.all_gather_object(configs, dict(space))
            if any(c != configs[0] for c in configs[1:]):
                raise RuntimeError("AutoMAX ranks selected different configurations.")
        if self._is_main:
            logger.info(f"Starting {trial_name} ({self.completed + 1}/{self.config.n_trials}) with configuration: {space}.")
        result = run_trial(self.trainer, space, trial_name, getattr(self.config, "seeds_per_trial", 1))
        return self._record_result(space, trial_name, result)

    def _record_result(self, space, trial_name: str, result: Dict[str, Any]) -> float:
        """Log a finished trial, keep its checkpoints if it is the best so far."""
        trail_log = dict(result["summary"])
        trail_log["space"] = space
        trail_log["trial"] = trial_name

        score = trail_log['val']
        if self.distributed:
            # Every rank must tell SMAC the same cost to stay in lockstep.
            shared = [score]
            dist.broadcast_object_list(shared, src=0)
            score = trail_log['val'] = shared[0]
        trial_dir, best_dir = result["trial_dir"], result["best_dir"]
        trail_log["trial_dir"] = trial_dir
        if "seed_summaries" in result:
            trail_log["seed_summaries"] = result["seed_summaries"]

        if getattr(self.config, "keep_top_k", 1) > 1:
            if score > self.best_score:
                self.best_score = score
                if self._is_main:
                    logger.info(f"Found new best configuration ({trial_name}, val {self.target} = {score:.6f})!")
            if self._is_main:
                self._update_top_k(trial_name, score, trial_dir, best_dir, space)
        elif score > self.best_score:
            if self._is_main:
                logger.info(f"Found new best configuration ({trial_name}, val {self.target} = {score:.6f})! Updating the best checkpoint.")
            self.best_score = score
            if self._is_main:
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                if os.path.exists(trial_dir):
                    os.rename(trial_dir, best_dir)
        elif self._is_main and os.path.exists(trial_dir):
            # Only the best trial's checkpoints are kept.
            shutil.rmtree(trial_dir)

        self._append_log(trail_log)
        if self.distributed:
            dist.barrier()
        return -score  # Return negative for minimization

    def _update_top_k(self, trial_name, score, trial_dir, best_dir, space):
        """Keep the checkpoint folders of the k best trials, point <experiment>_best at the best one and
        write the ranking (with every seed's checkpoint) to <experiment>_top<k>.json."""
        k = self.config.keep_top_k
        if score == score and os.path.exists(trial_dir):
            self.top.append({"trial": trial_name, "score": float(score), "dir": trial_dir, "space": dict(space)})
        self.top.sort(key=lambda e: -e["score"])
        for entry in self.top[k:]:                       # fell out of the top k
            if os.path.exists(entry["dir"]):
                shutil.rmtree(entry["dir"])
        self.top = self.top[:k]
        if os.path.islink(best_dir):
            os.unlink(best_dir)
        elif os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        if self.top:
            os.symlink(os.path.relpath(self.top[0]["dir"], os.path.dirname(best_dir)), best_dir)

        def checkpoints(folder):
            best = sorted(glob.glob(os.path.join(folder, "**", "best.pt"), recursive=True))
            if best:
                return best
            latest = {}                                  # no early stopping: last epoch of each run
            for path in glob.glob(os.path.join(folder, "**", "epoch_*.pt"), recursive=True):
                epoch = int(os.path.basename(path)[6:-3])
                run = os.path.dirname(path)
                if epoch > latest.get(run, (-1, None))[0]:
                    latest[run] = (epoch, path)
            return sorted(p for _, p in latest.values())

        ranking = [{"rank": r + 1, "trial": e["trial"], "val": e["score"], "dir": e["dir"],
                    "space": {key: (value.item() if hasattr(value, "item") else value) for key, value in e["space"].items()},
                    "checkpoints": checkpoints(e["dir"])} for r, e in enumerate(self.top)]
        path = best_dir[: -len("_best")] + f"_top{k}.json"
        with open(path, "w") as f:
            json.dump(ranking, f, indent=1, default=str)
        logger.info(f"Top {k} trials: " + ", ".join(f"{e['trial']} ({e['score']:.4f})" for e in self.top) + f" -> {path}")

    def _record_crash(self, space, trial_name: str, error: str):
        self._append_log({"val": float('nan'), "space": space, "trial": trial_name, "error": error})

    def _append_log(self, entry: Dict[str, Any]):
        self.finished += 1
        self.log.append(entry)
        with open(self._state_path(), "wb") as f:
            pickle.dump(self.log, f)

    def optimize(self):
        """
        Run the optimization process.

        Returns:
            Best configuration found
        """

        if self.config.is_parallel:
            incumbent = self._optimize_parallel()
        else:
            if self.config.devices:
                import torch
                torch.cuda.set_device(int(self.config.devices[0]))

            incumbent = self._optimize_serial()
        if not self._is_main:
            return incumbent
        logger.info(f"Best configuration found: {incumbent}")

        # Print trial history as a table
        has_test = any("test" in entry for entry in self.log)
        header = f"{'Trial':<8} {'Val ' + self.target:<20}" + (f" {'Test ' + self.target:<20}" if has_test else "")
        separator = "-" * len(header)
        print("\nTrial History:")
        print(separator)
        print(header)
        print(separator)
        for i, entry in enumerate(self.log, 1):
            val = f"{entry.get('val', float('nan')):.6f}"
            row = f"{i:<8} {val:<20}"
            if has_test:
                test = f"{entry.get('test', float('nan')):.6f}"
                row += f" {test:<20}"
            print(row)
        print(separator)

        return incumbent

    def _optimize_serial(self):
        """Ask/tell loop running one trial at a time in this process."""
        # Trials left running by an interrupted search are re-run first.
        resubmit = list(self.smac.runhistory.get_running_trials())
        # Under torchrun a rank-local failure leaves the process group unusable, so don't retry there.
        retries = 0 if self.distributed else self.config.trial_retries
        while resubmit or self._budget_left():
            info = resubmit.pop(0) if resubmit else self.smac.ask()
            trial_name = self._trial_name(info.config)
            start = time.time()
            for attempt in range(retries + 1):
                try:
                    cost, status = self.train(info.config, seed=info.seed), StatusType.SUCCESS
                    break
                except Exception:
                    error = traceback.format_exc()
                    if attempt < retries:
                        logger.warning(f"{trial_name} crashed; restarting it (retry {attempt + 1}/{retries}):\n{error}")
                        continue
                    cost, status = self.smac.scenario.crash_cost, StatusType.CRASHED
                    self._record_crash(info.config, trial_name, error)
                    logger.error(f"{trial_name} crashed:\n{error}")
            self.smac.tell(info, TrialValue(cost=cost, time=time.time() - start, status=status))
        return self.smac.intensifier.get_incumbent()

    def _optimize_parallel(self):
        """Ask/tell loop running up to ``n_workers`` trials concurrently."""
        cfg = self.config
        slots = make_slots(cfg.devices, cfg.n_workers, cfg.gpus_per_trial)
        log_dir = os.path.join(cfg.output_directory, cfg.name, "trial_logs")
        pool = TrialPool(self.trainer, slots, log_dir=log_dir, start_method=cfg.start_method,
                         seeds_per_trial=getattr(cfg, "seeds_per_trial", 1))
        logger.info(
            f"Parallel search: {len(slots)} concurrent trial(s), {cfg.gpus_per_trial} GPU(s) each, "
            f"GPU slots {slots}, start method '{pool.start_method}'. Per-trial logs in {log_dir}"
        )

        # Trials left running by an interrupted search are re-run first.
        resubmit = list(self.smac.runhistory.get_running_trials())
        pending: Dict[str, TrialInfo] = {}
        attempts: Dict[str, int] = {}

        try:
            while pending or resubmit or self._budget_left():
                while pool.has_free_slot() and (resubmit or self._budget_left(len(pending))):
                    info = resubmit.pop(0) if resubmit else self.smac.ask()
                    trial_name = self._trial_name(info.config)
                    gpus, log_path = pool.submit(trial_name, dict(info.config), trial_name)
                    pending[trial_name] = info
                    logger.info(f"Started {trial_name} on GPU(s) {gpus}: {dict(info.config)} (log: {log_path})")

                outcome = pool.wait()
                info = pending.pop(outcome.key)
                if not outcome.ok and attempts.get(outcome.key, 0) < cfg.trial_retries and not self._gave_up:
                    # Often an environment problem (out of memory, a wiped /dev/shm); the
                    # trial resumes from its own checkpoints when the recipe allows it.
                    attempts[outcome.key] = attempts.get(outcome.key, 0) + 1
                    logger.warning(f"{outcome.key} crashed after {outcome.elapsed:.0f}s (see {outcome.log_path}); "
                                   f"restarting it (retry {attempts[outcome.key]}/{cfg.trial_retries}):\n{outcome.error}")
                    resubmit.insert(0, info)
                    continue
                if outcome.ok:
                    cost = self._record_result(info.config, outcome.key, outcome.result)
                    status = StatusType.SUCCESS
                    summary = ", ".join(f"{k} {self.target} = {v:.6f}" for k, v in outcome.result["summary"].items())
                    logger.info(f"Finished {outcome.key} in {outcome.elapsed:.0f}s: {summary} "
                                f"[{self.completed}/{cfg.n_trials}]")
                else:
                    cost = self.smac.scenario.crash_cost
                    status = StatusType.CRASHED
                    self._record_crash(info.config, outcome.key, outcome.error)
                    logger.error(f"{outcome.key} crashed after {outcome.elapsed:.0f}s "
                                 f"(see {outcome.log_path}):\n{outcome.error}")
                self.smac.tell(info, TrialValue(cost=cost, time=outcome.elapsed, status=status))
        finally:
            pool.shutdown()

        return self.smac.intensifier.get_incumbent()
