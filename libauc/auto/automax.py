"""
AutoMAX optimization framework.
"""
import os
import sys
import time
import logging
from typing import Dict, Any, Optional
from smac import HyperparameterOptimizationFacade, Scenario
from smac.initial_design import DefaultInitialDesign
from .configuration import AutoMAXConfigration
import shutil
from smac.runhistory import StatusType, TrialInfo, TrialValue
import pickle

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

        # Setup SMAC scenario. Parallelism is handled by AutoMAX itself
        # (ask/tell + TrialPool), so SMAC always sees a single worker.
        scenario = Scenario(
            self.configspace,
            name=config.name,
            output_directory=config.output_directory,
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

        assert len(self.log) == self.finished

    def _state_path(self) -> str:
        return os.path.join(self.config.output_directory, self.config.name, "state.pkl")

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
        logger.info(f"Starting {trial_name} ({self.finished + 1}/{self.config.n_trials}) with configuration: {space}.")
        try:
            result = run_trial(self.trainer, space, trial_name)
        except Exception as exc:
            # SMAC records the trial as crashed; keep the log aligned with it.
            self._record_crash(space, trial_name, str(exc))
            raise
        return self._record_result(space, trial_name, result)

    def _record_result(self, space, trial_name: str, result: Dict[str, Any]) -> float:
        """Log a finished trial, keep its checkpoints if it is the best so far."""
        trail_log = dict(result["summary"])
        trail_log["space"] = space
        trail_log["trial"] = trial_name

        score = trail_log['val']
        trial_dir, best_dir = result["trial_dir"], result["best_dir"]

        if score > self.best_score:
            logger.info(f"Found new best configuration ({trial_name}, val {self.target} = {score:.6f})! Updating the best checkpoint.")
            self.best_score = score
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            if os.path.exists(trial_dir):
                os.rename(trial_dir, best_dir)
        elif os.path.exists(trial_dir):
            # Only the best trial's checkpoints are kept.
            shutil.rmtree(trial_dir)

        self._append_log(trail_log)
        return -score  # Return negative for minimization

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

            for trial_info in self.smac.runhistory.get_running_trials():
                # Re-run the trial
                cost = self.train(trial_info.config, seed=trial_info.seed)
                trial_value = TrialValue(cost=cost, time=0.0)
                self.smac.tell(trial_info, trial_value)

            incumbent = self.smac.optimize()
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

    def _optimize_parallel(self):
        """Ask/tell loop running up to ``n_workers`` trials concurrently."""
        cfg = self.config
        slots = make_slots(cfg.devices, cfg.n_workers, cfg.gpus_per_trial)
        log_dir = os.path.join(cfg.output_directory, cfg.name, "trial_logs")
        pool = TrialPool(self.trainer, slots, log_dir=log_dir, start_method=cfg.start_method)
        logger.info(
            f"Parallel search: {len(slots)} concurrent trial(s), {cfg.gpus_per_trial} GPU(s) each, "
            f"GPU slots {slots}, start method '{pool.start_method}'. Per-trial logs in {log_dir}"
        )

        # Trials left running by an interrupted search are re-run first.
        resubmit = list(self.smac.runhistory.get_running_trials())
        pending: Dict[str, TrialInfo] = {}

        try:
            while self.finished + len(pending) < cfg.n_trials or pending:
                while pool.has_free_slot() and self.finished + len(pending) < cfg.n_trials:
                    info = resubmit.pop(0) if resubmit else self.smac.ask()
                    trial_name = self._trial_name(info.config)
                    gpus, log_path = pool.submit(trial_name, dict(info.config), trial_name)
                    pending[trial_name] = info
                    logger.info(f"Started {trial_name} on GPU(s) {gpus}: {dict(info.config)} (log: {log_path})")

                outcome = pool.wait()
                info = pending.pop(outcome.key)
                if outcome.ok:
                    cost = self._record_result(info.config, outcome.key, outcome.result)
                    status = StatusType.SUCCESS
                    summary = ", ".join(f"{k} {self.target} = {v:.6f}" for k, v in outcome.result["summary"].items())
                    logger.info(f"Finished {outcome.key} in {outcome.elapsed:.0f}s: {summary} "
                                f"[{self.finished}/{cfg.n_trials}]")
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
