import logging
import torch
import numpy as np
from libauc.sampler import DualSampler, TriSampler, DistributedDualSampler, DistributedTriSampler
import os
from typing import Callable, List, Optional, Mapping
from torch.utils.data import Dataset
import importlib
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from libauc.utils import is_distributed

from ..config.args import TrainingArguments
from ..helpers import set_seed
from .callbacks import CallbackHandler, TrainerCallback, TrainerState
from .distributed import ShardSampler, all_reduce_mean, gather_shards

logger = logging.getLogger(__name__)

# Keys that identify classification-head parameters in both built-in libauc
# models (fc / linear) and HuggingFace models (classifier / head).
_HEAD_PARTS: frozenset = frozenset({"fc", "linear", "classifier", "head"})

# Losses that need no cross-rank synchronisation beyond DDP's gradient
# averaging (AUCMLoss's a/b are all-reduced by PESG).
_DDP_NATIVE_LOSSES: frozenset = frozenset({"AUCMLoss", "BCELoss", "CrossEntropyLoss"})
# Optimizers that update loss parameters without all-reducing them.
_DDP_UNSYNCED_OPTIMIZERS: frozenset = frozenset({"PDSCA"})

class _ChannelExpand:
    """DataLoader collate that expands single-channel images to 3 channels.

    Used when the dataset contains grayscale images (1 channel) but the
    model expects RGB input (3 channels).  The single channel is repeated
    three times: ``(N, 1, H, W)`` → ``(N, 3, H, W)``.

    Defined at module level for DataLoader worker pickling.
    """

    def __call__(self, batch):
        from torch.utils.data import default_collate
        images, targets, indices = default_collate(batch)
        images = images.expand(-1, 3, -1, -1).contiguous()
        return images, targets, indices


class _HFCollate:
    """DataLoader collate function that applies an ``AutoImageProcessor`` to a batch.

    Defined at module level so it is picklable by DataLoader worker processes.

    Image preprocessing (resize + model-specific normalisation) happens here,
    on CPU inside a DataLoader worker, **before** the batch is transferred to
    the GPU.  This avoids the costly GPU→CPU→GPU round-trip that occurs when
    preprocessing runs inside :meth:`~_HFModelWrapper.forward`.

    Single-channel (grayscale) inputs are automatically expanded to 3 channels
    before the processor is called, so the dataset requires no special handling.

    .. note::
        Assumes images are ``float32`` tensors in the ``[0, 1]`` range (i.e.
        the output of ``torchvision.transforms.ToTensor()``).
        ``do_rescale=False`` is set to avoid multiplying by 1/255 again.

    Args:
        processor: An ``AutoImageProcessor`` instance.
    """

    def __init__(self, processor) -> None:
        self.processor = processor

    def __call__(self, batch):
        from torch.utils.data import default_collate
        images, targets, indices = default_collate(batch)
        # Expand grayscale (1 ch) to RGB (3 ch) before the processor.
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1).contiguous()
        out = self.processor(images=images, return_tensors="pt", do_rescale=False)
        return out["pixel_values"], targets, indices


class _HFModelWrapper(torch.nn.Module):
    """Thin wrapper so HuggingFace image-classification models return raw logits.

    HuggingFace models return a ``SequenceClassifierOutput`` (or similar
    ``ModelOutput``) whose ``.logits`` field holds the raw class scores.
    This wrapper unpacks that so the rest of the Trainer sees a plain
    :class:`torch.Tensor`, identical in shape to a built-in libauc backbone.

    Image preprocessing (resize + normalise) is handled upstream by
    :class:`_HFCollate` in the DataLoader worker.  By the time
    :meth:`forward` is called the tensor is already in the model's expected
    format, so no CPU/GPU transfers are needed here.

    Args:
        hf_model (torch.nn.Module): Any ``transformers`` model that accepts
            a ``pixel_values`` keyword argument and returns an object with a
            ``.logits`` attribute (e.g. loaded via
            ``AutoModelForImageClassification``).
    """

    def __init__(self, hf_model: torch.nn.Module) -> None:
        super().__init__()
        self.hf_model = hf_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits as a plain tensor."""
        return self.hf_model(pixel_values=x).logits


class Trainer:
    r"""
    Full training loop for image-classification models supported by libauc.

    ``Trainer`` wires together a model, an AUC-specific loss function, a libauc
    optimizer, dual/tri-sampled data loaders, and an optional evaluation
    pipeline behind a unified :meth:`train` entry point.  Progress is
    surfaced through a :class:`~trainer.core.callbacks.CallbackHandler` so
    any number of :class:`~trainer.core.callbacks.TrainerCallback` subclasses
    can observe or alter the training loop without touching ``Trainer``
    internals.

    The class is intentionally thin: heavy lifting (data loading, model
    construction, loss/optimizer instantiation) is delegated to private
    helpers so subclasses like :class:`~trainer.core.graph_trainer.GraphTrainer`
    can override only the parts they need.

    Distributed training: when a ``torch.distributed`` process group with
    more than one rank is initialised before construction (e.g. under
    ``torchrun``, see :func:`~libauc.trainer.init_distributed`), the model is
    wrapped in ``DistributedDataParallel``, training uses
    :class:`~libauc.sampler.DistributedDualSampler` /
    :class:`~libauc.sampler.DistributedTriSampler` with
    ``train_args.batch_size`` as the **per-GPU** batch size, evaluation is
    sharded across ranks and gathered, and only rank 0 logs and writes
    checkpoints. Callbacks run on every rank; use
    ``state.is_world_process_zero`` to restrict side effects to rank 0.

    Args:
        train_args (TrainingArguments): Fully populated training configuration
            produced by :class:`~trainer.config.args.TrainingArguments`.
        model_cfg (dict): Architecture config forwarded to
            :meth:`_build_model`.  Must contain at least a ``"name"`` key.
            Built-in architectures: ``resnet20``, ``resnet18``,
            ``densenet121``.  Any HuggingFace model repo ID containing
            a ``/`` is also accepted (e.g.
            ``"google/vit-base-patch16-224"``,
            ``"openai/clip-vit-base-patch32"``).  For HF models the
            ``AutoImageProcessor`` is applied inside the trainer — the
            dataset needs no HF-specific transforms.
        train_dataset (Dataset): PyTorch ``Dataset`` for the training split.
            Must expose a ``.targets`` attribute (list or array of labels).
        eval_dataset (list[Dataset], optional): One or more evaluation
            datasets.  ``None`` disables evaluation (default: ``None``).
        metric (callable, optional): ``(y_true, y_pred) -> dict[str, float]``
            function returned by :func:`~trainer.helpers.build_metric`.
            ``None`` disables metric computation (default: ``None``).
        callbacks (list[TrainerCallback], optional): Callbacks invoked at
            every lifecycle hook.  When ``None`` the handler is created with
            an empty list (default: ``None``).

    Example::

        >>> from trainer.config.args import TrainingArguments
        >>> from trainer.core.image_trainer import Trainer
        >>> from trainer.core.callbacks import CLICallback
        >>> train_args = TrainingArguments(
        ...     optimizer="PESG", optimizer_kwargs={"lr": 0.1},
        ...     loss="AUCMLoss", loss_kwargs={"margin": 1.0},
        ...     SEED=42, batch_size=128, eval_batch_size=128,
        ...     sampling_rate=0.5, epochs=50, decay_epochs=[],
        ...     num_workers=2, output_path="./output", num_tasks=1,
        ...     resume_from_checkpoint=False, save_checkpoint_every=5,
        ...     project_name="libauc", experiment_name="demo", verbose=1,
        ... )
        >>> trainer = Trainer(
        ...     train_args=train_args,
        ...     model_cfg={"name": "resnet18"},
        ...     train_dataset=train_ds,
        ...     eval_dataset=[val_ds],
        ...     metric=metric_fn,
        ...     callbacks=[CLICallback()],
        ... )
        >>> log = trainer.train()

    HuggingFace model — no dataset changes needed, the trainer preprocesses
    internally::

        >>> trainer = Trainer(
        ...     train_args=train_args,
        ...     model_cfg={
        ...         "name": "google/vit-base-patch16-224",
        ...     },
        ...     train_dataset=train_ds,
        ...     eval_dataset=[val_ds],
        ...     metric=metric_fn,
        ... )
        >>> log = trainer.train()
    """
    
    def __init__(self,
                 train_args: TrainingArguments, 
                 model_cfg: dict,
                 train_dataset: Dataset, 
                 eval_dataset: Optional[List[Dataset]] = None, 
                 metric: Optional[Callable[[torch.Tensor, torch.Tensor], Mapping[str, float]]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None):
        """
        Initialize the trainer.

        Args:
            train_args: Training configuration arguments.
            model_cfg: Model architecture config (dict with at least a ``"name"`` key).
            train_dataset: Training dataset.
            eval_dataset: Optional list of evaluation datasets.
            metric: Evaluation metric function ``(y_true, y_pred) -> dict``.
            callbacks: Optional list of training callbacks.
        """
        
        self.args = train_args
        self.distributed = bool(is_distributed())
        self.rank        = dist.get_rank() if self.distributed else 0
        self.world_size  = dist.get_world_size() if self.distributed else 1
        # Seed before building the model so every trial / every DDP rank
        # starts from the same, reproducible initialisation.
        set_seed(self.args.SEED)

        self.train_dataset = train_dataset   # set before _build_model so it
        self._build_model(model_cfg)         # can infer in_channels from data
        self.eval_dataset = eval_dataset
        self.state = TrainerState()
        self.state.total_epoch = self.args.epochs
        self._set_dist_state()
        
        # Setup data loaders
        self.sampler, self.trainloader = self._get_train_dataloader(self.args)
        self.evalloaders = []
        if self.eval_dataset:
            for dataset in self.eval_dataset:
                self.evalloaders.append(self._get_eval_dataloader(dataset, self.args))
        
        # Calculate dataset statistics
        if isinstance(self.sampler.pos_len, list):
            self.data_len = self.sampler.pos_len[0] + self.sampler.neg_len[0]
        else:
            self.data_len = self.sampler.pos_len + self.sampler.neg_len

        self.pos_len = self.sampler.pos_len
        self.neg_len = self.sampler.neg_len

        # Setup optimizer and loss
        self.loss_fn, self.optimizer = self._construct_optimizer_and_loss(self.model, train_args)

        if self.distributed:
            self._check_ddp_support()
            self.model = DDP(self.model, device_ids=[torch.cuda.current_device()])

        # Setup metric and callbacks
        self.metric = metric
        if callbacks is None:
            self.callback_handler = CallbackHandler([], self.model, self.optimizer, self.loss_fn)
        else:
            self.callback_handler = CallbackHandler(callbacks, self.model, self.optimizer, self.loss_fn)
        self.callback_handler.on_init_end(self.args, self.state)

    def add_callback(self, callback):
        """Add a callback to the trainer."""
        self.callback_handler.add_callback(callback)

    # ------------------------------------------------------------------
    # Distributed helpers
    # ------------------------------------------------------------------

    @property
    def is_world_process_zero(self) -> bool:
        """``True`` on the process that logs and writes checkpoints."""
        return self.rank == 0

    @property
    def unwrapped_model(self) -> torch.nn.Module:
        """The underlying model, without the ``DistributedDataParallel`` wrapper."""
        return self.model.module if isinstance(self.model, DDP) else self.model

    # LR divisor applied at each epoch in ``args.decay_epochs``.
    decay_factor: float = 10.0

    def _decay_epochs(self) -> List[int]:
        """Epochs at which :meth:`_apply_lr_decay` decays the learning rate."""
        return self.args.decay_epochs

    def _apply_lr_decay(self, epoch: int) -> bool:
        """Decay the LR by :attr:`decay_factor` if *epoch* is scheduled.

        The single place the decay schedule is applied (it used to live in
        ``CLICallback``, so a Trainer without that callback never decayed).
        Optimizers with an epoch-level regulariser (PESG's ``model_ref``)
        also reset their reference model. Returns ``True`` if it decayed.
        """
        if epoch not in self._decay_epochs():
            return False
        if getattr(self.optimizer, "model_ref", None) is not None:
            self.optimizer.update_regularizer(decay_factor=self.decay_factor)
        else:
            self.optimizer.update_lr(decay_factor=self.decay_factor)
        if self.is_world_process_zero:
            logger.info(f"Epoch {epoch}: LR decayed by /{self.decay_factor}. New LR = {self.optimizer.param_groups[0]['lr']:.6f}")
        return True

    def _set_dist_state(self):
        self.state.is_world_process_zero = self.is_world_process_zero
        self.state.world_size = self.world_size

    def _sync_buffers(self):
        """Copy rank 0's buffers (e.g. BatchNorm running stats) to every rank.

        DDP broadcasts buffers only at the start of a training forward, so
        after the last step each rank holds stats updated on its own shard.
        Syncing before evaluation makes every shard use the weights that rank
        0 checkpoints.
        """
        for buf in self.unwrapped_model.buffers():
            dist.broadcast(buf, src=0)

    def _check_ddp_support(self):
        """Warn when the loss/optimizer pair has no distributed implementation.

        DDP averages model gradients, but losses with auxiliary state (e.g.
        ``a``/``b``/``alpha`` or per-sample moving averages) must be
        synchronised by the loss/optimizer itself; see LibAUC's
        ``examples/20_Distributed_Training_*`` for the supported pairs.
        """
        loss_ok = (
            getattr(self.loss_fn, "distributed", False)
            or getattr(getattr(self.loss_fn, "loss_fn", None), "distributed", False)  # pAUCLoss wrapper
            or self.args.loss in _DDP_NATIVE_LOSSES
        )
        if not loss_ok or self.args.optimizer in _DDP_UNSYNCED_OPTIMIZERS:
            logger.warning(
                f"{self.args.loss} / {self.args.optimizer} has no distributed-aware "
                "implementation in libauc; auxiliary loss/optimizer state will not be "
                "synchronised across GPUs and results may differ from single-GPU training."
            )
    
    def _build_model(self, model_cfg: dict):
        """
        Build a model from the ``model`` config block.

        Expected keys
        -------------
        name              Architecture name.  Use one of the built-in shortcuts
                          (``resnet20``, ``resnet18``, ``densenet121``) **or**
                          any HuggingFace model repo ID that contains a ``/``
                          (e.g. ``"google/vit-base-patch16-224"``,
                          ``"openai/clip-vit-base-patch32"``).
                          HF models are loaded via
                          ``AutoModelForImageClassification.from_pretrained``
                          and wrapped in :class:`_HFModelWrapper`.  The
                          ``AutoImageProcessor`` is applied in the DataLoader
                          collate function so the external dataset requires
                          **no** HF-specific transforms — a plain
                          ``ToTensor()`` is sufficient.
        pretrained        bool — load a local checkpoint from
                          ``pretrained_path`` (default ``False``).
        pretrained_remote bool — use the architecture's own pretrained
                          ImageNet weights where supported (default ``False``).
                          Ignored for HF models.
        pretrained_path   str — path to a local ``.pt`` checkpoint.  Required
                          when ``pretrained=True``.  Head parameters identified
                          by :data:`_HEAD_PARTS` are excluded from loading and
                          re-initialised so the model can be fine-tuned on a
                          different number of classes.

        Attributes set on ``self``
        --------------------------
        image_processor   ``AutoImageProcessor`` instance for the HF model
                          (``None`` for built-in models).  The trainer uses
                          it via :class:`_HFCollate` in the DataLoader —
                          do **not** apply it again in your dataset transforms.
        """

        name              = model_cfg.get("name", "")
        name_lower        = name.lower()
        pretrained        = model_cfg.get("pretrained", False)
        pretrained_remote = model_cfg.get("pretrained_remote", False)
        is_hf_model       = "/" in name

        num_classes = 1 if self.args.num_tasks <= 2 else self.args.num_tasks

        in_channels = 3  # sensible default (RGB)
        try:
            img = self._example_input()
            if hasattr(img, "shape") and img.ndim >= 3:
                in_channels = img.shape[0]
        except Exception:
            pass

        if in_channels not in (1, 3):
            raise ValueError(
                f"Dataset images have {in_channels} channels. "
                f"Only 1 (grayscale) and 3 (RGB) are supported."
            )

        # If grayscale, flag expansion; all model constructors receive in_channels=3.
        self._expand_to_rgb = (in_channels == 1)
        if self._expand_to_rgb:
            logger.info(
                "Grayscale input (1 channel) detected — images will be "
                "expanded to 3 channels (L→RGB repeat) in the DataLoader "
                "collate function."
            )
            in_channels = 3

        logger.info(f"Auto-detected: num_classes={num_classes}, in_channels={in_channels}")

        self.image_processor = None   # public reference (for inspection)
        self._hf_processor   = None   # used by _make_collate_fn()

        if name_lower == "resnet20":
            from libauc.models import resnet20
            model = resnet20(last_activation=None, num_classes=num_classes)
        elif name_lower == "resnet18":
            from libauc.models import resnet18
            model = resnet18(pretrained=pretrained_remote, last_activation=None, num_classes=num_classes)
        elif name_lower == "densenet121":
            from libauc.models import densenet121
            model = densenet121(pretrained=pretrained_remote, last_activation=None, activations='relu', num_classes=num_classes)
        elif is_hf_model:
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            try:
                _proc = AutoImageProcessor.from_pretrained(name)
                self.image_processor = _proc   # public reference
                self._hf_processor   = _proc   # picked up by DataLoaders
                _sz = getattr(_proc, "size", "unknown")
                logger.info(
                    f"HF image processor for '{name}': native size = {_sz}. "
                    "Preprocessing applied in DataLoader collate_fn."
                )
            except Exception as exc:
                logger.warning(
                    f"Could not load AutoImageProcessor for '{name}': {exc}. "
                    "Raw tensors will be passed directly to the model — ensure "
                    "your dataset already matches the model's expected resolution "
                    "and normalisation."
                )

            # ── Model ────────────────────────────────────────────────────────
            hf_model = AutoModelForImageClassification.from_pretrained(
                name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            model = _HFModelWrapper(hf_model)
            logger.info(f"Loaded HuggingFace model '{name}' | num_labels={num_classes}")
        else:
            raise ValueError(
                f"Unknown model '{name}'. Use a built-in name "
                f"(resnet20, resnet18, densenet121) or a HuggingFace repo ID "
                f"containing '/' (e.g. 'google/vit-base-patch16-224')."
            )

        model = model.cuda()

        if pretrained:
            pretrained_path = model_cfg.get("pretrained_path")
            if not pretrained_path:
                raise ValueError(
                    "pretrained=True but 'pretrained_path' is not set in model_cfg."
                )
            state_dict = torch.load(pretrained_path, weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            filtered = {
                k: v for k, v in state_dict.items()
                if not any(part in k.split(".") for part in _HEAD_PARTS)
            }
            msg = model.load_state_dict(filtered, strict=False)
            logger.info(msg)

            # Re-initialise the head so it starts from scratch.
            if is_hf_model:
                inner = model.hf_model
                for attr in ("classifier", "head"):
                    head = getattr(inner, attr, None)
                    if head is not None and hasattr(head, "reset_parameters"):
                        head.reset_parameters()
            else:
                for attr in ("fc", "linear"):
                    head = getattr(model, attr, None)
                    if head is not None and hasattr(head, "reset_parameters"):
                        head.reset_parameters()

        self.model = model

    def _example_input(self):
        """One model input (e.g. a ``(C, H, W)`` image) used to infer ``in_channels``."""
        first_sample = self.train_dataset[0]
        return first_sample[0] if isinstance(first_sample, (tuple, list)) else first_sample

    @staticmethod
    def _get_optimizer(name: str):
        """Return the optimizer class *name* from ``libauc.optimizers``."""
        opt = importlib.import_module("libauc.optimizers")
        return getattr(opt, name, None)

    @staticmethod
    def _get_loss(name: str):
        """Return the loss class *name*, checking ``libauc.losses`` then ``torch.nn``."""
        libauc_losses = importlib.import_module("libauc.losses")
        loss_cls = getattr(libauc_losses, name, None)
        if loss_cls is None:
            import torch.nn as nn
            loss_cls = getattr(nn, name, None)
        if loss_cls is None:
            raise ValueError(f"Loss function '{name}' not found in libauc.losses or torch.nn")
        return loss_cls

    def _construct_optimizer_and_loss(self, model, train_args: TrainingArguments):
        """Construct optimizer and loss function based on configuration."""
        # Setup loss function
        loss_cls = self._get_loss(train_args.loss)
        if train_args.loss in ["BCELoss", "CrossEntropyLoss"]:
            train_args.loss_kwargs.pop("num_labels", None)
        if train_args.loss in ["pAUCLoss", "MultiLabelpAUCLoss"]:
            if train_args.loss_kwargs["mode"] in ['SOPA']:
                loss_fn = loss_cls(data_len=self.data_len, pos_len=self.pos_len, **train_args.loss_kwargs)
            else:
                loss_fn = loss_cls(data_len=self.data_len, **train_args.loss_kwargs)
        elif train_args.loss in ["mAPLoss", "APLoss", "pAUC_DRO_Loss", "tpAUC_KL_Loss", "MIDAMLoss"]:
            loss_fn = loss_cls(data_len=self.data_len, **train_args.loss_kwargs)
        elif train_args.loss in ["pAUC_CVaR_Loss"]:
            loss_fn = loss_cls(data_len=self.data_len, pos_len=self.pos_len, **train_args.loss_kwargs)
        elif train_args.loss in ["tpAUC_CVaR_loss"]:
            loss_fn = loss_cls(data_length=self.data_len, **train_args.loss_kwargs)
        else:
            loss_fn = loss_cls(**train_args.loss_kwargs)

        # Setup optimizer
        opt_cls = self._get_optimizer(train_args.optimizer)
        optimizer = opt_cls(model.parameters(), loss_fn=loss_fn, **train_args.optimizer_kwargs)
    
        return loss_fn, optimizer

    def _make_collate_fn(self):
        """Return the appropriate collate function for this trainer instance.

        Priority:
        - HF model → :class:`_HFCollate` (handles both channel expansion and
          processor-based resize/normalise).
        - Built-in model + grayscale input → :class:`_ChannelExpand` (repeats
          the single channel three times to produce RGB tensors).
        - Built-in model + RGB input → ``None`` (PyTorch default collation).
        """
        proc   = getattr(self, "_hf_processor",  None)
        expand = getattr(self, "_expand_to_rgb", False)
        if proc is not None:
            return _HFCollate(proc)   # _HFCollate handles expansion internally
        if expand:
            return _ChannelExpand()
        return None

    def _get_train_dataloader(self, train_args: TrainingArguments):
        """Create training data loader with dual sampling.

        In distributed mode ``batch_size`` is per GPU and each rank draws a
        disjoint slice of every global mini-batch.
        """
        if self.distributed:
            if train_args.num_tasks >= 3:
                sampler = DistributedTriSampler(self.train_dataset, train_args.batch_size, sampling_rate=train_args.sampling_rate)
            else:
                sampler = DistributedDualSampler(self.train_dataset, train_args.batch_size, sampling_rate=train_args.sampling_rate)
        elif train_args.num_tasks >= 3:
            sampler = TriSampler(self.train_dataset, train_args.batch_size, sampling_rate=train_args.sampling_rate)
        else:
            sampler = DualSampler(self.train_dataset, train_args.batch_size, sampling_rate=train_args.sampling_rate)
        trainloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=train_args.batch_size,
            sampler=sampler,
            num_workers=train_args.num_workers,
            collate_fn=self._make_collate_fn(),
        )
        return sampler, trainloader

    def _get_eval_dataloader(self, dataset, train_args: TrainingArguments):
        """Create evaluation data loader."""
        evalloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=train_args.eval_batch_size,
            shuffle=False,
            sampler=ShardSampler(dataset, self.rank, self.world_size) if self.distributed else None,
            num_workers=train_args.num_workers,
            collate_fn=self._make_collate_fn(),
        )
        return evalloader
    
    def train(self):
        """
        Main training loop.
        
        Returns:
            List of training logs with metrics for each epoch
        """
        self.callback_handler.on_train_begin(self.args, self.state)
        model = self.model.cuda()
        self.loss_fn = self.loss_fn.cuda()

        # Load checkpoint if resuming
        if self.args.resume_from_checkpoint:
            latest_checkpoint = self.get_latest_checkpoint(os.path.join(self.args.output_path, self.args.experiment_name))
            if latest_checkpoint:
                self.load_checkpoint(latest_checkpoint)
                logger.info(f"Resuming training from epoch {self.state.epoch}")
            else:
                logger.info("No checkpoint found in output folder, starting from scratch")
        
        for epoch in range(self.state.epoch, self.args.epochs):
            if hasattr(self.sampler, "set_epoch"):
                self.sampler.set_epoch(epoch)
            self.callback_handler.on_epoch_begin(self.args, self.state)
            self._apply_lr_decay(epoch)
            step_losses = []
            model.train()

            # Training loop
            for batch in self.trainloader:
                self.callback_handler.on_step_begin(self.args, self.state)

                loss = self._compute_loss(model, batch)

                # Optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                step_losses.append(loss.item())

                self.callback_handler.on_step_end(self.args, self.state)

            # Evaluation
            model.eval()
            avg_train_loss = all_reduce_mean(float(np.mean(step_losses)))
            metrics, test_true, test_pred = self.evaluate_loop(model)

            self.callback_handler.on_epoch_end(
                self.args, self.state,
                metrics=metrics,
                train_loss=avg_train_loss,
                lr=self.optimizer.lr,
                test_true=test_true,
                test_pred=test_pred,
            )
            
            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_checkpoint_every == 0 or (epoch + 1) == self.args.epochs:
                checkpoint_path = os.path.join(self.args.output_path, self.args.experiment_name, f"epoch_{epoch + 1}.pt")
                self.save_checkpoint(checkpoint_path)

        self.callback_handler.on_train_end(self.args, self.state)

        return self.state.train_log

    def _compute_loss(self, model, batch):
        """Forward one training batch and return the loss (override for other data layouts)."""
        data, targets, index = batch
        data, targets = data.cuda(), targets.cuda()
        y_pred = model(data)

        if self.args.loss == "CrossEntropyLoss":
            return self.loss_fn(y_pred, targets)
        y_pred = torch.sigmoid(y_pred)
        if self.args.loss == "BCELoss":
            return self.loss_fn(y_pred, targets)
        if isinstance(index, list):  # Multi-label
            index, task_id = index
            return self.loss_fn(y_pred, targets, index=index.cuda(), task_id=task_id)
        return self.loss_fn(y_pred, targets, index=index.cuda())

    def evaluate(self, loader, model):
        """
        Evaluate model on a given data loader.
        
        Args:
            loader: Data loader for evaluation
            model: Model to evaluate
            
        Returns:
            Tuple of (dictionary of evaluation metrics, test_true, test_pred)
        """
        test_pred_list = []
        test_true_list = []
        
        with torch.no_grad():
            for test_data, test_targets, _ in loader:
                test_data = test_data.cuda()
                test_pred = model(test_data)
                # Apply sigmoid to convert logits to probabilities
                test_pred = torch.sigmoid(test_pred)
                test_pred_list.append(test_pred.cpu().detach().numpy())
                test_true_list.append(test_targets.numpy())
            
        test_true = np.concatenate(test_true_list)
        test_pred = np.concatenate(test_pred_list)
        if self.distributed:
            test_true, test_pred = gather_shards(test_true, test_pred)
        # Flatten if needed (for binary classification)
        if test_pred.ndim > 1:
            test_pred = test_pred.flatten()
        if test_true.ndim > 1:
            test_true = test_true.flatten()
        result = self.metric(test_true, test_pred)
        return result, test_true, test_pred

    def evaluate_loop(self, model):
        """
        Evaluate model on all evaluation datasets.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Tuple of (dictionary of metrics from all evaluation datasets, test_true, test_pred)
            test_true and test_pred are from the first evaluation dataset, or None if no eval datasets
        """
        metrics = []
        test_true = None
        test_pred = None

        if self.distributed:
            self._sync_buffers()
        
        if not self.evalloaders:
            self.callback_handler.on_evaluate(self.args, self.state)
            return metrics, test_true, test_pred
        
        for loader in self.evalloaders:
            result, eval_true, eval_pred = self.evaluate(loader, model)
            metrics.append(result)

            # Store test_true and test_pred from the first evaluation dataset
            if test_true is None:
                test_true = eval_true
                test_pred = eval_pred
        
        self.callback_handler.on_evaluate(self.args, self.state)
        return metrics, test_true, test_pred

    def save_checkpoint(self, checkpoint_path: str):
        # Every rank holds identical weights; only rank 0 writes.
        if not self.is_world_process_zero:
            return
        # Ensure checkpoint directory exists
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            # Unwrapped so checkpoints load with or without DDP.
            'model_state_dict': self.unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'loss_fn': self.loss_fn,
            'state': self.state,
            'args': self.args
        }
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        map_location = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=map_location)
                
        self.unwrapped_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if hasattr(self.loss_fn, 'a') and hasattr(self.loss_fn, 'b') and hasattr(self.loss_fn, 'alpha'):
            self.loss_fn.load_state_dict(checkpoint['loss_fn_state_dict'])
        else:
            self.loss_fn = checkpoint['loss_fn']
        
        self.state = checkpoint['state']
        self._set_dist_state()
        # have to check if the args are the same as the current args
        self.args = checkpoint['args']
                
        logger.info(f"Checkpoint loaded successfully. Resuming from epoch {self.state.epoch}")
        return checkpoint

    def get_latest_checkpoint(self, output_path: str):
        if not os.path.exists(output_path):
            return None
        
        checkpoint_files = []
        for file in os.listdir(output_path):
            if file.startswith("epoch_") and file.endswith(".pt"):
                try:
                    epoch_num = int(file.split("_")[1].split(".")[0])
                    checkpoint_files.append((epoch_num, os.path.join(output_path, file)))
                except (ValueError, IndexError):
                    continue
        
        if not checkpoint_files:
            return None
        
        # Sort by epoch number and return the latest
        checkpoint_files.sort(key=lambda x: x[0])
        return checkpoint_files[-1][1]