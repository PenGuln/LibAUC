import logging
from types import SimpleNamespace
from typing import Callable, List, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from libauc.utils import collate_fn

from ..config.args import TrainingArguments
from ..data.medical_study_dataset import study_windows
from .callbacks import TrainerCallback
from .distributed import ShardSampler, gather_shards
from .image_trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["MILTrainer", "MILHead"]


def _as_tensor(bag) -> torch.Tensor:
    return bag if isinstance(bag, torch.Tensor) else torch.from_numpy(np.asarray(bag))


def _gather_keep_grad(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """All-gather *tensor* along dim 0, keeping autograd for this rank's part."""
    parts = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(parts, tensor.detach().contiguous())
    parts[rank] = tensor
    return torch.cat(parts)


def _strip_head(model: torch.nn.Module) -> int:
    """Replace the final classification ``Linear`` of *model* with ``Identity``.

    Looks for the head under the names used by libauc's backbones and
    HuggingFace models (``fc``, ``linear``, ``classifier``, ``head``) so the
    model outputs its pooled features. Returns the feature dimension.
    """
    root = getattr(model, "hf_model", model)
    for attr in ("fc", "linear", "classifier", "head"):
        head = getattr(root, attr, None)
        if not isinstance(head, torch.nn.Module):
            continue
        if isinstance(head, torch.nn.Linear):
            setattr(root, attr, torch.nn.Identity())
            return head.in_features
        linears = [(n, m) for n, m in head.named_modules() if isinstance(m, torch.nn.Linear)]
        if linears:
            path, last = linears[-1]           # e.g. HF ResNet: Sequential(Flatten, Linear)
            parent = head.get_submodule(path.rpartition(".")[0]) if "." in path else head
            setattr(parent, path.rpartition(".")[2], torch.nn.Identity())
            return last.in_features
    raise ValueError(f"Cannot find the classification head of {type(root).__name__} to replace.")


class MILHead(torch.nn.Module):
    """Instance-level MIL model: a feature backbone plus a scoring head.

    * attention pooling — returns ``(score, weight)`` per instance, with
      ``weight = exp(w_a^T tanh(V f))`` as in ``ResNet20_stoc_att``;
    * softmax pooling — returns ``sigmoid(score)`` per instance.
    """

    def __init__(self, backbone: torch.nn.Module, feat_dim: int, attention: bool, attention_dim: int = 128):
        super().__init__()
        self.backbone   = backbone
        self.classifier = torch.nn.Linear(feat_dim, 1)
        self.attention  = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, attention_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(attention_dim, 1),
        ) if attention else None

    def forward(self, x):
        features = self.backbone(x).flatten(1)
        scores = self.classifier(features)
        if self.attention is None:
            return torch.sigmoid(scores)
        return scores, torch.exp(self.attention(features))


class MILTrainer(Trainer):
    r"""
    Training loop for multiple instance learning (MIL). It runs in one of two
    modes:

    * **bag level** (default) — :class:`~libauc.losses.MIDAMLoss` /
      :class:`~libauc.optimizers.MIDAM` on bags with one binary label, as
      described below;
    * **study level** (``num_slices`` set) — multi-label medical studies from
      :class:`~libauc.trainer.data.MedicalStudyCSVDataset` (one or more image
      series per study), see *Study-level mode*.

    Each sample is a *bag* of instances with one bag-level label. Every
    training step uniformly samples up to ``instance_batch_size`` instances
    from each bag and pools their predictions stochastically, as in the MIDAM
    paper; evaluation pools every instance of each bag. The pooling follows
    ``train_args.loss_kwargs["mode"]``:

    * ``"attention"`` (default) — attention pooling; the model returns
      ``(instance_scores, attention_weights)``.
    * ``"softmax"`` — smoothed-max pooling with temperature ``tau``; the model
      returns sigmoid instance scores.

    Supported models (``model_cfg["name"]``), built for the pooling mode:

    * ``ffnn`` — tabular bags of shape ``(n_instances, ..., n_features)``;
      optional ``hidden_sizes`` (default: one layer of ``n_features`` units).
    * ``resnet20`` — image bags of shape ``(n_instances, C, H, W)``; LibAUC's
      MIDAM models (``ResNet20_stoc_att`` / ``resnet20``), fed the raw bag
      values as in the MIDAM tutorials.
    * any other image backbone the image :class:`Trainer` builds —
      ``resnet18``, ``densenet121`` or a HuggingFace model ID — with the same
      ``pretrained`` / ``pretrained_remote`` / ``pretrained_path`` options.
      Its classifier is replaced by a :class:`MILHead` (hidden size
      ``attention_dim``, default ``128``, for attention pooling). ``uint8``
      bags are scaled to ``[0, 1]``, grayscale is expanded to RGB, and HF
      models get their image processor applied to the sampled instances.

    Study-level mode: each sample is a dict ``{"series", "targets", "index"}``.
    Every training step draws ``num_slices`` random slices from the study's
    concatenated series; evaluation scores every slice. The model is a
    :class:`~libauc.models.MILClassifier` over an encoder from
    :func:`~libauc.models.build_mil_encoder`, configured by ``model_cfg``:
    ``name`` (``resnet18``, ``coatnet``, ``radeo``, ``biovfm``, ``omnirad``,
    ``imagenet_vit``), ``pretrained``, ``pretrained_path``, ``config_path``,
    ``cache_dir``, ``tuning_mode`` (``full_finetune`` | ``linear_probe``),
    ``image_size``, ``modality``, ``body_part``, ``pool`` (``mean``, ``max``,
    ``attention``, ``gated_attention``, ``class_attention``,
    ``class_gated_attention``, ``query_attention``), ``query_mode``,
    ``query_heads``, ``query_rmsnorm``, ``attention_dim``,
    ``separate_classifier_heads`` and ``encoder_chunk_size``. Any loss works;
    :class:`~libauc.losses.MultiLabelMIDAMLoss` (``pool: class_attention``)
    trains on the stochastic MIDAM pooling of the sampled slices. Under DDP
    the MIDAM statistics are gathered over the global batch, so the per-study
    moving averages and ``a``/``b``/``alpha`` stay identical on every rank.

    Args:
        train_args (TrainingArguments): Training configuration. In bag-level
            mode ``loss`` must be ``"MIDAMLoss"`` and ``optimizer`` ``"MIDAM"``.
        model_cfg (dict): ``{"name": "ffnn" | "resnet20", ...}``.
        train_dataset: Bag dataset yielding ``(bag, label, index)`` with a
            ``.targets`` attribute, e.g. :class:`~libauc.datasets.CustomDataset`
            with ``return_index=True``.
        eval_dataset (list, optional): Bag datasets for evaluation.
        metric (callable, optional): ``(y_true, y_pred) -> dict[str, float]``.
        callbacks (list[TrainerCallback], optional): Training callbacks.
        instance_batch_size (int): Instances sampled per bag per step
            (default: ``4``).
        eval_instance_batch_size (int): Instances per forward pass when a
            whole bag is scored during evaluation (default: ``256``).
        smoothing_decay_factor (float): At every LR decay epoch the loss's
            moving-average parameter ``gamma`` is divided by this factor
            (default: ``2``).
        num_slices (int, optional): Slices sampled per study per training
            step; setting it selects study-level mode (default: ``None``).

    Example::

        >>> trainer = MILTrainer(
        ...     train_args=train_args,          # loss="MIDAMLoss", optimizer="MIDAM"
        ...     model_cfg={"name": "ffnn"},
        ...     train_dataset=train_bags,
        ...     eval_dataset=[val_bags],
        ...     metric=metric_fn,
        ...     callbacks=[CLICallback()],
        ...     instance_batch_size=4,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        train_args:             TrainingArguments,
        model_cfg:              dict,
        train_dataset,
        eval_dataset:           Optional[List]                               = None,
        metric:                 Optional[Callable[..., Mapping[str, float]]] = None,
        callbacks:              Optional[List[TrainerCallback]]              = None,
        instance_batch_size:    int                                          = 4,
        smoothing_decay_factor: float                                        = 2.0,
        eval_instance_batch_size: int                                        = 256,
        num_slices:             Optional[int]                                = None,
    ):
        self.num_slices               = num_slices
        self.instance_batch_size      = instance_batch_size
        self.smoothing_decay_factor   = smoothing_decay_factor
        self.eval_instance_batch_size = eval_instance_batch_size
        self._generic_backbone        = False
        super().__init__(
            train_args    = train_args,
            model_cfg     = model_cfg,
            train_dataset = train_dataset,
            eval_dataset  = eval_dataset,
            metric        = metric,
            callbacks     = callbacks,
        )

    @property
    def study_level(self) -> bool:
        return self.num_slices is not None

    @property
    def _study_midam(self) -> bool:
        return self.study_level and self.args.loss == "MultiLabelMIDAMLoss"

    @property
    def pooling(self) -> str:
        return self.args.loss_kwargs.get("mode", "attention")

    @property
    def _tau(self) -> float:
        # MIDAMLoss wraps the mode-specific loss, which holds tau.
        return getattr(getattr(self.loss_fn, "loss_fn", self.loss_fn), "tau", 0.1)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _example_input(self):
        # The first instance of the first bag, so the base Trainer infers
        # in_channels from an image rather than from a whole bag.
        return _as_tensor(self.train_dataset[0][0])[0]

    def _build_model(self, model_cfg: dict):
        if self.study_level:
            return self._build_study_model(model_cfg)

        from libauc.models import FFNN, FFNN_stoc_att, ResNet20_stoc_att, resnet20

        if self.args.loss != "MIDAMLoss":
            raise ValueError(f"MILTrainer requires loss 'MIDAMLoss', got '{self.args.loss}'.")
        if self.pooling not in ("attention", "softmax"):
            raise ValueError(f"MIDAMLoss mode must be 'attention' or 'softmax', got '{self.pooling}'.")

        name      = model_cfg.get("name", "").lower()
        attention = self.pooling == "attention"
        bag       = np.asarray(self.train_dataset[0][0])

        if bag.ndim >= 4 and name != "resnet20":
            # Generic image backbone: built by the image Trainer, head swapped
            # for an MIL head.
            super()._build_model(model_cfg)
            feat_dim = _strip_head(self.model)
            self.model = MILHead(self.model, feat_dim, attention,
                                 attention_dim=model_cfg.get("attention_dim", 128)).cuda()
            self._generic_backbone = True
            logger.info(f"Built MIL model '{model_cfg.get('name')}' (features {feat_dim}) for {self.pooling} pooling "
                        f"(bag shape {bag.shape})")
            return

        if bag.ndim >= 4:  # (n_instances, C, H, W)
            channels = bag.shape[1]
            if attention:
                model = ResNet20_stoc_att(inchannels=channels, num_classes=1)
            else:
                if channels != 3:
                    raise ValueError("Softmax pooling with resnet20 supports 3-channel images only.")
                model = resnet20(num_classes=1, last_activation="sigmoid")
        else:              # (n_instances, [1,] n_features)
            if name != "ffnn":
                raise ValueError(f"Tabular bags need model 'ffnn', got '{name}' "
                                 "(image backbones need bags of shape (n_instances, C, H, W)).")
            input_dim    = bag.shape[-1]
            hidden_sizes = tuple(model_cfg.get("hidden_sizes", (input_dim,)))
            if attention:
                model = FFNN_stoc_att(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=1)
            else:
                model = FFNN(input_dim=input_dim, hidden_sizes=hidden_sizes, num_classes=1,
                             last_activation="sigmoid")

        logger.info(f"Built MIL model '{name}' for {self.pooling} pooling (bag shape {bag.shape})")
        self.model = model.cuda()

    def _build_study_model(self, model_cfg: dict):
        from libauc.models import MILClassifier, build_mil_encoder

        # 2.5D windows (adjacent slices as channels) and slot embeddings; off by default.
        self.window_slices  = int(model_cfg.get("window_slices", 1))
        self.slot_embedding = bool(model_cfg.get("slot_embedding", False))
        if self.window_slices not in (1, 3):
            raise ValueError(f"window_slices must be 1 or 3, got {self.window_slices}")

        if self._study_midam and model_cfg.get("pool") != "class_attention":
            raise ValueError("MultiLabelMIDAMLoss requires model pool 'class_attention'.")

        encoder, feature_dim = build_mil_encoder(
            model_cfg.get("name", "resnet18"),
            pretrained      = model_cfg.get("pretrained", False),
            pretrained_path = model_cfg.get("pretrained_path"),
            config_path     = model_cfg.get("config_path"),
            cache_dir       = model_cfg.get("cache_dir"),
            tuning_mode     = model_cfg.get("tuning_mode", "full_finetune"),
            image_size      = model_cfg.get("image_size", 224),
            modality        = model_cfg.get("modality", "MRI"),
            body_part       = model_cfg.get("body_part", "N/A"),
        )
        self.model = MILClassifier(
            encoder                   = encoder,
            feature_dim               = feature_dim,
            num_classes               = 1 if self.args.num_tasks <= 2 else self.args.num_tasks,
            pool                      = model_cfg.get("pool", "mean"),
            attention_dim             = model_cfg.get("attention_dim", 128),
            encoder_chunk_size        = model_cfg.get("encoder_chunk_size", 64),
            separate_classifier_heads = model_cfg.get("separate_classifier_heads", False),
            query_heads               = model_cfg.get("query_heads", 1),
            query_rmsnorm             = model_cfg.get("query_rmsnorm", True),
            query_mode                = model_cfg.get("query_mode"),
            num_slots                 = len(self.train_dataset.series_slots) if self.slot_embedding else 0,
        )
        reset_head = bool(model_cfg.get("init_reset_head", False))
        if model_cfg.get("init_checkpoint"):
            # Two-stage training (e.g. BCE, then an AUC loss): start from a previous run's weights. The optimizer is
            # built afterwards, so PESG's epoch-decay reference is this starting point. With init_reset_head the final
            # per-class classifier is not loaded and keeps its fresh initialisation (LibAUC's two-stage recipe);
            # everything else must match exactly.
            state = torch.load(model_cfg["init_checkpoint"], map_location="cpu", weights_only=False)
            state = state.get("model_state_dict", state)
            head = ("classifier.", "classifiers.")
            if reset_head:
                state = {k: v for k, v in state.items() if not k.startswith(head)}
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            bad_missing = [k for k in missing if not (reset_head and k.startswith(head))]
            if bad_missing or unexpected:
                raise RuntimeError(f"init_checkpoint does not match the model: missing {bad_missing}, unexpected {unexpected}")
            logger.info(f"Model initialised from {model_cfg['init_checkpoint']}"
                        + (f" (classifier re-initialised: {len(missing)} tensors)" if reset_head else ""))
            if reset_head and model_cfg.get("init_bias_from_prior", False):
                self._init_bias_from_prior(self.model)
        elif reset_head:
            raise ValueError("init_reset_head needs init_checkpoint")
        elif model_cfg.get("init_bias_from_prior", False):
            self._init_bias_from_prior(self.model)
        self.model = self.model.cuda()
        logger.info(f"Built study-level MIL model '{model_cfg.get('name')}' "
                    f"(features {feature_dim}, pool {model_cfg.get('pool', 'mean')}, {self.num_slices} slices, "
                    f"{self.window_slices}-slice windows, slot embedding {self.slot_embedding}, "
                    f"prior bias {bool(model_cfg.get('init_bias_from_prior', False))})")

    def _init_bias_from_prior(self, model):
        """Start each class logit at logit(mean training target), so the first steps don't spend on base rates."""
        targets = np.asarray(self.train_dataset.targets, dtype=np.float64).reshape(len(self.train_dataset.targets), -1)
        prior = torch.logit(torch.as_tensor(targets.mean(0), dtype=torch.float32).clamp(1e-3, 1 - 1e-3))
        with torch.no_grad():
            if model.separate_classifier_heads:
                for c, head in enumerate(model.classifiers):
                    head.bias.fill_(float(prior[c]))
            else:
                model.classifier.bias.copy_(prior[: model.classifier.bias.numel()])
        logger.info(f"Classifier bias initialised from training base rates: {np.round(torch.sigmoid(prior).numpy(), 3).tolist()}")

    def _check_ddp_support(self):
        if self._study_midam:
            return  # synchronised in _compute_loss / _optimizer_step
        if self.study_level and self.args.optimizer == "PESG" and self.args.loss in ("AUCMLoss", "MultiLabelAUCMLoss"):
            return  # PESG all-reduces the a/b gradients; _optimizer_step averages alpha
        super()._check_ddp_support()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_train_dataloader(self, train_args: TrainingArguments):
        if self.study_level:
            return self._get_study_train_dataloader(train_args)

        from libauc.sampler import DualSampler, DistributedDualSampler

        sampler_cls = DistributedDualSampler if self.distributed else DualSampler
        sampler = sampler_cls(self.train_dataset, train_args.batch_size, sampling_rate=train_args.sampling_rate)
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size  = train_args.batch_size,
            sampler     = sampler,
            num_workers = train_args.num_workers,
            collate_fn  = collate_fn,   # bags differ in size: keep them as lists
        )
        return sampler, loader

    def _get_study_train_dataloader(self, train_args: TrainingArguments):
        # Studies are drawn uniformly (no positive oversampling): with many
        # labels per study there is no single class to balance.
        sampler = DistributedSampler(self.train_dataset, shuffle=True, seed=train_args.SEED) if self.distributed else None
        loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size  = train_args.batch_size,
            shuffle     = sampler is None,
            sampler     = sampler,
            num_workers = train_args.num_workers,
            collate_fn  = list,         # studies differ in slice count
        )
        targets = np.asarray(self.train_dataset.targets)
        binary = (targets.reshape(len(targets), -1) >= 0.5)
        # The base Trainer reads pos_len / neg_len from the sampler.
        stats = sampler if sampler is not None else SimpleNamespace()
        stats.pos_len = binary.sum(0).tolist()
        stats.neg_len = (~binary).sum(0).tolist()
        return stats, loader

    def _get_eval_dataloader(self, dataset, train_args: TrainingArguments):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size  = train_args.eval_batch_size,
            shuffle     = False,
            sampler     = ShardSampler(dataset, self.rank, self.world_size) if self.distributed else None,
            num_workers = train_args.num_workers,
            collate_fn  = list if self.study_level else collate_fn,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _prepare_instances(self, x: torch.Tensor) -> torch.Tensor:
        """Instances ``(n, ...)`` -> model input on the GPU."""
        if not self._generic_backbone:
            return x.float().cuda()      # ffnn / resnet20: raw values, as in the tutorials
        is_uint8 = x.dtype == torch.uint8
        x = x.float()
        if is_uint8:
            x = x / 255.0
        if x.shape[1] == 1 and (getattr(self, "_expand_to_rgb", False) or self._hf_processor is not None):
            x = x.expand(-1, 3, -1, -1)
        if self._hf_processor is not None:
            x = self._hf_processor(images=x, return_tensors="pt", do_rescale=False)["pixel_values"]
        return x.cuda()

    def _study_inputs(self, sample, count=None, random=False):
        """``(windows, slot_ids)`` of one study; every slice as a centre when ``count`` is None."""
        return study_windows(sample["series"], count=count, random=random,
                             window_slices=getattr(self, "window_slices", 1))

    def _model_kwargs(self, slot_ids):
        return {"slot_ids": slot_ids} if getattr(self, "slot_embedding", False) else {}

    def _compute_study_loss(self, model, batch):
        inputs  = [self._study_inputs(sample, self.num_slices, random=True) for sample in batch]
        studies = torch.stack([w for w, _ in inputs]).cuda()
        extra   = self._model_kwargs(torch.stack([s for _, s in inputs]).cuda())
        y_true = torch.stack([sample["targets"] for sample in batch]).float().cuda()
        index  = torch.as_tensor([sample["index"] for sample in batch]).cuda()

        if self._study_midam:
            sn, sd = model(studies, return_midam=True, **extra)
            if self.distributed:
                # MIDAM's loss couples the whole batch, so it is computed on the
                # global batch; gradients flow back through this rank's part only.
                sn, sd = (_gather_keep_grad(t, self.rank, self.world_size) for t in (sn, sd))
                y_true, index = (_gather_keep_grad(t, self.rank, self.world_size) for t in (y_true, index))
            binary  = y_true >= self.loss_fn.label_threshold
            task_id = torch.where(binary.any(dim=0) & (~binary).any(dim=0))[0]
            return self.loss_fn((sn, sd), y_true, index=index, task_id=task_id)

        logits = model(studies, **extra)
        if self.args.loss == "CrossEntropyLoss":
            return self.loss_fn(logits, y_true)
        y_pred = torch.sigmoid(logits)
        if self.args.loss == "BCELoss":
            return self.loss_fn(y_pred, y_true)
        # AUC losses compare positives with negatives: binary targets (soft report labels thresholded at 0.5) and, for
        # multi-label losses, only the findings that have both classes in this batch (a rare finding often has no
        # positive among a few studies, which would make its term undefined).
        y_bin = (y_true >= 0.5).float()
        if y_bin.shape[-1] > 1 and type(self.loss_fn).__name__.startswith("MultiLabel"):
            task_id = torch.where((y_bin == 1).any(0) & (y_bin == 0).any(0))[0]
            if len(task_id) == 0:
                return (y_pred * 0).sum()       # nothing to rank; keeps every DDP rank's backward in step
            return self.loss_fn(y_pred, y_bin, task_id=task_id, index=index)
        return self.loss_fn(y_pred, y_bin, index=index)

    def _optimizer_step(self, model, loss):
        if self.distributed and not self._study_midam and isinstance(getattr(self.loss_fn, "alpha", None), torch.Tensor):
            # PESG averages a/b gradients over ranks but updates alpha from each rank's own batch; average it so the
            # ranks keep one shared AUC-margin state.
            super()._optimizer_step(model, loss)
            with torch.no_grad():
                dist.all_reduce(self.loss_fn.alpha.data)
                self.loss_fn.alpha.data.div_(self.world_size)
            return
        if not (self._study_midam and self.distributed):
            return super()._optimizer_step(model, loss)
        self.optimizer.zero_grad()
        loss.backward()
        # Each rank backpropagated the global loss through its own studies;
        # DDP averaged those parts, so scale back to their sum. a/b already
        # see the whole batch on every rank (the all-reduce keeps them equal).
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(self.world_size)
        for p in (self.loss_fn.a, self.loss_fn.b):
            if p.grad is not None:
                dist.all_reduce(p.grad)
                p.grad.div_(self.world_size)
        self.optimizer.step()
        for t in (self.loss_fn.a, self.loss_fn.b, self.loss_fn.alpha):
            dist.broadcast(t.data, src=0)

    def _compute_loss(self, model, batch):
        """Stochastic pooling over ``instance_batch_size`` instances per bag.

        Equivalent to :func:`libauc.utils.MIL_sampling` per bag, but the
        sampled instances of all bags go through the model in one forward
        pass (faster, and a single forward per step as DDP expects).
        """
        if self.study_level:
            return self._compute_study_loss(model, batch)

        bags, labels, index = batch
        sampled, counts = [], []
        for bag in bags:
            x = _as_tensor(bag)
            k = min(x.shape[0], self.instance_batch_size)
            ids = torch.multinomial(torch.ones(x.shape[0]), k, replacement=False)
            sampled.append(x[ids])
            counts.append(k)
        x = self._prepare_instances(torch.cat(sampled))

        def bag_mean(values):
            return torch.stack([v.mean() for v in values.view(-1).split(counts)]).view(-1, 1)

        if self.pooling == "attention":
            scores, weights = model(x)
            y_pred = (bag_mean(scores.view(-1) * weights.view(-1)), bag_mean(weights))
        else:
            y_pred = bag_mean(torch.exp(model(x).view(-1) / self._tau))

        y_true = torch.as_tensor(np.asarray(labels, dtype=np.float32)).view(-1, 1).cuda()
        index  = torch.as_tensor(np.asarray(index)).view(-1).cuda()
        return self.loss_fn(y_pred=y_pred, y_true=y_true, index=index)

    def _apply_lr_decay(self, epoch: int) -> bool:
        decayed = super()._apply_lr_decay(epoch)
        if decayed and hasattr(self.loss_fn, "update_smoothing"):
            self.loss_fn.update_smoothing(decay_factor=self.smoothing_decay_factor)
        return decayed

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _pool_bag(self, net, bag) -> float:
        """Bag score from all its instances (cf. :func:`libauc.utils.MIL_aggregation`)."""
        chunks = [net(self._prepare_instances(x)) for x in _as_tensor(bag).split(self.eval_instance_batch_size)]
        if self.pooling == "attention":
            scores  = torch.cat([c[0].view(-1) for c in chunks])
            weights = torch.cat([c[1].view(-1) for c in chunks])
            # Weighted instance score, squashed like the loss does (sigmoid(sn / sd)).
            return torch.sigmoid((scores * weights).sum() / weights.sum()).item()
        out = torch.cat([c.view(-1) for c in chunks])
        tau = self._tau
        return (tau * torch.log(torch.mean(torch.exp(out / tau)))).item()

    def _evaluate_studies(self, loader):
        net = self.unwrapped_model
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                for sample in batch:
                    study, slot_ids = self._study_inputs(sample)            # every slice as a centre
                    logits = net(study.unsqueeze(0).cuda(), **self._model_kwargs(slot_ids.unsqueeze(0).cuda()))
                    if self.args.loss == "CrossEntropyLoss" and logits.shape[-1] > 1:
                        prob = torch.softmax(logits, dim=-1)
                    else:
                        prob = torch.sigmoid(logits)
                    preds.append(prob.cpu().numpy()[0])
                    trues.append(sample["targets"].numpy())
        test_pred = np.asarray(preds, dtype=np.float32)
        test_true = (np.asarray(trues) >= 0.5).astype(np.int64)
        if self.distributed:
            test_true, test_pred = gather_shards(test_true, test_pred)
        # Binary: 1-D scores; multi-label: keep (n_studies, n_labels).
        if self.args.num_tasks <= 2:
            test_true, test_pred = test_true.reshape(-1), test_pred.reshape(-1)
        return self.metric(test_true, test_pred), test_true, test_pred

    def evaluate(self, loader, model):
        if self.study_level:
            return self._evaluate_studies(loader)

        net = self.unwrapped_model
        preds, trues = [], []
        with torch.no_grad():
            for bags, labels, _ in loader:
                preds.extend(self._pool_bag(net, bag) for bag in bags)
                trues.extend(np.asarray(label).reshape(-1)[0] for label in labels)
        test_true = np.asarray(trues, dtype=np.float32)
        test_pred = np.asarray(preds, dtype=np.float32)
        if self.distributed:
            test_true, test_pred = gather_shards(test_true, test_pred)
        return self.metric(test_true, test_pred), test_true, test_pred

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = super().load_checkpoint(checkpoint_path)
        # MIDAM's loss state (a, b, alpha, per-bag moving averages) are plain
        # tensors, not module state; restore them in place so the optimizer,
        # which references a/b/alpha, stays linked to the same tensors.
        saved = getattr(checkpoint["loss_fn"], "loss_fn", checkpoint["loss_fn"])
        current = getattr(self.loss_fn, "loss_fn", self.loss_fn)
        for attr in ("a", "b", "alpha", "s", "sn", "sd"):
            if hasattr(saved, attr) and hasattr(current, attr):
                getattr(current, attr).data.copy_(getattr(saved, attr).data)
        if hasattr(saved, "gamma"):
            current.gamma = saved.gamma
        return checkpoint
