import logging
from typing import Callable, List, Mapping, Optional

import numpy as np
import torch

from libauc.utils import collate_fn

from ..config.args import TrainingArguments
from .callbacks import TrainerCallback
from .distributed import ShardSampler, gather_shards
from .image_trainer import Trainer

logger = logging.getLogger(__name__)

__all__ = ["MILTrainer", "MILHead"]


def _as_tensor(bag) -> torch.Tensor:
    return bag if isinstance(bag, torch.Tensor) else torch.from_numpy(np.asarray(bag))


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
    Training loop for multiple instance learning (MIL) with
    :class:`~libauc.losses.MIDAMLoss` and :class:`~libauc.optimizers.MIDAM`.

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

    Args:
        train_args (TrainingArguments): Training configuration; ``loss`` must be
            ``"MIDAMLoss"`` and ``optimizer`` ``"MIDAM"``.
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
    ):
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

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _get_train_dataloader(self, train_args: TrainingArguments):
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

    def _get_eval_dataloader(self, dataset, train_args: TrainingArguments):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size  = train_args.eval_batch_size,
            shuffle     = False,
            sampler     = ShardSampler(dataset, self.rank, self.world_size) if self.distributed else None,
            num_workers = train_args.num_workers,
            collate_fn  = collate_fn,
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

    def _compute_loss(self, model, batch):
        """Stochastic pooling over ``instance_batch_size`` instances per bag.

        Equivalent to :func:`libauc.utils.MIL_sampling` per bag, but the
        sampled instances of all bags go through the model in one forward
        pass (faster, and a single forward per step as DDP expects).
        """
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
        if decayed:
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

    def evaluate(self, loader, model):
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
