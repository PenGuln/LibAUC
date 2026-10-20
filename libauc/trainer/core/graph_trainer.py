import logging
from typing import Callable, List, Mapping, Optional

import numpy as np
import torch

from .image_trainer import Trainer
from ..config.args import TrainingArguments
from .callbacks import TrainerCallback
from .distributed import ShardSampler, gather_shards

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Architectures available in libauc.models for graph learning
# ---------------------------------------------------------------------------
_GNN_REGISTRY = {
    "gcn":        "GCN",
    "gin":        "GIN",
    "gine":       "GINE",
    "graphsage":  "GraphSAGE",
    "gat":        "GAT",
    "mpnn":       "MPNN",
    "deepergcn":  "DeeperGCN",
    "pna":        "PNA",
}

class GraphTrainer(Trainer):
    r"""
    Training loop for graph neural networks built with libauc's GNN model zoo.

    ``GraphTrainer`` extends :class:`~trainer.core.image_trainer.Trainer` with
    graph-aware overrides:

    * :meth:`_build_model` — looks up the requested GNN architecture in
      :data:`_GNN_REGISTRY` and constructs it via ``libauc.models``, then
      infers whether the model expects edge features (``supports_edge_attr``).
    * :meth:`_get_train_dataloader` / :meth:`_get_eval_dataloader` — use
      ``torch_geometric.loader.DataLoader`` instead of the standard PyTorch
      one, while keeping the same :class:`~libauc.sampler.DualSampler` for
      positive/negative balancing.
    * :meth:`_forward` — dispatches to the correct GNN forward signature
      (with or without ``edge_attr``).
    * :meth:`_decay_epochs` — decays the learning rate by ``decay_factor`` at
      ``decay_epochs`` (defaults to ``train_args.decay_epochs``).

    Supported GNN architectures
    ---------------------------
    ``gcn``, ``gin``, ``gine``, ``graphsage``, ``gat``, ``mpnn``,
    ``deepergcn``, ``pna``

    Args:
        train_args (TrainingArguments): Training configuration.
        model_cfg (dict): GNN model configuration.  Required key: ``name``
            (one of the architectures listed above).  Optional keys:
            ``emb_dim`` (default ``256``), ``num_layers`` (default ``5``),
            ``graph_pooling``, ``dropout``,
            ``atom_features_dims``, ``bond_features_dims``, ``act``, ``norm``,
            ``jk``, ``v2`` (GAT-only), ``aggr`` / ``t`` / ``learn_t`` / ``p``
            / ``learn_p`` / ``block`` (DeeperGCN-only), ``pretrained`` (bool),
            ``pretrained_path`` (str).
        train_dataset: PyG-compatible graph dataset (train split).
        eval_dataset (list, optional): PyG-compatible graph datasets for
            evaluation splits (default: ``None``).
        metric (callable, optional): ``(y_true, y_pred) -> dict[str, float]``
        callbacks (list[TrainerCallback], optional): Training callbacks.
        decay_epochs (list, optional): Epochs at which the learning rate is
            divided by ``decay_factor``; floats are fractions of
            ``train_args.epochs``. Defaults to ``train_args.decay_epochs``.
            This is the only decay schedule applied (the two are not
            combined).
        decay_factor (float): LR divisor at each decay epoch (default: ``10.0``).
        train_eval_dataset: Optional dataset for an unbiased train-split
            evaluation; falls back to ``train_dataset`` when ``None``.

    Example::

        >>> trainer = GraphTrainer(
        ...     train_args=train_args,
        ...     model_cfg={"name": "gin", "emb_dim": 300},
        ...     train_dataset=train_ds,
        ...     eval_dataset=[val_ds, test_ds],
        ...     metric=metric_fn,
        ...     callbacks=[CLICallback()],
        ...     decay_epochs=[100, 150],
        ...     decay_factor=10.0,
        ... )
        >>> log = trainer.train()
    """

    def __init__(
        self,
        train_args:         TrainingArguments,
        model_cfg:          dict,
        train_dataset,
        eval_dataset:       Optional[List]                               = None,
        metric:             Optional[Callable[..., Mapping[str, float]]] = None,
        callbacks:          Optional[List[TrainerCallback]]              = None,
        decay_epochs:       Optional[List[int]]                          = None,
        decay_factor:       float                                         = 10.0,
        train_eval_dataset                                                = None,
    ):
        """
        Parameters
        ----------
        train_args          : TrainingArguments
        model_cfg           : dict passed to build_gnn_model()
        train_dataset       : PyG-compatible graph dataset (train split)
        eval_dataset        : list of PyG-compatible graph datasets (eval splits)
        metric              : callable (y_true, y_pred) -> dict of floats
        callbacks           : list of TrainerCallback instances
        decay_epochs        : epochs at which to decay the LR, e.g. [100, 200];
                              defaults to train_args.decay_epochs
        decay_factor        : LR divisor applied at each decay epoch (default 10)
        train_eval_dataset  : optional separate dataset for unbiased train-split
                              evaluation; falls back to train_dataset when None
        """
        # Stash GNN-specific state before super().__init__ runs, because the
        # parent calls _get_train_dataloader() which we override below.
        self._model_cfg_gnn      = model_cfg
        self._with_edge_features = False
        self.decay_epochs        = None if decay_epochs is None else [
            int(e * train_args.epochs) if isinstance(e, float) else e for e in decay_epochs
        ]
        self.decay_factor        = decay_factor
        self.train_eval_dataset  = train_eval_dataset

        # The parent will call build_model() (CNN path) and produce a
        # placeholder model.  We swap it out at the start of train().
        super().__init__(
            train_args    = train_args,
            model_cfg     = model_cfg,
            train_dataset = train_dataset,
            eval_dataset  = eval_dataset,
            metric        = metric,
            callbacks     = callbacks,
        )

    def _build_model(self, model_cfg: dict):
        """
        Build a GNN model from libauc.models.

        Required keys
        -------------
        name        : one of gcn | gin | gine | graphsage | gat | mpnn | deepergcn | pna

        Optional keys (inferred automatically)
        ---------------------------------------
        num_tasks   : inferred from ``self.args.num_tasks``; ignored if present in model_cfg
        emb_dim     : node/edge embedding size (default 256)
        num_layers  : number of message-passing layers (default 5)

        Optional keys (forwarded verbatim to the model constructor)
        -----------------------------------------------------------
        graph_pooling, dropout, atom_features_dims, bond_features_dims,
        act, norm, jk, v2 (GAT only),
        aggr / t / learn_t / p / learn_p / block (DeeperGCN only),
        pretrained (bool), pretrained_path (str)

        Returns
        -------
        (model, with_edge_features: bool)
            with_edge_features is inferred from the model's `supports_edge_attr`
            class-level flag, which every model in libauc.models declares.
        """
        name = model_cfg.get("name", "").lower()

        # ── HuggingFace Hub path ────────────────────────────────────────────────
        if "/" in name:
            from torch_geometric.nn import to_captum  # noqa: ensure PyG available
            from transformers import AutoModel

            model = AutoModel.from_pretrained(name, trust_remote_code=True).cuda()

            # Infer edge-feature support from the loaded model
            if hasattr(model, "supports_edge_attr"):
                self._with_edge_features = bool(model.supports_edge_attr)
            else:
                self._with_edge_features = False

            logger.info(f"Loaded HF model '{name}' | with_edge_features={self._with_edge_features}")
            self.model = model
            return
        
        if name not in _GNN_REGISTRY:
            raise ValueError(
                f"Unknown GNN model '{name}'. "
                f"Supported: {list(_GNN_REGISTRY.keys())}"
                f"or a HuggingFace model ID (e.g. 'user/my-gcn')"
            )

        # Import the class from libauc.models
        import libauc.models as libauc_models
        cls_name  = _GNN_REGISTRY[name]
        model_cls = getattr(libauc_models, cls_name, None)
        if model_cls is None:
            raise ImportError(
                f"'{cls_name}' not found in libauc.models. "
                "Make sure your libauc version includes GNN support."
            )

        _META_KEYS = {"name", "num_tasks", "pretrained", "pretrained_remote", "pretrained_path",
                      "init_checkpoint", "init_reset_head", "init_bias_from_prior"}   # not constructor arguments

        constructor_kwargs = {
            k: v for k, v in model_cfg.items() if k not in _META_KEYS
        }
        constructor_kwargs["num_tasks"] = self.args.num_tasks
        constructor_kwargs.setdefault("emb_dim",    256)
        constructor_kwargs.setdefault("num_layers", 5)

        model = model_cls(**constructor_kwargs).cuda()

        if hasattr(model_cls, "supports_edge_attr"):
            with_edge_features = bool(model_cls.supports_edge_attr)
        else:
            with_edge_features = name in {"deepergcn", "gine", "gat", "mpnn", "pna"}

        # ---- optional warm-start -----------------------------------------------
        if model_cfg.get("pretrained", False):
            pretrained_path = model_cfg.get("pretrained_path")
            if not pretrained_path:
                raise ValueError(
                    "pretrained=True but 'pretrained_path' is not set in model_cfg."
                )
            state_dict = torch.load(pretrained_path, weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            msg = model.load_state_dict(state_dict, strict=False)
            logger.info(f"GNN pretrained weights loaded: {msg}")
            model.graph_pred_linear.reset_parameters()

        logger.info(
            f"Built {cls_name} | emb_dim={constructor_kwargs['emb_dim']} "
            f"| num_layers={constructor_kwargs['num_layers']} "
            f"| with_edge_features={with_edge_features}"
        )
        
        self._with_edge_features  = with_edge_features
        self.model = model

    def _decay_epochs(self) -> List[int]:
        return self.args.decay_epochs if self.decay_epochs is None else self.decay_epochs

    # ------------------------------------------------------------------
    # DataLoader overrides – use PyG's DataLoader
    # ------------------------------------------------------------------

    def _get_train_dataloader(self, train_args: TrainingArguments):
        from torch_geometric.loader import DataLoader as PyGDataLoader
        from libauc.sampler import DualSampler, DistributedDualSampler

        sampler_cls = DistributedDualSampler if self.distributed else DualSampler
        sampler = sampler_cls(
            self.train_dataset,
            train_args.batch_size,
            sampling_rate=train_args.sampling_rate,
        )

        loader = PyGDataLoader(
            self.train_dataset,
            batch_size  = train_args.batch_size,
            sampler     = sampler,
            num_workers = train_args.num_workers,
        )
        return sampler, loader

    def _get_eval_dataloader(self, dataset, train_args: TrainingArguments):
        from torch_geometric.loader import DataLoader as PyGDataLoader

        return PyGDataLoader(
            dataset,
            batch_size  = train_args.eval_batch_size,
            shuffle     = False,
            sampler     = ShardSampler(dataset, self.rank, self.world_size) if self.distributed else None,
            num_workers = train_args.num_workers,
        )

    # ------------------------------------------------------------------
    # GNN forward-pass helper
    # ------------------------------------------------------------------

    def _forward(self, model, batch):
        """
        Run a forward pass using the signature appropriate for this architecture.

            supports_edge_attr=False → model(x, edge_index, batch)
            supports_edge_attr=True  → model(x, edge_index, edge_attr, batch)

        Returns sigmoid probabilities, shape [N, num_tasks].
        """
        if self._with_edge_features:
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        else:
            logits = model(batch.x, batch.edge_index, batch.batch)
        return torch.sigmoid(logits)

    # ------------------------------------------------------------------
    # Training (the loop itself is Trainer.train)
    # ------------------------------------------------------------------

    def _compute_loss(self, model, batch):
        """Loss of one PyG training batch (``_forward`` returns sigmoid probabilities)."""
        data, targets, index = batch
        data, targets, index = data.cuda(), targets.cuda(), index.cuda()
        pred = self._forward(model, data)
        if self.args.loss == "CrossEntropyLoss":
            raise ValueError("CrossEntropyLoss not supported for GNNs.")
        if self.args.loss == "BCELoss":
            return self.loss_fn(pred, targets.reshape(-1, 1))
        return self.loss_fn(pred, targets, index=index)

    # ------------------------------------------------------------------
    # Model config / initialisation hooks (see Trainer._initialize_model)
    # ------------------------------------------------------------------

    def _model_cfg_keys(self):
        return None     # remaining keys are forwarded to the GNN constructor

    def _is_head_key(self, key: str) -> bool:
        return key.startswith("graph_pred_linear.")

    def _head_layers(self, model):
        head = getattr(model, "graph_pred_linear", None)
        if head is None:
            return super()._head_layers(model)
        return [head] if isinstance(head, torch.nn.Linear) else [m for m in head.modules() if isinstance(m, torch.nn.Linear)][-1:]

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _eval_single_loader(self, model, loader):
        """
        Run inference over *loader* and compute the registered metric.

        Returns
        -------
        (metrics_dict, y_true np.ndarray, y_pred np.ndarray)
        """
        pred_list, true_list = [], []

        with torch.no_grad():
            for batch in loader:
                data, targets, index = batch
                data = data.cuda()
                targets = targets.cuda()
                pred    = self._forward(model, data)
                pred_list.append(pred.cpu().detach().numpy())
                true_list.append(targets.cpu().detach().numpy())

        y_true = np.concatenate(true_list)
        y_pred = np.concatenate(pred_list)
        if self.distributed:
            y_true, y_pred = gather_shards(y_true, y_pred)

        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        if y_true.ndim > 1:
            y_true = y_true.flatten()

        metrics = self.metric(y_true, y_pred) if self.metric else {}
        return metrics, y_true, y_pred

    def evaluate(self, loader, model):
        """
        Override base Trainer.evaluate() to use the GNN forward pass.

        Args
        ----
        loader : PyG DataLoader
        model  : GNN model

        Returns
        -------
        (metrics_dict, y_true, y_pred)
        """
        return self._eval_single_loader(model, loader)