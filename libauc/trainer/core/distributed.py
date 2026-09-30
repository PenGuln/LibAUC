"""
Helpers for running :class:`~libauc.trainer.Trainer` under
``torch.distributed`` (one process per GPU, ``DistributedDataParallel``).

The Trainer switches to distributed mode automatically when a process group
with more than one rank is initialised before it is constructed, e.g. by
launching with ``torchrun``::

    torchrun --nproc_per_node 4 -m libauc.trainer.run_image_trainer --config_file cfg.yaml

or by AutoMAX's trial workers (:mod:`libauc.auto.parallel`).
"""
import logging
import os
from typing import Iterator, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler

from libauc.utils import is_distributed

logger = logging.getLogger(__name__)

__all__ = ["init_distributed", "cleanup_distributed", "get_rank", "get_world_size", "ShardSampler"]


def init_distributed() -> bool:
    """Initialise the default NCCL process group from ``torchrun``-style env vars.

    Reads ``LOCAL_RANK`` / ``RANK`` / ``WORLD_SIZE`` (plus ``MASTER_ADDR`` /
    ``MASTER_PORT``), pins this process to ``cuda:LOCAL_RANK`` and creates the
    process group. A no-op when ``WORLD_SIZE`` is unset or ``1``, or when the
    group already exists.

    Returns:
        bool: ``True`` if the process is part of a multi-rank group.
    """
    if is_distributed():
        return True
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    logger.info(f"Initialised process group: rank {dist.get_rank()}/{world_size}, cuda:{local_rank}")
    return True


def cleanup_distributed() -> None:
    """Destroy the default process group if one exists."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


class ShardSampler(Sampler):
    """Evaluation sampler giving rank ``r`` the indices ``r, r + W, r + 2W, ...``.

    Unlike :class:`torch.utils.data.distributed.DistributedSampler` it never
    pads shards with duplicate samples, so metrics computed after
    :func:`gather_shards` match single-GPU evaluation exactly.
    """

    def __init__(self, dataset, rank: int, world_size: int):
        self.num_samples = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.num_samples, self.world_size))

    def __len__(self) -> int:
        return len(range(self.rank, self.num_samples, self.world_size))


def gather_shards(*local_arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    """All-gather arrays produced over a :class:`ShardSampler` shard.

    Every rank receives the full arrays, re-interleaved into dataset order
    (rank ``r`` holds rows ``r::W``). Arrays may differ in length by one
    between ranks.
    """
    world_size = dist.get_world_size()
    parts = [None] * world_size
    dist.all_gather_object(parts, local_arrays)

    gathered = []
    for i in range(len(local_arrays)):
        shards = [np.asarray(p[i]) for p in parts]
        total = sum(len(s) for s in shards)
        out = np.empty((total,) + shards[0].shape[1:], dtype=shards[0].dtype)
        for rank, shard in enumerate(shards):
            out[rank::world_size] = shard
        gathered.append(out)
    return tuple(gathered)


def all_reduce_mean(value: float) -> float:
    """Average a Python scalar across ranks (identity when not distributed)."""
    if not is_distributed():
        return value
    t = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(t)
    return (t / dist.get_world_size()).item()
