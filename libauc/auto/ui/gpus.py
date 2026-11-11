"""
GPU status for the GUI's GPU picker, read from ``nvidia-smi`` (no NVML
bindings needed, and no CUDA context is created in the GUI process).

GPUs are identified by UUID: a worker is pinned to its GPUs with
``CUDA_VISIBLE_DEVICES=GPU-<uuid>,...``, which means the same GPU to every
process regardless of index order.
"""
import os
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import psutil

_TIMEOUT = 10  # seconds


@dataclass
class GPUProcess:
    pid: int
    user: str
    memory_mib: Optional[float]


@dataclass
class GPUInfo:
    index: int
    uuid: str
    name: str
    utilization: Optional[float]  # percent
    memory_used: Optional[float]  # MiB
    memory_total: Optional[float]  # MiB
    temperature: Optional[float]  # Celsius
    allowed: bool = True  # inside this process's CUDA_VISIBLE_DEVICES
    processes: List[GPUProcess] = field(default_factory=list)


def _number(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None


def _rows(output: str) -> List[List[str]]:
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line.strip()]


def _run(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=_TIMEOUT, check=True).stdout


def _allowed(gpu: GPUInfo, visible: Optional[str]) -> bool:
    """Whether ``CUDA_VISIBLE_DEVICES`` (of this process, e.g. set by Slurm for
    the job's allocation) includes ``gpu``. Integer entries are compared with
    nvidia-smi's index, which matches CUDA's order on the usual identical-GPU
    nodes."""
    if visible is None:
        return True
    for entry in (e.strip() for e in visible.split(",")):
        if entry.startswith(("GPU-", "MIG-")) and gpu.uuid.startswith(entry):
            return True
        if entry.isdigit() and int(entry) == gpu.index:
            return True
    return False


def _user(pid: int) -> str:
    try:
        return psutil.Process(pid).username()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "?"


def query_gpus() -> Tuple[List[GPUInfo], Optional[str]]:
    """
    Returns ``(gpus, error)``: the GPUs on this machine, or an empty list and
    why they couldn't be listed (no driver, no nvidia-smi, ...).
    """
    try:
        gpu_rows = _rows(_run(
            "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ))
        app_rows = _rows(_run("--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"))
    except FileNotFoundError:
        return [], "nvidia-smi was not found"
    except subprocess.CalledProcessError as exc:
        message = (exc.stdout or exc.stderr or "").strip().splitlines()
        return [], message[0] if message else f"nvidia-smi exited with code {exc.returncode}"
    except subprocess.TimeoutExpired:
        return [], f"nvidia-smi did not answer within {_TIMEOUT}s"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = []
    for row in gpu_rows:
        if len(row) < 7:
            continue
        index, uuid, name, utilization, used, total, temperature = row[:7]
        gpu = GPUInfo(int(index), uuid, name, _number(utilization), _number(used), _number(total), _number(temperature))
        gpu.allowed = _allowed(gpu, visible)
        gpus.append(gpu)

    by_uuid = {gpu.uuid: gpu for gpu in gpus}
    for row in app_rows:
        if len(row) < 3 or row[0] not in by_uuid or not row[1].isdigit():
            continue
        pid = int(row[1])
        by_uuid[row[0]].processes.append(GPUProcess(pid, _user(pid), _number(row[2])))
    return gpus, None
