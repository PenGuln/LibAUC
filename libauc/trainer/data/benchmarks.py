"""
Medical evaluation benchmarks: the image classification tasks used to
evaluate the Medicine_Foundation_Model encoders (RADEO etc.), with the same
data, splits and preprocessing as its ``lincls.py`` linear-probe runs.

Two families, both resolved through :func:`libauc.trainer.load_dataset`:

* CSV benchmarks (``aod``, ``brisc2025``, ...): a ``path,class`` CSV per
  split under ``root_path`` (the ``Evaluation_Datasets_Medical_Project``
  folder). ``val`` is a stratified 80:20 split of the train CSV, as
  ``lincls.py`` does, so model selection never sees the test CSV.
* MedMNIST at 224px (``chestmnist_224``, ``dermamnist_224``, ...): the
  official MedMNIST+ splits, read from ``<root_path>/<flag>_224.npz``.

Multi-class targets are one-hot encoded, so they train as one-vs-rest
multi-label AUC tasks; binary tasks keep a single 0/1 target.

Each entry also records the imaging ``modality`` and ``body_part`` that
modality-conditioned encoders such as RADEO take as input.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_SIZE = 224
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# name -> CSVs (relative to root_path), metadata, and for binary tasks the
# class treated as positive. Names are lower-case: load_dataset() lowercases.
CSV_BENCHMARKS: Dict[str, dict] = {
    "aod": {"title": "AOD", "train": "AOD/AOD_train.csv", "test": "AOD/AOD_test.csv",
            "modality": "Fundus", "body_part": "N/A"},
    "brisc2025": {"title": "brisc2025", "train": "brisc2025/brisc2025/brisc2025_train.csv",
                  "test": "brisc2025/brisc2025/brisc2025_test.csv", "modality": "MRI", "body_part": "N/A"},
    "nct-crc-he-100k": {"title": "NCT-CRC-HE-100K", "train": "NCT-CRC-HE-100K/NCT-CRC-HE-100K_train.csv",
                        "test": "NCT-CRC-HE-100K/NCT-CRC-HE-100K_test.csv", "modality": "Microscopy", "body_part": "N/A"},
    "fetal_planes": {"title": "fetal_planes", "train": "fetal_planes/fetal_planes_train.csv",
                     "test": "fetal_planes/fetal_planes_test.csv", "modality": "Ultrasound", "body_part": "N/A"},
    "fitzpatrick17k_nine": {"title": "fitzpatrick17k_nine", "train": "fitzpatrick17k/finalfitz17k_nine_train.csv",
                            "test": "fitzpatrick17k/finalfitz17k_nine_test.csv", "modality": "Dermoscopy", "body_part": "N/A"},
    "kvasir-capsule": {"title": "Kvasir-Capsule", "train": "Kvasir-Capsule/Kvasir-Capsule_train.csv",
                       "test": "Kvasir-Capsule/Kvasir-Capsule_test.csv", "modality": "Endoscopy", "body_part": "N/A"},
    "oasis": {"title": "OASIS", "train": "OASIS/OASIS_train.csv", "test": "OASIS/OASIS_test.csv",
              "modality": "MRI", "body_part": "N/A"},
    "sarscov2_ctscan": {"title": "sarscov2_ctscan", "train": "sarscov2_ctscan/sarscov2_ctscan_train.csv",
                        "test": "sarscov2_ctscan/sarscov2_ctscan_test.csv", "modality": "CT", "body_part": "N/A",
                        "positive": "COVID"},
    "lc25000": {"title": "lc25000", "train": "lc25000/lc25k_train.csv", "test": "lc25000/lc25k_test.csv",
                "modality": "Microscopy", "body_part": "N/A"},
    "oct-c8": {"title": "OCT-C8", "train": "OCT-C8/oct_c8_train.csv", "test": "OCT-C8/oct_c8_test.csv",
               "modality": "OCT", "body_part": "N/A"},
    "rsna-pneumonia": {"title": "rsna-pneumonia", "train": "rsna-pneumonia/rsna_pd_train.csv",
                       "test": "rsna-pneumonia/rsna_test.csv", "modality": "X-Ray", "body_part": "N/A",
                       "positive": "1"},
}

# name -> MedMNIST flag + metadata (as in lincls.py's MEDMNIST_2D_METADATA).
MEDMNIST_BENCHMARKS: Dict[str, dict] = {
    "chestmnist_224": {"title": "ChestMNIST (224)", "flag": "chestmnist", "modality": "X-Ray", "body_part": "Chest"},
    "dermamnist_224": {"title": "DermaMNIST (224)", "flag": "dermamnist", "modality": "Dermoscopy", "body_part": "Skin"},
    "organamnist_224": {"title": "OrganAMNIST (224)", "flag": "organamnist", "modality": "CT", "body_part": "Abdomen"},
    "organcmnist_224": {"title": "OrganCMNIST (224)", "flag": "organcmnist", "modality": "CT", "body_part": "Abdomen"},
    "organsmnist_224": {"title": "OrganSMNIST (224)", "flag": "organsmnist", "modality": "CT", "body_part": "Abdomen"},
    "bloodmnist_224": {"title": "BloodMNIST (224)", "flag": "bloodmnist", "modality": "Microscopy", "body_part": "N/A"},
    "tissuemnist_224": {"title": "TissueMNIST (224)", "flag": "tissuemnist", "modality": "Microscopy", "body_part": "N/A"},
}

BENCHMARKS: Dict[str, dict] = {**CSV_BENCHMARKS, **MEDMNIST_BENCHMARKS}


def _transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """lincls.py's linear-probe transforms (ImageNet normalisation)."""
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
    train = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        normalize,
    ])
    return train, test


def _encode(class_ids: np.ndarray, num_classes: int) -> np.ndarray:
    """0/1 targets for binary tasks, one-hot ``[N, K]`` targets otherwise."""
    class_ids = np.asarray(class_ids).reshape(-1).astype(np.int64)
    if num_classes == 2:
        return class_ids.astype(np.float32)
    return np.eye(num_classes, dtype=np.float32)[class_ids]


class _BenchmarkDataset(Dataset):
    """Yields ``(image, target, index)``; ``index`` may be the
    ``(index, task_id)`` pair TriSampler passes for multi-label targets."""

    targets: np.ndarray

    def _image(self, idx: int) -> Image.Image:
        raise NotImplementedError

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        task_id = None
        if isinstance(idx, (tuple, list)):
            idx, task_id = idx
        image = self._image(idx)
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[idx].reshape(-1), (idx, task_id) if task_id is not None else idx


class ClassCSVDataset(_BenchmarkDataset):
    """A ``path,class`` CSV (class names as strings, one class per image).
    Relative image paths are resolved against ``image_root``."""

    def __init__(self, df: pd.DataFrame, classes: List[str], transform, image_root: str = "",
                 positive: Optional[str] = None):
        self.paths = df["path"].tolist()
        self.image_root = image_root
        self.transform = transform
        self.classes = list(classes)
        if positive is not None:
            # Binary task: put the positive class at index 1.
            self.classes = [c for c in self.classes if c != positive] + [positive]
        index = {c: i for i, c in enumerate(self.classes)}
        unknown = sorted(set(df["class"]) - set(index))
        if unknown:
            raise ValueError(f"Classes {unknown} are not in the training classes {self.classes}")
        self.targets = _encode(df["class"].map(index).to_numpy(), len(self.classes))

    def _image(self, idx):
        path = self.paths[idx]
        if not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        return Image.open(path).convert("RGB")


class MedMNISTDataset(_BenchmarkDataset):
    """A MedMNIST split, with labels read from the ``.npz`` (not by loading
    every image) and multi-class labels one-hot encoded."""

    def __init__(self, flag: str, split: str, root: str, transform):
        import medmnist

        info = medmnist.INFO[flag]
        self.data = getattr(medmnist, info["python_class"])(
            split=split, root=root, size=IMAGE_SIZE, as_rgb=True, download=False,
        )
        self.transform = transform
        self.classes = [info["label"][k] for k in sorted(info["label"], key=int)]
        labels = np.asarray(self.data.labels)
        if info["task"] == "multi-label, binary-class":
            self.targets = labels.astype(np.float32)
        else:
            self.targets = _encode(labels, len(self.classes))

    def _image(self, idx):
        return self.data[idx][0]  # medmnist returns (PIL image, label) with transform=None


def _stratified_split(df: pd.DataFrame, val_frac: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class shuffle and split, matching lincls.py's CSVDataset.stratified_split."""
    train_parts, val_parts = [], []
    for _, group in df.groupby("class"):
        group = group.sample(frac=1.0, random_state=seed)
        n_val = max(1, int(round(len(group) * val_frac))) if len(group) > 1 else 0
        val_parts.append(group.iloc[:n_val])
        train_parts.append(group.iloc[n_val:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(val_parts).reset_index(drop=True)


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "path" not in df.columns or "class" not in df.columns:
        raise ValueError(f"{path}: expected 'path' and 'class' columns, got {list(df.columns)}")
    df["class"] = df["class"].astype(str)
    return df


def load_benchmark_dataset(name: str, splits: List[str], root_path: str, val_frac: float = 0.2,
                           val_seed: int = 42, **_) -> Tuple[Dataset, List[Dataset]]:
    """
    Load benchmark ``name`` (a key of :data:`BENCHMARKS`).

    Args:
        name: benchmark name, lower-case.
        splits: eval splits to return: ``"val"`` and/or ``"test"``.
        root_path: the ``Evaluation_Datasets_Medical_Project`` folder for
            CSV benchmarks, or the folder holding ``<flag>_224.npz`` for
            MedMNIST ones.
        val_frac, val_seed: size and seed of the stratified val split carved
            from a CSV benchmark's train CSV.

    Returns:
        ``(train_dataset, eval_datasets)``, as :func:`load_dataset`.
    """
    train_tf, test_tf = _transforms()

    if name in MEDMNIST_BENCHMARKS:
        flag = MEDMNIST_BENCHMARKS[name]["flag"]
        npz = os.path.join(root_path, f"{flag}_{IMAGE_SIZE}.npz")
        if not os.path.isfile(npz):
            raise FileNotFoundError(f"{npz} not found; root_path must hold the MedMNIST+ {IMAGE_SIZE}px files")
        train = MedMNISTDataset(flag, "train", root_path, train_tf)
        evals = []
        for split in splits:
            if split not in ("val", "test"):
                raise NotImplementedError(f"Split '{split}' is not implemented for dataset '{name}'.")
            evals.append(MedMNISTDataset(flag, split, root_path, test_tf))
        return train, evals

    spec = CSV_BENCHMARKS[name]
    train_csv = os.path.join(root_path, spec["train"])
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"{train_csv} not found; root_path must be the evaluation datasets folder")
    image_root = os.path.dirname(train_csv)
    positive = spec.get("positive")

    full_train = _read_csv(train_csv)
    train_df, val_df = _stratified_split(full_train, val_frac, val_seed)
    classes = sorted(full_train["class"].unique())
    train = ClassCSVDataset(train_df, classes, train_tf, image_root, positive)
    logger.info(f"[{spec['title']}] {len(classes)} classes, train {len(train_df)} / val {len(val_df)}")

    evals = []
    for split in splits:
        if split == "val":
            df = val_df
        elif split == "test":
            df = _read_csv(os.path.join(root_path, spec["test"]))
        else:
            raise NotImplementedError(f"Split '{split}' is not implemented for dataset '{name}'.")
        evals.append(ClassCSVDataset(df, classes, test_tf, image_root, positive))
    return train, evals
