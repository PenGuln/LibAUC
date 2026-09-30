"""
Custom CSV-backed image dataset support for the GUI's "Custom Dataset..."
option (image backend only -- GNN custom datasets are not yet supported).

Shared between ``auto_trainer.py``'s ``build_pipeline()`` (actual training,
run inside the GUI's training-worker subprocess) and the GUI's "Inspect
Dataset" preview, so both use identical loading logic.
"""
from typing import Any, Dict, List, Tuple


def load_custom_csv_dataset(kwargs: Dict[str, Any], splits: List[str]) -> Tuple[Any, List[Any]]:
    """
    Build a train dataset + one eval dataset per requested split from a
    user-supplied CSV/image-folder layout, via libauc's
    :class:`~libauc.trainer.data.datasets.MedicalImageCSVDataset`.

    Args:
        kwargs: dataset.kwargs from the config -- expected keys:
            ``train_csv`` (required), ``val_csv``/``test_csv`` (required for
            whichever splits are requested), ``image_root``, ``image_col``
            (default ``"image_path"``), ``label_col`` (default ``"label"``),
            ``image_size`` (default ``224``).
        splits: eval splits to build, e.g. ``["val"]``.

    Returns:
        ``(train_dataset, eval_datasets)``, matching
        ``libauc.trainer.load_dataset()``'s return shape.
    """
    from libauc.trainer.data.datasets import (
        MedicalImageCSVDataset,
        _medical_test_transform,
        _medical_train_transform,
    )

    image_size = int(kwargs.get("image_size", 224))
    image_root = kwargs.get("image_root", "")
    image_col = kwargs.get("image_col", "image_path")
    label_col = kwargs.get("label_col", "label")

    train_csv = kwargs.get("train_csv")
    if not train_csv:
        raise ValueError("Custom dataset requires a 'train_csv' path in dataset.kwargs.")

    train_dataset = MedicalImageCSVDataset(
        source=train_csv, image_root=image_root, image_col=image_col,
        label_col=label_col, transform=_medical_train_transform(image_size),
    )

    split_csv_key = {"val": "val_csv", "test": "test_csv"}
    eval_datasets = []
    for split in splits:
        csv_key = split_csv_key.get(split)
        csv_path = kwargs.get(csv_key) if csv_key else None
        if not csv_path:
            raise ValueError(
                f"Custom dataset requires a '{csv_key or split}' CSV path in "
                f"dataset.kwargs to evaluate split '{split}'."
            )
        eval_datasets.append(MedicalImageCSVDataset(
            source=csv_path, image_root=image_root, image_col=image_col,
            label_col=label_col, transform=_medical_test_transform(image_size),
        ))
    return train_dataset, eval_datasets
