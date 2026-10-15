import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .series_transforms import build_series_transform

logger = logging.getLogger(__name__)


class MedicalStudyCSVDataset(Dataset):
    """One item is a study made from several cached image series."""

    def __init__(
        self,
        frame,
        target_cols,
        study_id_col="study_id",
        series_path_col="series_path",
        image_root="",
        npz_key="volume",
        transform=None,
        series_slots=(),
        random_fallback=False,
        metadata_cols=None,
        return_metadata=False,
        drop_missing_targets=True,
    ):
        frame = frame.copy()
        target_cols = list(target_cols)

        if drop_missing_targets:
            targets = frame.groupby(study_id_col)[target_cols].first()
            valid_studies = targets.dropna().index
            frame = frame[frame[study_id_col].isin(valid_studies)]

        self.frame = frame.reset_index(drop=True)
        self.target_cols = target_cols
        self.study_id_col = study_id_col
        self.series_path_col = series_path_col
        self.image_root = image_root
        self.npz_key = npz_key
        self.transform = transform
        self.series_slots = list(series_slots)
        self.random_fallback = random_fallback
        self.metadata_cols = metadata_cols or []
        self.return_metadata = return_metadata

        # Group the frame by study ID and store the groups in a dictionary
        self.groups = {
            study_id: group.reset_index(drop=True)
            for study_id, group in self.frame.groupby(study_id_col, sort=False)
        }

        self.study_ids = [
            study_id
            for study_id, group in self.groups.items()
            if self._select_series(group)
        ]
        self.targets = np.stack([
            self.groups[study_id][target_cols].iloc[0].to_numpy(dtype=np.float32)
            for study_id in self.study_ids
        ])

        logger.info(
            "Loaded %d studies, %d series", len(self.study_ids), len(self.frame)
        )

    def __len__(self):
        return len(self.study_ids)

    def _select_series(self, group):
        # Select series for study based on the defined slots. Should change this later.
        selected = []
        used = set()

        for index, slot in enumerate(self.series_slots):
            available = group[~group.index.isin(used)]
            matches = available["plane"].str.lower() == slot["plane"].lower()
            if "fluid" in slot:
                matches &= available["fluid"] == slot["fluid"]
            if "fatsat" in slot:
                matches &= available["fatsat"] == slot["fatsat"]
            if "weight" in slot:
                weights = slot["weight"]
                if not isinstance(weights, list):
                    weights = [weights]
                matches &= available["weight"].str.lower().isin(
                    [weight.lower() for weight in weights]
                )
            candidates = available[matches]

            if candidates.empty and slot.get("fallback") == "non_fatsat":
                candidates = available[
                    (available["plane"].str.lower() == slot["plane"].lower())
                    & (available["fatsat"] == False)
                ]
            if candidates.empty:
                continue
            # Prefer series with more slices and smaller pixel spacing
            chosen = candidates.sort_values(
                ["n_slices", "pixel_spacing_row", "pixel_spacing_column"],
                ascending=[False, True, True],
                na_position="last",
            ).iloc[0]
            selected.append((index, chosen))
            used.add(chosen.name)

        filled = {index for index, _ in selected}
        empty_slots = [
            index for index in range(len(self.series_slots)) if index not in filled
        ]
        remaining = group[~group.index.isin(used)]
        if self.random_fallback:
            remaining = remaining.iloc[torch.randperm(len(remaining)).tolist()]
        else:
            remaining = remaining.sort_values(
                ["n_slices", "pixel_spacing_row", "pixel_spacing_column"],
                ascending=[False, True, True],
                na_position="last",
            )
        for slot, (_, row) in zip(empty_slots, remaining.iterrows()):
            selected.append((slot, row))
        return selected

    def _load_series(self, path):
        if not os.path.isabs(path):
            path = os.path.join(self.image_root, path)
        with np.load(path) as data:
            volume = data[self.npz_key]
        if volume.ndim == 3:
            volume = volume[:, None]
        volume = torch.from_numpy(volume).float()
        return self.transform(volume) if self.transform else volume

    def __getitem__(self, index):

        study_id = self.study_ids[index]

        # RSNA KNEE task specific selection. Should be changed to use all series in the future.
        selected = self._select_series(self.groups[study_id])
        series = [None] * len(self.series_slots)
        for slot, row in selected:
            series[slot] = self._load_series(str(row[self.series_path_col]))

        # Not removed None, because we want to keep the order of series_slots, incase if we want to use that information
        sample = {
            "series": series,
            "targets": torch.from_numpy(self.targets[index]),
            "index": index
        }
        if self.return_metadata:
            row = self.groups[study_id].iloc[0]
            sample["metadata"] = {name: row[name] for name in self.metadata_cols}
        return sample


def sample_study_slices(series, count, random=False):
    # Will be used in training loop to sample slices from series.
    volume = torch.cat([volume for volume in series if volume is not None])
    if random:
        indices = torch.multinomial(
            torch.ones(len(volume)), count, replacement=len(volume) < count
        ).sort().values
    else:
        indices = torch.linspace(0, len(volume) - 1, count).round().long()
    return volume[indices]


def study_windows(series, count=None, random=False, window_slices=1):
    """Model inputs of one study: ``(windows [N, C, H, W], slot ids [N])``.

    Centres are drawn like :func:`sample_study_slices` over the study's
    concatenated slices (``count`` of them, or every slice when ``count`` is
    ``None``), so ``window_slices=1`` returns the same slices. With
    ``window_slices=3`` each window is 2.5D: channel ``c`` comes from slice
    ``centre - 1 + c`` of the same series (repeated at the series edges). The
    series are expected to be channel-replicated and normalised per channel
    already (``channels`` + ``normalize`` transforms), so this equals stacking
    three grey slices and then normalising. The slot id is the index of the
    window's series in ``series`` (its ``series_slots`` position).
    """
    present = [(slot, volume) for slot, volume in enumerate(series) if volume is not None]
    lengths = torch.tensor([len(volume) for _, volume in present])
    total = int(lengths.sum())
    if count is None:
        centres = torch.arange(total)
    elif random:
        centres = torch.multinomial(torch.ones(total), count, replacement=total < count).sort().values
    else:
        centres = torch.linspace(0, total - 1, count).round().long()
    starts = torch.cumsum(lengths, 0) - lengths
    which = torch.searchsorted(starts, centres, right=True) - 1        # series of each centre
    local = centres - starts[which]
    windows, slots = [], []
    for j, (slot, volume) in enumerate(present):
        k = local[which == j]
        if len(k) == 0:
            continue
        if window_slices == 1:
            windows.append(volume[k])
        else:
            if volume.shape[1] != window_slices:
                raise ValueError(f"window_slices={window_slices} needs {window_slices}-channel series "
                                 f"(add a 'channels: {window_slices}' transform), got {volume.shape[1]}")
            half = window_slices // 2
            n = len(volume)
            windows.append(torch.stack([volume[(k + offset).clamp(0, n - 1), channel]
                                        for channel, offset in enumerate(range(-half, half + 1))], dim=1))
        slots.append(torch.full((len(k),), slot, dtype=torch.long))
    return torch.cat(windows), torch.cat(slots)


def load_medical_study_csv_datasets(kwargs, splits, preprocessing=None):
    kwargs = dict(kwargs)
    csv_path = kwargs.pop("csv_path")
    split_col = kwargs.pop("split_col", "split")
    train_split = kwargs.pop("train_split", "train")
    frame = pd.read_csv(csv_path)
    preprocessing = preprocessing or {}

    train = MedicalStudyCSVDataset(
        frame[frame[split_col] == train_split],
        transform=build_series_transform(preprocessing.get("train")),
        random_fallback=True,
        **kwargs,
    )

    # Right evaluation is selecting series based on slots. Should be changed to use all series in the future.
    evaluation = [
        MedicalStudyCSVDataset(
            frame[frame[split_col] == split],
            transform=build_series_transform(
                preprocessing.get(split, preprocessing.get("eval"))
            ),
            random_fallback=False,
            **kwargs,
        )
        for split in splits
    ]
    return train, evaluation
