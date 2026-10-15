"""Offline preprocessing for DICOM series."""

import os
import re
from pathlib import Path

import numpy as np
import torch

from .series_transforms import cv2_resize


def percentile_scale(volume, lower=1.0, upper=99.0):
    """Clip one complete series to percentile bounds and scale it to [0, 1]."""
    volume = np.asarray(volume, dtype=np.float32)
    low, high = np.percentile(volume, [lower, upper])
    if high <= low:
        raise ValueError("Series has no intensity variation.")
    return np.clip((volume - low) / (high - low), 0.0, 1.0).astype(np.float32)


def crop_series_to_physical_fov(volume, pixel_spacing, field_of_view_mm=130.0):
    """Center-crop every slice to a fixed physical field of view."""
    volume = np.asarray(volume, dtype=np.float32)
    spacing = np.asarray(pixel_spacing, dtype=np.float64)
    crop_rows = max(1, int(round(field_of_view_mm / spacing[0])))
    crop_columns = max(1, int(round(field_of_view_mm / spacing[1])))
    _, rows, columns = volume.shape
    row_padding = max(0, crop_rows - rows)
    column_padding = max(0, crop_columns - columns)
    padded_flag = (row_padding > 0) or (column_padding > 0)
    pad_top = row_padding // 2
    pad_left = column_padding // 2
    padded = np.pad( 
        volume, 
        ((0, 0), (pad_top, row_padding - pad_top), (pad_left, column_padding - pad_left),),
        constant_values=float(volume.min()),
    )
    top = (padded.shape[1] - crop_rows) // 2
    left = (padded.shape[2] - crop_columns) // 2
    return padded[:, top:top + crop_rows, left:left + crop_columns], padded_flag


def read_dicom_series(series_directory, return_metadata=False, threads=1):
    """Read, rescale, and anatomically order one DICOM series.

    ``threads > 1`` reads the files concurrently (same result): on network or
    FUSE-mounted storage each file open is latency-bound, so a series of ~30
    files otherwise pays that latency 30 times in a row.
    """
    
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut

    paths = sorted(path for path in Path(series_directory).rglob("*") if path.is_file())
    dicom_paths = [path for path in paths if path.suffix.lower() == ".dcm"]
    paths = dicom_paths or paths
    if not paths:
        raise FileNotFoundError(f"No DICOM files found under {series_directory}.")

    def read_one(path):
        dataset = pydicom.dcmread(str(path))
        pixels = apply_modality_lut(dataset.pixel_array, dataset).astype(np.float32)
        photometric = str(getattr(dataset, "PhotometricInterpretation", "")).upper()
        if photometric == "MONOCHROME1":
            pixels = pixels.max() - pixels

        spacing = getattr(dataset, "PixelSpacing", None)
        position = getattr(dataset, "ImagePositionPatient", None)
        orientation = getattr(dataset, "ImageOrientationPatient", None)
        return {
            "path": str(path),
            "pixels": pixels,
            "pixel_spacing": None if spacing is None else np.asarray(spacing, dtype=float),
            "position": None if position is None else np.asarray(position, dtype=float),
            "orientation": None if orientation is None else np.asarray(orientation, dtype=float),
            "slice_location": getattr(dataset, "SliceLocation", None),
            "instance_number": getattr(dataset, "InstanceNumber", None),
        }

    if threads > 1 and len(paths) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(int(threads), len(paths))) as pool:
            slices = list(pool.map(read_one, paths))      # input order kept
    else:
        slices = [read_one(path) for path in paths]

    orientation = slices[0]["orientation"]
    if orientation is not None and all(x["position"] is not None for x in slices):
        normal = np.cross(orientation[:3], orientation[3:])
        normal = normal / np.linalg.norm(normal)
        positions = [float(np.dot(x["position"], normal)) for x in slices]
    elif all(x["slice_location"] is not None for x in slices):
        positions = [float(x["slice_location"]) for x in slices]
    else:
        positions = [float(x["instance_number"]) for x in slices]

    ordered = sorted(zip(positions, slices), key=lambda x: (x[0], x[1]["path"]))
    positions = [position for position, _ in ordered]
    slices = [item for _, item in ordered]
    spacing = slices[0]["pixel_spacing"]
    if spacing is None:
        raise ValueError("PixelSpacing is required for the physical crop.")

    volume = np.stack([item["pixels"] for item in slices])
    metadata = {
        "pixel_spacing": spacing.astype(np.float32),
        "slice_positions": np.asarray(positions, dtype=np.float32),
    }
    return (volume, metadata) if return_metadata else volume


_T1 = re.compile(r"\bt1\b|\bt1w\b|\bt1ffe\b")
_T2 = re.compile(r"\bt2\b|\bt2w\b")
_PD = re.compile(r"\bpd\b|\bpdw\b|proton|\bdp\b|dens")


def _number(dataset, tag):
    try:
        return float(getattr(dataset, tag))
    except (TypeError, ValueError, AttributeError):
        return float("nan")


def probe_dicom_series(series_directory):
    """Series-level metadata used to pick series for a study.

    Reads the header of the middle file only and returns ``weight`` (``T1``,
    ``T2``, ``PD``, ``GRE`` or ``UNK``, from the description, falling back to
    TR/TE), ``n_slices`` and the pixel spacing, or ``None`` if the header
    can't be read.
    """
    import pydicom

    series_directory = Path(series_directory)
    files = sorted(path for path in series_directory.iterdir() if path.is_file())
    dicom_files = [path for path in files if path.suffix.lower() == ".dcm"]
    files = dicom_files or files

    try:
        dataset = pydicom.dcmread(str(files[len(files) // 2]), stop_before_pixels=True)
    except Exception:
        return None

    description = str(getattr(dataset, "SeriesDescription", ""))
    sequence_name = str(getattr(dataset, "SequenceName", ""))
    text = re.sub(r"[_\-./]", " ", f"{description} {sequence_name}".lower())

    repetition_time = _number(dataset, "RepetitionTime")
    echo_time = _number(dataset, "EchoTime")

    t1 = bool(_T1.search(text))
    t2 = bool(_T2.search(text))
    pd = bool(_PD.search(text))
    if t1 and not t2 and not pd:
        weight = "T1"
    elif t2 and not pd:
        weight = "T2"
    elif pd:
        weight = "PD"
    elif "GR" in str(getattr(dataset, "ScanningSequence", "")).upper():
        weight = "GRE"
    elif repetition_time < 800:
        weight = "T1"
    elif echo_time > 60:
        weight = "T2"
    elif repetition_time >= 800:
        weight = "PD"
    else:
        weight = "UNK"

    spacing = np.asarray(getattr(dataset, "PixelSpacing", [np.nan, np.nan]), dtype=float)
    if spacing.size < 2:
        spacing = np.repeat(spacing, 2)[:2]
    return {
        "weight": weight,
        "n_slices": len(files),
        "pixel_spacing_row": float(spacing[0]),
        "pixel_spacing_column": float(spacing[1]),
    }


def load_dicom_series(
    series_directory,
    image_size=None,
    field_of_view_mm=130.0,
    lower_percentile=1.0,
    upper_percentile=99.0,
    threads=1,
):
    """Read one DICOM series and preprocess it as :func:`preprocess_dicom_series` caches it.

    Returns ``(volume, metadata)``: a float32 ``(slices, H, W)`` volume with a
    fixed physical field of view and intensities in [0, 1], and the series
    metadata plus ``padded_flag``.
    """
    volume, metadata = read_dicom_series(series_directory, return_metadata=True, threads=threads)
    volume, padded_flag = crop_series_to_physical_fov(
        volume,
        metadata["pixel_spacing"],
        field_of_view_mm=field_of_view_mm,
    )
    volume = percentile_scale(volume, lower_percentile, upper_percentile)
    if image_size is not None:
        volume = cv2_resize(
            torch.from_numpy(volume).unsqueeze(1), image_size, "area"
        )[:, 0].numpy()
    metadata["padded_flag"] = padded_flag
    return volume.astype(np.float32), metadata


def preprocess_dicom_series(
    series_directory,
    output_path,
    image_size=None,
    field_of_view_mm=130.0,
    lower_percentile=1.0,
    upper_percentile=99.0,
    compressed=True,
):
    """Convert one DICOM series to a model-neutral ``.npz`` cache.

    The cache contains a fixed physical field of view with series-relative
    intensities in [0, 1]. Model-specific channel expansion, augmentation, and
    mean/std normalization intentionally remain runtime transforms.
    """
    volume, metadata = load_dicom_series(
        series_directory,
        image_size=image_size,
        field_of_view_mm=field_of_view_mm,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    padded_flag = metadata["padded_flag"]
    payload = {
        "volume": volume.astype(np.float32),
        "slice_positions": metadata["slice_positions"],
        "requested_fov_mm": np.asarray(field_of_view_mm, dtype=np.float32),
        "output_image_size": np.asarray(
            -1 if image_size is None else image_size, dtype=np.int32
        ),
        "resize_interpolation": np.asarray("area"),
        "lower_percentile": np.asarray(lower_percentile, dtype=np.float32),
        "upper_percentile": np.asarray(upper_percentile, dtype=np.float32),
        "padded_flag": np.asarray(padded_flag),
    }

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if compressed:
        np.savez_compressed(output_path, **payload)
    else:
        np.savez(output_path, **payload)
    return output_path
