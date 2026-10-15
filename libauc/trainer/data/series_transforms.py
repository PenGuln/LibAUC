from functools import partial

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, InterpolationMode, Resize
from torchvision.transforms import functional as TF


def resize(x, size, interp="bilinear"):
    return Resize(
        (size, size), interpolation=InterpolationMode(interp), antialias=True
    )(x)


def cv2_resize(x, size, interp="area"):
    modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
    }
    shape = x.shape
    images = x.reshape(-1, shape[-2], shape[-1]).numpy()
    images = np.stack([
        cv2.resize(image, (size, size), interpolation=modes[interp])
        for image in images
    ])
    return torch.from_numpy(images).reshape(*shape[:-2], size, size)


def rotate(x, degrees, interp="bilinear"):
    angle = torch.empty(1).uniform_(-degrees, degrees).item()
    return TF.rotate(x, angle, interpolation=InterpolationMode(interp))


def affine(x, degrees, scale, translate, interp="bilinear"):
    angle = torch.empty(1).uniform_(-degrees, degrees).item()
    scale = torch.empty(1).uniform_(scale[0], scale[1]).item()
    shift = [
        int(torch.empty(1).uniform_(-translate[0], translate[0]).item() * x.shape[-1]),
        int(torch.empty(1).uniform_(-translate[1], translate[1]).item() * x.shape[-2]),
    ]
    return TF.affine(
        x, angle, shift, scale, [0.0, 0.0],
        interpolation=InterpolationMode(interp),
    )


def contrast(x, amount, scope="series"):
    shape = (len(x), 1, 1, 1) if scope == "slice" else (1, 1, 1, 1)
    factor = x.new_empty(shape).uniform_(1 - amount, 1 + amount)
    mean = x.mean((-2, -1), keepdim=True)
    return (mean + factor * (x - mean)).clamp(0, 1)


def intensity(x, brightness=0.0, contrast=0.0, scope="series"):
    shape = (len(x), 1, 1, 1) if scope == "slice" else (1, 1, 1, 1)
    brightness = x.new_empty(shape).uniform_(1 - brightness, 1 + brightness)
    contrast = x.new_empty(shape).uniform_(1 - contrast, 1 + contrast)
    mean = x.mean((-2, -1), keepdim=True)
    return ((mean + contrast * (x - mean)) * brightness).clamp(0, 1)


def gamma(x, lo, hi, scope="series"):
    shape = (len(x), 1, 1, 1) if scope == "slice" else (1, 1, 1, 1)
    value = x.new_empty(shape).uniform_(lo, hi)
    return x.clamp(0, 1).pow(value)


def channels(x, count):
    if count == 3 and x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    return x


def normalize(x, mean, std):
    mean = x.new_tensor(mean)[None, :, None, None]
    std = x.new_tensor(std)[None, :, None, None]
    return (x - mean) / std


def percentile_normalize(x, lo=1.0, hi=99.0, scope="series"):
    if scope == "slice":
        values = x.flatten(1)
        low = torch.quantile(values, lo / 100, dim=1)[:, None, None, None]
        high = torch.quantile(values, hi / 100, dim=1)[:, None, None, None]
    else:
        low, high = torch.quantile(x, x.new_tensor([lo / 100, hi / 100]))
    return ((x - low) / (high - low + 1e-8)).clamp(0, 1)


TRANSFORMS = {
    "resize": resize,
    "cv2_resize": cv2_resize,
    "rotate": rotate,
    "affine": affine,
    "contrast": contrast,
    "intensity": intensity,
    "gamma": gamma,
    "channels": channels,
    "normalize": normalize,
    "percentile_normalize": percentile_normalize,
}


def build_series_transform(config):
    transforms = []
    for item in config or []:
        item = dict(item)
        name = item.pop("name")
        transforms.append(partial(TRANSFORMS[name], **item))
    return Compose(transforms)
