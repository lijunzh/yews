from typing import Any, Callable, Dict, List
import numpy as np
import torch
from numpy import ndarray
from scipy import signal
from scipy.special import expit
from torch import Tensor

__all__ = [
    "Transform",
    "ToTensor",
    "ToInt",
    "ZeroMean",
    "CutWaveform",
    "SoftClip",
    "RemoveTrend",
    "Taper",
    "BandpassFilter",
]


class Transform:
    """An abstract class representing a Transform.

    All other transform should subclass it. All subclasses should override
    ``__call__`` which performs the transform.

    Note:
        A transform-like object has ``__call__`` implmented. Typical
        transform-like objects include python functions and methods.

    """

    def __call__(self, data: Any) -> Any:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = self.__class__.__name__
        content = [f"{key} = {val}" for key, val in self.__dict__.items()]
        body = ", ".join(content)
        return f"{head}({body})"


class Compose(Transform):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.ZeroMean(),
        >>>     transforms.ToTensor(),
        >>> ])

    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data: Any) -> Any:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class ToTensor(Transform):
    """Converts a numpy.ndarray C x S) to a torch.FloatTensor of shape (C x S)."""

    def __call__(self, data: ndarray) -> Tensor:
        data = data[None, :] if data.ndim == 1 else data
        return torch.from_numpy(data).float()


class ToInt(Transform):
    """Convert a label to int based on the given lookup table.

    Args:
        lookup (dict): Lookup table to convert a label to int.

    """

    def __init__(self, lookup_table: Dict[Any, int]):
        self.lookup_table = lookup_table

    def __call__(self, label: Any) -> int:
        return self.lookup_table[label]


class SoftClip(Transform):
    """Soft clip input to compress large amplitude signals."""

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, data: ndarray) -> ndarray:
        return expit(data * self.scale)


class ZeroMean(Transform):
    """Remove mean from each waveforms."""

    def __call__(self, data: ndarray) -> ndarray:
        data = data.astype(float).T
        data -= data.mean(axis=0)
        return data.T


class CutWaveform(Transform):
    """Cut a portion of the input waveform."""

    def __init__(self, sample_start: int, sample_end: int):
        self.start = sample_start
        self.end = sample_end

    def __call__(self, data: ndarray) -> ndarray:
        return data[:, self.start : self.end]


class RemoveTrend(Transform):
    """Remove trend from each waveforms."""

    def __call__(self, data: ndarray) -> ndarray:
        data = signal.detrend(data, axis=-1)
        return data


class Taper(Transform):
    """Add taper in both ends of each waveforms."""

    def __call__(self, data: ndarray, half_taper: float = 0.05) -> ndarray:

        [x, y] = data.shape
        tukey_win = signal.tukey(y, alpha=2 * half_taper, sym=True)
        data = data * tukey_win
        return data


class BandpassFilter(Transform):
    """Apply Bandpass filter to each waveforms."""

    def __call__(
        self,
        data: ndarray,
        delta: float = 0.01,
        order: int = 4,
        lowfreq: int = 2,
        highfreq: int = 16,
    ) -> ndarray:

        nyq = 0.5 * (1 / delta)
        low = lowfreq / nyq
        high = highfreq / nyq
        b, a = signal.butter(order, [low, high], btype="bandpass")
        data = signal.filtfilt(b, a, data, axis=-1, padtype=None, padlen=None, irlen=None)

        return data
