from . import functional as F
from .base import BaseTransform

try:
    from scipy.special import expit
except ModuleNotFoundError:
    # fake expit if scipy is not installed
    import numpy as np
    expit = lambda x: 1 / (1 + np.exp(-x))

__all__ = [
    "ToTensor",
    "ToInt",
    "Select",
    "ZeroMean",
    "CutWaveform",
    "SoftClip",
]

class ToTensor(BaseTransform):
    """Converts a numpy.ndarray C x S) to a torch.FloatTensor of shape (C x S).

    """

    def __call__(self, wav):
        return F._to_tensor(wav)


class ToInt(BaseTransform):
    """Convert a label to int based on the given lookup table.

    Args:
        lookup (dict): Lookup table to convert a label to int.

    """

    def __init__(self, lookup):
        if isinstance(lookup, dict):
            self.lookup = lookup
        else:
            raise ValueError("Lookup table needs to be a dictionary.")
        if any([not isinstance(val, int) for val in self.lookup.values()]):
            raise ValueError("Values of the lookup table need to be Int.")

    def __call__(self, label):
        return self.lookup[label]


class Select(BaseTransform):
    """Select an item from the iterable of each datapoint.

    Args:
        index (int): Index to select the item.
    """

    def __init__(self, index):
        if isinstance(index, int):
            self.index = index
        else:
            raise ValueError("Index needs to be a int.")

    def __call__(self, label):
        return label[self.index]


class SoftClip(BaseTransform):
    """Soft clip input to compress large amplitude signals.

    """

    def __init__(self, scale=1.):
        if isinstance(scale, float) or isinstance(scale, int):
            self.scale = scale
        else:
            raise ValueError("Scale needs to be a number.")

    def __call__(self, wav):
        return expit(wav * self.scale)


class ZeroMean(BaseTransform):
    """Remove mean from each waveforms.

    """

    def __call__(self, wav):
        wav = wav.astype(float).T
        wav -= wav.mean(axis=0)
        return wav.T


class CutWaveform(BaseTransform):
    """Cut a portion of the input waveform.

    """

    def __init__(self, samplestart, sampleend):
        self.start = int(samplestart)
        self.end = int(sampleend)

    def __call__(self, wav):
        return wav[:, self.start:self.end]
