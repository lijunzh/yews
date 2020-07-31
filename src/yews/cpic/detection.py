import numpy as np
import torch
from scipy.special import expit

from .utils import chunks
from .utils import compute_probs
from .utils import sliding_window_view

def find_nonzero_runs(a):
    # source: https://stackoverflow.com/
    # questions/31544129/extract-separate-non-zero-blocks-from-array

    # Create an array that is 1 where a is nonzero, and pad each end with an extra 0.
    isnonzero = np.concatenate(([0], (np.asarray(a) != 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(isnonzero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges - [0, 1]


def detect(waveform, fs, wl, model, transform, g, threshold=0.5,
           batch_size=None, size_limit=None):
    """size_limit is the maximum number of waveform array elements in the
    long dimension to be processed at a time. Can be used when working with
    memory constraints. Should be an integer multiple of fs*wl"""
    if size_limit:
        if not (isinstance(size_limit, int)):
            raise TypeError("size_limit must be type integer")
        if size_limit % (fs*wl) != 0:
            raise ValueError("size_limit must be integer multiple of fs*wl")
        probs_list = []
        offset = int(fs*(wl - g))
        for chunk in chunks(waveform, size_limit, offset):
            probs = compute_probs(model, transform, chunk,
                                  shape=[3, fs * wl],
                                  step=[1, int(g * fs)],
                                  batch_size=batch_size)
            probs_list.append(probs)
        probs = np.concatenate(probs_list, axis=1)

    else:
        probs = compute_probs(model, transform, waveform,
                              shape=[3, fs * wl],
                              step=[1, int(g * fs)],
                              batch_size=batch_size)

    probs[probs < threshold] = 0
    p_prob, s_prob = probs[1:]

    # detect window length
    p = find_nonzero_runs(p_prob)
    s = find_nonzero_runs(s_prob)

    detect_results = {
        'p': p * g + 5,
        's': s * g + 5,
        'detect_p': p_prob,
        'detect_s': s_prob,
    }

    return detect_results
