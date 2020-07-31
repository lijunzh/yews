import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter1d
from scipy.special import expit


def probs2cfs(probs, sigma=3):
    cf_p = np.log10(probs[1] / (probs[0] + 1e-5))
    cf_s = np.log10(probs[2] / (probs[0] + 1e-5))
    cf_p[probs.argmax(axis=0) != 1] = 0
    cf_s[probs.argmax(axis=0) != 2] = 0
    cf_p = gaussian_filter1d(cf_p, sigma=sigma)
    cf_s = gaussian_filter1d(cf_s, sigma=sigma)

    return cf_p, cf_s


def chunks(array, size, offset=0):
    """Yield successive n-sized chunks from array, starting at an offset before
    the end of the previous chunk."""
    if not (isinstance(size, int) and isinstance(offset, int)):
        raise TypeError("Arguments 'size' and 'offset' must be type integer")
    slice_axis = array.ndim - 1
    axis_length = array.shape[slice_axis]
    for i in range(0, axis_length - offset, size - offset):
        range_end = min(i + size, axis_length)
        yield array.take(indices=range(i, range_end), axis=slice_axis)


def compute_probs(model, transform, waveform, shape, step, batch_size=None):
    model.eval()
    with torch.no_grad():
        windows = np.squeeze(sliding_window_view(waveform, shape, step))
        windows = torch.stack([transform(window) for window in windows])
        if batch_size:
            outputs = []
            for batch in chunks(windows, batch_size):
                outputs.append(model(batch))
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs = model(windows)

    if next(model.parameters()).is_cuda:
        outputs = outputs.cpu().numpy()
    else:
        outputs = outputs.numpy()

    probs = expit(outputs).T
    probs /= probs.sum(axis=0)

    return probs


# Copyright: Fanjin Zeng, obtained from https://gist.github.com/Fnjn/b061b28c05b5b0e768c60964d2cafa8d#file-sliding_window_view-py
def sliding_window_view(x, shape, step=None, subok=False, writeable=False):
    """
    Create sliding window views of the N dimensions array with the given window
    shape. Window slides across each dimension of `x` and provides subsets of `x`
    at any window position.
    Parameters
    ----------
    x : ndarray
        Array to create sliding window views.
    shape : sequence of int
        The shape of the window. Must have same length as number of input array dimensions.
    step: sequence of int, optional
        The steps of window shifts for each dimension on input array at a time.
        If given, must have same length as number of input array dimensions.
        Defaults to 1 on all dimensions.
    subok : bool, optional
        If True, then sub-classes will be passed-through, otherwise the returned
        array will be forced to be a base-class array (default).
    writeable : bool, optional
        If set to False, the returned array will always be readonly view.
        Otherwise it will return writable copies(see Notes).
    Returns
    -------
    view : ndarray
        Sliding window views (or copies) of `x`. view.shape = (x.shape - shape) // step + 1
    See also
    --------
    as_strided: Create a view into the array with the given shape and strides.
    broadcast_to: broadcast an array to a given shape.
    Notes
    -----
    ``sliding_window_view`` create sliding window views of the N dimensions array
    with the given window shape and its implementation based on ``as_strided``.
    Please note that if writeable set to False, the return is views, not copies
    of array. In this case, write operations could be unpredictable, so the return
    views is readonly. Bear in mind, return copies (writeable=True), could possibly
    take memory multiple amount of origin array, due to overlapping windows.
    For some cases, there may be more efficient approaches, such as FFT based algo discussed in #7753.
    Examples
    --------
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> sliding_window_view(x, shape)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 1,  2],
             [11, 12]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[11, 12],
             [21, 22]],
            [[12, 13],
             [22, 23]]]])
    >>> i, j = np.ogrid[:3,:4]
    >>> x = 10*i + j
    >>> shape = (2,2)
    >>> step = (1,2)
    >>> sliding_window_view(x, shape, step)
    array([[[[ 0,  1],
             [10, 11]],
            [[ 2,  3],
             [12, 13]]],
           [[[10, 11],
             [20, 21]],
            [[12, 13],
             [22, 23]]]])
    """
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    try:
        shape = np.array(shape, np.int)
    except:
        raise TypeError("`shape` must be a sequence of integer")
    else:
        if shape.ndim > 1:
            raise ValueError("`shape` must be one-dimensional sequence of integer")
        if len(x.shape) != len(shape):
            raise ValueError("`shape` length doesn't match with input array dimensions")
        if np.any(shape <= 0):
            raise ValueError("`shape` cannot contain non-positive value")

    if step is None:
        step = np.ones(len(x.shape), np.intp)
    else:
        try:
            step = np.array(step, np.intp)
        except:
            raise TypeError("`step` must be a sequence of integer")
        else:
            if step.ndim > 1:
                raise ValueError("`step` must be one-dimensional sequence of integer")
            if len(x.shape) != len(step):
                raise ValueError(
                    "`step` length doesn't match with input array dimensions"
                )
            if np.any(step <= 0):
                raise ValueError("`step` cannot contain non-positive value")

    o = (np.array(x.shape) - shape) // step + 1  # output shape
    if np.any(o <= 0):
        raise ValueError("window shape cannot larger than input array shape")

    strides = x.strides
    view_strides = strides * step

    view_shape = np.concatenate((o, shape), axis=0)
    view_strides = np.concatenate((view_strides, strides), axis=0)
    view = np.lib.stride_tricks.as_strided(
        x, view_shape, view_strides, subok=subok, writeable=writeable
    )

    if writeable:
        return view.copy()
    else:
        return view
