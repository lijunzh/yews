import math
import os
import tarfile
from pathlib import Path
from urllib import request
import numpy as np
from torch.utils.model_zoo import tqdm

try:
    from obspy import UTCDateTime, read

    has_obspy = True
except ModuleNotFoundError:
    has_obspy = False

###############################################################################
#
#                           Path related utils
#
###############################################################################


def get_files_under_dir(root, pattern):
    """Construct list of path objects given pattern under the root directory."""
    root = Path(root)
    if root.exists():
        return [p for p in root.glob(pattern) if p.is_file()]
    else:
        raise FileNotFoundError(f"Direcotry {root} does not exist.")


################################################################################
#
#                       ObsPy related reading and loading
#
################################################################################
def stream2array(st):
    """Convert seismic frame from obspy.Stream to numpy.ndarray."""
    if has_obspy:
        return np.stack([tr.data[: int(np.floor(len(tr.data) / 10) * 10)] for tr in st])
    else:
        raise ModuleNotFoundError("Consider installing ObsPy for seismic I/O.")


def read_frame_obspy(path, **kwargs):
    """Read a seismic frame using ObsPy read.

    Args:
        path (path): Path to seismic files (SAC, mseed, etc.).
        starttime (UTCDateTime, optional): Frame starting time.
        endtime (UTCDateTime, optional): Frame ending time.

    Returns:
        Cutted frame numpy array of single or multiple component seismogram.

    """
    if has_obspy:
        return stream2array(read(path, **kwargs))
    else:
        raise ModuleNotFoundError("Consider installing ObsPy for seismic I/O.")


################################################################################
#
#                           Utilities for URLs
#
################################################################################


def test_url(url):
    try:
        with request.urlopen(url) as req:
            return req
    except Exception:
        return None


def gen_bar_update():
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def sizeof_fmt(num, suffix="B"):
    """Get human-readable file size

    Args:
        num (int): Number of unit size.
        suffix (str): Unit of size (default: B).

    """
    mag = int(math.floor(math.log(num, 1024)))
    val = num / math.pow(1024, mag)
    if mag > 7:
        return f"{val:.1f}Y{suffix}"
    else:
        return f"{val:3.1f}{['','K','M','G','T','P','E','Z'][mag]}{suffix}"


class URL(object):
    def __init__(self, url):
        self.url = url
        req = test_url(url)
        if req:
            self.req = req
            self.size = req.length
            self.url_filename = self.get_filename()
        else:
            raise ValueError(f"{url} is not a valid URL or unreachable.")

    def get_filename(self):
        return [
            a.split("=")[1]
            for a in self.req.info()["content-disposition"].split("; ")
            if a.startswith("filename=")
        ][0].strip('"')

    def __repr__(self):
        return f"{self.url_filename}, {sizeof_fmt(self.size)}, from <{self.url}>"

    def download(self, root, filename=None):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)  # mkdir if not exists)
        if not filename:
            filename = self.url_filename
        fpath = root / filename

        print(f"Downloading {self.url} to {fpath}")
        request.urlretrieve(self.url, fpath, reporthook=gen_bar_update())


################################################################################
#
#                           Utilities for Tarfile
#
################################################################################


def extract_tar(infile, outdir=".", mode="r:*"):
    with tarfile.open(infile, mode=mode) as tar:
        tar.extractall(outdir)
