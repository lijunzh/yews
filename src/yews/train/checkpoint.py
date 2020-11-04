from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

_Path = Union[str, Path]


def get_checkpoints(path: _Path, prefix: str = "", ext: str = ""):
    """Get paths to checkpoints in a directory."""
    return Path(path).glob(prefix + "*" + ext)


def get_checkpoint_path(
    checkpoint_dir: _Path,
    epoch: int,
    prefix: str = "",
    ext: str = "",
    extra: Optional[str] = None,
):
    if extra:
        return Path(checkpoint_dir) / f"{prefix}_{extra}_epoch{epoch:03d}{ext}"
    else:
        return Path(checkpoint_dir) / f"{prefix}_epoch{epoch:03d}{ext}"


def save_checkpoint(
    path: _Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: LRScheduler = None,
    strip_module: bool = True,
    epoch: int = 0,
) -> None:
    """
    Save checkpoint for current model, optimizer, and [optional]
    lr_scheduler.

    """

    # create directory if not exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model_state = (
        model.module.state_dict() if strip_module else model.state_dict()
    )
    optimizer_state = optimizer.state_dict()
    if lr_scheduler:
        ls_state = lr_scheduler.state_dict()
    else:
        ls_state = None
    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": optimizer_state,
        "lr_scheduler_state": ls_state,
    }
    # write the checkpoint
    torch.save(checkpoint, path)


def load_checkpoint(
    path: _Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None,
    lr_scheduler: LRScheduler = None,
    strip_module: bool = True,
    filter_fun: Optional[Callable] = None,
) -> int:
    """Loads the checkpoint from the given file."""

    # read checkpoint from file
    checkpoint = torch.load(path, map_location=torch.device("cpu"))

    # load model state
    model = model.module if strip_module else model
    if filter_fun:
        filtered_state_dict = dict(
            filter(filter_fun, checkpoint["model_state"])
        )
        model.load_state_dict(filtered_state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint["model_state"])

    # load optimizer and lr_scheduler if requested
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state"])

    # always return current epoch in the checkpoint
    return checkpoint["epoch"]
