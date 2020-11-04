from typing import Callable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .distributed import all_reduce
from .logging import get_logger
from .meter import ScalarMeter
from .timer import Timer

logger = get_logger(__name__)


def setup_env(
    rng_seed: Optional[int] = None,
    cudnn_benchmark: bool = False,
) -> None:
    """Sets up environment for training or testing."""

    # fix the RNG seeds
    if rng_seed:
        if isinstance(rng_seed, int):
            np.random.seed(rng_seed)
            logger.debug("Random seed of Numpy is set to %r.", rng_seed)
            torch.manual_seed(rng_seed)
            logger.debug("Random seed of PyTorch is set to %r.", rng_seed)
        else:
            logger.warning("Invalid rng_seed: %r.", rng_seed)
    else:
        logger.debug("Random seed is not initialized.")
    # configure the CUDNN backend
    torch.backends.cudnn.benchmark = cudnn_benchmark
    logger.debug("CUDNN benchmark is enabled: %r", cudnn_benchmark)


def setup_model(
    model: torch.nn.Module,
    distributed: bool = True,
) -> torch.nn.Module:
    """Sets up a model for training or testing and log the results."""

    # prepare model for GPU devices
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)
    logger.debug(
        "Model %r is moved to device: %s.", model.__class__, cur_device
    )
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device
        )
        logger.debug(
            "Model %r is wrappered in DistributedDataParallel on device %s.",
            model.__class__,
            cur_device,
        )
    else:
        model = torch.nn.DataParallel(model)
        logger.debug(
            "Model %r is wrappered in DataParallel on %d devices.",
            model.__class__,
            torch.cuda.device_count(),
        )
    return model


def set_lr(optimizer: torch.optim.Optimizer, learning_rate: float):
    """Set lr on all param_group of given optimizer."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def get_lr(optimizer: torch.optim.Optimizer, group: Optional[int] = None):
    """Get lr from the param_group in given optimizer."""
    if group:
        lrs = optimizer.param_groups[group]["lr"]
    else:
        lrs = [group["lr"] for group in optimizer.param_groups]
    if any(lr != lrs[0] for lr in lrs):
        logger.critical(
            "Learning rates of all parameter groups need to be the same: lr = %r",
            lrs,
        )
        cur_lr = None
    else:
        cur_lr = lrs[0]
    return cur_lr


def shuffle(loader, cur_epoch):
    """"Shuffles the data."""
    # RandomSampler handles shuffling automatically
    if isinstance(loader.sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        loader.sampler.set_epoch(cur_epoch)
        logger.debug("Sampler are manually shuffled at epoch %d.", cur_epoch)
    else:
        logger.debug(
            "Sampler are not manually shuffled at epoch%d.", cur_epoch
        )


@torch.no_grad()
def computer_targets(
    inputs: torch.Tensor, model: torch.nn.Module, use_amp: bool = False
) -> torch.Tensor:
    """Computer targets given inputs and model."""
    model.eval()
    if use_amp:
        with autocast():
            targets = model(inputs).detach()
    else:
        targets = model(inputs).detach()
    return targets


def train_epoch(
    train_loader: DataLoader,
    model: Module,
    loss_fun: Callable,
    optimizer: Optimizer,
    cur_epoch: int,
    num_gpus: int,
    get_targets: Optional[Callable] = None,
    distributed: bool = False,
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
    log_period: int = 1,
):
    """Train model for one epoch."""

    # meters
    timer = Timer(name=cur_epoch, logger=None)
    loss_epoch = ScalarMeter(log_period)

    # shuffle the dataset on all GPUs in distributed mode
    if distributed:
        shuffle(train_loader, cur_epoch)

    # put model in training mode
    model.train()
    timer.start()
    for cur_iter, (inputs, labels) in enumerate(train_loader):
        # Transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

        # computer targets from teacher model
        if get_targets:
            targets = get_targets(inputs)
        else:
            targets = labels

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward, backward, and update
        if use_amp:
            with autocast(enabled=True):
                outputs = model(inputs)
                loss = loss_fun(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_fun(outputs, targets)
            loss.backward()
            optimizer.step()

        # sync loss between GPUs
        if distributed:
            loss = all_reduce([loss])[0] / num_gpus

        # sync loss from GPU to CPU
        loss = loss.item()

        # update stats
        loss_epoch.add_value(loss)

        # display intra batch results
        if (cur_iter + 1) % log_period == 0:
            elapsed_time_display = timer.stop()
            logger.info(
                # "Epoch %3d time %5.1f lr = %.8f median loss = %8.6f",
                "Epoch %3d time %5.1f lr = %.8f avg loss = %8.6f",
                cur_epoch,
                elapsed_time_display,
                get_lr(optimizer),
                # loss_epoch.get_win_median(),
                loss_epoch.get_global_avg(),
            )
            # restart lap timer after display
            timer.start()

    # stop timer after one epoch
    timer.stop()
    return loss_epoch


@torch.no_grad()
def test_epoch(
    test_loader: DataLoader,
    model: Module,
    distributed: bool,
    use_amp: bool,
):
    """Evaluates the model on the test set."""

    # initiate meters
    test_correct = 0
    test_total = 0

    # enable eval mode
    model.eval()
    for inputs, labels in test_loader:
        # transfer the data to the current GPU device
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)

        if use_amp:
            with autocast(enabled=True):
                # compute the predictions
                outputs = model(inputs)
        else:
            outputs = model(inputs)

        # count correct predictions
        _, predictions = torch.max(outputs.data, 1)
        if distributed:
            batch_total = torch.Tensor([labels.shape[0]]).cuda()
            batch_correct = (predictions == labels).sum()
            batch_total, batch_correct = all_reduce(
                [batch_total, batch_correct]
            )
            batch_total, batch_correct = (
                batch_total.item(),
                batch_correct.item(),
            )
        else:
            batch_total = labels.size(0)
            batch_correct = (predictions == labels).sum().item()

        # update meters
        test_total += batch_total
        test_correct += batch_correct

    return test_correct / test_total
