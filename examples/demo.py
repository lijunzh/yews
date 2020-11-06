import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

import yews.datasets as dsets
import yews.transforms as transforms
from yews import train
from yews.models import cpic_v1, cpic_v2, cpic_v3, cpic_v4
from yews.train import Timer

use_dist = True
num_epochs = 5


def build_data_loaders(distributed):

    waveform_transform = transforms.Compose(
        [
            transforms.ZeroMean(),
            transforms.SoftClip(1e-3),
            transforms.ToTensor(),
        ]
    )

    dsets.set_memory_limit(10 * 1024 ** 3)  # first number is GB
    dset = dsets.Wenchuan(
        path="wenchuan/", download=False, sample_transform=waveform_transform,
    )

    # Split datasets into training and validation
    train_length = int(len(dset) * 0.8)
    val_length = len(dset) - train_length
    train_set, val_set = random_split(dset, [train_length, val_length])

    # Prepare dataloaders
    if distributed:
        sampler_train = DistributedSampler(train_set) if distributed else None
        train_loader = DataLoader(
            train_set,
            batch_size=500,
            shuffle=(not sampler_train),
            sampler=sampler_train,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        sampler_val = DistributedSampler(val_set) if distributed else None
        val_loader = DataLoader(
            val_set,
            batch_size=500,
            shuffle=False,
            sampler=sampler_val,
            num_workers=12,
            pin_memory=True,
            drop_last=False,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=1000,
            shuffle=True,
            num_workers=12,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1000,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, val_loader


def train_model(num_epochs, use_dist=False):

    dist_master_proc = (not use_dist) or train.distributed.is_master_proc()

    logger = train.get_logger()
    train.logging.setup_logger(enabled=dist_master_proc)
    logger.info("Logging setup.")

    train.setup_env(rng_seed=1, cudnn_benchmark=True)
    logger.info("Computing env setup.")

    loss_fun = CrossEntropyLoss()
    model = train.setup_model(cpic_v1(), distributed=use_dist)
    optimizer = Adam(model.parameters(), lr=0.1)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    train_loader, val_loader = build_data_loaders(distributed=use_dist)

    for cur_epoch in range(0, num_epochs):

        loss_epoch = train.train_epoch(
            train_loader,
            model,
            loss_fun,
            optimizer,
            cur_epoch,
            num_gpus=2,
            log_period=100,
        )

        timer = Timer(name=cur_epoch, logger=None)
        timer.start()
        accuracy_epoch = train.test_epoch(
            val_loader, model, use_dist, use_amp=False
        )
        timer.stop()

        elapsed_time_epoch = Timer.timers[cur_epoch]
        logger.info(
            "Epoch %3d time %6.1f lr = %.8f avg loss = %8.6f acc = %2.2f",
            cur_epoch + 1,
            elapsed_time_epoch,
            train.get_lr(optimizer),
            loss_epoch.get_global_avg(),
            accuracy_epoch * 100,
        )

        lr_scheduler.step()


if __name__ == "__main__":
    if use_dist:
        train.distributed.multi_proc_run(
            2,
            "nccl",
            "localhost",
            [10000, 65000],
            train_model,
            fun_args=(num_epochs, use_dist),
        )
    else:
        train_model(num_epochs)
