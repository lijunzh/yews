import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, random_split

import yews.datasets as dsets
import yews.transforms as transforms
from yews import train
from yews.models import cpic_v1, cpic_v2, cpic_v3
from yews.train import Timer

logger = train.logging.get_logger()


def train_model():

    num_epochs = 75
    use_dist = True
    dist_master_proc = (not use_dist) or train.distributed.is_master_proc()

    train.logging.setup_logger(logger, enabled=dist_master_proc)
    logger.info("Logging setup.")

    train.setup_env()
    logger.info("Computing env setup.")

    waveform_transform = transforms.Compose(
        [
            transforms.ZeroMean(),
            transforms.SoftClip(1e-3),
            transforms.ToTensor(),
        ]
    )

    dsets.set_memory_limit(10 * 1024 ** 3)  # first number is GB
    dset = dsets.Wenchuan(
        path="/scratch1/china/lijun/wenchuan",
        download=False,
        sample_transform=waveform_transform,
    )

    # Split datasets into training and validation
    train_length = int(len(dset) * 0.8)
    val_length = len(dset) - train_length
    train_set, val_set = random_split(dset, [train_length, val_length])

    # Prepare dataloaders
    train_loader = DataLoader(
        train_set, batch_size=1000, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_set, batch_size=2000, shuffle=False, num_workers=4
    )

    loss_fun = CrossEntropyLoss()
    model = cpic_v1().cuda()
    optimizer = Adam(model.parameters(), lr=0.2)
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    for cur_epoch in range(0, num_epochs):

        loss_epoch = train.train_epoch(
            train_loader,
            model,
            loss_fun,
            optimizer,
            cur_epoch,
            num_gpus=2,
            log_period=20,
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
            # lr_scheduler.get_last_lr()[0],
            train.get_lr(optimizer),
            loss_epoch.get_global_avg(),
            accuracy_epoch * 100,
        )

        lr_scheduler.step()


if __name__ == "__main__":
    train.distributed.multi_proc_run(
        2, "nccl", "localhost", [10000, 65000], train_model, fun_args=(),
    )
