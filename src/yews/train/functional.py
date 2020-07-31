import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .utils import CumulativeMovingAverageMeter

def get_torch_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_on_device(model, device):
    return torch.nn.DataParallel(model.to(device))

def model_off_device(model):
    return model.module.to(torch.device('cpu'))

def generate_tmp_name(prefix):
    return f"{prefix}_{np.floor(time.time()):.0f}"


def rm_dir_content(path):
    path = Path(path)
    shutil.rmtree(path)

def rm_content(path):
    for content in Path(path).glob('*'):
        shutil.rmtree(content)

def save_checkpoint(state, is_best=False, filename='checkpoint.pth.tar'):
    """Save checkpoint to pth.tar file"""

    filename = Path(filename)
    print(f"=> saving checkpoint")
    torch.save(state, filename)
    print(f"=> checkpoint saved to '{filename}'")
    if is_best:
        shutil.copyfile(filename, filename.parent / 'model_best.pth.tar')
        print("=> best model updated.")

def load_checkpoint(filename):
    """Load checkpoint from pth.tar file"""

    if Path(filename).is_file():
        print(f"=> loading checkpoint '{filename}'")

        checkpoint = torch.load(filename)
        print(f"=> loaded checkpoint '{filename}' "
              f"(epoch {checkpoint['epoch']})")

        return checkpoint
    else:
        print(f"=> no checkpoint found at '{filename}'")

        return None

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
    return res

def train(model, train_loader, criterion, optimizer, epoch, print_freq):
    # performance metrics
    batch_time = CumulativeMovingAverageMeter()
    data_time  = CumulativeMovingAverageMeter()
    losses     = CumulativeMovingAverageMeter()
    top1       = CumulativeMovingAverageMeter()

    # swtich to train mode
    model.train()

    end = time.time()
    optimizer.zero_grad()
    for i, (samples, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # model target to cuda
        if next(model.parameters()).is_cuda:
            target = target.cuda(non_blocking=True)

        output = model(samples)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1 = accuracy(output, target)
        losses.update(loss.item(), samples.size(0))
        top1.update(prec1[0], samples.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq:
            if (i % print_freq == 0):
                print(f'Epoch: [{epoch}][{i} / {len(train_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    return top1.avg, losses.avg


def validate(model, val_loader, criterion, print_freq=None):
    batch_time = CumulativeMovingAverageMeter()
    losses = CumulativeMovingAverageMeter()
    top1 = CumulativeMovingAverageMeter()

    # switch to evaluate model
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (samples, target) in enumerate(val_loader):
            if next(model.parameters()).is_cuda:
                target = target.cuda(non_blocking=True)

            output = model(samples)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1 = accuracy(output, target)
            losses.update(loss.item(), samples.size(0))
            top1.update(prec1[0], samples.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_freq:
                if i % print_freq == 0:
                    print(f"Test [{i}/{len(val_loader)}]\t"
                          f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                          f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})")

    print(f" * Prec@1\t{top1.avg:.3f}")

    return top1.avg, losses.avg
