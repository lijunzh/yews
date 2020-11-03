import torch
from torch import optim

from . import functional as F

class Trainer(object):
    """Mega class for training process.

    """
    def __init__(self, model, criterion, lr=0.1):
        # default device
        self.device = F.get_torch_device()

        # model
        self.model = model

        # optimizer
        self.criterion = criterion
        self.lr = lr
        self.optimizer = None
        self.scheduler = None

        # training process
        self.start_epoch = 0
        self.end_epoch = None

        # results
        self.best_acc = None
        self.best_model_state = None
        self.train_loss = None
        self.train_acc = None
        self.val_loss = None
        self.val_acc = None

        # reset
        self.reset()

    def reset(self):
        self.arch = self.model.__class__
        self.model = F.model_on_device(self.model, self.device)

        self.reset_optimizer()
        self.reset_scheduler()

        self._reset_results()

    def reset_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def reset_scheduler(self):
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              verbose=True)

    def update_scheduler(self):
        self.scheduler.step(self.val_loss[-1])


    def _reset_results(self):
        self.best_acc = 0.
        self.best_model = None
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def update_train_results(self, acc, loss):
        self.train_acc.append(acc)
        self.train_loss.append(loss)

    def update_val_results(self, acc, loss):
        self.val_acc.append(acc)
        self.val_loss.append(loss)

    def save_checkpoint(self, path=None):
        print("=> Pulling checkpoint from Trainer ...")
        self.model = F.model_off_device(self.model)
        checkpoint = {
            'arch': self.arch,
            'best_acc': self.best_acc,
            'current_model': self.model.state_dict(),
            'best_model': self.best_model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        self.model = F.model_on_device(self.model, self.device)
        if path:
            print("=> Saving checkpoint ...")
            torch.save(checkpoint, path)
            print(f"=> Checkpoint saved to '{path}'")

    def load_checkpoint(self, path):
        print(f"=> Loading checkpoint from '{path}' ... ")
        checkpoint = torch.load(path)

        if self.arch != checkpoint['arch']:
            raise ValueError(f"Architecture {checkpoint['arch']} in checkpoint does not match that on model ({self.arch})")
        self.model = F.model_off_device(self.model)
        self.model.load_state_dict(checkpoint['current_model'])
        self.model = F.model_on_device(self.model, self.device)
        self.best_acc = checkpoint['best_acc']
        self.best_model_state = checkpoint['best_model']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def results(self, path=None):
        results = {
            'model': self.best_model_state,
            'train_loss': self.train_loss,
            'train_acc': self.train_acc,
            'val_loss': self.val_loss,
            'val_acc': self.val_acc,
        }

        if path:
            torch.save(results, path)

        return results

    def validate(self, loader, print_freq=None):
        """Validate the model on a given datset using the dataloader provided.

        """

        return F.validate(self.model, loader, self.criterion, print_freq=print_freq)

    def train_one_epoch(self, loader, epoch, print_freq=None):
        return F.train(self.model, loader, self.criterion, self.optimizer,
                       epoch, print_freq=print_freq)

    def train(self, train_loader, val_loader, epochs=100,
              print_freq=None, resume=None):
        """Train the model on a given datset using the dataloader provided.

        """
        start_epoch = 0
        end_epoch = epochs

        if resume:
            self.load_checkpoint(resume)

        # record results for initial model
        print("Validation on training set.")
        acc, loss = self.validate(train_loader)
        self.update_train_results(acc, loss)
        print("Validation on valiation set.")
        acc, loss = self.validate(val_loader)
        self.update_val_results(acc, loss)

        # training loop
        print("Start training ...")
        for epoch in range(start_epoch, end_epoch):
            # update learning rate
            self.update_scheduler()

            # train model for one epoch
            acc, loss = self.train_one_epoch(train_loader, epoch, print_freq=print_freq)
            self.update_train_results(acc, loss)

            # validate model for current epoch
            acc, loss = self.validate(val_loader)
            self.update_val_results(acc, loss)

            # save checkpoint
            self.save_checkpoint(path=f'checkpoint_current.pth.tar')

            # preserve best model and accuracy
            is_best = self.val_acc[-1] > self.best_acc
            self.best_acc = max(self.val_acc[-1], self.best_acc)
            print(f" @ Best\t\t{self.best_acc:.3f}")
            if is_best:
                self.model = F.model_off_device(self.model)
                self.best_model_state = self.model.state_dict()
                self.model = F.model_on_device(self.model, self.device)
                self.save_checkpoint(path='checkpoint_best.pth.tar')

        # training finished
        print(f"Training fisihed. Best accuracy is {self.best_acc}")
