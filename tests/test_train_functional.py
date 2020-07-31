import time

import pytest
import torch
from yews.train import functional as F

class TestTorchDevice():

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2),
        torch.nn.Linear(2, 2)
    )

    model_on_cpu = torch.nn.DataParallel(model.to(torch.device('cpu')))

    def test_get_torch_device(self):
        assert F.get_torch_device() == torch.device('cpu')

    def test_model_on_device(self):
        assert F.model_on_device(self.model, torch.device('cpu')).module == self.model

    def test_model_off_device(self):
        assert F.model_off_device(self.model_on_cpu) == self.model


class TestFileDir():

    def test_generate_tmp_name(self):
        tmp_name0 = F.generate_tmp_name('tmp')
        time.sleep(1)
        tmp_name1 = F.generate_tmp_name('tmp')
        assert tmp_name0 != tmp_name1
