import pytest
import torch
from yews import models

@pytest.mark.internet
class TestCPICModel:

    def test_cpic_model(self):
        dummpy_input = torch.randn(1, 3, 2000)
        model = models.cpic(pretrained=True)
        model(dummpy_input)
        model = models.cpic(pretrained=False)
        model(dummpy_input)
