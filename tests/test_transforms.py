import numpy as np
import pytest
import torch
import yews.transforms as transforms
import yews.transforms.functional as F

class TestIsTransform():

    class GoodTransform(object):

        def __call__(self):
            pass

    class BadTransform(object):

        pass

    def test_is_transform(self):
        assert transforms.is_transform(self.GoodTransform()) == True
        assert transforms.is_transform(self.BadTransform()) == False



class DummpyBaseTransform(transforms.BaseTransform):

    def __init__(self, a=0, b=1):
        self.a = a
        self.b = b

    def __call__(self, data):
        return data


class TestIsNumpyWaveform:

    def test_single_channel_waveform_vector(self):
        wav = np.empty(10)
        assert F._is_numpy_waveform(wav)

    def test_single_channel_waveform_matrix(self):
        wav = np.empty((1, 10))
        assert F._is_numpy_waveform(wav)

    def test_multi_channel_waveform(self):
        wav = np.empty((3, 10))
        assert F._is_numpy_waveform(wav)

    def test_invalid_waveform_wrong_type(self):
        wav = torch.tensor(10)
        assert not F._is_numpy_waveform(wav)


class TestToTensor:

    def test_type_exception(self):
        wav = torch.tensor(10)
        with pytest.raises(TypeError):
            F._to_tensor(wav)

    def test_single_channel_waveform(self):
        wav = np.zeros(10)
        tensor = torch.zeros(1, 10,dtype=torch.float)
        assert torch.allclose(F._to_tensor(wav), tensor)

    def test_multi_channel_waveform(self):
        wav = np.zeros((3, 10))
        tensor = torch.zeros(3, 10,dtype=torch.float)
        assert torch.allclose(F._to_tensor(wav), tensor)


@pytest.mark.smoke
class TestBaseTransform:

    def test_raise_call_notimplementederror(self):
        with pytest.raises(NotImplementedError):
            t = transforms.BaseTransform()
            t(0)

    def test_repr(self):
        t = transforms.BaseTransform()
        assert type(t.__repr__()) is str


@pytest.mark.smoke
class TestMandatoryMethods:

    def test_call_method(self):
        assert all([hasattr(getattr(transforms, t), '__call__') for t in
                    transforms.transforms.__all__])

    def test_repr_method(self):
        assert all([hasattr(getattr(transforms, t), '__repr__') for t in
                    transforms.transforms.__all__])


@pytest.mark.smoke
class TestComposeTransform:

    def test_repr(self):
        t = transforms.Compose([DummpyBaseTransform()])
        assert type(t.__repr__()) is str

class TestTransformCorrectness:
    def test_compose(self):
        wav = np.array([1, 3])
        assert torch.allclose(transforms.Compose([
            transforms.ZeroMean(),
            transforms.ToTensor(),
        ])(wav), torch.tensor([[-1, 1]], dtype=torch.float))

    def test_to_tensor_dtype(self):
        wav = np.array([1])
        assert torch.allclose(torch.tensor([[1]], dtype=torch.float),
                                transforms.ToTensor()(wav))
    def test_to_tensor_shape(self):
        wav = np.array([1])
        assert transforms.ToTensor()(wav).shape == (1,1)

    def test_zero_mean_channel_mean(self):
        wav = np.random.rand(3, 2000)
        assert all([np.allclose(ch.mean(), 0) for ch in transforms.ZeroMean()(wav)])

    def test_cut_waveform_shape(self):
        wav = np.random.randn(3, 2000)
        assert transforms.CutWaveform(100, 1900)(wav).shape == (3, 1800)

    def test_cut_waveform_value(self):
        wav = np.random.randn(3, 2000)
        assert np.allclose(transforms.CutWaveform(100, 1900)(wav),
                           wav[:, 100:1900])

    def test_select(self):
        wav = [1, 2, 3]
        assert np.allclose(transforms.Select(1)(wav), 2)
        with pytest.raises(ValueError):
            transforms.Select(1.0)

    def test_soft_clip(self):
        wav = np.array([-1, -0.5, 0, 0.5, 1])
        assert np.allclose(transforms.SoftClip()(wav),
                           np.array([0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]))
        assert np.allclose(transforms.SoftClip(1)(wav),
                           np.array([0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]))
        assert np.allclose(transforms.SoftClip(1.)(wav),
                           np.array([0.26894142, 0.37754067, 0.5, 0.62245933, 0.73105858]))
        with pytest.raises(ValueError):
            transforms.SoftClip('a')

    def test_lookup_table(self):
        with pytest.raises(ValueError):
            transforms.ToInt(lookup=[])
        with pytest.raises(ValueError):
            transforms.ToInt(lookup={'a': '0'})
        assert transforms.ToInt(lookup={'a': 0})('a') == 0
