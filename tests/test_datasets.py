import os
from pathlib import Path

import numpy as np
import pytest
import yews.datasets as datasets
import yews.transforms as transforms

root_dir = Path('tests/assets').resolve()

def rm(path):
    try:
        os.remove(path)
    except OSError:
        pass


class DummpyDatasetlike(object):

    def __init__(self, size=1):
        self.size = size
        self.data = ['a item'] * self.size

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.size


@pytest.mark.smoke
def test_is_dataset():
    assert not datasets.is_dataset(0)
    assert datasets.is_dataset([])


@pytest.mark.smoke
class TestMandatoryMethods:

    def test_call_method(self):
        assert all([hasattr(getattr(datasets, t), '__getitem__') for t in
                    datasets.__all__])

    def test_repr_method(self):
        assert all([hasattr(getattr(datasets, t), '__len__') for t in
                    datasets.__all__])

@pytest.mark.smoke
class TestBaseDataset:

    class DummyBaseDataset(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()


    class DummyBaseDatasetNoSamples(datasets.BaseDataset):

        def build_dataset(self):
            return 0, DummpyDatasetlike()


    class DummyBaseDatasetNoTargets(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), 0


    class DummyBaseDatasetWrongLength(datasets.BaseDataset):

        def build_dataset(self):
            return DummpyDatasetlike(1), DummpyDatasetlike(2)

    class DummyTransform(transforms.BaseTransform):

        def __call__(self, data):
            return "transformed"

    class DummyNumericDataset(datasets.BaseDataset):

        def build_dataset(self):
            samples = np.ones((10, 3, 100))
            targets = np.ones(10) * 2
            return samples, targets


    def test_export_dataset(self):
        rm('samples.npy')
        rm('targets.npy')
        dset = self.DummyNumericDataset(root='.')
        dset.export_dataset('.')
        samples = np.load('samples.npy')
        targets = np.load('targets.npy')
        assert samples.shape == (10, 3, 100)
        assert np.allclose(samples, np.ones((10, 3, 100)))
        assert targets.shape == (10, )
        assert np.allclose(targets, np.ones(10) * 2)
        rm('samples.npy')
        rm('targets.npy')

    def test_empty_construct(self):
        dset = datasets.BaseDataset()
        assert len(dset) == 0

    def test_noempty_constrct(self):
        dset = self.DummyBaseDataset(root='.')
        assert len(dset) == 1

    def test_raise_notimplmenetederror(self):
        with pytest.raises(NotImplementedError):
            dset = datasets.BaseDataset('.')

    def test_no_samples(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetNoSamples(root='.')

    def test_no_targets(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetNoTargets(root='.')

    def test_samples_targets_not_match(self):
        with pytest.raises(ValueError):
            dset = self.DummyBaseDatasetWrongLength(root='.')

    def test_getitem_with_transform(self):
        dset = self.DummyBaseDataset(root='.',
                                     sample_transform=self.DummyTransform(),
                                     target_transform=self.DummyTransform())
        assert dset[0] == ('transformed', 'transformed')
        dset = self.DummyBaseDataset(root='.')
        assert dset[0] == ('a item', 'a item')

    def test_repr(self):
        dset = self.DummyBaseDataset(root='.',
                                     sample_transform='t',
                                     target_transform='tt')
        assert type(dset.__repr__()) is str
        dset = self.DummyBaseDataset()
        assert type(dset.__repr__()) is str


@pytest.mark.smoke
class TestPathDataset:

    class DummyPathDataset(datasets.PathDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_root_is_path(self):
        # check existing path
        dset = self.DummyPathDataset(path='.')
        # check path resolved
        assert dset.root == Path(dset.root).resolve()
        # check non-existing path
        with pytest.raises(ValueError):
            dset = self.DummyPathDataset(path='abc')


@pytest.mark.smoke
class TestDirDataset:

    class DummyDirDataset(datasets.DirDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_invalid_root(self):
        dset = self.DummyDirDataset(path='.')
        with pytest.raises(ValueError):
            dset = self.DummyDirDataset(path='setup.py')


@pytest.mark.smoke
class TestDataset:

    def test_loading_npy(self):
        dset = datasets.Dataset(path=root_dir / 'array_folder')
        assert all([dset[0][0].shape == (3, 100), dset[0][1].shape == ()])
        default_memory_limit = datasets.utils.get_memory_limit()
        datasets.utils.set_memory_limit(1)
        dset = datasets.Dataset(path=root_dir / 'array_folder')
        assert all([dset[0][0].shape == (3, 100), dset[0][1].shape == ()])
        datasets.utils.set_memory_limit(default_memory_limit)

    def test_invalid_root(self):
        with pytest.raises(ValueError):
            datasets.Dataset(path=root_dir)


@pytest.mark.smoke
class TestFileDataset:

    class DummpyFileDataset(datasets.FileDataset):

        def build_dataset(self):
            return DummpyDatasetlike(), DummpyDatasetlike()

    def test_file_check(self):
        dset = self.DummpyFileDataset(path='setup.py')
        with pytest.raises(ValueError):
            dset = self.DummpyFileDataset(path='.')


class TestPackagedDataset:

    class DummyPackagedDataset(datasets.PackagedDataset):

        url = 'https://www.dropbox.com/s/kvnphsnjtnhlrrx/dummy_packaged_dataset.tar.bz2?dl=1'

    def test_download_flag(self):
        with pytest.raises(ValueError):
            self.DummyPackagedDataset(path='.', download=None)
        rm('samples.npy')
        rm('targets.npy')
        with pytest.raises(ValueError):
            self.DummyPackagedDataset(path='.')

    @pytest.mark.internet
    def test_dummy_download(self):
        # prepare root folder
        rm('samples.npy')
        rm('targets.npy')
        rm('dummy_packaged_dataset.tar.bz2')
        # test download and extract
        self.DummyPackagedDataset(path='.', download=True)
        # test extract only
        rm('samples.npy')
        rm('targets.npy')
        self.DummyPackagedDataset(path='.', download=True)
        # test ready dataset
        dset = self.DummyPackagedDataset(path='.', download=True)
        assert dset[0][1] == 0
        assert len(dset) == 1
        # clean root folder
        rm('samples.npy')
        rm('targets.npy')
        rm('dummy_packaged_dataset.tar.bz2')


class TestWenchuanDataset:

    @pytest.mark.internet
    @pytest.mark.large_download
    def test_url(self):
        rm('samples.npy')
        rm('targets.npy')
        rm('wenchuan.tar.bz2')
        datasets.Wenchuan(path='.', download=True)
        rm('samples.npy')
        rm('targets.npy')
        rm('wenchuan.tar.bz2')


class TestSCSNDataset:

    @pytest.mark.internet
    @pytest.mark.large_download
    def test_url(self):
        rm('samples.npy')
        rm('targets.npy')
        rm('scsn.tar.bz2')
        datasets.SCSN(path='.', download=True)
        rm('samples.npy')
        rm('targets.npy')
        rm('scsn.tar.bz2')


@pytest.mark.password
class TestMarianaDataset:

    @pytest.mark.internet
    @pytest.mark.large_download
    def test_url(self):
        rm('samples.npy')
        rm('targets.npy')
        rm('mariana.tar.bz2')
        datasets.Mariana(path='.', download=True)
        rm('samples.npy')
        rm('targets.npy')
        rm('mariana.tar.bz2')
