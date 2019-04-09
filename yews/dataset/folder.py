from pathlib import Path
from .classification import ClassificationDataset


class ClassificationDatasetFolder(ClassificationDataset):
    """A generic dataloader for classification task where the samples are
    arranaged in folder with following format:

        root/class_x/xxx
        root/class_x/xxy
        root/class_x/xxz
        root/class_y/123
        root/class_y/nsdf3
        root/class_y/asd932_

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, transform=None, target_transform=None):
        super(ClassificationDatasetFolder, self).__init__(root, transform=transform,
                                                          target_transform=target_transform)
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


    def _find_classes(self):
        """
        Finds the class folders in a dataset.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        classes = [d.name for d in Path(self.source).iterdir() if d.is_dir()]

        return classes.sort()

    def _make_dataset(self):
        samples = []
        for target in class_to_idx.keys():
            dir = Path(self.source) / target
            fnames = [f.stem for f in dir.iterdir() if f.is_file()]
            for fname in sorted(fnames):
                path = str(dir / fanme) + '.*'
                samples.append((path, self.class_to_idx[target]))

        return samples

