from typing import Callable, List, Optional
from torchvision.datasets import CIFAR10, CIFAR100


class CIFARLabelledUtilityBase:
    def __init__(self, init_indices, original_transform, train):
        self.indices = init_indices
        self.original_transform = original_transform

        # If this is a train set, we can choose what is labelled
        if train:
            self.indices = init_indices
        # Otherwise, it must all be labelled
        elif init_indices is None:
            self.indices = list(range(len(self.data)))
        else:
            raise ValueError('Cannot pass init_indices with ')

    def add_index(self, index):
        self.indices.append(index)

    def remove_index(self, index):
        self.indices.remove(index)


class CIFARSubsetBase(CIFARLabelledUtilityBase):

    """self.indices dictate what data is presented at all - used for autoencoder configuration DAF"""

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]
        return super().__getitem__(real_index)

    def get_original_data(self, index):
        img = self.data[index]
        return self.original_transform(img)

    def get_original_label(self, index):
        target = self.targets[index]
        return target


class CIFARLabelledClassificationBase(CIFARLabelledUtilityBase):

    """self.indices dictate what the label is presented at all - used for classifier configuration DAF"""

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        _data = super().__getitem__(index)
        return _data[0], int(index in self.indices)

    def get_original_data(self, index):
        return super().__getitem__(index)[0]
        
    def get_original_label(self, index):
        return super().__getitem__(index)[1]



class CIFAR10Subset(CIFARSubsetBase, CIFAR10):
    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        CIFAR10.__init__(self, root, train=train, transform=transform, target_transform=target_transform, download=download)
        CIFARSubsetBase.__init__(self, init_indices, original_transform, train)


class CIFAR100Subset(CIFARSubsetBase, CIFAR100):
    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        CIFAR100.__init__(self, root, train=train, transform=transform, target_transform=target_transform, download=download)
        CIFARSubsetBase.__init__(self, init_indices, original_transform, train)



class CIFAR10LabelledClassification(CIFARLabelledClassificationBase, CIFAR10):
    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        CIFAR10.__init__(self, root, train=train, transform=transform, target_transform=target_transform, download=download)
        CIFARLabelledClassificationBase.__init__(self, init_indices, original_transform, train)


class CIFAR100LabelledClassification(CIFARLabelledClassificationBase, CIFAR100):
    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        CIFAR100.__init__(self, root, train=train, transform=transform, target_transform=target_transform, download=download)
        CIFARLabelledClassificationBase.__init__(self, init_indices, original_transform, train)
