from typing import Callable, List, Optional
from torchvision.datasets import CIFAR10, CIFAR100

class CIFAR10Subset(CIFAR10):

    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        self.indices = init_indices
        self.original_transform = original_transform
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        
        # If this is a train set, we can choose what is labelled
        if train:
            self.indices = init_indices
        # Otherwise, it must all be labelled
        elif init_indices is None:
            self.indices = list(range(len(self.data)))
        else:
            raise ValueError('Cannot pass init_indices with ')

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

    def add_index(self, index):
        self.indices.append(index)

    def remove_index(self, index):
        self.indices.remove(index)


class CIFAR100Subset(CIFAR100):

    def __init__(self, root, train, transform, target_transform, download, original_transform, init_indices: List[int] = None):
        self.indices = init_indices
        self.original_transform = original_transform
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # If this is a train set, we can choose what is labelled
        if train:
            self.indices = init_indices
        # Otherwise, it must all be labelled
        elif init_indices is None:
            self.indices = list(range(len(self.data)))
        else:
            raise ValueError('Cannot pass init_indices with ')

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

    def add_index(self, index):
        self.indices.append(index)

    def remove_index(self, index):
        self.indices.remove(index)
