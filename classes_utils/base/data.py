import random
import torch
from torch.utils.data import DataLoader, Sampler
from util_functions.data import coll_fn_utt


class EvenBinaryClassificationBatchSampler(Sampler):

    """
        For an uneven binary dataset of labels {0,1}, this will ensure that each batch is
        a close as possible to 50-50
        
        Each batch is:
            50%: The minority class is randomly sampled without replacement
            50%: The majority class is randomly sampled without replacement
            This works by effectively keeping two independent samplers, as shown by methods
        
        Epoch lengths are the size of the minority class * 2
            i.e. not all majority class data will be sampled each epoch

    """

    def __init__(self, data_source, batch_size, class1_indices):
        self.data_source = data_source
        self.batch_size = batch_size
        class1_indices = set(class1_indices)
        class0_indices = set(range(len(data_source))) - set(class1_indices)

        # Make the class with more instances the majority class, and vice versa
        # Typically, `unlabelled' is the minoirty class
        self.minority_indices, self.majority_indices = sorted([class0_indices, class1_indices], key=len)
        super(EvenBinaryClassificationBatchSampler, self).__init__(data_source)

    @staticmethod
    def make_sampler(indices_list):
        n = len(indices_list)
        generator = torch.Generator()
        generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
        position_list = torch.randperm(n, generator=generator).tolist()
        yield from list(map(list(indices_list).__getitem__, position_list))
        
    def __iter__(self):
        batch = []
        majority_sampler = self.make_sampler(self.majority_indices)
        minority_sampler = self.make_sampler(self.minority_indices)
        for idx_pair in zip(majority_sampler, minority_sampler):
            batch.extend(idx_pair)
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0:
            random.shuffle(batch)
            yield batch

    def __len__(self):
        # Keep going until minority class is expended
        num_effective = 2*len(self.minority_indices)
        return (num_effective + self.batch_size - 1) // self.batch_size


class ClassificationDAFDataloader(DataLoader):
    def __init__(self, dataset, collate_fn=coll_fn_utt, **kwargs):
        batch_sampler = EvenBinaryClassificationBatchSampler(
            data_source=dataset, batch_size=kwargs['batch_size'], class1_indices=dataset.indices
        )
        del kwargs['batch_size']
        self.spent = False
        super(ClassificationDAFDataloader, self).__init__(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, **kwargs
        )

    def __iter__(self):
        if self.spent:
            pass
            # raise Exception('Classification DAF Dataloader can only be used once, to prevent out of date labelled vector')
        else:
            self.spent = True
        return super().__iter__()


if __name__ == '__main__':

    from util_functions.data import coll_fn_utt
    from classes_utils.audio.data import _make_test_labelled_classification_dataset
    
    
    dataset = _make_test_labelled_classification_dataset(1000)
    batch_size = 64

    dataloader = ClassificationDAFDataloader(dataset, batch_size=batch_size)

    for batch in dataloader:
        print(batch['labelled'].sum()/batch['labelled'].size(0))

    print(f"Size of dataset: {len(dataset)}")
    print(f"Of which class 1 (labelled): {len(dataset.indices)}")
    print(f"Effective length: {len(dataset.indices)*2}, in {len(dataset.indices)*2/batch_size} batches")
    print(f"Classification DAF Dataloader length: {len(dataloader)} batches")

    import pdb; pdb.set_trace()

    for batch in dataloader:
        pass
