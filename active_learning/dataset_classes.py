from numpy import pi, unique
import torch
from torch import nn
from tqdm import tqdm
from torch.nn.utils.rnn import pad_packed_sequence, pack_sequence

from active_learning.util_classes import RenovationError

from .annotation_classes import SentenceIndex
from .data_utils import *

TQDM_MODE = True


class ALAttribute:
    def __init__(self, name: str, initialisation: torch.tensor, cache: bool = False):
        # might change cache to arbitrary length
        self.name = name
        self.attr = initialisation
        self.cache = cache
        if cache:
            self.prev_attr = initialisation.clone()

    def __getitem__(self, idx):
        return self.attr[idx]

    def __setitem__(self, idx, value):
        raise Exception("Use update_attr_with_instance instead of setting item")

    def __len__(self):
        return len(self.attr)

    def generate_nans(self, new_data):
        nans = [float('nan') for nd in new_data] if isinstance(new_data, list) else float('nan')
        raise RenovationError('Need to find a way to add more data under pytorch implementation')

    def get_attr_by_window(self, window):
        return self.attr[window.i][window.slice]

    # Put asserts in here!!!
    def update_attr_with_window(self, window, new_attr):
        assert self.get_attr_by_window(window).shape == new_attr.shape
        if self.cache:
            self.prev_attr[window.i][window.slice] = self.attr[window.i][
                window.slice
            ].clone()
        self.attr[window.i][window.slice] = new_attr

    def update_attr_with_instance(self, i, new_attr):
        # assert self.attr[i].shape == new_attr.shape
        if self.cache:
            try:
                self.prev_attr[i] = self.attr[i].clone()
            except AttributeError:
                self.prev_attr[i] = self.attr[i]
        self.attr[i] = new_attr


class StochasticAttribute(ALAttribute):
    def __init__(
        self,
        name: str,
        initialisation: list,
        C: int,
        entropy_function = None,
        cache: bool = False
    ):
        # size [N, C, d1, ...]
        super(StochasticAttribute, self).__init__(name, initialisation, cache)
        #assert all(initialisation[i].shape == self.assertion_dim for i in range(len(initialisation))), \
        #    f"Initialisation for {self.name} requires list of datapoints, each either first dimension C (={C})"
        self.C = C
        if entropy_function:
            self.entropy_function = entropy_function

    def __setitem__(self, idx, value):
        raise Exception("Use update_attr_with_instance instead of setting item")

    def get_attr_by_window(self, window):
        return [draw[window.slice] for draw in self.attr[window.i]]

    def update_attr_with_window(self, window, new_attr):
        assert len(new_attr) == self.C, f"Require list of C MC draws for {self.name}"
        # assert self.get_attr_by_window(window).shape == new_attr.shape
        if self.cache: 
            for c in range(self.C):
                self.prev_attr[window.i][c][window.slice] = self.attr[window.i][c][window.slice].clone()
        for c in range(self.C):
            self.attr[window.i][c][window.slice] = new_attr[c][window.slice]

    def update_attr_with_instance(self, i, new_attr):
        # assert self.attr[i].shape == new_attr.shape
        # assert new_attr.shape == self.assertion_dim, f"New attr of shape {new_attr.shape} for  {self.name}"
        self.attr[i] = torch.stack(new_attr)
        if self.cache: self.prev_attr[i] = self.attr[i]



class VAEEnsembleGaussiansAttribute(StochasticAttribute):
    def __init__(
        self,
        name: str,
        initialisation: list,
        C: int,
        z_dim: int,
        cache: bool = False,
    ):
        # For the Gaussins of size z_dim*2 ==> calculates data uncertainty term
        self.z_dim = z_dim
        super(VAEEnsembleGaussiansAttribute, self).__init__(
            name, initialisation, C, cache=cache, assertion_dim=(C, int(z_dim*2))
        )
    
    def entropy_function(self, attr):
        # This is of length C, not M, so some of the Gaussians may be repeated
        # There is probably a more efficient way of doing this, but it's only marginal
        log_vars = attr[:,self.z_dim:]
        std_devs = torch.exp(0.5*log_vars)
        determinant_term = sum([torch.prod(sigma) for sigma in std_devs]) / self.C
        coefficient_term = self.z_dim * (1 + torch.log(2*pi))
        return 0.5*(determinant_term + coefficient_term)


class VAEEnsembleEmbeddingsAttribute(StochasticAttribute):
    def __init__(
        self,
        name: str,
        initialisation: list,
        C: int,
        z_dim: int,
        cache: bool = False,
    ):
        # For the embeddings of size z_dim ==> calculates total uncertainty term
        self.z_dim = z_dim
        super(VAEEnsembleEmbeddingsAttribute, self).__init__(
            name, initialisation, C, cache=cache, assertion_dim=(C, z_dim)
        )
    
    def entropy_function(self, attr):
        # This is of length C, not M, but none of them will be repeats
        z = torch.mean(attr, dim=0)
        diffs = [a - z for a in attr]
        sigma_ens = sum([torch.outer(d, d) for d in diffs]) / (self.C - 1)
        coefficient_term = self.z_dim * (1 + torch.log(2*pi))
        return 0.5*(sigma_ens+coefficient_term)
        

class ActiveLearningDatasetBase:
    
    def __init__(self, attrs_dict, index_class, main_indexer):
        self.attrs = attrs_dict
        self.main_indexer = main_indexer
        self.index = index_class(self)
        self.total_cost = sum([self.get_cost_by_index(i) for i in range(len(self))])
    
    def __getattr__(self, attr):
        return self.attrs[attr]

    def add_attribute(self, new_attribute):
        attr_name = new_attribute.name
        if attr_name in self.attrs:
            raise AttributeError(f"Dataset already has attribute {new_attribute.name}")
        else:
            self.attrs[attr_name] = new_attribute

    def add_labels(self, window, labels):
        self.labels.update_attr_with_window(window, labels)

    def add_temp_labels(self, window, temp_labels):
        self.temp_labels.update_attr_with_window(window, temp_labels)

    def data_from_window(self, window):
        return self.data.get_attr_by_window(window)

    def labels_from_window(self, window):
        return self.labels.get_attr_by_window(window)

    def update_attributes(self, batch_indices, new_attr, sizes):
        raise NotImplementedError

    def update_preds(self, batch_indices, preds, lengths):
        raise NotImplementedError

    def process_scores(self, scores, lengths):
        raise NotImplementedError

    # def get_temporary_labels(self, i):
    #     temp_data = self.data[i]
    #     temp_labels = [self.data[i][j] if j in self.index.labelled_idx[i]
    #                    else self.temp_labels[i][j] for j in range(len(temp_data))]
    #     return temp_data, temp_labels

    def get_cost_by_window(self, window):
        unit_costs = self.cost[window.i][window.slice]
        return total_sum(unit_costs)

    def get_cost_by_index(self, i):
        unit_costs = self.cost[i]
        return total_sum(unit_costs)

    def __getitem__(self, idx):
        return self.attrs[self.main_indexer][idx]

    def __len__(self):
        return len(self.attrs[self.main_indexer])
    
    
class KaldiMetaDataset(ActiveLearningDatasetBase):

    def __init__(self, part_ids, utt_ids, L1s, bands, durs, **kwargs):
        """
            This is for "offline" active learning scripts where the metadata is only used 
            for diversity rule setting/accessing a decoding file
        """
        attrs_dict = self.generate_attrs_dict(part_ids, utt_ids=utt_ids, durs=durs, L1s=L1s, bands=bands, **kwargs)
        super(KaldiMetaDataset, self).__init__(attrs_dict, index_class=SentenceIndex, main_indexer="part_ids")
        
    def update_attributes(self, batch_indices, new_attr_dict, lengths):
        if {"durs", "L1s", "bands"}.intersection(set(new_attr_dict.keys())):
            raise Exception("Metadata shouldn't be getting updated")
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in tqdm(enumerate(batch_indices), disable=not TQDM_MODE):
                self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j])

    def generate_attrs_dict(self, part_ids, durs, **kwargs):
        unique_part_ids = unique(part_ids)
        used_dict = {k: [] for k in kwargs.keys()}
        used_part_ids = []
        used_durs = []
        indices_dict = {}
        print('generating data 1')
        for j, pi in tqdm(enumerate(part_ids), disable=not TQDM_MODE):
            if pi in indices_dict:
                indices_dict[pi].append(j)
            else:
                indices_dict[pi] = [j]
        print('generating data 2')
        for upi in tqdm(unique_part_ids, disable=not TQDM_MODE):
            indices = indices_dict[upi]
            used_part_ids.append([part_ids[i] for i in indices])
            used_durs.append([durs[i] for i in indices])
            for k in used_dict.keys():
                used_dict[k].append([kwargs[k][i] for i in indices])
        attrs_dict = {
            "part_ids": ALAttribute(name="part_ids", initialisation=used_part_ids),
            "cost": ALAttribute(name="cost", initialisation=used_durs)
        }
        attrs_dict.update(
            {k: ALAttribute(name=k, initialisation=used_dict[k]) for k in kwargs.keys()}
        )
        return attrs_dict

    def get_batch(self, batch_indices, labels_important: bool):
        out_list = []
        for bi in batch_indices:
            out_list.append([self.part_ids[bi][0], self.utt_ids[bi]])
        return out_list, None, None, None


class ActiveLearningDataset(ActiveLearningDatasetBase):
    def __init__(
        self,
        data,
        labels,
        costs,
        index_class,
        semi_supervision_agent,
        al_attributes=None,
    ):

        # When initialised with labels, they must be for all the data.
        # Data without labels (i.e. in the real, non-simulation case), must be added later with self.data

        if al_attributes is None: al_attributes = []
        nan_init = torch.ones_like(labels)*float('nan')

        attrs_dict = {
            "data": ALAttribute(name="data", initialisation=data),
            "labels": ALAttribute(name="labels", initialisation=labels),
            "cost": ALAttribute(name="cost", initialisation=costs),
            "temp_labels": ALAttribute(name="temp_labels", initialisation=nan_init.clone()),
            "last_logits": ALAttribute(name="last_logits", initialisation=nan_init.clone(), cache=True),
        }

        attrs_dict.update({ala.name: ala for ala in al_attributes})
        super(ActiveLearningDataset, self).__init__(attrs_dict, index_class, "data")
        
        self.semi_supervision_agent = semi_supervision_agent
        
    def add_data(self, new_data):
        raise RenovationError('need to update total cost')
        new_data = [torch.array(nd) for nd in new_data]
        [v.add_new_data(new_data) if k == 'data' else v.expand_size(new_data) for k, v in self.attrs.items()]

    def get_batch(self, batch_indices, labels_important: bool):
        raise NotImplementedError


class DimensionlessDataset(ActiveLearningDataset):
    def __init__(
        self,
        data,
        labels,
        costs,
        index_class,
        semi_supervision_agent,
        data_reading_method=lambda x: x,
        label_reading_method=lambda x: x,
        al_attributes=[],
        is_stochastic=False
    ):
        super(DimensionlessDataset, self).__init__(
            data=data,
            labels=labels,
            costs=costs,
            index_class=index_class,
            semi_supervision_agent=semi_supervision_agent,
            al_attributes=al_attributes,
        )
        self.data_reading_method = data_reading_method
        self.label_reading_method = label_reading_method
        self.is_stochastic=is_stochastic

    def update_attributes(self, batch_indices, new_attr_dict, *args, **kwargs):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                if not self.is_stochastic:
                    # We receive a tensor of attributes, so interate with j
                    self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j])
                else:
                    # Each attr_value is actually a list of samples, each of shape [len(batch_indices), ...]
                    self.__getattr__(attr_name).update_attr_with_instance(i, [sample[j] for sample in attr_value])


    def process_scores(self, scores, lengths=None):
        return scores

    def get_batch(self, batch_indices, labels_important: bool):

        X = torch.stack([torch.tensor(self.data_reading_method(self.data[i])) for i in batch_indices])
        y = torch.stack([torch.tensor(self.label_reading_method(self.labels[i])) for i in batch_indices])
        semi_supervision_mask = torch.ones(y.shape)

        if labels_important:
            # Fill in with semi-supervision labels            
            y, semi_supervision_mask = self.semi_supervision_agent.process_labels(y, semi_supervision_mask)
            return X, y, [None for x in X], semi_supervision_mask

            for j, label in enumerate(y):
                instance_index = batch_indices[j]
                if self.index.labelled_idx[instance_index]:
                    pass
                elif self.index.temp_labelled_idx[instance_index]:
                    y[j] = self.temp_labels[instance_index]
                elif self.index.unlabelled_idx[instance_index]:
                    y[j] = torch.exp(torch.tensor(self.last_logits[instance_index]))
                    semi_supervision_mask[j] = self.semi_supervision_multiplier
                else:
                    raise Exception(
                        "Instance index does not appear in any of the annotation status lists"
                    )

        else:
            return (
                X,
                torch.tensor([]),
                [None for x in X],  # was lengths
                semi_supervision_mask,
            )


class AudioReconstructionDimensionlessDataset(DimensionlessDataset):

    def __init__(self, data, labels, costs, index_class, semi_supervision_agent, 
            data_reading_method=lambda x: x, label_reading_method=lambda x: x, al_attributes=[], 
            is_stochastic=False, max_seq_len=300
        ):
        super().__init__(data, labels, costs, index_class, semi_supervision_agent, 
            data_reading_method=data_reading_method, label_reading_method=label_reading_method, 
            al_attributes=al_attributes, is_stochastic=is_stochastic
        )
        self.max_seq_len = max_seq_len

    def update_attributes(self, batch_indices, new_attr_dict, lengths, *args, **kwargs):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                length = lengths[j]
                if not self.is_stochastic:
                    # We receive a tensor of attributes, so interate with j
                    self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j][:length])
                else:
                    # Each attr_value is actually a list of samples, each of shape [len(batch_indices), ...]
                    self.__getattr__(attr_name).update_attr_with_instance(i, [sample[j][:length] for sample in attr_value])

    def get_batch(self, batch_indices, labels_important: bool):

        Xs = [torch.tensor(self.data_reading_method(self.data[i])) for i in batch_indices]
        packed_X = nn.utils.rnn.pack_sequence(list(map(lambda x: x.permute(1, 2, 0), Xs)), enforce_sorted=False)
        X, lengths = nn.utils.rnn.pad_packed_sequence(packed_X, batch_first=True, total_length=self.max_seq_len)
        X = X.permute(0, 3, 1, 2)
        y = X
        semi_supervision_mask = torch.ones(y.shape)

        if labels_important:
            # Fill in with semi-supervision labels            
            y, semi_supervision_mask = self.semi_supervision_agent.process_labels(y, semi_supervision_mask)
            return X, y, lengths, semi_supervision_mask

            for j, label in enumerate(y):
                ...

        else:
            return (
                X,
                torch.tensor([]),
                lengths,
                semi_supervision_mask,
            )



class ImageClassificationDataset(ActiveLearningDataset):
    def __init__(
        self, data, labels, index_class, semi_supervision_agent, al_attributes=None
    ):
        al_attributes.append(ALAttribute(name="data", initialisation=torch.tensor(data)))

        super(ImageClassificationDataset, self).__init__(
            data, labels, index_class, semi_supervision_agent, al_attributes
        )

    def update_attributes(self, batch_indices, new_attr_dict, sizes):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                # Implement windows at some point, but not in this class!
                self.__getattr__(attr_name).update_attr_with_instance(i, attr_value[j])

    def process_scores(self, scores, sizes=None):
        return scores

    def get_batch(self, batch_indices, labels_important: bool):

        X = torch.tensor(self.data[batch_indices])
        y = torch.tensor([int(self.labels[i]) for i in batch_indices])
        semi_supervision_mask = torch.ones(y.shape)

        if labels_important:
            # Fill in with semi-supervision labels            
            y, semi_supervision_mask = self.semi_supervision_agent.process_labels(y, semi_supervision_mask)
            return X, y, torch.tensor([]), semi_supervision_mask
            # Fill in with semi-supervision labels
            for j, label in enumerate(y):
                instance_index = batch_indices[j]
                if self.index.labelled_idx[instance_index]:
                    pass
                elif self.index.temp_labelled_idx[instance_index]:
                    y[j] = self.temp_labels[instance_index]
                elif self.index.unlabelled_idx[instance_index]:
                    y[j] = torch.exp(torch.tensor(self.last_logits[instance_index]))
                    semi_supervision_mask[j] = self.semi_supervision_multiplier
                else:
                    raise Exception(
                        "Instance index does not appear in any of the annotation status lists"
                    )

        else:
            return (
                X,
                torch.tensor([]),
                [None for x in X],  # was lengths
                semi_supervision_mask,
            )


class OneDimensionalSequenceTaggingDataset(ActiveLearningDataset):
    def __init__(
        self,
        data,
        labels,
        index_class,
        semi_supervision_agent,
        padding_token,
        empty_tag,
        al_attributes=[],
    ):
        super().__init__(
            data, labels, index_class, semi_supervision_agent, al_attributes
        )
        self.empty_tag = empty_tag
        self.padding_token = padding_token

    def update_attributes(self, batch_indices, new_attr_dict, lengths):
        for attr_name, attr_value in new_attr_dict.items():
            for j, i in enumerate(batch_indices):
                self.__getattr__(attr_name)[i] = attr_value[j][: lengths[j]]

    def process_scores(self, scores, lengths):
        return [scores[i, :length].reshape(-1) for i, length in enumerate(lengths)]

    def get_batch(
        self, batch_indices, labels_important: bool
    ):  # batch_indices is a list, e.g. one of labelled_set
        """
        labels_important flag just to save a bit of time
        """

        sequences, tags = [self.data[i] for i in batch_indices], [self.labels[i] for i in batch_indices]

        padded_sentences, lengths = pad_packed_sequence(
            pack_sequence(
                [torch.LongTensor(_) for _ in sequences], enforce_sorted=False
            ),
            batch_first=True,
            padding_value=self.padding_token,
        )
        padded_tags, _ = pad_packed_sequence(
            pack_sequence([torch.LongTensor(_) for _ in tags], enforce_sorted=False),
            batch_first=True,
            padding_value=self.empty_tag,
        )

        semi_supervision_mask = torch.ones(padded_tags.shape)

        if labels_important:
            # Fill in the words that have not been queried
            tags, semi_supervision_mask = self.semi_supervision_agent.process_labels(tags, semi_supervision_mask)
            return padded_sentences, tags, lengths, semi_supervision_mask

            for j, sentence_tags in enumerate(padded_tags):
                sentence_index = batch_indices[j]
                for token_idx in range(int(lengths[j])):
                    if token_idx in self.index.labelled_idx[sentence_index]:
                        pass
                    elif token_idx in self.index.temp_labelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = torch.tensor(self.temp_labels[sentence_index][token_idx])
                    elif token_idx in self.index.unlabelled_idx[sentence_index]:
                        padded_tags[j, token_idx] = torch.exp(torch.tensor(self.last_logits[sentence_index][token_idx]))
                        semi_supervision_mask[
                            j, token_idx
                        ] = self.semi_supervision_multiplier
                    else:  # Padding
                        continue

        return padded_sentences, torch.tensor([]), lengths, semi_supervision_mask
