import random
import torch, pickle

__all__ = [
    'AudioUtteranceDataset',
    'SubsetAudioUtteranceDataset',
    'LabelledClassificationAudioUtteranceDataset'
]


class AudioUtteranceDataset(torch.utils.data.Dataset):
    def __init__(self, audio, utt_ids, dim_means, dim_stds, **kwargs):
        assert not set(kwargs.keys()).intersection({'audio', 'utt_ids'})
        super(AudioUtteranceDataset, self).__init__()
        self.audio = audio
        self.utt_ids = utt_ids
        self.kwargs = kwargs
        self.dim_means = self.load_pickle(dim_means) if isinstance(dim_means, str) else dim_means
        self.dim_stds = self.load_pickle(dim_stds) if isinstance(dim_stds, str) else dim_stds        

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        means, stds = self.get_mean_and_std(utt_id)
        base_res = {
            "audio": (self.audio[index]-means)/stds,
            "utt_id": utt_id
        }
        kwargs_res = {k: v[index] for k, v in self.kwargs.items()}
        base_res.update(kwargs_res)
        return base_res

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_mean_and_std(self, utt_id):
        speaker_id = utt_id.split('-XXXXXX')[0] if utt_id[0] == 'C' else utt_id.split('-')[1]
        return self.dim_means[speaker_id], self.dim_stds[speaker_id]

    def pop(self, utt_id):
        raise Exception('Include all attributes')
        new_audio, new_utt_ids, new_text = [], [], []
        rem_audio, rem_utt_ids, rem_text = [], [], []
        for a, u, t in zip(self.audio, self.utt_ids, self.text):
            if u != utt_id:
                new_audio.append(a)
                new_utt_ids.append(u)
                new_text.append(t)
            else:
                rem_audio.append(a)
                rem_utt_ids.append(u)
                rem_text.append(t)
        self.audio=new_audio
        self.utt_ids=new_utt_ids
        self.text=new_text
        return rem_audio, rem_utt_ids, rem_text
    
    def push(self, audios, utt_ids, texts):
        raise Exception('Include all attributes')
        for audio, utt_id, text in zip(audios, utt_ids, texts):
            self.audio.append(audio)
            self.utt_ids.append(utt_id)
            self.text.append(text)


class SubsetAudioUtteranceDataset(AudioUtteranceDataset):
    
    def __init__(self, audio, utt_ids, dim_means, dim_stds, init_indices, **kwargs):
        super(SubsetAudioUtteranceDataset, self).__init__(
            audio, utt_ids, dim_means, dim_stds, **kwargs
        )
        self.indices = init_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]
        return super().__getitem__(real_index)

    def get_original_data(self, index):
        return super().__getitem__(index)


class LabelledClassificationAudioUtteranceDataset(AudioUtteranceDataset):

    def __init__(self, audio, utt_ids, dim_means, dim_stds, init_labelled_indices, **kwargs):
        assert not set(kwargs.keys()).intersection({'labelled'})
        super(LabelledClassificationAudioUtteranceDataset, self).__init__(
            audio, utt_ids, dim_means, dim_stds, **kwargs
        )
        self.indices = init_labelled_indices

    def __getitem__(self, index):
        res = super().__getitem__(index)
        res['labelled'] = int(index in self.indices)
        return res



def _make_test_labelled_classification_dataset(dataset_size):

    seq_len = 4000
    num_dims = 40
    speakers = ['A', 'B', 'C', 'D', 'E']

    audio = 20 + 5*torch.randn(dataset_size, seq_len, num_dims)
    utt_ids = [f"utt_{i+1}-speaker{random.choice(speakers)}" for i in range(dataset_size)]
    dim_means = {f"speaker{X}": 20*torch.ones(num_dims) for X in speakers}
    dim_stds = {f"speaker{X}": 5*torch.ones(num_dims) for X in speakers}
    init_labelled_indices = random.sample(range(dataset_size), int(dataset_size/3))
    kwargs = {
        "attribute_a": list(map(lambda x: x + "'s attribute a", utt_ids)),
        "attribute_b": list(map(lambda x: x + "'s attribute b", utt_ids))
    }

    dataset = LabelledClassificationAudioUtteranceDataset(
        audio, utt_ids, dim_means, dim_stds, init_labelled_indices, **kwargs
    )

    return dataset



if __name__ == '__main__':

    dataset = _make_test_labelled_classification_dataset(128)

    import pdb; pdb.set_trace()

    labelled_arr = [l['labelled'] for l in dataset]