import torch, pickle

__all__ = [
    'AudioRAEUtteranceDataset',
    'AudioRAEWordDataset'
]

class AudioRAEWordDataset(torch.utils.data.Dataset):
    def __init__(self, audio, words, word_indices, utt_ids):
        super(AudioRAEWordDataset, self).__init__()
        self.audio = audio
        self.words = words
        self.word_indices = word_indices
        self.utt_ids = utt_ids

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        return (
            self.audio[index],
            self.words[index],
            self.utt_ids[index],
            self.word_indices[index],
        )


class AudioRAEUtteranceDataset(torch.utils.data.Dataset):
    def __init__(self, audio, utt_ids, text, dim_means, dim_stds):
        super(AudioRAEUtteranceDataset, self).__init__()
        self.audio = audio
        self.utt_ids = utt_ids
        self.text = text
        self.dim_means = self.load_pickle(dim_means)
        self.dim_stds = self.load_pickle(dim_stds)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        means, stds = self.get_mean_and_std(utt_id)
        return (
            (self.audio[index]-means)/stds,
            self.text[index],
            utt_id
        )

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_mean_and_std(self, utt_id):
        speaker_id = utt_id.split('-XXXXXX')[0] if utt_id[0] == 'C' else utt_id.split('-')[1]
        return self.dim_means[speaker_id], self.dim_stds[speaker_id]

    def pop(self, utt_id):
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
        for audio, utt_id, text in zip(audios, utt_ids, texts):
            self.audio.append(audio)
            self.utt_ids.append(utt_id)
            self.text.append(text)


class SubsetAudioRAEUtteranceDataset(AudioRAEUtteranceDataset):
    
    def __init__(self, audio, utt_ids, text, dim_means, dim_stds, init_indices, add_channel):
        super(SubsetAudioRAEUtteranceDataset, self).__init__(
            audio, utt_ids, text, dim_means, dim_stds
        )
        self.indices = init_indices
        self.add_channel = add_channel

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = self.indices[index]
        audio, text, utt_id = super().__getitem__(real_index)
        if self.add_channel:
            audio = audio.reshape(1, *audio.shape)
        return audio, text, utt_id

    def get_original_data(self, index):
        audio, text, utt_id = super().__getitem__(index)
        if self.add_channel:
            audio = audio.reshape(1, *audio.shape)
        return audio, text, utt_id



class AudioCertaintyUtteranceDataset(torch.utils.data.Dataset):
    def __init__(self, audio, utt_ids, text, certainties, dim_means, dim_stds):
        super(AudioCertaintyUtteranceDataset, self).__init__()
        self.audio = audio
        self.utt_ids = utt_ids
        self.text = text
        self.certainties = certainties
        self.dim_means = self.load_pickle(dim_means)
        self.dim_stds = self.load_pickle(dim_stds)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def get_mean_and_std(self, utt_id):
        speaker_id = utt_id.split('-XXXXXX')[0] if utt_id[0] == 'C' else utt_id.split('-')[1]
        return self.dim_means[speaker_id], self.dim_stds[speaker_id]

    def __len__(self):
        return len(self.audio)

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        utt_id = self.utt_ids[index]
        means, stds = self.get_mean_and_std(utt_id)
        return (
            (self.audio[index]-means)/stds,
            self.text[index],
            utt_id,
            self.certainties[index]
        )
