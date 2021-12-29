import torch

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


## REPLACED BY KWARGS IN AudioRAEUtteranceDataset
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
