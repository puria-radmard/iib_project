import torch
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)

__all__ = [
    'MLMDataset',
    'TokensDataset',
    'BERTWordEmbeddingCache',
    'BERTUttEmbeddingCache'
]


# https://www.youtube.com/watch?v=R6hcxMMOrPE
class MLMDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


class TokensDataset(torch.utils.data.Dataset):
    def __init__(self, tokens_dict):
        self.tokens_dict = tokens_dict

    def __len__(self):
        return len(self.tokens_dict["part_sizes"])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.tokens_dict.items()}


class BERTWordEmbeddingCache:
    def __init__(self, embedding_dict):
        self.embedding_dict = embedding_dict

    def __call__(self, batch):
        utt_ids = batch["utterance_segment_ids"]
        indices = batch["word_indices"]
        out_embeddings = []
        for u, utt_id in enumerate(utt_ids):
            out_embeddings.append(self.embedding_dict[utt_id][indices[u]])
        return torch.stack(out_embeddings)


class BERTUttEmbeddingCache:
    def __init__(self, embedding_dict):
        self.embedding_dict = embedding_dict

    def __call__(self, batch):
        utt_ids = batch["utterance_segment_ids"]
        out_embeddings = []
        for utt_id in utt_ids:
            out_embeddings.append(self.embedding_dict[utt_id])
        return torch.stack(out_embeddings)
