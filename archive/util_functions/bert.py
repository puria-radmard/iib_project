import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import BertTokenizer, BertForMaskedLM, AdamW, AutoModel
device = (
    torch.device('cuda') if torch.cuda.is_available() 
    else torch.device('cpu')
)
from classes_utils.audio.bert import MLMDataset, TokensDataset, BERTUttEmbeddingCache, BERTWordEmbeddingCache

relevant_keys = ["input_ids", "attention_mask", "token_type_ids"]


def get_all_sentences(subset_text_path, pool_alignment_path):

    with open(subset_text_path, "r") as f:
        lines = f.read().split("\n")[:-1]
    all_sentences = [" ".join(l.split()[1:]) for l in lines]

    pool_alignment_df = pd.read_csv(pool_alignment_path, sep=" ", header=None)
    pool_sentences = []
    for pool_id in tqdm(np.unique(pool_alignment_df[0].tolist()), disable=True):
        utterance_df = pool_alignment_df[pool_alignment_df[0] == pool_id]
        all_sentences.append(" ".join(utterance_df[4].tolist()))

    return all_sentences


def mlm_mask(inputs, m_prop=0.15):

    cls_id, sep_id, pad_id, msk_id = 101, 102, 0, 103

    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (
        (rand < m_prop)
        * (inputs.input_ids != cls_id)
        * (inputs.input_ids != pad_id)
        * (inputs.input_ids != sep_id)
    )

    for i in range(mask_arr.shape[0]):
        selection = torch.flatten(mask_arr[i].nonzero()).tolist()
        inputs.input_ids[i, selection] = msk_id

    return inputs


def mlm_training_iteration(batch, model, optim):

    optim.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["token_type_ids"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optim.step()

    return loss.item()


def mlm_training(model, inputs, num_epochs):

    inputs = mlm_mask(inputs)
    dataset = MLMDataset(inputs)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32)

    model.to(device)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch in dataloader:
            loss = mlm_training_iteration(batch, model, optim)

            print(f"BERT pretraining epoch {epoch}, loss {loss}", flush=True)

    torch.cuda.empty_cache()
    return model.config


def pretrain_bert(all_sentences, bert_epochs):

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForMaskedLM.from_pretrained("bert-base-uncased")

    inputs = tokenizer(
        all_sentences,
        return_tensors="pt",
        max_length=256,
        truncation=True,
        padding="max_length",
    )

    print("\n", flush=True)
    print("-" * 90, flush=True)
    mlm_config = mlm_training(model, inputs, bert_epochs)
    print("-" * 90, flush=True)
    print("\n", flush=True)

    word_encoder = AutoModel.from_config(model.config)
    word_encoder.to(device)

    return word_encoder


def generate_token_information(tokenize, data_dict):

    tokens = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "part_sizes": [],
        "utterance_ids": [],
    }
    partition_counts = lambda s: [
        int(tokenize(w).attention_mask.sum() - 2) for w in s.split()
    ]

    utt_ids = data_dict["utterance_segment_ids"]
    word_ix = data_dict["word_index"]
    words = data_dict["words"]
    assert (
        sum(
            [
                word_ix[index]
                for index, _ in enumerate(utt_ids)
                if utt_ids[index] != utt_ids[index - 1]
            ]
        )
        == 0
    )
    indexes = [
        index for index, _ in enumerate(utt_ids) if utt_ids[index] != utt_ids[index - 1]
    ]
    indexes.append(len(utt_ids))
    sentences = [
        " ".join(words[indexes[i] : indexes[i + 1]])
        for i, _ in enumerate(indexes)
        if i != len(indexes) - 1
    ]

    for j, sentence in tqdm(enumerate(sentences), disable=True):
        new_tokens = tokenize(sentence)
        part_counts = partition_counts(sentence)
        num_spaces = len(sentence.split()) - 1

        assert sum(part_counts) == int(new_tokens.attention_mask.sum() - 2), (
            sum(part_counts),
            new_tokens.attention_mask.sum(),
        )
        for k, v in new_tokens.items():
            tokens[k].append(v[0])
        tokens["part_sizes"].append(part_counts)
        tokens["utterance_ids"].append(utt_ids[indexes[j]])

    tokens.update({k: torch.stack(tokens[k]).to(device) for k in relevant_keys})

    return tokens


def index_by_parts_sizes(item, part_sizes):
    start_idx = 1
    ret = []
    for ps in part_sizes:
        ret.append(item[start_idx : start_idx + ps])
        start_idx += ps
    return ret


def generate_word_embedding_cache(word_encoder, data_dict, mode):

    if mode not in ["utt", "word"]:
        raise ValueError(f"No cache type {mode}")

    with torch.no_grad():

        print("Collecting word embeddings", flush=True)

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenize_func = lambda s: tokenizer.encode_plus(
            s,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding="max_length",
        )

        # Only need to generate word embeddings for unlabelled set, which is held in data_dict from before.
        # (since we only have audio words for the unlabelled set anyway)
        tokens = generate_token_information(tokenize_func, data_dict)
        token_dataset = TokensDataset(tokens)
        token_dataloader = torch.utils.data.DataLoader(
            token_dataset, collate_fn=tokens_collate_fn, batch_size=32, shuffle=False
        )

        cache_dict = {}
        for token_batch in token_dataloader:
            torch.cuda.empty_cache()
            outputs = word_encoder(**{k: token_batch[k] for k in relevant_keys})
            part_counts = token_batch["part_sizes"]

            for i, part_count in enumerate(part_counts):
                ungrouped_embeddings = index_by_parts_sizes(
                    outputs.last_hidden_state[i], part_count
                )
                word_embeddings = [uem.mean(0).detach() for uem in ungrouped_embeddings]
                if mode == "word":
                    cache_dict[token_batch["utterance_ids"][i]] = word_embeddings
                elif mode == "utt":
                    cache_dict[token_batch["utterance_ids"][i]] = torch.mean(
                        word_embeddings
                    )

            del outputs

        print("Collection done", flush=True)

        torch.cuda.empty_cache()
        if mode == "word":
            return BERTWordEmbeddingCache(cache_dict)
        elif mode == "utt":
            return BERTUttEmbeddingCache(cache_dict)


def tokens_collate_fn(instances):
    output_dict = {k: [] for k in instances[0].keys()}
    for instance in instances:
        for k in instance.keys():
            output_dict[k].append(instance[k])
    output_dict.update(
        {k: torch.stack(output_dict[k]).to(device) for k in relevant_keys}
    )
    return output_dict