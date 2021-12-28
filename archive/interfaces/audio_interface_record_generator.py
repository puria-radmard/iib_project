import json, sys, pickle, os, torch
from classes_utils.audio.data import AudioRAEUtteranceDataset

from interfaces.audio_interface import get_key_dictionaries, configure_dataloaders
from classes_utils.architecture import AudioEncoderDecoderEnsemble
from util_functions.data import coll_fn_utt

class Config:
    def __init__(self, d):
        self.__dict__.update(d)
    def update(self, d):
        self.__dict__.update(d)

def load_model(config, log_root):

    ensemble = AudioEncoderDecoderEnsemble(
        ensemble_type=config['ensemble_type'],
        encoder_type=config['encoder_architecture'],
        decoder_type='basic_LSTM',
        encoder_ensemble_kwargs={
            "mfcc_dim": config['feature_dim'],
            "embedding_dim": config['hidden_size'],
            "dropout_rate": config['dropout'],
            "ensemble_size": config['ensemble_size'],
            "variational": 'variational' in config['task_type'],
        },
        decoder_ensemble_kwargs={
            "mfcc_dim": config['feature_dim'],
            "embedding_dim": config['hidden_size'] if 'variational' not in config['task_type'] else config['hidden_size']//2,
            "dropout_rate": config['dropout'],
            "mean_first": config['decoder_mean_first']
        },
        weights_path=os.path.join(log_root, 'ensemble.mdl')
    ).to('cuda')

    saved = torch.load(os.path.join(log_root, 'ensemble.mdl'))
    for k, v in ensemble.state_dict().items():
        saved_v = saved[k]
        assert (saved_v != v).sum().item() == 0

    ensemble.eval()

    return ensemble


def get_log_info(log_path):
    with open(log_path, 'r') as f:
        lines = f.read()[:-1].split('\n')
    
    config_prefix = 'Config dir : '
    seed_prefix = 'Splitting dataset using seed '
    
    cnf_path = [l for l in lines if l[:len(config_prefix)] == config_prefix][0][len(config_prefix):]
    seed_val = [l for l in lines if l[:len(seed_prefix)] == seed_prefix][0][len(seed_prefix):]
    seed_val = float(seed_val)

    with open(os.path.join(cnf_path, "config.json")) as jfile:
        config = json.load(jfile)
    
    return cnf_path, seed_val, config


def save_key_dictionaries(ood_data_dict, train_dataloader, model, config, test_dataloader, log_root):
    
    ood_dataset = AudioRAEUtteranceDataset(ood_data_dict['mfcc'], ood_data_dict['utterance_segment_ids'], ood_data_dict['text'])
    ood_dataloader = torch.utils.data.dataloader.DataLoader(ood_dataset, 512, collate_fn = coll_fn_utt)

    train_anchor, train_reconstruction, train_encs = \
            get_key_dictionaries(train_dataloader, model, config, activate_tqdm=True)

    test_anchor, test_reconstruction, test_encs = \
        get_key_dictionaries(test_dataloader, model, config, activate_tqdm=True)

    ood_anchor, ood_reconstruction, ood_encs = \
        get_key_dictionaries(ood_dataloader, model, config, activate_tqdm=True)

    save_dict = {
        "train_anchor": train_anchor,
        "train_reconstruction": train_reconstruction,
        "train_encs": train_encs,
        "test_anchor": test_anchor,
        "test_reconstruction": test_reconstruction,
        "test_encs": test_encs,
        "ood_anchor": ood_anchor,
        "ood_reconstruction": ood_reconstruction,
        "ood_encs": ood_encs
    }

    for k, v in save_dict.items():
        _path = os.path.join(log_root, f"{k}.pkl")
        with open(_path, 'wb') as handle:
            pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    log_path = sys.argv[1]
    log_root, data_seed, config = get_log_info(log_path)
    model = load_model(config, log_root)

    (
        train_dataloader,
        test_dataloader,
        ood_data_dict,
        data_dict,
    ) = configure_dataloaders(Config(config), data_seed)
    
    save_key_dictionaries(ood_data_dict, train_dataloader, model, Config(config), test_dataloader, log_root)