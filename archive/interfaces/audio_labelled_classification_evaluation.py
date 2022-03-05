# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib
# python -m interfaces.audio_labelled_classification_comparison

import torch, sys, os, json
from torch.utils.data import DataLoader
from config import device
from tqdm import tqdm

from util_functions.data import coll_fn_utt, generate_data_dict_utt, add_certainties_to_data_dict
from config.ootb.las_reg import *
from interfaces.audio_labelled_classification import labelled_classification_test_train_datasets

def get_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res


def run_evaluations(evaluation_model, evalutation_dataloader, get_certainties):
    all_preds = torch.empty(0, 2)
    all_embeddings = torch.empty(0, 256)
    all_utt_ids = []
    all_certainties = torch.empty(0)

    with torch.no_grad():

        for i, batch in tqdm(enumerate(evalutation_dataloader)):

            if i > 0 and i%10 == 0:
                print(f'Batch {i} | {len(evalutation_dataloader)} done')

            embeddings, preds = evaluation_model(batch['padded_features'].to(device))
            
            all_preds = torch.cat([all_preds, preds[0].detach().cpu()])
            all_embeddings = torch.cat([all_embeddings, embeddings[0].detach().cpu()])
            all_utt_ids.extend(batch['utt_id'])
            if get_certainties:
                all_certainties = torch.cat([all_certainties, batch['certainties'].detach().cpu()])
    
    if get_certainties:
        return all_preds, all_embeddings, all_utt_ids, all_certainties
    return all_preds, all_embeddings, all_utt_ids


if __name__ == '__main__':

    config_path_base = sys.argv[1] # "/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/config-1"
    save_path = sys.argv[2] # "/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/config-1/evaluations.pkl"

    config_path = os.path.join(config_path_base, 'config.json')
    model_path = os.path.join(config_path_base, 'model.mdl')

    config = get_json(config_path)
    arch_name = config['architecture_name']
    _model = eval(arch_name)(dropout=0, use_logits=False)
    print(arch_name + ':', sum(p.numel() for p in _model.parameters())/1000000.0, "M parameters")

    _model.load_state_dict(torch.load(model_path))
    _model.eval()
    _model.to(device)

    save_dict = {}

    ctm_path = '/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.ctm'
    original_data_dict = generate_data_dict_utt(config['features_paths'], text_path=None)
    data_dict = add_certainties_to_data_dict(original_data_dict, [ctm_path])

    unlabelled_dataset, _ = labelled_classification_test_train_datasets(
        data_dict=data_dict,
        max_sequence_length=config['max_seq_len'],
        labelled_list_path=None,
        unlabelled_list_path=config['unlabelled_list'],
        test_prop=0
    )
    assert len(unlabelled_dataset.indices) == 0
    unlabelled_dataloader = DataLoader(unlabelled_dataset, shuffle = False, collate_fn=coll_fn_utt, batch_size=config['batch_size'])

    print("starting unlabelled")
    unlabelled_preds, unlabelled_embeddings, unlabelled_utt_ids, unlabelled_certainties = \
        run_evaluations(_model, unlabelled_dataloader, True)
    save_dict["unlabelled_preds"] = unlabelled_preds
    save_dict["unlabelled_embeddings"] = unlabelled_embeddings
    save_dict["unlabelled_utt_ids"] = unlabelled_utt_ids
    save_dict["unlabelled_certainties"] = unlabelled_certainties


    labelled_dataset, _ = labelled_classification_test_train_datasets(
        data_dict=original_data_dict,
        max_sequence_length=config['max_seq_len'],
        labelled_list_path=config['labelled_list'],
        unlabelled_list_path=None,
        test_prop=0
    )
    labelled_dataloader = DataLoader(labelled_dataset, shuffle = False, collate_fn=coll_fn_utt, batch_size=config['batch_size'])
    print("starting labelled")
    labelled_preds, labelled_embeddings, labelled_utt_ids = \
        run_evaluations(_model, labelled_dataloader, False)
    save_dict["labelled_preds"] = labelled_preds
    save_dict["labelled_embeddings"] = labelled_embeddings
    save_dict["labelled_utt_ids"] = labelled_utt_ids

    torch.save(save_dict, save_path)
    print('Saved in', save_path)
