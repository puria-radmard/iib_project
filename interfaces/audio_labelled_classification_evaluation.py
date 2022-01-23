# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib
# python -m interfaces.audio_labelled_classification_comparison

import torch, sys, os, json
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import device

from util_functions.data import coll_fn_utt, generate_data_dict_utt, add_certainties_to_data_dict
from config.ootb import (
    default_blstm_listener_self_attention_regression_architecture, 
    default_blstm_listener_transformer_regression_architecture
)
from interfaces.audio_labelled_classification import labelled_classification_test_train_datasets

def get_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res

config_path_base = sys.argv[1] # "/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/config-1"
save_path = sys.argv[2] # "/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/config-1/evaluations.pkl"

if __name__ == '__main__':

    config_path = os.path.join(config_path_base, 'config.json')
    model_path = os.path.join(config_path_base, 'model.mdl')

    config = get_json(config_path)
    arch_name = config['architecture_name']
    model = eval(arch_name)(dropout=0, use_logits=False)
    print(arch_name + ':', sum(p.numel() for p in model.parameters())/1000000.0, "M parameters")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    ctm_path = '/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.ctm'
    data_dict = generate_data_dict_utt(config['features_paths'], text_path=None)
    data_dict = add_certainties_to_data_dict(data_dict, [ctm_path])

    train_dataset, _ = labelled_classification_test_train_datasets(
        data_dict=data_dict,
        max_sequence_length=config['max_seq_len'],
        labelled_list_path=None,
        unlabelled_list_path=config['unlabelled_list'],
        test_prop=0
    )

    assert len(train_dataset.indices) == 0

    dataloader = DataLoader(train_dataset, collate_fn=coll_fn_utt, batch_size=config['batch_size'])
    
    all_preds = torch.empty(0, 2)
    all_embeddings = torch.empty(0, 256)
    all_utt_ids = []
    all_certainties = torch.empty(0)

    with torch.no_grad():

        for i, batch in enumerate(dataloader):

            if i%10 == 0:
                print(f'Batch {i} | {len(dataloader)} done')

            embeddings, preds = model(batch['padded_features'].to(device))
            
            all_preds = torch.cat([all_preds, preds[0].detach().cpu()])
            all_embeddings = torch.cat([all_embeddings, embeddings[0].detach().cpu()])
            all_utt_ids.extend(batch['utt_id'])
            all_certainties = torch.cat([all_certainties, batch['certainties'].detach().cpu()])

    save_dict = {
        'all_preds': all_preds,
        'all_embeddings': all_embeddings,
        'all_utt_ids': all_utt_ids,
        'all_certainties': all_certainties,
    }
    torch.save(save_dict, save_path)
    print('Saved in', save_path)
    
