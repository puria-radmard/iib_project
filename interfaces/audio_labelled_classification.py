import os, json
from torch import nn
from classes_utils.audio.data import LabelledClassificationAudioUtteranceDataset
from classes_utils.base.data import ClassificationDAFDataloader
from training_scripts.audio_regression_scripts import audio_regression_script
from config.ootb_architectures import listen_and_attend
from util_functions.data import (
    train_test_split_data_dict, coll_fn_utt, generate_data_dict_utt, 
    combine_data_dicts, data_dict_length_split, split_data_dict_by_labelled
)
from config import *
import argparse
from util_functions.base import config_savedir

parser = argparse.ArgumentParser()
parser.add_argument("--architecture_name", required=True, type=str)
parser.add_argument("--lr", required=True, type=float, help="Learning rate of RAE optimizer")
parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
parser.add_argument("--scheduler_proportion", required=True, type=float)
parser.add_argument("--dropout", required=True, type=float)
parser.add_argument("--weight_decay",required=True,type=float)
parser.add_argument("--batch_size", required=True, type=int)
parser.add_argument("--num_epochs", required=True, type=int)
parser.add_argument("--features_paths",required=True,nargs="+",help="List of paths where .ark files are")
parser.add_argument("--labelled_list",required=True,type=str,help="Path to list of labelled utts")
parser.add_argument("--unlabelled_list",required=True,type=str,help="Path to list of unlabelled utts")
parser.add_argument("--max_seq_len",required=True,type=int)
parser.add_argument("--test_prop",required=True,type=float)
parser.add_argument("--save_dir", required=False, default=None)
 
if __name__ == '__main__':

    args = parser.parse_args()
    
    # Make training objects
    model = getattr(listen_and_attend, args.architecture_name)(args.dropout, use_logits = True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.scheduler_proportion)
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # Get the data dicts and split by labelled and unlabelled
    data_dict = generate_data_dict_utt(args.features_paths, text_path=None)    
    data_dict = data_dict_length_split(data_dict, args.max_seq_len)
    labelled_data_dict, unlabelled_data_dict = split_data_dict_by_labelled(
        data_dict, args.labelled_list, args.unlabelled_list
    )

    # Split both subsets by train-test
    train_labelled_data_dict, test_labelled_data_dict = train_test_split_data_dict(labelled_data_dict, args.test_prop)
    train_unlabelled_data_dict, test_unlabelled_data_dict = train_test_split_data_dict(unlabelled_data_dict, args.test_prop)
    
    # Recombine them
    train_data_dict = combine_data_dicts(train_labelled_data_dict, train_unlabelled_data_dict)
    test_data_dict = combine_data_dicts(test_labelled_data_dict, test_unlabelled_data_dict)

    # Put data dicts into datasets
    train_dataset = LabelledClassificationAudioUtteranceDataset(
        audio=train_data_dict['mfcc'],
        utt_ids=train_data_dict['utterance_segment_ids'],
        dim_means="config/per_speaker_mean.pkl",
        dim_stds="config/per_speaker_std.pkl",
        init_labelled_indices=set(range(len(train_labelled_data_dict['mfcc']))),
    )
    test_dataset = LabelledClassificationAudioUtteranceDataset(
        audio=test_data_dict['mfcc'],
        utt_ids=test_data_dict['utterance_segment_ids'],
        dim_means="config/per_speaker_mean.pkl",
        dim_stds="config/per_speaker_std.pkl",
        init_labelled_indices=set(range(len(test_labelled_data_dict['mfcc']))),
    )

    # Put datasets into dataloaders
    train_dataloader = ClassificationDAFDataloader(train_dataset, batch_size=args.batch_size)
    test_dataloader = ClassificationDAFDataloader(test_dataset, batch_size=args.batch_size)

    save_dir = config_savedir(args.save_dir, args)

    model, results = audio_regression_script(
        ensemble=model,
        optimizer=opt,
        scheduler=scheduler,
        scheduler_epochs=args.scheduler_epochs,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=args.num_epochs,
        target_attribute_name="labelled",
        is_regression=False,
        show_print=True
    )

    with open(os.path.join(save_dir, 'results.json'), 'w') as jf:
        json.dump(results, jf)

    torch.save(model.state_dict(), os.path.join(save_dir, 'model.mdl'))
