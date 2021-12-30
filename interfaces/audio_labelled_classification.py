import os, pickle, json
from torch import nn
import numpy as np
from classes_losses.reconstrution import MovingReconstructionLoss, ReconstructionLoss
from classes_utils.audio.data import AudioUtteranceDataset
from training_scripts.audio_regression_scripts import audio_regression_script
from classes_utils.architecture import AudioEncoderDecoderEnsemble
from config.ootb_architectures import listen_and_attend
from util_functions.data import *
from config import *
import argparse
 
if __name__ == '__main__':
    print('CONFIGURING ARGS', flush=True)

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
    parser.add_argument("--labelled_list",required=True,nargs="+",help="Path to list of labelled utts")
    parser.add_argument("--unlabelled_list",required=True,nargs="+",help="Path to list of unlabelled utts")
    parser.add_argument("--max_seq_len",required=True,type=int)
    parser.add_argument("--test_prop",required=True,type=float)
    parser.add_argument("--save_dir", required=False, default=None)

    args = parser.parse_args()
    
    # use args.architecture_name to make model here
    model = getattr(listen_and_attend, args.architecture_name)(args.dropout, use_logits = True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=args.scheduler_proportion)

    criterion = nn.CrossEntropyLoss(reduction='mean')

    print('DONE CONFIGURING ARGS\n', flush=True)

    print('CONFIGURING DATA', flush=True)

    data_dict = generate_data_dict_utt(args.features_paths, text_path=None)
    
    print(f'limiting mfcc sequence length to {args.max_seq_len}')
    data_dict = data_dict_length_split(data_dict, args.max_seq_len)

    master_dataset = AudioUtteranceDataset(
        data_dict["mfcc"], data_dict["utterance_segment_ids"], data_dict["text"],
        "config/per_speaker_mean.pkl",
        "config/per_speaker_std.pkl"
    )
    test_length = int(np.floor(args.test_prop*len(master_dataset)))
    train_length = len(master_dataset) - test_length
    datasettrn, datasettst = torch.utils.data.random_split(master_dataset, [train_length, test_length])
    train_dataloader = torch.utils.data.DataLoader(
        datasettrn, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasettst, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
    )

    print('DONE CONFIGURING DATA\n', flush=True)

    i=0
    while True:
        try:
            save_dir=f"{args.save_dir}-{i}"
            os.mkdir(save_dir)
            break
        except:
            i+=1
        if i>50:
            raise Exception("Too many folders!")

    saveable_args = vars(args)
    config_json_path = os.path.join(save_dir, "config.json")

    print(f"Config dir : {save_dir}\n", flush=True)

    with open(config_json_path, "w") as jfile:
        json.dump(saveable_args, jfile)

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
