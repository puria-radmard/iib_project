import os, pickle, json

import numpy as np
from classes_losses.reconstrution import MovingReconstructionLoss, ReconstructionLoss
from classes_utils.audio.data import AudioRAEUtteranceDataset
from training_scripts.audio_unsupervised_recalibration_scripts import unsupervised_recalibration_script
from classes_utils.architecture_integration import AudioEncoderDecoderEnsemble
from util_functions.data import *
from config import *
import argparse

from util_functions.data import coll_fn_utt

if __name__ == '__main__':
    print('CONFIGURING ARGS', flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--minibatch_seconds", required=True, type=int)
    parser.add_argument("--total_added_seconds", required=True, type=int)
    parser.add_argument("--total_start_seconds", required=True, type=int)
    parser.add_argument("--encoder_architecture", required=True, type=str)
    parser.add_argument("--decoder_architecture", required=True, type=str)
    parser.add_argument("--cell_type", required=True, type=str)
    parser.add_argument("--encoder_lstm_sizes", required=True, type=int, nargs='+')
    parser.add_argument("--encoder_lstm_layers", required=True, type=int, nargs='+')
    parser.add_argument("--encoder_fc_hidden_dims", required=True, type=int, nargs='+')
    parser.add_argument("--metric_function", required=True, type=str)
    parser.add_argument("--labelled_utt_list_path", required=True, type=str)
    parser.add_argument("--unlabelled_utt_list_path", required=True, type=str)
    parser.add_argument("--feature_dim", required=True, type=int, help="Audio feature vector size")
    parser.add_argument("--embedding_dim", required=True, type=int, help="Hidden size of LSTM encoder")
    parser.add_argument("--initial_lr", required=True, type=float, help="Learning rate of RAE optimizer")
    parser.add_argument("--finetune_lr", required=True, type=float, help="Learning rate of RAE optimizer")
    parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
    parser.add_argument("--scheduler_proportion", required=True, type=float)
    parser.add_argument("--dropout", required=True, type=float)
    parser.add_argument("--weight_decay",required=True,type=float)
    parser.add_argument("--finetune_weight_decay",required=True,type=float)
    parser.add_argument("--mult_noise",required=True,type=float)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--num_initial_epochs", required=True, type=int)
    parser.add_argument("--num_finetune_epochs", required=True, type=int)
    parser.add_argument("--data_dict_path",required=False)
    parser.add_argument("--features_paths",required=False,nargs="+",help="List of paths where .ark files are")
    parser.add_argument("--max_seq_len",required=True,type=int)
    parser.add_argument("--save_dir", required=False, default=None)
    parser.add_argument("--use_dim_means", required=False, default=True)
    parser.add_argument("--use_dim_stds", required=False, default=True)

    args = parser.parse_args()
    args.moving_encoder = (args.encoder_architecture in moving_encoder_types)

    encoder_ensemble_kwargs = {
        "mfcc_dim": args.feature_dim, "embedding_dim": args.embedding_dim, 
        "dropout_rate": args.dropout, "variational": False, 
        "lstm_sizes": args.encoder_lstm_sizes,
        "lstm_layers": args.encoder_lstm_layers,
        "fc_hidden_dims": args.encoder_fc_hidden_dims,
        "cell_type": args.cell_type
    }
    decoder_ensemble_kwargs = {
        "mfcc_dim": args.feature_dim,
        "embedding_dim": args.embedding_dim,
        "dropout_rate": args.dropout, "mean_first": False,
        "cell_type": args.cell_type
    }

    ensemble = AudioEncoderDecoderEnsemble(
        "basic", args.encoder_architecture, args.decoder_architecture,
        1, encoder_ensemble_kwargs, decoder_ensemble_kwargs, args.mult_noise
    ).to(device)

    opt = torch.optim.Adam(
        ensemble.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay
    )
    finetune_opt = torch.optim.Adam(
        ensemble.parameters(), lr=args.finetune_lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=1, gamma=args.scheduler_proportion
    )

    decodings_criterion = (
        MovingReconstructionLoss(num_frames=args.num_frames, stride=args.stride, mean_decodings=False)
        if args.moving_encoder else ReconstructionLoss(mean_decodings=False)
    ).to(DEFAULT_DTYPE)

    print('DONE CONFIGURING ARGS\n', flush=True)

    print('CONFIGURING DATA', flush=True)

    if args.features_paths:
        print('loading from features path')
        data_dict = generate_data_dict_utt(args.features_paths, text_path=None)
        if not args.use_dim_means:
            print('removing mean')
            data_dict['mfcc'] = data_demeaning(data_dict['mfcc'])
    elif args.data_dict_path:
        print('loading from data dict path')
        with open(args.data_dict_path, 'rb') as handle:
            data_dict = pickle.load(handle)
    else:
        raise argparse.ArgumentError(None, 'Need either data_dict_path or features_paths')
    
    print(f'limiting mfcc sequence length to {args.max_seq_len}')
    data_dict = data_dict_length_split(data_dict, args.max_seq_len)

    print(f'splitting data into labelled and unlabelled')
    labelled_data_dict, unlabelled_data_dict = split_data_dict(
        data_dict, args.labelled_utt_list_path, args.unlabelled_utt_list_path
    )

    labelled_dataset = AudioRAEUtteranceDataset(
        labelled_data_dict["mfcc"],
        labelled_data_dict["utterance_segment_ids"],
        labelled_data_dict["text"],
        "config/per_speaker_mean.pkl",
        "config/per_speaker_std.pkl"
    )
    unlabelled_dataset = AudioRAEUtteranceDataset(
        unlabelled_data_dict["mfcc"],
        unlabelled_data_dict["utterance_segment_ids"],
        unlabelled_data_dict["text"],
        "config/per_speaker_mean.pkl",
        "config/per_speaker_std.pkl"
    )

    labelled_dataloader = torch.utils.data.DataLoader(
        labelled_dataset, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
    )
    unlabelled_dataloader = torch.utils.data.DataLoader(
        unlabelled_dataset, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
    )

    print('DONE CONFIGURING DATA\n', flush=True)

    metric_function = {
        "reconstruction_loss": MovingReconstructionLoss(num_frames=args.num_frames, stride=args.stride, mean_decodings=False, mean_losses=False)
                if args.moving_encoder else ReconstructionLoss(mean_decodings=False, mean_losses=False)
    }[args.metric_function]

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

    unsupervised_recalibration_script(
        ensemble=ensemble,
        optimizer=opt,
        scheduler=scheduler,
        scheduler_epochs=args.scheduler_epochs,
        encodings_criterion=None,
        decodings_criterion=decodings_criterion,
        anchor_criterion=None,
        labelled_dataloader=labelled_dataloader,
        unlabelled_dataloader=unlabelled_dataloader,
        num_initial_epochs=args.num_initial_epochs,
        num_finetune_epochs=args.num_finetune_epochs,
        finetune_optimizer=finetune_opt,
        minibatch_seconds=args.minibatch_seconds,
        total_seconds=args.total_added_seconds,
        metric_function=metric_function,
        save_dir=save_dir,
        show_print=True,
    )
