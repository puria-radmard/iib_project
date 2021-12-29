import os, pickle, json
from torch import nn
import numpy as np
from classes_losses.reconstrution import MovingReconstructionLoss, ReconstructionLoss
from classes_utils.audio.data import AudioCertaintyUtteranceDataset
from training_scripts.audio_uncertainty_regression import audio_uncertainty_regression_script
from classes_utils.architecture import AudioEncoderDecoderEnsemble
from util_functions.data import *
from config import *
import argparse

if __name__ == '__main__':
    print('CONFIGURING ARGS', flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", required=True, type=str)
    parser.add_argument("--encoder_lstm_sizes", required=True, type=int, nargs='+')
    parser.add_argument("--encoder_lstm_layers", required=True, type=int, nargs='+')
    parser.add_argument("--decoder_fc_hidden_dims", required=True, type=int, nargs='+')
    parser.add_argument("--feature_dim", required=True, type=int, help="Audio feature vector size")
    parser.add_argument("--embedding_dim", required=True, type=int, help="Hidden size of LSTM encoder")
    parser.add_argument("--lr", required=True, type=float, help="Learning rate of RAE optimizer")
    parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
    parser.add_argument("--scheduler_proportion", required=True, type=float)
    parser.add_argument("--dropout", required=True, type=float)
    parser.add_argument("--weight_decay",required=True,type=float)
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--num_epochs", required=True, type=int)
    parser.add_argument("--features_paths",required=True,nargs="+",help="List of paths where .ark files are")
    parser.add_argument("--alignment_paths",required=True,nargs="+",help="(mapped) CTM file(s) with certainty scores, e.g. ../active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm")
    parser.add_argument("--max_seq_len",required=True,type=int)
    parser.add_argument("--test_prop",required=True,type=float)
    parser.add_argument("--save_dir", required=False, default=None)
    parser.add_argument("--use_dim_means", required=False, default=True)
    parser.add_argument("--use_dim_stds", required=False, default=True)

    args = parser.parse_args()
    # args.moving_encoder = (args.encoder_architecture in moving_encoder_types)

    encoder_ensemble_kwargs = {
        "mfcc_dim": args.feature_dim, "embedding_dim": args.embedding_dim, 
        "dropout_rate": args.dropout, "variational": False, 
        "lstm_sizes": args.encoder_lstm_sizes,
        "lstm_layers": args.encoder_lstm_layers,
        "fc_hidden_dims": [],
        "cell_type": args.cell_type
    } 
    decoder_ensemble_kwargs = {
        "embedding_dim": args.embedding_dim,
        "layer_dims": args.decoder_fc_hidden_dims,
        # Need sigmoid for regression
        "nonlinearities": ['tanh' for _ in args.decoder_fc_hidden_dims[:-1]] + ['sigmoid'],
        "dropout_rate": args.dropout, "mean_first": False,
    }

    ensemble = AudioEncoderDecoderEnsemble(
        "basic", "basic_bidirectional_LSTM", "fc_decoder",
        1, encoder_ensemble_kwargs, decoder_ensemble_kwargs
    ).to(device)

    opt = torch.optim.Adam(
        ensemble.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=1, gamma=args.scheduler_proportion
    )

    criterion = nn.MSELoss(reduction='mean')

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
        raise Exception
        with open(args.data_dict_path, 'rb') as handle:
            data_dict = pickle.load(handle)
    else:
        raise argparse.ArgumentError(None, 'Need either data_dict_path or features_paths')

    data_dict = add_certainties_to_data_dict(data_dict, args.alignment_paths)
    
    print(f'limiting mfcc sequence length to {args.max_seq_len}')
    data_dict = data_dict_length_split(data_dict, args.max_seq_len)

    master_dataset = AudioCertaintyUtteranceDataset(
        data_dict["mfcc"], data_dict["utterance_segment_ids"], data_dict["text"],
        "config/per_speaker_mean.pkl",
        "config/per_speaker_std.pkl",
        targets = data_dict["certainties"]
    )
    test_length = int(np.floor(args.test_prop*len(master_dataset)))
    train_length = len(master_dataset) - test_length
    datasettrn, datasettst = torch.utils.data.random_split(master_dataset, [train_length, test_length])
    train_dataloader = torch.utils.data.DataLoader(
        datasettrn, collate_fn=coll_fn_utts, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        datasettst, collate_fn=coll_fn_utts, batch_size=args.batch_size, shuffle=True
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

    ensemble, results = audio_uncertainty_regression_script(
        ensemble=ensemble,
        optimizer=opt,
        scheduler=scheduler,
        scheduler_epochs=args.scheduler_epochs,
        criterion=criterion,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=args.num_epochs,
        show_print=True
    )

    with open(os.path.join(save_dir, 'results.json'), 'w') as jf:
        json.dump(results, jf)

    torch.save(ensemble.state_dict(), os.path.join(save_dir, 'ensemble.mdl'))
