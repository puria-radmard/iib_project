import argparse, pickle, torch, os, json
import numpy as np
from tqdm import tqdm
from classes_losses.reconstrution import MovingReconstructionLoss

from config import *
from config.audio_lists import *

from util_functions.data import *
from util_functions.audio_transformations import TransformationDistribution

from classes_losses import *

from classes_utils.ensemble import *
from classes_utils.audio.bert import *
from classes_utils.audio.data import *
from classes_utils.architecture_integration import AudioEncoderDecoderEnsemble

from training_scripts.audio_training_scripts import *

possible_encoder_architectures = list(encoder_types.keys())
possible_ensemble_types = list(ensemble_method_dict.keys())

__all__ = [
    'parse_args',
    'verify_args',
    'configure_ensemble_and_opt',
    'configure_dataloaders',
    'configure_training_script_and_losses',
    'config_args',
    'main'
]


def parse_args():

    main_parser = argparse.ArgumentParser()
    task_parser = main_parser.add_argument_group("task")

    match_latent = main_parser.add_argument_group("match_latent")
    sim_clr_parser = main_parser.add_argument_group("sim_clr")
    autoencoder_parser = main_parser.add_argument_group("autoencoder")
    add_noise_parser = main_parser.add_argument_group("add_noise")
    mult_noise_parser = main_parser.add_argument_group("mult_noise")
    moving_parser = main_parser.add_argument_group("moving_parser")

    replication_ensemble_variational_encoder_parser = main_parser.add_argument_group("replication_ensemble_variational_encoder")
    ensemble_encoder_parser = main_parser.add_argument_group("ensemble_encoder")
    word_parser = main_parser.add_argument_group("word")
    utt_parser = main_parser.add_argument_group("utt")

    main_parser.add_argument("--feature_dim", required=True, type=int, help="Audio feature vector size")
    main_parser.add_argument("--hidden_size", required=True, type=int, help="Hidden size of LSTM encoder")
    main_parser.add_argument('--decoder_do_mean_first', dest='decoder_mean_first', action='store_true')
    main_parser.add_argument('--decoder_do_not_mean_first', dest='decoder_mean_first', action='store_false')
    main_parser.set_defaults(decoder_mean_first=False)
    main_parser.add_argument("--lr", required=True, type=float, help="Learning rate of RAE optimizer")

    main_parser.add_argument("--scheduler_epochs", required=True, nargs="+", type=int)
    main_parser.add_argument("--scheduler_proportion", required=True, type=float)
    main_parser.add_argument("--dropout", required=True, type=float, help="Dropout rate of RAE")
    main_parser.add_argument("--weight_decay",required=True,type=float,help="Optimiser weight decay of RAE optimizer",)
    main_parser.add_argument("--batch_size", required=True, type=int, help="Batch size for training RAE")
    main_parser.add_argument("--num_epochs", required=True, type=int, help="Number of epochs of RAE training")
    main_parser.add_argument("--data_dict_path",required=False,help="Pre-prepared data_dict, overrides all other data arguments")
    main_parser.add_argument("--features_paths",required=False,nargs="+",help="List of paths where .ark files are")
    main_parser.add_argument("--test_prop", required=True, type=float)
    main_parser.add_argument("--ensemble_size", required=False, type=int)
    main_parser.add_argument("--save_dir", required=False, default=None)
    main_parser.add_argument("--encoder_architecture", required=True, type=str, choices=possible_encoder_architectures)
    main_parser.add_argument("--ensemble_type", required=True, type=str, choices=possible_ensemble_types)

    task_parser.add_argument("--datalevel", required=True, choices=possible_scope)
    task_parser.add_argument("--task_type", required=True, choices=possible_tasks)
    task_parser.add_argument("--encoder_task", required=False, choices=possible_encoder_tasks)

    match_latent.add_argument("--text_encoder",required=False,type=str,choices=["bert", "zeros"],help="Type of word_encoder",)
    match_latent.add_argument("--bert_epochs",required=False,type=int,help="Number of epochs to pretrain text BERT with",)

    sim_clr_parser.add_argument("--sim_clr_temperature",required=False,type=float)
    sim_clr_parser.add_argument("--comparison_dim",required=False,type=int)

    autoencoder_parser.add_argument('--do_mean_autoencoder_decodings', dest='mean_autoencoder_decodings', action='store_true')
    autoencoder_parser.add_argument('--do_not_mean_autoencoder_decodings', dest='mean_autoencoder_decodings', action='store_false')
    autoencoder_parser.set_defaults(mean_autoencoder_decodings=False)

    replication_ensemble_variational_encoder_parser.add_argument("--target_encoder_config_dir", required=False, type=str)
    replication_ensemble_variational_encoder_parser.add_argument("--replication_loss", required=False, type=str, choices = ve_loss_dict.keys())

    ensemble_encoder_parser.add_argument("--encoder_loss_mult", required=False, type=float)
    ensemble_encoder_parser.add_argument("--decoder_loss_mult", required=False, type=float)
    ensemble_encoder_parser.add_argument("--anchor_mult", required=False, type=float)

    add_noise_parser.add_argument("--add_noise_a", required=False, type=float)
    mult_noise_parser.add_argument("--mult_noise_a", required=False, type=float)

    moving_parser.add_argument("--hidden_dims", required=False, type=int, nargs="+")
    moving_parser.add_argument("--num_frames", required=False, type=int)
    moving_parser.add_argument("--stride", required=False, type=int)

    word_parser.add_argument("--alignment_path",required=False,type=str,help="Path to most recent decoded audio",)
    word_parser.add_argument("--utt2dur_path", required=False, type=str, help="Path to utt2dur")  # move to main scope???

    utt_parser.add_argument("--text_path",required=False,type=str,help="Path to text file (utt --> full text)")
    main_parser.add_argument("--max_len",required=False, type=int, help="Maximum MFCC sequence length")

    args = main_parser.parse_args()

    args.moving_encoder = (args.encoder_architecture in moving_encoder_types)

    verify_args(args, main_parser)

    return args


def verify_args(args, main_parser):

    inactive_tasks = (
        set(possible_tasks + possible_ensemble_types + possible_scope + possible_ensemble_types)
        - {args.ensemble_type} - {args.task_type} - {args.datalevel} - {args.encoder_task} - {None}
    )

    if args.task_type == 'ensemble_variational_encoder':
        inactive_tasks.remove('ensemble_encoder') # These tasks share the same arguments (enc/dec/anc mults)

    for group in main_parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        if group.title in inactive_tasks:
            erroneously_active_options = [
                k for k, v in group_dict.items() if v is not None
            ]
            if erroneously_active_options:
                message = f"Active task/data level is {args.task_type}, but argument(s) {erroneously_active_options} for {group.title} are used"
                raise argparse.ArgumentError(None, message)
        else:
            for k, v in group_dict.items():
                if v is not None:
                    print(k, "\t\t", v, flush=True)

    if args.task_type == 'ensemble_ve' and args.autoencoder_loss:
        raise argparse.ArgumentError(None, "Script is training VE, but an autoencoder option is passed")

    if args.datalevel == 'word' and args.max_len:
        raise argparse.ArgumentError(None, "Cannot set max sequence length for words")

    if args.text_encoder == "bert":
        assert (
            args.bert_epochs is not None
        ), "Require --bert_epochs if text_encoder = bert"

    if args.moving_encoder:
        assert None not in [args.hidden_dims, args.num_frames, args.stride], "Moving encoder requires all of these"

    if args.data_dict_path and (args.features_paths or args.text_path or args.alignment_path or args.utt2dur_path):
        raise argparse.ArgumentError(None, "Cannot provide two data sources (data_dict and features/text/alignment)")


def configure_ensemble_and_opt(args):

    encoder_ensemble_kwargs = {
        "mfcc_dim": args.feature_dim, "embedding_dim": args.hidden_size, "dropout_rate": args.dropout, 
        "variational": args.task_type in ['ensemble_variational_encoder', 'replication_ensemble_variational_encoder'],
        "num_frames": args.num_frames, "stride": args.stride, "hidden_dims": args.hidden_dims
    }
    decoder_ensemble_kwargs = {
        "mfcc_dim": args.feature_dim,
        "hidden_dims": args.hidden_dims,
        "num_frames": args.num_frames,
        "embedding_dim": (
            args.hidden_size if args.task_type != 'ensemble_variational_encoder' 
            else args.hidden_size//2
        ), 
        "comparison_dim": args.comparison_dim,
        "dropout_rate": args.dropout,
        "mean_first": args.decoder_mean_first,
    }

    if args.mult_noise_a:
        encoder_ensemble_kwargs['a'] = args.mult_noise_a
    elif args.add_noise_a:
        encoder_ensemble_kwargs['a'] = args.add_noise_a


    if args.decoder_loss_mult == 0:
        decoder_type = "no_decoder"

    elif args.task_type == 'ensemble_encoder':
        if args.encoder_task in ['autoencoder', 'match_latent']:
            decoder_type = "simple_sliding_nn" if args.moving_encoder else "basic_LSTM"
        elif args.encoder_task == 'sim_clr':
            decoder_type = "basic_simclr"
        else:
            decoder_type = "no_decoder"

    elif args.task_type == 'replication_ensemble_variational_encoder':
        decoder_type = "no_decoder"

    elif args.task_type == 'ensemble_variational_encoder':
        if args.encoder_task == 'autoencoder':
            decoder_type = "basic_LSTM"

    ensemble = AudioEncoderDecoderEnsemble(
        args.ensemble_type, args.encoder_architecture, decoder_type,
        args.ensemble_size, encoder_ensemble_kwargs, decoder_ensemble_kwargs
    )

    adam_opt = torch.optim.Adam(
        ensemble.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        adam_opt, step_size=1, gamma=args.scheduler_proportion
    )

    return ensemble, adam_opt, scheduler, encoder_ensemble_kwargs, decoder_ensemble_kwargs, decoder_type


def configure_dataloaders(args, split_seed = None):

    if not split_seed:
        random_seed = round(torch.randn(1)[0].item(), 5)
        print(f'\nSplitting dataset using seed {random_seed}\n')
        torch.manual_seed(random_seed)
    else:
        print(f'\nSplitting dataset using seed {split_seed}\n')
        torch.manual_seed(split_seed)

    ood_data_dict = None

    # CHECK FOR CACHED DATADICTS HERE
    if args.datalevel == "word":
        if args.encoder_task == 'sim_clr':
            raise ValueError("SimCLR not implemented for words yet")
        if not args.data_dict_path:
            data_dict = generate_data_dict_words(args.features_paths, args.alignment_path, args.utt2dur_path)
            data_dict["mfcc"] = process_features(data_dict["mfcc"])
        else:
            with open(args.data_dict_path, 'rb') as handle:
                data_dict = pickle.load(handle)

        master_dataset = AudioRAEWordDataset(data_dict["mfcc"], data_dict["words"], data_dict["word_index"], data_dict["utterance_segment_ids"])
        test_length = int(np.floor(args.test_prop*len(master_dataset)))
        train_length = len(master_dataset) - test_length
        datasettrn, datasettst = torch.utils.data.random_split(master_dataset, [train_length, test_length])

        train_dataloader = torch.utils.data.DataLoader(
            datasettrn,
            collate_fn=coll_fn_words,
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_dataloader = torch.utils.data.DataLoader(
            datasettst,
            collate_fn=coll_fn_words,
            batch_size=args.batch_size,
            shuffle=True,
        )

    elif args.datalevel == "utt":
        if not args.data_dict_path:
            data_dict = generate_data_dict_utt(args.features_paths, args.text_path)
            data_dict = process_data_dict(data_dict, args.max_len)
        else:
            with open(args.data_dict_path, 'rb') as handle:
                data_dict = pickle.load(handle)

        in_domain = list(map(lambda x: x[0] != 'A', data_dict['utterance_segment_ids']))
        out_domain = list(map(lambda x: x[0] == 'A', data_dict['utterance_segment_ids']))
        ood_data_dict = {k: np.array(v)[out_domain] for k, v in data_dict.items()}
        data_dict = {k: np.array(v)[in_domain] for k, v in data_dict.items()}

        master_dataset = AudioRAEUtteranceDataset(data_dict["mfcc"], data_dict["utterance_segment_ids"], data_dict["text"])
        test_length = int(np.floor(args.test_prop*len(master_dataset)))
        train_length = len(master_dataset) - test_length
        datasettrn, datasettst = torch.utils.data.random_split(master_dataset, [train_length, test_length])

        if args.encoder_task != 'sim_clr':
            train_dataloader = torch.utils.data.DataLoader(
                datasettrn, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
            )
            test_dataloader = torch.utils.data.DataLoader(
                datasettst, collate_fn=coll_fn_utt, batch_size=args.batch_size, shuffle=True
            )

        else:
            band_size = 6
            data_augmenter = TransformationDistribution({
                "frequency": [(1+i*band_size, 1+(i+1)*band_size, 0.9) for i in range(args.feature_dim//band_size)]
            }, args.feature_dim)
            train_dataloader = torch.utils.data.DataLoader(
                datasettrn, collate_fn=generate_coll_fn_simclr_utt(data_augmenter), batch_size=args.batch_size, shuffle=True
            )
            test_dataloader = torch.utils.data.DataLoader(
                datasettst, collate_fn=generate_coll_fn_simclr_utt(data_augmenter), batch_size=args.batch_size, shuffle=True
            )

    torch.manual_seed(torch.initial_seed())

    return train_dataloader, test_dataloader, ood_data_dict, data_dict


def configure_training_script_and_losses(args):

    output_args = {}

    if args.task_type == 'ensemble_encoder':

        training_func = train_autoencoder_ensemble

        anc_crit = EncoderEnsembleAnchorLoss()
        output_args["anchor_criterion"] = (lambda *x, **kx: args.anchor_mult*anc_crit(*x, **kx)) if args.anchor_mult > 0 else None

        if args.encoder_task == 'autoencoder':
            rec_crit = (
                MovingReconstructionLoss(num_frames=args.num_frames, stride=args.stride, mean_decodings=args.mean_autoencoder_decodings)
                if args.moving_encoder else ReconstructionLoss(mean_decodings=args.mean_autoencoder_decodings)
            )
            output_args["decodings_criterion"] = (lambda *x, **kx: args.decoder_loss_mult * rec_crit(*x, **kx)) if args.decoder_loss_mult else None
            output_args["encodings_criterion"] = None

        elif args.encoder_task == 'match_latent':
            rec_crit = ReconstructionLoss()
            sem_crit = TextEncoderLoss()
            output_args["decodings_criterion"] = (lambda *x, **kx: args.decoder_loss_mult * rec_crit(*x, **kx)) if args.decoder_loss_mult else None
            output_args["encodings_criterion"] = (lambda *x, **kx: args.encoder_loss_mult * sem_crit(*x, **kx)) if args.dencder_loss_mult else None

        elif args.encoder_task == 'sim_clr':
            sim_clr_crit = SimCLREnsemblanceLoss(args.sim_clr_temperature, args.batch_size)
            output_args["decodings_criterion"] = (lambda *x, **kx: args.decoder_loss_mult * sim_clr_crit(*x, **kx)) if args.decoder_loss_mult else None
            output_args["encodings_criterion"] = None       

    
    elif args.task_type == 'ensemble_variational_encoder':

        training_func = train_variational_encoder_ensemble

        output_args["anchor_criterion"] = (lambda *x, **kx: args.anchor_mult*anc_crit(*x, **kx)) if args.anchor_mult > 0 else None

        if args.encoder_task == 'autoencoder':
            rec_crit = ReconstructionLoss(mean_decodings=args.mean_autoencoder_decodings)
            output_args["decodings_criterion"] = (lambda *x, **kx: args.decoder_loss_mult * rec_crit(*x, **kx)) if args.decoder_loss_mult else None

            reg_crit = VAEEnsemblePriorLoss(args.hidden_size)
            output_args["encodings_criterion"] = (lambda *x, **kx: args.encoder_loss_mult * reg_crit(*x, **kx)) if args.decoder_loss_mult else None


    elif args.task_type == 'replication_ensemble_variational_encoder':
        
        training_func = train_replication_variational_encoder_ensemble

        output_args['replication_criterion'] = ve_loss_dict[args.replication_loss](args.target_encoder_config_dir)

        if args.anchor_mult > 0:
            anc_crit = VariationalEncoderEnsembleAnchorLoss()
            output_args["anchor_criterion"] = lambda *x, **kx: args.anchor_mult*anc_crit(*x, **kx)
        else:
            output_args["anchor_criterion"] = None

    return training_func, output_args


def config_args(args, split_seed = None):

    print('\n', '-'*40, '\n', flush = True)
    print("Generating data features and model", flush = True)

    train_args = {"num_epochs": args.num_epochs}
    (
        train_args["ensemble"],
        train_args["optimizer"],
        train_args["scheduler"],
        encoder_ensemble_kwargs,
        decoder_ensemble_kwargs,
        decoder_type
    ) = configure_ensemble_and_opt(args)

    train_args["scheduler_epochs"] = args.scheduler_epochs

    (
        train_args["train_dataloader"],
        train_args["test_dataloader"],
        ood_data_dict,
        data_dict,
    ) = configure_dataloaders(args, split_seed)

    print("Finished generating data features and model", flush = True)
    print('\n', '-'*40, '\n', flush = True)

    training_func, new_train_args = configure_training_script_and_losses(args)
    train_args.update(new_train_args)

    train_args['ensemble'] = train_args['ensemble'].to(device)

    return train_args, training_func, encoder_ensemble_kwargs, decoder_ensemble_kwargs, decoder_type, data_dict, ood_data_dict



def get_key_dictionaries(_dataloader, _ensemble, args, device=device, activate_tqdm = False):

    anc_crit = EncoderEnsembleAnchorLoss(sum = False)
    rec_crit = (
        MovingReconstructionLoss(num_frames=args.num_frames, stride=args.stride, mean_decodings=args.mean_autoencoder_decodings, mean_losses=False) 
        if args.moving_encoder else ReconstructionLoss(args.mean_autoencoder_decodings, False)
    )
    anc_dict, rec_dict, enc_dict = {}, {}, {}

    for batch in tqdm(_dataloader, disable = not activate_tqdm):
        
        _encodings, *z_list, _decodings = _ensemble(batch['padded_features'].to(device))

        _anc_losses = anc_crit(
            _encodings if not _ensemble.variational else z_list[0], sum_losses = True
        ).sum(axis = -1)**0.5
        anc_dict.update({
            batch['utterance_segment_ids'][i]: _anc_losses[i].cpu() for i in range(len(_anc_losses))
        })

        enc_dict.update({
            batch['utterance_segment_ids'][i]: [e[i].detach().cpu() for e in _encodings] for i in range(len(_anc_losses))
        })

        if args.encoder_task == 'autoencoder':
            _rec_losses = rec_crit(_decodings, batch)
            _rec_losses = torch.mean(torch.stack(_rec_losses), 0)
            rec_dict.update({
                batch['utterance_segment_ids'][i]: _rec_losses[i].cpu() for i in range(len(_rec_losses))
            })
        
        torch.cuda.empty_cache()
        
    return anc_dict, rec_dict, enc_dict


def main():

    args = parse_args()
    train_args, training_func, encoder_ensemble_kwargs, decoder_ensemble_kwargs, decoder_type, data_dict, ood_data_dict = \
        config_args(args)

    if not args.save_dir:
        print('WARNING: this script will NOT save ensemble weights/config', flush = True)
    else:
        i=0
        while True:
            try:
                os.mkdir(f"{args.save_dir}-{i}")
                break
            except:
                i+=1
            if i>50:
                raise Exception("Too many folders!")

        saveable_args = vars(args)
        config_json_path = os.path.join(f"{args.save_dir}-{i}", "config.json")
        model_path = os.path.join(f"{args.save_dir}-{i}", "ensemble.mdl")

        print(f"Config dir : {args.save_dir}-{i}")

        if args.task_type == 'ensemble_encoder':
            saveable_args['__target_autoencoder_kwargs'] = {
                "ensemble_type": args.ensemble_type,
                "encoder_type": args.encoder_architecture,
                "decoder_type": decoder_type,
                "encoder_ensemble_kwargs": encoder_ensemble_kwargs,
                "decoder_ensemble_kwargs": decoder_ensemble_kwargs,
            }

        with open(config_json_path, "w") as jfile:
            json.dump(saveable_args, jfile)

    ensemble = training_func(**train_args)
    if args.save_dir:
        torch.save(ensemble.state_dict(), model_path)


    with torch.no_grad():
    
        ensemble.eval()

        ood_dataset = AudioRAEUtteranceDataset(ood_data_dict['mfcc'], ood_data_dict['utterance_segment_ids'], ood_data_dict['text'])
        ood_dataloader = torch.utils.data.dataloader.DataLoader(ood_dataset, 512, collate_fn = coll_fn_utt)
        
        train_anchor, train_reconstruction, train_encs = \
            get_key_dictionaries(train_args["train_dataloader"], ensemble, args)
        test_anchor, test_reconstruction, test_encs = \
            get_key_dictionaries(train_args["test_dataloader"], ensemble, args)
        ood_anchor, ood_reconstruction, ood_encs = \
            get_key_dictionaries(ood_dataloader, ensemble, args)

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
        _path = os.path.join(f"{args.save_dir}-{i}", f"{k}.pkl")
        with open(_path, 'wb') as handle:
            pickle.dump(v, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    main()
