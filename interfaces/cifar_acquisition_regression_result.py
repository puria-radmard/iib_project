import numpy as np
from glob import glob
from tqdm import tqdm
import seaborn as sns
import sys, os, torch, json
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
from active_learning.acquisition.logit_acquisition import LowestConfidenceAcquisition, MaximumEntropyAcquisition

from interfaces.cifar_acquisition_regression import generate_acquisition_regression_dataset, get_acquisition_classes, get_acquisition_prediction_criterion_class

sns.set_style('darkgrid')


## REALLY convoluted way of getting the `pseudo losses' - TODO: Fix
def recreate_loss_function(
    training_mode, objective_uniformalisation, logit_bank_path, acquisition_classes, dataset, batch_size, include_trainset, byol_daf, multitask_training_mode
    ):

    acquisition_classes = get_acquisition_classes(training_mode, False, multitask_training_mode)

    # Recreate (nearly) full dataset - fair enough approximation for 0.1 test set
    # TODO: SORT THIS OUT FOR LARGER TEST PROPS WHERE THIS IS NOT A GOOD APPROXIMATION
    full_set_loader, _, _ = generate_acquisition_regression_dataset(
        logit_bank_path, acquisition_classes, dataset, batch_size, 0.01, include_trainset, byol_daf
    )

    # Get the criterion class we wanted
    criterion = get_acquisition_prediction_criterion_class(
        multitask_training_mode, dataset, objective_uniformalisation, full_set_loader, 
        second_training_mode=None, args=None, save_dir=None
    )

    return criterion


def simulate_acqusition_loss(opts, targets, preds, training_mode, multitask_training_mode, posterior_acquisition_type):

    if training_mode == 'posterior_distillation_multitask':
        # Make sure we are doing the right posterior loss
        assert posterior_acquisition_type.lower() in opts['multitask_training_mode'], f"Trying to simulate {posterior_acquisition_type} with a {opts['multitask_training_mode']} system!"

    # The thing we are recreating from the distilled posterior
    acquisition_classes = [MaximumEntropyAcquisition if posterior_acquisition_type=='entropy' else LowestConfidenceAcquisition if posterior_acquisition_type=='LC' else None]

    # So let's recreate the last loss that we would have achieved, had we done this
    criterion = recreate_loss_function(
        training_mode=training_mode, objective_uniformalisation=opts['objective_uniformalisation'], logit_bank_path=opts['logit_bank_path'], 
        acquisition_classes=acquisition_classes, dataset=opts['dataset'], batch_size=opts['batch_size'], 
        include_trainset=opts['include_trainset'], byol_daf=opts['byol_daf'], multitask_training_mode=multitask_training_mode, 
    ).cpu()

    # Posterior head not trained to predict transformed target, so we have to take the transform first!
    preds = criterion._transform_target(torch.tensor(preds))
    targets = torch.tensor(targets)

    acq_loss = [criterion(preds, targets).item()]

    return acq_loss




def entropy_from_list(list_of_logits):

    # Stack the list into size [N, C]
    logits = torch.cat(list_of_logits)
    pmfs = torch.distributions.Categorical(logits = logits)

    # Get entropies
    return pmfs.entropy().numpy()


def lcs_from_list(list_of_logits):

    # Stack the list into size [N, C]
    logits = torch.cat(list_of_logits)
    pmfs = torch.distributions.Categorical(logits = logits)

    # Get entropies
    return - pmfs.probs.min(axis=1).values.numpy()


extraction_function_dict = {
    'entropy_bce': entropy_from_list,
    'lc_histogram_jacobian_bce': lcs_from_list
}


def spearmans_r_from_config_path(evaluation_dict, training_mode, posterior_acquisition_type):

    ## Get information from dict
    if training_mode == 'posterior_distillation_multitask' and posterior_acquisition_type == 'None':
        # Preds and real targets are of form [posterior, acquisition]
        targets = torch.cat([pair[1] for pair in evaluation_dict['all_acquisition']]).numpy()
        preds = torch.cat([pair[1].reshape(-1) for pair in evaluation_dict['all_preds']]).numpy()

    elif training_mode == 'posterior_distillation_multitask' and posterior_acquisition_type != 'None':
        # Preds and real targets are of form [posterior, acquisition]
        print('NOTE: USING MULTITASK POSTERIOR HEAD\n\n')
        targets = extraction_function_dict[posterior_acquisition_type]([pair[0] for pair in evaluation_dict['all_acquisition']])
        preds = extraction_function_dict[posterior_acquisition_type]([pair[0] for pair in evaluation_dict['all_preds']])

    elif training_mode == 'posterior_distillation':
        # All derived, like the one above, but no need to index pairs
        targets = extraction_function_dict[posterior_acquisition_type](evaluation_dict['all_acquisition'])
        preds = extraction_function_dict[posterior_acquisition_type](evaluation_dict['all_preds'])

    else:
        # Non posterior distillation based => just get acq prediction
        targets = torch.cat(evaluation_dict['all_acquisition'])
        preds = torch.cat(evaluation_dict['all_preds']).reshape(-1)

    # This training routine trains model to predict negative LC values
    # NB: the other lc based methods predict a (transformed) LC + 1 => don't need this!!
    if training_mode == 'lc_bce':
        preds = - preds

    return spearmanr(targets, preds), targets, preds


def get_result_line_old(args, eval_path):

    # Get predictions
    evaluations = torch.load(eval_path)

    sp = spearmans_r_from_config_path(evaluations, args['training_mode'])

    return (args['training_mode'], args['densenet_daf_depth']), args['test_prop'], float(sp[0])


def get_result_line_new(args, eval_path, loss_results_path, multitask_loss_results_path, posterior_acquisition_type):

    # Get predictions
    evaluations = torch.load(eval_path)

    # Odd case within args, overwritten in the right places
    secondary_acq_type = '-'

    # Get the spearmans, but also get the target acquisition values, and the preds used to rank against them
    # These only need to be used for the  args['training_mode'] == 'posterior_distillation_multitask' && posterior_acquisition_type != 'None'
    # case, where we are recreating the loss function
    sp, targets, preds = spearmans_r_from_config_path(evaluations, args['training_mode'], posterior_acquisition_type)

    # Get the right type of loss result
    if args['training_mode'] == 'posterior_distillation_multitask':
        
        # Need to get two losses here, both of which are saved
        results_df = pd.read_csv(multitask_loss_results_path)

        # Losses are recorded for each batch, and the column 'Training' tells us if this was 
        # a training or testing batch. We split the dataset based on where the training mode changes,
        # and filter out just the loss on testing batches
        results_df['group'] = results_df['Training'].ne(results_df['Training'].shift()).cumsum()
        grouped_results_df = results_df.groupby('group')

        # Get just the dfs we care about, i.e. Training column is False
        dfs = list(filter(lambda x: ~x.Training.iloc[0], [data for _, data in grouped_results_df]))

        # Get the batch-wise mean of each of them, for each loss we want
        kl_loss = list(map(lambda x: x.Distillation.mean().round(4), dfs))

        if posterior_acquisition_type == 'None':

            # We are using what the multitask outputs anyway!
            secondary_acq_type = args['multitask_training_mode']

            acq_loss = list(map(lambda x: x.Secondary.mean().round(4), dfs))

        elif posterior_acquisition_type != 'None':

            # We are using the posterior head to derive the acquisition function
            secondary_acq_type = args['multitask_training_mode'] + '-' + 'simulated'

            acq_loss = simulate_acqusition_loss(args, targets, preds, args['training_mode'], args['multitask_training_mode'], posterior_acquisition_type)
            

        acq_weight = args.get('multitask_acquisition_target_weight')
        dist_temp = args.get('distillation_temperature')

    elif args['training_mode'] == 'posterior_distillation':

        # Label this
        secondary_acq_type = posterior_acquisition_type + '-' + 'simulated'

        # This time acq is purely sythetic, not been trained on at all
        acq_loss = simulate_acqusition_loss(args, targets, preds, posterior_acquisition_type, posterior_acquisition_type, posterior_acquisition_type)

        # Need to get one loss here, as the KL loss
        with open(loss_results_path, 'r') as f:
            results_history = json.load(f)
            
            kl_loss = [r['test dec loss'] * args["batch_size"] for r in results_history]

        acq_weight = '--'
        dist_temp = args.get('distillation_temperature')

    else:

        # Need to get one loss here, as the acq loss
        with open(loss_results_path, 'r') as f:
            results_history = json.load(f)
            
            kl_loss = '-'
            acq_loss = [r['test dec loss'] * args["batch_size"] for r in results_history]

        acq_weight = '--'
        dist_temp = '--'

    byol_arg = "BYOL embs" if args['byol_daf'] else "DenseNet embs"

    return (
        args['training_mode'], secondary_acq_type, byol_arg, args['lr'], acq_weight, dist_temp, args['test_prop']
    ), float(sp[0]), acq_loss, kl_loss


def graphical_results_set(results_dictionary, save_directory, experiment_tag):
    # This is the 'old method' of plotting results: 
    #   x axis is test proportion
    #   y axis is spearmans
    #   lines aere grouped based on architecture (depth of densenet)
    #   byol not accounted for, and learning rate assumed constant

    for (task, layers), (points) in sorted(results_dictionary.items(), key = lambda x: x[0][1]):

        # points = {prop: corr, prop: corr, ...}
        props = points.keys()
        corrs = points.values()

        # Sort them
        props, corrs = zip(*sorted(zip(props, corrs)))

        # To be plotted/filled
        mean_line = np.array([np.mean(v) for v in corrs])
        std_bound = np.array([np.std(v) for v in corrs])        

        plt.plot(props, mean_line, label = layers)
        plt.fill_between(props, mean_line - std_bound, mean_line + std_bound, alpha = 0.2)

    plt.legend(title="Regression DenseNet layers")
    plt.title(" ".join(experiment_tag.split('_')[1:]))

    plt.xlabel('Test proportion')
    plt.ylabel('spearmans coefficient')

    plt.savefig(os.path.join(save_directory, f'{experiment_tag}_figure.png'))


def tabular_results_set(results_dictionary):
    # New method of plotting:

    print('Task', '|', 'Secondary task (if multiask)', '|', 'Daf form', '|', 'lr', '|', 'acq weight (if multitask)', '|', 'temp (if distillation)', '|', 'test_prop', '|', 'Spearmans', '|', 'acq loss', '|', 'KL loss (if multitask)')

    for cols, results in sorted(results_dictionary.items(), key = lambda x: x[0][1]):

        sr_mean = round(np.mean([r[0] for r in results]), 3)
        sr_std = round(np.std([r[0] for r in results]), 3)
        sr_str = f'{sr_mean} ± {2*sr_std}'

        if isinstance(results[0][1], str):
            acq_str = "--"
        else:
            smoothed_acq_losses = [np.mean(r[1][-6:]) for r in results]
            acq_mean = round(np.mean(smoothed_acq_losses), 4)
            acq_std = round(np.std(smoothed_acq_losses), 4)
            acq_str = f'{acq_mean} ± {2*acq_std}'

        if isinstance(results[0][2], str):
            kl_str = "--"
        else:
            smoothed_kl_losses = [np.mean(r[2][-6:]) for r in results]
            kl_mean = round(np.mean(smoothed_kl_losses), 4)
            kl_std = round(np.std(smoothed_kl_losses), 4)
            kl_str = f'{kl_mean} ± {2*kl_std}'

        print(cols, '|', sr_str, '|', acq_str, '|', kl_str)


def new_collate_results(config_directories, min_epochs, posterior_acquisition_type):

    # Dictionary schema = {(task, layers): [(test_prop1, spearmans_coef1), ...]}
    results_dict = {}

    # Iterate over experiments
    for config_dir in tqdm(config_directories):

        # Get the config and results paths
        json_path = os.path.join(config_dir, 'config.json')
        eval_path = os.path.join(config_dir, 'acquisition_results.pkl')
        loss_results_path = os.path.join(config_dir, 'results.json')
        multitask_loss_results_path = os.path.join(config_dir, 'multitask_loss_hist.txt')

        # Check if viable (enough epochs, results exist)
        with open(json_path, 'r') as f:
            args_dict = json.load(f)
        results_collected = os.path.exists(eval_path)

        # If viable experiment, add results to results_dict
        if results_collected and args_dict['num_epochs'] >= min_epochs:
            
            try:
                new_key, *res = get_result_line_new(args_dict, eval_path, loss_results_path, multitask_loss_results_path, posterior_acquisition_type)
            except AssertionError as e:
                print(e)
                continue

            # Get current line
            existing_results = results_dict.get(new_key, [])

            existing_results += [res]

            # Add line back to all results
            results_dict[new_key] = existing_results

    return results_dict


def old_collate_results(config_directories, min_epochs):

    # Dictionary schema = {(task, layers): [(test_prop1, spearmans_coef1), ...]}
    results_dict = {}

    # Iterate over experiments
    for config_dir in tqdm(config_directories):

        # Get the config and results paths
        json_path = os.path.join(config_dir, 'config.json')
        eval_path = os.path.join(config_dir, 'acquisition_results.pkl')

        # Check if viable (enough epochs, results exist)
        with open(json_path, 'r') as f:
            args_dict = json.load(f)
        results_collected = os.path.exists(eval_path)

        # If viable experiment, add results to results_dict
        if results_collected and args_dict['num_epochs'] >= min_epochs:

            new_key, test_prop, res = get_result_line_old(args_dict, eval_path)

            # Get current line
            existing_results = results_dict.get(new_key, {})

            # Get current trials for point
            existing_trials = existing_results.get(test_prop, [])
            
            # Add trial to line
            existing_results[test_prop] = existing_trials + [res]

            # Add line back to all results
            results_dict[new_key] = existing_results

    return results_dict



if __name__ == '__main__':

    exp_tag = sys.argv[1] # e.g. distillation_entropy_mse
    minimum_epochs = int(sys.argv[2])   # 50
    dataset = sys.argv[3] # cifar10 or cifar100
    results_type = sys.argv[4]  # graphical or tabular

    # Where to look for specifics
    suffix = '100' if dataset == 'cifar100' else '10'

    if results_type == 'old':
        base_dir = f'/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_{suffix}'

        # * path for glob search
        config_dir_search = os.path.join(base_dir, exp_tag + "-*/")

        # Get all config dirs
        config_dirs = glob(config_dir_search)

        # Dictionary schema = {(task, layers): [(test_prop1, spearmans_coef1), ...]}
        results_dict = old_collate_results(config_dirs, minimum_epochs)

        graphical_results_set(results_dict, base_dir, exp_tag)

    elif results_type == 'new':

        # Posterior 
        POSTERIOR_ACQUISITION_TYPE = sys.argv[5]

        base_dir = f'/home/alta/BLTSpeaking/exp-pr450/lent_logs/second_baseline_posterior_distillation_{suffix}'

        # * path for glob search
        config_dir_search = os.path.join(base_dir, exp_tag + "-*/")

        # Get all config dirs
        config_dirs = glob(config_dir_search)

        # Dictionary schema = {(task, layers): [(test_prop1, spearmans_coef1), ...]}
        results_dict = new_collate_results(config_dirs, minimum_epochs, POSTERIOR_ACQUISITION_TYPE)

        tabular_results_set(results_dict)

        
