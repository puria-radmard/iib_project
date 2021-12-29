import os, json, pickle, torch
import matplotlib.pyplot as plt
import seaborn as sns
from training_scripts.cifar_autoencoder_scripts import train_autoencoder_ensemble

sns.set_style('darkgrid')


def update_indices(previously_trained_indices, agent):
    # We are using the pytorch dataloader, so need to move the batches to the CIFARSubset classes
    new_indices = []
    for ls in agent.labelled_set:
        # The agent will continually add more data to the labelled set
        # We want to filter out the previously seen indices, only finetuning on new data
        unseen_indices = list(filter(lambda x: x not in previously_trained_indices, ls))
        new_indices.extend(unseen_indices)
        previously_trained_indices.update(unseen_indices)  
    return previously_trained_indices, new_indices



def make_metric_dictionary(all_round_windows, round_num, labelled_indices, newly_added_indices, save_dir):
    
    # Initialise the distributions
    labelled_distribution, unlabelled_distribution, new_added_distribution = [], [], []

    # Selector stores all window scores anyway, which are what the `active learning' is based on
    for window in all_round_windows:
        # Get score
        metric, index = window.score, window.i
        # Add to correct distributions
        is_labelled = index in labelled_indices
        labelled_distribution.append(metric) if is_labelled else unlabelled_distribution.append(metric)
        # Another distribution of indices about to be labelled
        if index in newly_added_indices:
            new_added_distribution.append(metric)
    
    if round_num%3 == 1:
        fig_path = os.path.join(save_dir, f"dis_plot_{round_num}_{len(labelled_distribution)}.png")
        fig, axs = plt.subplots(1, figsize=(10, 10))
        sns.histplot(unlabelled_distribution, label = "Unseen set", ax=axs, color='r')
        sns.histplot(labelled_distribution, label = "Seen set", ax=axs, color='b')
        sns.histplot(new_added_distribution, label = "New added set", ax=axs, color='g')
        axs.legend()
        fig.savefig(fig_path)

    pickle_path = os.path.join(save_dir, f"distributions_{round_num}_{len(labelled_distribution)}.pkl")
    with open(pickle_path, 'wb') as h:
        pickle.dump(
            {'seen': labelled_distribution, 'unseen': unlabelled_distribution},
            h, pickle.HIGHEST_PROTOCOL
        )



def unsupervised_recalibration_script(
        agent, args, model_init_method, dataloader_init_method,
        train_dataset, encodings_criterion, decodings_criterion, 
        anchor_criterion, save_dir, device
    ):

    print('WARNING: Please check that agent is initialised before unsupervised_recalibration_script is called!')

    round_num = 0
    # Keep track of the images we have already trained on
    # At each finetuning stage we finetune only on the added set
    previously_trained_indices = set()

    for _ in agent:

        # Lag in when the metric values are actually updated!
        if round_num > 0:
            make_metric_dictionary(agent.selector.all_round_windows, round_num, previously_trained_indices, set(new_indices), save_dir)

        previously_trained_indices, new_indices = update_indices(previously_trained_indices, agent)
        
        ## Sanity check on this
        # unlabelled_scores = list(filter(lambda x: x.i not in previously_trained_indices, agent.selector.all_round_windows))
        # unlabelled_scores = sorted(unlabelled_scores, key=lambda w: w.score, reverse = True)
        # set(map(lambda x: x.i, unlabelled_scores[:500])) == set(new_indices)

        if args.reinitialise_autoencoder_ensemble:
            # Reinit - we finetune on all data so far
            train_dataset.indices = list(previously_trained_indices)
        else:
            # Don't reinit - we only finetune on unseen data
            train_dataset.indices = list(new_indices)
        # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        train_dataloader = dataloader_init_method(train_dataset, batch_size=args.batch_size)

        # Make logging path based on save_dir
        round_num += 1

        with open(os.path.join(save_dir, f'labelled_set_{round_num}.txt'), 'w') as f:
            for i in new_indices:
                f.write(str(i))
                f.write('\n')

        print(f'\n\nRound: {round_num} | {len(train_dataset)} labelled')

        if args.reinitialise_autoencoder_ensemble or round_num == 1:
            agent.model.model = model_init_method()
            agent.model = agent.model.to(device)

        # Need to return logits drectly while in training/val script
        agent.model.midloop = True

        if round_num == 1:
            initial_optimizer = torch.optim.SGD(agent.model.parameters(), lr=args.initial_lr, \
                momentum=args.momentum, weight_decay=args.weight_decay)
            agent.model, results = train_autoencoder_ensemble(
                ensemble=agent.model,
                optimizer=initial_optimizer,
                scheduler=None,
                scheduler_epochs=[],
                encodings_criterion=encodings_criterion,
                decodings_criterion=decodings_criterion,
                anchor_criterion=anchor_criterion,
                train_dataloader=train_dataloader,
                test_dataloader=[],
                num_epochs=args.num_initial_epochs,
            )

        else:
            finetune_optimizer = torch.optim.SGD(agent.model.parameters(), lr=args.finetune_lr, \
                momentum=args.momentum, weight_decay=args.finetune_weight_decay)
            agent.model, results = train_autoencoder_ensemble(
                ensemble=agent.model,
                optimizer=finetune_optimizer,
                scheduler=None,
                scheduler_epochs=[],
                encodings_criterion=encodings_criterion,
                decodings_criterion=decodings_criterion,
                anchor_criterion=anchor_criterion,
                train_dataloader=train_dataloader,
                test_dataloader=[],
                num_epochs=args.num_finetune_epochs,
            )
            
        results_path = os.path.join(save_dir, f'round_{round_num}_results.json')
        with open(results_path, 'w') as jfile:
            json.dump(results, jfile)

        # Need to return attribute dictionary while for active learning agent
        agent.model.midloop = False

        # When this loop ends, model attributes and scores are updated, using a model that is trained on everything so far
        # So we need to visualise the distributions after this point

    return agent