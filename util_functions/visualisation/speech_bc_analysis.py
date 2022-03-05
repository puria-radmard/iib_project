import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import json, torch, os
import active_learning as al
al.disable_tqdm()

import sys
sys.path.insert(0, '../active_learning-pr450')
from alignment_utils import DecodingModelSurrogate


class CachedClassifierPreds(DecodingModelSurrogate):
    
    def __init__(self, utt2dur_path, grapheme_lookup_path, metadata_path, lookup_path, eval_dict):
        super(CachedClassifierPreds, self).__init__(utt2dur_path, grapheme_lookup_path, metadata_path, lookup_path)
        utts = set(self.df.utt.tolist())
        self.unlabelled_preds_dict = {
            k: v for k, v 
            in zip(eval_dict['unlabelled_utt_ids'], eval_dict['unlabelled_preds'])
            if k in utts
        }
        
    def __call__(self, batch):
        # BATCHES NEED TO BE IN FORM [(part_id, [utt_id1, ..., utt_idn]), (...)]
        
        # Part ids in batch
        part_ids = [b[0] for b in batch]
        
        # Template output dictionary
        output_dict = {'classifier_preds': []}
        
        # Iterate over parts: b = (part_id, [utt_id1, ..., utt_idn])
        for b in batch:
            
            # Preds for utts in this part
            part_classifier_preds = []
            
            for utt in b[1]:
                # Might not be in df/dict as well
                try:
                    part_classifier_preds.append(self.unlabelled_preds_dict[utt])
                except:
                    continue
                
            # Add to main output
            output_dict['classifier_preds'].append(part_classifier_preds.copy())
            
        # Incorporate 'traditional' features from base class
        remainder_dict = super(CachedClassifierPreds, self).__call__(batch)
        output_dict.update(remainder_dict)
        
        return {k: v for k, v in output_dict.items()}


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    return sys.stdout


def enablePrint(new_std):
    sys.stdout = new_std


def get_json(path):
    with open(path, 'r') as f:
        j = json.load(f)
    return j


def cdf_plot(data, numbins, axs, label):
    # Data is the raw data distribution, num bins will be equally spread on data range
    res = stats.cumfreq(data, numbins=numbins)
    cum_data = res.lowerlimit + np.linspace(0, res.binsize*res.cumcount.size, res.cumcount.size)
    axs.plot(cum_data, res.cumcount/res.cumcount[-1], label=label)
    return cum_data, res.cumcount/res.cumcount[-1]


def filter_ami_from_labelled_eval(evaluation_dict):
    # Collect data from evaluation_dict['labelled_preds'] only if the corresponding 
    # evaluation_dict['labelled_utt_ids'] IS NOT from ami
    res = torch.empty(0, 2)
    for pred, utt_id in zip(evaluation_dict['labelled_preds'], evaluation_dict['labelled_utt_ids']):
        if utt_id[0] != 'A':
            res = torch.cat([res, pred.reshape(1, 2)])
    return res


def filter_bulats_from_labelled_eval(evaluation_dict):
    # Collect data from evaluation_dict['labelled_preds'] only if the corresponding 
    # evaluation_dict['labelled_utt_ids'] IS from ami
    res = torch.empty(0, 2)
    for pred, utt_id in zip(evaluation_dict['labelled_preds'], evaluation_dict['labelled_utt_ids']):
        if utt_id[0] == 'A':
            res = torch.cat([res, pred.reshape(1, 2)])
    return res


def scatter_daf_scores_against_lc_asr(eval_dictionary, axes, ylabel, bins = [100, 100]):

    # Scatter 'ASR time weighted confidence' on the x axis of provided axes
    # and the corresponding output value on the y - label for y axis is an argument

    # Assume dictionary with torch tensor values as the input
    x = eval_dictionary['unlabelled_certainties'].numpy()
    y = eval_dictionary['unlabelled_preds'][:,1].numpy()

    # Label the axes - one is fixed, the other is an argument
    axes.set_xlabel('ASR time weighted confidence')
    axes.set_ylabel(ylabel)

    # Makes sense here
    axes.set_xlim(0, 1)

    # Calculate densities
    hh, locx, locy = np.histogram2d(x, y, bins=bins)
    hh = np.log(hh)
    
    # Sort the points by density, so that the densest points are plotted last
    z = np.array([hh[np.argmax(a<=locx[1:]),np.argmax(b<=locy[1:])] for a,b in zip(x,y)])
    idx = z.argsort()
    x2, y2, z2 = x[idx], y[idx], z[idx]

    # Scatter the coloured data
    plt.scatter(x2, y2, c=z2, cmap='jet', marker='.')  

    return x, y


def get_acq_set(acq, config, evaluation_dict, word_alignment_path,):
    
    time_budget = 50*60*60
    
    diversity_policy = al.batch_querying.NoPolicyBatchQuerying()

    model = CachedClassifierPreds(
        utt2dur_path = '/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/labelled_b450/fbk.asr+man/utt2dur', 
        metadata_path = '/home/alta/BLTSpeaking/convert-v2/lib/scores/BULATS_Speaking_MetaData_July_2013_Nov_2017.v1.txt',
        lookup_path = "/home/alta/BLTSpeaking/convert-v2/4/lib/spId/BLXXXgrp17-map.lst",
        grapheme_lookup_path = '/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/lang-LM1-int_b450/phones/align_lexicon.txt',
        eval_dict = evaluation_dict
    )
    
    bulats_df = model.df[model.df.utt.str.startswith('C')].reset_index()

    assert all(bulats_df.utt.str.slice(0, 24) == bulats_df.part)                    # All parts are just the first 24 characters of a part
    assert set(bulats_df.utt.to_list()) > set(set(evaluation_dict['unlabelled_utt_ids']))    # All utterances in the evaluation set are in the BULATS only df

    L1_list = model.df.FirstLanguage.tolist()
    band_list = model.df.SpeakingBand.tolist()
    metadata_list = [[L1, band_list[i]] for i, L1 in enumerate(L1_list)]
    init_list = [None] * len(model.df)

    train_set = al.dataset_classes.KaldiMetaDataset(
        part_ids=model.df.part.tolist(), utt_ids=model.df.utt.tolist(), speakers=model.df.speaker.tolist(),
        L1s=L1_list, bands=band_list, durs=model.df.dur.tolist(), metadata=metadata_list,
        phoneme_confidence_scores=init_list.copy(), phoneme_durations=init_list.copy(), phoneme_sequence=init_list.copy(), 
        grapheme_confidence_scores=init_list.copy(), grapheme_durations=init_list.copy(), grapheme_sequence=init_list.copy(),
        word_sequence=init_list.copy(), word_confidence_scores=init_list.copy(), word_durations=init_list.copy(),
        grapheme_trigrams=init_list.copy(), phoneme_trigrams=init_list.copy(),
        classifier_preds=init_list.copy()
    )
    
    if acq == 'conf':
        acquisition = al.acquisition.TimeWeightedWordConfidence(train_set)
        selector = al.selector.FullSequenceSelector(1, time_budget, acquisition, diversity_policy, 'argmax')
        
    elif acq == 'bc':
        acquisition = al.acquisition.BinaryClassifierPredictionAcquisition(train_set)
        selector = al.selector.FullSequenceSelector(1, time_budget, acquisition, diversity_policy, 'argmax')
        
    elif acq == 'tnbc':
        acquisition = al.acquisition.TimeNormalisedBinaryClassifierPredictionAcquisition(train_set)
        selector = al.selector.FullSequenceSelector(1, time_budget, acquisition, diversity_policy, 'argmax')
        
    namer = al.agent.AutomaticKaldiNameIncrementer(
        model_run_dir="",
        data_run_dir="",
        log_run_dir="",
        base_model_paths={},
        base_data_paths={"labelled_utts": config['labelled_list'], "unlabelled_utts": config['unlabelled_list']},
        base_log_paths={},
        constant_paths={},
        feature_names=[],
        make=False
    )
    
    model.update_alignment(
        #phoneme_alignment_path = namer.prev_iter.phone_align_mlf, #"/home/alta/BLTSpeaking/asr-outputs/GKTS4/rnnlm/phone-align/BLXXXeval3.mlf",  
        #grapheme_alignment_path = agent.namer.prev_iter.phone_align_mlf, #"/home/alta/BLTSpeaking/asr-outputs/GKTS4/rnnlm/graph-align/BLXXXeval3.mlf", 
        word_alignment_path = word_alignment_path
    )
    
    agent = al.agent.KaldiAgent(train_set, 512, selector, model, 'cpu', namer, time_budget, '', call_path=False)
    agent.init(config['labelled_list'])
    
    return agent, model


def make_acquisition(_agent):

    __stdout = blockPrint()
    # MAKE THE ACQUISITION
    _agent.step()
    enablePrint(__stdout)
    
    # Simple checks that things are working

    print('Time added', sum(map(lambda x: x.cost, _agent.selector.round_selection))/60/60)
    lowest_score_added = min(map(lambda x: x.score, _agent.selector.round_selection))
    assert lowest_score_added == _agent.selector.round_selection[-1].score.item()
    
    return _agent


def acquisition_proportions_history(_agent, _model, return_utts = False):

    # Translates all bands to standard form
    bands_translater = {
        'pre-A1': 'A1',
        'A1': 'A1', 
        'A1_High': 'A1', 
        'A2': 'A2', 
        'A2_High': 'A2', 
        'B1': 'B1', 
        'B1_High': 'B1',
        'B2': 'B2', 
        'B2_High': 'B2',
        'C1': 'C', 
        'C2': 'C',
        'C1_High': 'C'
    }

    # Maps utterances of utts
    utt2band = {row.utt: row.SpeakingBand for row in _model.df.iloc}
    utt2dur = {row.utt: row.dur for row in _model.df.iloc}

    # Get all the utts flattened out
    utts_in_order = []
    for window in _agent.selector.round_selection:
        new_utts = _agent.train_set.utt_ids.get_attr_by_window(window)
        utts_in_order.extend(new_utts)

    ## Iterate through the utts and add to proportion history
    # Initialised counts history: empty list for all (target) bands
    counts_history = {k: [0] for k in set(bands_translater.values())}

    # Duration for x axis
    cumulative_duration = [0]

    for utt in utts_in_order:

        try:

            # Get band and dur of latest utt
            new_band = bands_translater[utt2band[utt]]
            new_dur = utt2dur[utt]

            # For all possible bands, only increment if new utt belongs to it
            # k = band name, v = counts so far
            for k, v in counts_history.items():
                new_val = v[-1] + new_dur if k == new_band else v[-1]
                counts_history[k].append(new_val)

            cumulative_duration.append(cumulative_duration[-1] + new_dur)

        except:
            print(utt)

    # Generate the proportions histrory
    counts_history_np = {k: np.array(v) for k, v in counts_history.items()}
    totals = sum(counts_history_np.values())
    proportions_history = {k: v/totals for k, v in counts_history_np.items()}
    
    if return_utts:
        return proportions_history, cumulative_duration, utts_in_order
    else:
        return proportions_history, cumulative_duration


def make_proportions_plot(durations, _props_history, axs=None):

    if axs == None:
        ## Begin plotting
        fig, axs = plt.subplots(1)

    # Order we want
    BANDS = ['A1', 'A2', 'B1', 'B2', 'C']

    # Start the base, which we 'build' on
    base = np.zeros_like(_props_history['A1'])

    # Go through bands and build on top of/update base
    for band in BANDS:

        # Fill between old and new base
        new_base = base + _props_history[band]
        axs.fill_between(durations, base, new_base, label = f"{band} - {round(_props_history[band][-1], 3)}")

        # Update base
        base = new_base

    axs.legend()


def get_conf_distribution(windows, _agent):
    
    # This will be a list of individual utterance scores
    res = []

    # Iterate over windows (part) selected this round
    for window in windows:
        
        # Get the confidences and durations of words in each utterance in that window
        word_confs = _agent.train_set.word_confidence_scores.get_attr_by_window(window)
        word_durs = _agent.train_set.word_durations.get_attr_by_window(window)
        
        # Dot product each one and add to list
        for conf_list, durs_list in zip(word_confs, word_durs):
            confs = np.array(conf_list)
            durs = np.array(durs_list)
            a = (confs@durs)/(durs.sum())
            res.append(a)
            
    return res


def plot_cumulative_average_confidence(axes, **agents):
    
    # Return the plots as is, for later use
    res_plots = {}
    
    for plot_label, _agent in agents.items():
        
        # Get all the confidences of the windows (parts) selected, in order of selection
        confs = get_conf_distribution(_agent.selector.round_selection, _agent)

        # Get all the durations of the windows (parts) selected, in order of selection
        durs = [window.cost for window in _agent.selector.round_selection]

        # Initialise plot
        cumulative_mean, cumulative_durs, dot_prod, total_time = [], [], 0, 0

        # Append (time weighted) mean so far
        for cf, dr in zip(confs, durs):

            # Keep track of all cf*dr, and of all dr
            dot_prod += cf * dr
            total_time += dr

            # Add new time weighted mean
            cumulative_durs.append(total_time)
            cumulative_mean.append(dot_prod/total_time)

        axes.plot(cumulative_durs, cumulative_mean, label = plot_label)
        
        # Save for later
        res_plots[plot_label] = [cumulative_durs, cumulative_mean]
        
    axes.legend()
    axes.set_xlabel('Total time')
    axes.set_ylabel('Running time weighted confidence average')
    
    return res_plots



def confidence_cdf_from_agents(main_model, axs, **agents):
    # main_model.df has all the confidences -> gives the full datset's cdf
    # agents is a dictionary mapping plot labels to the agents that provide the acquisition windows
    
    numbins = 50

    # Extract the full dataset's confidence distribution
    all_confs = []
    for _, row in main_model.df.iterrows():
        confs = np.array(row.word_confidence_scores)
        durs = np.array(row.word_durations)
        all_confs.append((confs@durs)/(durs.sum()))

    # The cdf plottables for the full set
    all_cdf_xys = {'Full set': cdf_plot(all_confs, numbins, axs, label='Full set')}

    # Plot and save the cdfs for all the agents provided
    for plot_label, _agent in agents.items():
        _dist = get_conf_distribution(_agent.selector.round_selection, _agent)
        all_cdf_xys[plot_label] = cdf_plot(_dist, numbins, axs, label=plot_label)

    axs.set_ylim(0, 1)
    axs.set_xlabel('ASR confidence')
    axs.set_ylabel('Cumulative')

    axs.legend()

    return all_cdf_xys
