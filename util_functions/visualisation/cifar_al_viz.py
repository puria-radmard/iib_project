from imp import acquire_lock
import pandas as pd
import os, json
from glob import glob
import numpy as np
from tqdm import tqdm
import seaborn as sns
sns.set_style('darkgrid')


def get_json(path):
    with open(path, 'r') as jf:
        jfile = json.load(jf)
    return jfile


def get_plot_point_baseline(log_base, round_num, key, epoch_limit=None):

    round_base = os.path.join(log_base, f'round_{round_num}')

    try:

        labelled_set_path = os.path.join(round_base, 'labelled_set.txt')
        with open(labelled_set_path, 'r') as f:
            lines = f.read().split('\n')[:-1]
        res_image = len(lines)

        valid_acc = pd.read_csv(os.path.join(round_base, 'log.txt'), sep='\t')['Valid Acc.']
        if len(valid_acc) < 300: print(key, res_image, len(valid_acc))
        if epoch_limit != None:
            valid_acc = valid_acc[:epoch_limit]
        res_performance = valid_acc.max()
    except:
        return None, None
    
    return res_image, res_performance


def visualise_performance_curves(performance_curves, axes, key_translation):

    for acq_name, curves in performance_curves.items():

        extent = min([len(c[0]) for c in curves])
        images = curves[0][0][:extent]
        performance = np.mean([c[1][:extent] for c in curves], 0)

        axes.plot(images, performance, label = key_translation.get(acq_name, acq_name))
    
    return axes


def get_plot_point_daf():
    pass


def cifar_baseline_al_curves(absolute_base, num_runs, max_round_nums, epoch_limit=None):

    all_perf_curves = {}

    for l in tqdm(range(num_runs)):

        try:
            log_base = f'{absolute_base}/round_history-{l}'
            config_path = os.path.join(log_base, 'config.json')
            config = get_json(config_path)
        except:
            continue

        performance, images = [], []

        acq = config['acquisitionFunction']
        key = (acq, config['roundProp'])

        for r in range(1, max_round_nums):
            new_image, new_performance = get_plot_point_baseline(log_base, r, key, epoch_limit)
            if new_image is None: break
            images.append(new_image)
            performance.append(new_performance)
        

        all_perf_curves[key] = all_perf_curves.get(key, []) + [(images, performance)]

    return all_perf_curves


def cifar_daf_al_curves(daf_bases):

    all_daf_images = []
    all_daf_performances = []

    for daf_base in daf_bases:
        num_rounds = len(glob(os.path.join(daf_base, 'round_*_*/log.txt')))

        performance, images = [], []

        for nr in range(1, num_rounds+1):
            round_base = glob(os.path.join(daf_bases, f'round_{nr}_*'))[0]
            images.append(int(round_base.split('_')[-1]))
            valid_acc = pd.read_csv(os.path.join(round_base, 'log.txt'), sep='\t')['Valid Acc.']
            performance.append(valid_acc.max())# iloc[-1])

        all_daf_images.append(images)
        all_daf_performances.append(performance)

    return all_daf_images, all_daf_performances
