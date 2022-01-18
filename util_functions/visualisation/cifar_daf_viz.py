from .cifar_al_viz import get_json
from glob import glob
import pandas as pd
import os

def cifar_daf_al_curves(absolute_daf_base):

    # Schema: {DAF architecture: [([image counts 1], [performances 1]), ([image counts 2], [performances 2])]}
    all_performance_curves = {}

    # Get all the round_histories under the base
    daf_bases = glob(os.path.join(absolute_daf_base, 'round_history-*'))

    # Iterate over the round histroy folders
    for daf_base in daf_bases:

        # Find out how many rounds we should iterate over
        num_rounds = len(glob(os.path.join(daf_base, 'round_*_*/log.txt')))

        # Initialise curve for this run
        performance, images = [], []

        # Find the DAF architecture
        architecture = get_json(os.path.join(daf_base, 'daf_config.json'))['architecture_name']

        # Iterate over rounds
        for nr in range(1, num_rounds+1):

            # We don't know the number of images for this round, so just use glob to get folder
            round_base = glob(os.path.join(daf_base, f'round_{nr}_*'))[0]

            # Now that we know the folder name, use it to get the number of images for that round
            num_images = int(round_base.split('_')[-1])
            images.append(num_images)

            # Add the best performance in that round to the curve
            valid_acc = pd.read_csv(os.path.join(round_base, 'log.txt'), sep='\t')['Valid Acc.']
            performance.append(valid_acc.max())# iloc[-1])

            if len(valid_acc) < 300: print(architecture, num_images, len(valid_acc))

        # If already logged this architecture, add to it, else initialise it in all_performance_curves
        current_log = all_performance_curves.get(architecture, [])
        all_performance_curves[architecture] = current_log + [(images, performance)]

    return all_performance_curves