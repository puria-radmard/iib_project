import json
import numpy as np
import pandas as pd
from glob import escape, glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_maxes_df(results, num_epochs):
    results = results.groupby(['jacobian_setting', 'reweigher_setting', 'test_prop']).aggregate(lambda x: np.squeeze(np.array(list(zip(x))), 1))
    results['rank_maxs'] = results.rank_history.apply(lambda x: x[:,:num_epochs].max(-1))
    results['rank_means'] = results.rank_maxs.apply(lambda x: x.mean(-1))
    results['rank_stds'] = results.rank_maxs.apply(lambda x: x.std(-1))
    results.drop(['train_loss', 'test_loss', 'rank_history', 'rank_maxs'], axis = 1, inplace = True)
    results = results[results.index.get_level_values(2).isin([0.2, 0.6])]
    return results

log_query = '/home/alta/BLTSpeaking/exp-pr450/jacobian_investigation/logs/*.json'

results_df = pd.DataFrame(columns=[])

for res_path in glob(log_query):
    
    with open(res_path, 'r') as f:
        results_dict = json.load(f)

    results_df = results_df.append(results_dict, ignore_index=True)

results = get_maxes_df(results_df, -1)
results['rank'] = results[['rank_means', 'rank_stds']].apply(lambda x: f'${round(x[0], 2)}\pm{round(x[1], 2)}$', axis = 1)

print(results.to_latex(escape=False))
