from glob import glob
import os, json, argparse, subprocess
from util_functions.base import config_savedir

parser = argparse.ArgumentParser()

parser.add_argument(
    "--baseDAFPath", required=True, type=str,
    help="farming daf acquisitions from here, e.g. /home/alta/BLTSpeaking/exp-pr450/logs/cifar_daf_recalibration/round_history-0"
)
parser.add_argument('--mirroredPath', type=str, required=True, help='The `normal\' acquisition file you want to mirror wrt acquisition sizes')
parser.add_argument('--saveDir', type=str, required=True, help='Where to make directories to save indices')
parser.add_argument('--dataset', type=str, required=True, help='Dataset, just to check everything is right')

args = parser.parse_args()
saveable_args = vars(args)
state = {k: v for k, v in args._get_kwargs()}

# Check all three provided datasets are consistent
with open(os.path.join(args.mirroredPath, 'config.json'), 'r') as jfile:
    mirrored_config = json.load(jfile)
    assert args.dataset == mirrored_config['dataset']

with open(os.path.join(args.baseDAFPath, 'config.json'), 'r') as jfile:
    daf_config = json.load(jfile)
    assert args.dataset == daf_config['dataset']

def round_number_text_path(base_path, num):
    return os.path.join(base_path, f"labelled_set_{num}.txt")

def indicies_from_text(path):
    with open(path, 'r') as f:
        text = f.read()[:-1]
    new_indices = [int(l) for l in text.split('\n')]
    return new_indices

def save_indices_to_path(args, indices):
    save_path = os.path.join(args.saveDir, f'indices_{len(indices)}.txt')
    with open(save_path, 'w') as f:
        for index in indices:
            f.write(str(index))
            f.write('\n')
    return save_path


# Configure save dir and save configs
args.saveDir = config_savedir(args.saveDir, args)

config_json_path = os.path.join(args.saveDir, "config.json")
with open(config_json_path, "w") as jfile:
    json.dump(saveable_args, jfile)

daf_config_json_path = os.path.join(args.saveDir, "daf_config.json")
with open(daf_config_json_path, "w") as jfile:
    json.dump(daf_config, jfile)

mirrored_config_json_path = os.path.join(args.saveDir, "mirrored_config.json")
with open(mirrored_config_json_path, "w") as jfile:
    json.dump(mirrored_config, jfile)


# Get the sizes we are aiming to copy with the minibatches
num_mirrored_acquisitions = len(glob(os.path.join(args.mirroredPath, 'round_*')))
target_set_sizes = []
for acq_num in range(1, num_mirrored_acquisitions+1):
    labelled_set_path = os.path.join(args.mirroredPath, f'round_{acq_num}', 'labelled_set.txt')
    target_set_sizes.append(len(indicies_from_text(labelled_set_path)))

# Initialise accumulated_indices
accumulated_indices = []

# Number of times the DAF has updated
num_minibatches = len(glob(os.path.join(args.baseDAFPath, 'labelled_set_*.txt')))

with open(os.path.join(args.saveDir, 'closeness_log.txt'), 'w') as log_file:

    print(f"Config dir : {args.saveDir}\n", file=log_file)
    
    subset_sizes = []
    subset_index_paths = []

    # Iterate over DAF acquisitions and add them to accumulated_indices
    for mb_num in range(1,num_minibatches+1):
        
        if len(accumulated_indices) >= target_set_sizes[0]:
            
            print(f'Target set size:', target_set_sizes[0], 'Actual set size:', len(accumulated_indices), file = log_file)
            print(f'\toff target by', len(accumulated_indices)-target_set_sizes[0], file = log_file)
            target_set_sizes = target_set_sizes[1:]
            subset_index_path = save_indices_to_path(args, accumulated_indices)

            subset_sizes.append(len(accumulated_indices))
            subset_index_paths.append(subset_index_path)
        
        if len(target_set_sizes) == 0:
            break

        set_path = round_number_text_path(args.baseDAFPath, mb_num)
        accumulated_indices.extend(indicies_from_text(set_path))

# Loop over all subsets and use their 
for i, (subset_size, subset_index_path) in enumerate(zip(subset_sizes, subset_index_paths), 1):

    log_base = os.path.join(args.saveDir, f"round_{i}_{subset_size}")
    os.mkdir(log_base)

    cmd = "/home/alta/BLTSpeaking/exp-pr450/shell_scripts/daf_active_learning/run_train_cifar_on_subset.sh "
    cmd += os.path.join(log_base, "output_log.txt") + " "
    cmd += str(subset_size) + " "
    cmd += log_base + " "
    cmd += subset_index_path + " "
    cmd +=  mirrored_config_json_path

    result = subprocess.Popen(cmd, shell=True)
    text = result.communicate()[0]
    return_code = result.returncode
 
    print(text)
    print(return_code)
    print('\n')
