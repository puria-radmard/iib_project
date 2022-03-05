#!/bin/bash
#$ -S /bin/bash
#!/bin/bash
#$ -S /bin/bash

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate easter_env
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/easter_env/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/easter_env/lib

dataset=$1
baseDAFPath=$2 # e.g. /home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_daf_recalibration/round_history-0
mirroredPath=$3 # e.g. /home/alta/BLTSpeaking/exp-pr450/lent_logs/densenet_al_baseline/round_history-0
saveDir=$4 # e.g. /home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_daf_active_learning/round_history

python -m interfaces.daf_acquisition_queuer \
    --run_path /home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/general_daf_active_learning/run_train_cifar_on_subset.sh \
    --dataset $dataset \
    --baseDAFPath $baseDAFPath \
    --mirroredPath $mirroredPath \
    --saveDir $saveDir
