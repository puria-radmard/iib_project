#!/bin/bash
#$ -S /bin/bash
#!/bin/bash
#$ -S /bin/bash

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate bias_investigation
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/bias_investigation/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib

baseDAFPath=$1 # e.g. /home/alta/BLTSpeaking/exp-pr450/logs/cifar_daf_recalibration/round_history-0
mirroredPath=$2 # e.g. /home/alta/BLTSpeaking/exp-pr450/logs/densenet_al_baseline/round_history-0

python -m interfaces.daf_acquisition_queuer \
    --dataset cifar10 \
    --baseDAFPath $baseDAFPath \
    --mirroredPath $mirroredPath \
    --saveDir /home/alta/BLTSpeaking/exp-pr450/logs/cifar_ensemble_daf_active_learning/round_history
