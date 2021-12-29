#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME

export PATH="/home/alta/Users/pr450/anaconda3/bin:$PATH"
echo $PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate bias_investigation
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/bias_investigation/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Ussers/pr450/anaconda3/envs/bias_investigation/lib

initProp=$1
roundProp=$2
totalBudgetProp=0.6
acquisitionFunction=$3
roundEpochs=$4
saveDir=$5

python -m training_scripts.cifar_active_learning \
    -a preresnet --dataset cifar100 --depth 110 --epochs 164 --schedule 81 122 --gamma 0.1 --wd 1e-4 \
    --checkpoint /home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline/checkpoint \
    --initProp $initProp \
    --roundProp $roundProp \
    --totalBudgetProp $totalBudgetProp \
    --acquisitionFunction $acquisitionFunction \
    --roundEpochs $roundEpochs \
    --saveDir $saveDir
