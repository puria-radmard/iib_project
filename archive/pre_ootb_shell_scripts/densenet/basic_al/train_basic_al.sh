#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME

export PATH="/home/alta/Users/pr450/anaconda3/bin:$PATH"
echo $PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate bias_investigation
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/bias_investigation/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib

initProp=$1
roundProp=$2
totalBudgetProp=0.6
acquisitionFunction=$3
roundEpochs=$4
saveDir=$5
dataset=$6

python -m training_scripts.cifar_active_learning \
    -a densenet --depth 100 --growthRate 12 --train-batch 64 --epochs 100 --schedule 75 --gamma 0.1 --wd 1e-4 \
    --checkpoint /home/alta/BLTSpeaking/exp-pr450/logs/preresnet_al_baseline/checkpoint \
    --dataset $dataset \
    --initProp $initProp \
    --roundProp $roundProp \
    --totalBudgetProp $totalBudgetProp \
    --acquisitionFunction $acquisitionFunction \
    --roundEpochs $roundEpochs \
    --saveDir $saveDir
