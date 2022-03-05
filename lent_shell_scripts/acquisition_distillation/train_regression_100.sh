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
conda activate easter_env
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/easter_env/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/easter_env/lib


densenet_daf_depth=$1
lr=0.01
logit_bank_path=$2
include_trainset=""
dataset="cifar100"
batch_size=64
training_mode=$3
objective_uniformalisation=$4
test_prop=$5
num_epochs=$6

save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_100/distillation_${training_mode}_${objective_uniformalisation}"


python -m interfaces.cifar_acquisition_regression \
    --densenet_daf_depth            $densenet_daf_depth \
    --lr                            $lr \
    --logit_bank_path               $logit_bank_path \
    $include_trainset   \
    --dataset                       $dataset \
    --batch_size                    $batch_size  \
    --training_mode                 $training_mode \
    --objective_uniformalisation    $objective_uniformalisation \
    --test_prop                     $test_prop   \
    --num_epochs                    $num_epochs \
    --save_dir                      $save_dir
