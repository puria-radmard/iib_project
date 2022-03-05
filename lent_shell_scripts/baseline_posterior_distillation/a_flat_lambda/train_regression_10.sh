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


lr=$1
training_mode=$2
multitask_training_mode=$3
distillation_temperature=$4
multitask_acquisition_target_weight=$5
byol_daf=$6

# This one was trained with 15001 images
logit_bank_path="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_10/trained_logits-2/evaluations.pkl"
densenet_daf_depth="130"
include_trainset=""
dataset="cifar10"
batch_size="64"
objective_uniformalisation="no_un"
test_prop="0.1"
num_epochs="50"

save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/second_baseline_posterior_distillation_10/distillation_${training_mode}"


python -m interfaces.cifar_acquisition_regression \
    --lr                                      $lr \
    --logit_bank_path                         $logit_bank_path    \
    --training_mode                           $training_mode  \
    --multitask_training_mode                 $multitask_training_mode    \
    --distillation_temperature                $distillation_temperature   \
    --multitask_acquisition_target_weight     $multitask_acquisition_target_weight    \
    --densenet_daf_depth                      $densenet_daf_depth \
    --dataset                                 $dataset    \
    --batch_size                              $batch_size \
    --objective_uniformalisation              $objective_uniformalisation \
    --test_prop                               $test_prop  \
    --num_epochs                              $num_epochs \
    --save_dir                                $save_dir   \
    $include_trainset   $byol_daf   \
