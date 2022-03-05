#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME

export PATH="/home/alta/Users/pr450/anaconda3/bin:$PATH"
echo $PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
# export CUDA_VISIBLE_DEVICES=3
echo $CUDA_VISIBLE_DEVICES
 
## DON'T FORGET TO TURN OFF CUDA LINE ABOVE
## Key training examples to try:
# train_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/baseline_posterior_distillation/b_lambda_scheduling/train_cifar_regression.sh"

## Previous set up, to check for refactor issues
# $train_script posterior_distillation_multitask entropy_histogram_jacobian_bce "--lambda_scheduler_option,no_lambda_scheduling,--multitask_acquisition_target_weight,0.75" cifar100
# $train_script entropy_histogram_jacobian_bce None "--lambda_scheduler_option,no_lambda_scheduling,--multitask_acquisition_target_weight,1.0" cifar10

## Coarsening CIFAR100 AND trying out asymptotic lambda
# $train_script posterior_distillation_multitask entropy_histogram_jacobian_bce "--lambda_scheduler_option,asymptotic_pretrainer_lambda_scheduling,--multitask_acquisition_target_weight,5" cifar100coarse

## Checking if soft_rank works
# $train_script posterior_distillation_multitask entropy_soft_rank "--lambda_scheduler_option,step_pretrainer_lambda_scheduling,--multitask_acquisition_target_weight,5" cifar100

CUDA_VERSION=10.1
test -z "$CUDA_VERSION" || {
        test -d "/usr/local/cuda-${CUDA_VERSION}" &&
        CUDA_DEFAULT="/usr/local/cuda-${CUDA_VERSION}"
}
CUDA_PATH="${CUDA_DEFAULT:-/usr/local/cuda}"
PATH="${CUDA_PATH}/bin:${PATH}"
LD_RUN_PATH="${CUDA_PATH}/lib64:${LD_RUN_PATH}"
LD_LIBRARY_PATH="${CUDA_PATH}/lib64:${LD_LIBRARY_PATH}"
export CUDA_PATH PATH LD_RUN_PATH LD_LIBRARY_PATH


source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate easter_env
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/easter_env/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/easter_env/lib


training_mode=$1
multitask_training_mode=$2
lambda_scheduler_option_and_parameter=$3 # This is in full form - e.g. "\"--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 0.75\""
dataset=$4

echo training_mode   $training_mode
echo multitask_training_mode  $multitask_training_mode
echo lambda_scheduler_option_and_parameter   $lambda_scheduler_option_and_parameter
echo dataset   $dataset



if [[ $dataset == "cifar10" ]]
then
   suffix1="10"
   suffix2="10"
elif [[ $dataset == "cifar100" ]]
then
   suffix1="100"
   suffix2="100"
elif [[ $dataset == "cifar100coarse" ]]
then
   suffix1="100"
   suffix2="100_coarse"
   coarse_cifar100_option="--coarse_cifar100"
   dataset="cifar100"
else
   echo "Bad dataset"
   exit 1
fi



# This one was trained with 15001 images
logit_bank_path="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_${suffix1}/trained_logits-2/evaluations.pkl"
densenet_daf_depth="64" # NB: smaller densenet this time!!
include_trainset=""
batch_size="256"
lr="0.01" # Best one found, or close enough
byol_daf="" # Not using BYOL DAF for now
distillation_temperature="3.5" # Best one found, or close enough
objective_uniformalisation="no_un"
test_prop="0.1"
num_epochs="50"

lambda_scheduler_option="$(cut -d',' -f2 <<<"$lambda_scheduler_option_and_parameter")"
save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/third_baseline_posterior_distillation_${suffix2}/distillation_${training_mode}_${lambda_scheduler_option}"


python -m interfaces.cifar_acquisition_regression \
   --lr                                     $lr \
   --logit_bank_path                        $logit_bank_path    \
   --training_mode                          $training_mode  \
   --multitask_training_mode                $multitask_training_mode    \
   --distillation_temperature               $distillation_temperature   \
   --densenet_daf_depth                     $densenet_daf_depth \
   --dataset                                $dataset    \
   --batch_size                             $batch_size \
   --objective_uniformalisation             $objective_uniformalisation \
   --test_prop                              $test_prop  \
   --num_epochs                             $num_epochs \
   --save_dir                               $save_dir   \
   $include_trainset   $byol_daf   $coarse_cifar100_option \
   `echo "$lambda_scheduler_option_and_parameter" | tr , " "`
