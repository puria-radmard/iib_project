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


total_added_prop=0.6
scheduler_epochs=100000
scheduler_proportion=1
dropout=0.3
weight_decay=0.0005
finetune_weight_decay=0.0005
batch_size=128

total_start_prop=0.05


dataset=$1
architecture_name=$2
minibatch_prop=$3
# Fixing lr since we will always reinitialise
initial_lr=$4
finetune_lr=$initial_lr
# Fixing epochs since we will always reinitialise
num_initial_epochs=$5
num_finetune_epochs=$num_initial_epochs
save_dir=$6


python -m interfaces.cifar_labelled_classification_recalibration_with_diversity \
    --dataset                           $dataset    \
    --architecture_name                 $architecture_name  \
    --minibatch_prop                    $minibatch_prop \
    --total_added_prop                  $total_added_prop   \
    --total_start_prop                  $total_start_prop   \
    --initial_lr                        $initial_lr \
    --finetune_lr                       $finetune_lr    \
    --scheduler_epochs                  $scheduler_epochs   \
    --scheduler_proportion              $scheduler_proportion   \
    --dropout                           $dropout    \
    --weight_decay                      $weight_decay   \
    --finetune_weight_decay             $finetune_weight_decay  \
    --batch_size                        $batch_size \
    --num_initial_epochs                $num_initial_epochs \
    --num_finetune_epochs               $num_finetune_epochs    \
    --save_dir                          $save_dir   \
    --do_reinitialise_autoencoder_ensemble
