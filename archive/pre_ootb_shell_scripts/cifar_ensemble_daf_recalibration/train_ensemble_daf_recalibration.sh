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

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib


dataset=${12}
metric_function=$1
minibatch_prop=$2
total_added_prop=0.5
total_start_prop=$3
initial_lr=$4
finetune_lr=$5
scheduler_epochs=100000
scheduler_proportion=1
dropout=0.3
weight_decay=0.0005
finetune_weight_decay=0.0005
mult_noise=$6
batch_size=128
num_initial_epochs=$7
num_finetune_epochs=$8
ensemble_size=$9
reinit_option=${10}
save_dir=${11}


python -m training_scripts.cifar_unsupervised_recalibration \
    --dataset $dataset \
    --ensemble_size $ensemble_size \
    --metric_function $metric_function \
    --minibatch_prop $minibatch_prop \
    --total_added_prop $total_added_prop \
    --total_start_prop $total_start_prop \
    --initial_lr $initial_lr \
    --finetune_lr $finetune_lr \
    --scheduler_epochs $scheduler_epochs \
    --scheduler_proportion $scheduler_proportion \
    --dropout $dropout \
    --weight_decay $weight_decay \
    --finetune_weight_decay $finetune_weight_decay \
    --mult_noise $mult_noise \
    --batch_size $batch_size \
    --num_initial_epochs $num_initial_epochs \
    --num_finetune_epochs $num_finetune_epochs \
    --$reinit_option \
    --save_dir $save_dir
