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


total_added_prop=0.6
scheduler_epochs=100000
scheduler_proportion=1
dropout=0.3
weight_decay=0.0005
finetune_weight_decay=0.0005
batch_size=128
ensemble_size=1

reinit_option=do_reinitialise_autoencoder_ensemble
total_start_prop=0.05

# !!
unet_skip_option=compressed_net
# !!


metric_function=$1
minibatch_prop=$2
initial_lr=$3
finetune_lr=$initial_lr
num_initial_epochs=$4
num_finetune_epochs=$num_initial_epochs
save_dir=$5
dataset=$6


python -m interfaces.cifar_unsupervised_recalibration \
    --dataset $dataset \
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
    --batch_size $batch_size \
    --ensemble_size $ensemble_size \
    --num_initial_epochs $num_initial_epochs \
    --num_finetune_epochs $num_finetune_epochs \
    --$reinit_option \
    --$unet_skip_option \
    --save_dir $save_dir
