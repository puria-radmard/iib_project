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


total_added_seconds=180000  # 50 hours total added
scheduler_epochs=100000
scheduler_proportion=1
dropout=0.3
weight_decay=0.0005
finetune_weight_decay=0.0005
batch_size=64
max_seq_len=300
ensemble_size=1

metric_function=$1
minibatch_seconds=$2
labelled_utt_list_path=$3
unlabelled_utt_list_path=$4
initial_lr=$5
finetune_lr=$6
unet_skip_option=$7
num_initial_epochs=$8
num_finetune_epochs=$9
reinit_option=${10}
save_dir=${11}


python -m training_scripts.audio_gru_unsupervised_recalibration \
    --metric_function   $metric_function \
    --minibatch_seconds $minibatch_seconds \
    --total_added_seconds   $total_added_seconds \
    --initial_lr    $initial_lr \
    --finetune_lr   $finetune_lr \
    --labelled_utt_list_path $labelled_utt_list_path \
    --unlabelled_utt_list_path $unlabelled_utt_list_path \
    --scheduler_epochs  $scheduler_epochs \
    --scheduler_proportion  $scheduler_proportion \
    --dropout   $dropout \
    --weight_decay  $weight_decay \
    --finetune_weight_decay     $finetune_weight_decay \
    --batch_size    $batch_size \
    --num_initial_epochs    $num_initial_epochs \
    --num_finetune_epochs   $num_finetune_epochs \
    --$reinit_option \
    --$unet_skip_option \
    --max_seq_len   $max_seq_len \
    --save_dir  $save_dir \
    --ensemble_size $ensemble_size \
    --features_paths "/home/alta/BLTSpeaking/exp-graphemic-kaldi-lw519/feats/htk_fbk/AMI-IHM-train-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-graphemic-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXtrn06-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXgrp13-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXeval3-fbk.ark"