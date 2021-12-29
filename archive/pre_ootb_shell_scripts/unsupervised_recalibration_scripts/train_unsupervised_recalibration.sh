#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME

export PATH="/home/alta/Users/pr450/anaconda3/bin:$PATH"
echo $PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate bias_investigation
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/bias_investigation/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/bias_investigation/lib


initial_lr=$1
finetune_lr=$2
labelled_utt_list_path=$3
unlabelled_utt_list_path=$4
num_initial_epochs=$5
embedding_dim=$6
encoder_lstm_sizes=`echo "$7" | tr , " "`
encoder_lstm_layers=`echo "$8" | tr , " "`
encoder_fc_hidden_dims=`echo "$9" | tr , " "`
mult_noise=${10}
cell_type=${11}
encoder_architecture=${12}
decoder_architecture=${13}


python -m interfaces.audio_unsupervised_recalibration_scripts \
    --minibatch_seconds 3600 \
    --total_added_seconds 90000 \
    --total_start_seconds 0 \
    --encoder_architecture $encoder_architecture \
    --decoder_architecture $decoder_architecture \
    --cell_type $cell_type \
    --metric_function reconstruction_loss \
    --encoder_lstm_sizes $encoder_lstm_sizes \
    --encoder_lstm_layers $encoder_lstm_layers \
    --encoder_fc_hidden_dims $encoder_fc_hidden_dims \
    --labelled_utt_list_path $labelled_utt_list_path \
    --unlabelled_utt_list_path $unlabelled_utt_list_path \
    --feature_dim 40 \
    --embedding_dim $embedding_dim \
    --initial_lr  $initial_lr \
    --finetune_lr $finetune_lr \
    --num_initial_epochs $num_initial_epochs \
    --scheduler_epochs 5 \
    --scheduler_proportion 0.5 \
    --mult_noise $mult_noise \
    --dropout 0.3 \
    --weight_decay 0.0005 \
    --finetune_weight_decay 0.0005 \
    --batch_size 32 \
    --num_finetune_epochs 1 \
    --max_seq_len 300 \
    --save_dir /home/alta/BLTSpeaking/exp-pr450/logs/active_learning_sweep/config \
    --features_paths "/home/alta/BLTSpeaking/exp-graphemic-kaldi-lw519/feats/htk_fbk/AMI-IHM-train-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-graphemic-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXtrn06-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXgrp13-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXeval3-fbk.ark"
