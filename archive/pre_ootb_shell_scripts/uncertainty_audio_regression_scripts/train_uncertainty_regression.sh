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


lr=$1
encoder_lstm_sizes=`echo "$2" | tr , " "`
encoder_lstm_layers=`echo "$3" | tr , " "`
embedding_dim=$4
decoder_fc_hidden_dims=`echo "$5" | tr , " "`
num_epochs=$6
cell_type=$7
alignment_paths=`echo "$8" | tr , " "`


python -m interfaces.uncertainty_audio_regression \
    --cell_type $cell_type \
    --encoder_lstm_sizes $encoder_lstm_sizes \
    --encoder_lstm_layers $encoder_lstm_layers \
    --decoder_fc_hidden_dims $decoder_fc_hidden_dims \
    --feature_dim 40 \
    --embedding_dim $embedding_dim \
    --lr $lr \
    --num_epochs $num_epochs \
    --scheduler_epochs 5 \
    --scheduler_proportion 0.5 \
    --dropout 0.3 \
    --weight_decay 0.0005 \
    --batch_size 32 \
    --max_seq_len 300 \
    --test_prop 0.25 \
    --save_dir /home/alta/BLTSpeaking/exp-pr450/logs/uncertainty_regression/config \
    --alignment_paths $alignment_paths \
    --features_paths "/home/alta/BLTSpeaking/exp-graphemic-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXtrn06-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXgrp13-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXeval3-fbk.ark"
