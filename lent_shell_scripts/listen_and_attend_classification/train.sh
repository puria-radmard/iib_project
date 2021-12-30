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


arch=$1
lr=$2
decoder_dropout=$3
labelled_list=$4
unlabelled_list=$5
save_dir=$6

scheduler_epochs=100000
scheduler_proportion=1
weight_decay=0.0005
batch_size=64
num_epochs=10
labelled_list=$labelled_list
unlabelled_list=$unlabelled_list
max_seq_len=10000 # Let's try this first
test_prop=0.25



python -m interfaces.audio_labelled_classification \
    --architecture_name      $arch \
    --lr                     $lr    \
    --scheduler_epochs       $scheduler_epochs  \
    --scheduler_proportion   $scheduler_proportion  \
    --dropout                $decoder_dropout   \
    --weight_decay           $weight_decay    \
    --batch_size             $batch_size    \
    --num_epochs             $num_epochs    \
    --labelled_list          $labelled_list  \
    --unlabelled_list        $unlabelled_list  \
    --max_seq_len            $max_seq_len  \
    --test_prop              $test_prop  \
    --save_dir               $save_dir    \
    --features_paths "/home/alta/BLTSpeaking/exp-graphemic-kaldi-lw519/feats/htk_fbk/AMI-IHM-train-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-graphemic-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXtrn06-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXgrp13-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXeval3-fbk.ark"
