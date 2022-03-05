#!/bin/bash
#$ -S /bin/bash

unset LD_PRELOAD
echo $HOSTNAME

export PATH="/home/alta/Users/pr450/anaconda3/bin:$PATH"
echo $PATH

export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=2
echo $CUDA_VISIBLE_DEVICES

## To test out
# train_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/las_multitask/train_multitask_speech.sh"

# $train_path moving_dnn lc_soft_rank None 0.00001 /home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm "--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 1.0"
# $train_path moving_dnn bag_of_phonemes_multitask lc_soft_rank 0.00001 /home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm "\"--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 0.75\""

source /home/alta/Users/pr450/anaconda3/etc/profile.d/conda.sh
conda activate easter_env
export PYTHONBIN=/home/alta/Users/pr450/anaconda3/envs/easter_env/bin/python

echo $0 $@

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/alta/Users/pr450/anaconda3/envs/easter_env/lib


listener_type=$1
criterion=$2
multitask_training_mode=$3
lr=$4
alignment_paths=$5
lambda_scheduler_option_and_parameter=$6

echo
echo listener_type $listener_type
echo criterion $criterion
echo multitask_training_mode $multitask_training_mode
echo lr $lr
echo alignment_paths $alignment_paths
echo lambda_scheduler_option_and_parameter $lambda_scheduler_option_and_parameter
echo

lambda_scheduler_option="$(cut -d' ' -f2 <<<"$lambda_scheduler_option_and_parameter")"
save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/las_multitask/distillation_${criterion}_${lambda_scheduler_option}"

unpacked_lambda_options="$lambda_scheduler_option_and_parameter" | tr -d '"'

python -m interfaces.audio_uncertainty_regression \
    --architecture_name             "default_${listener_type}_listener_transformer_regression_architecture" \
    --lr                            $lr \
    --scheduler_epochs              10000 `# No scheduling for now` \
    --scheduler_proportion          1.0 `# Keep it simple` \
    --dropout                       0 \
    --weight_decay                  0.0005 `# Best to maximise for torchsort criterion` \
    --batch_size                    64 `# Compared to 10 for binary classifier` \
    --num_epochs                    20 \
    --alignment_paths               $alignment_paths `# No need to split sequences with the LAS architectures` \
    --max_seq_len                   1000 `# Standard again` \
    --test_prop                     0.25 \
    --save_dir                      $save_dir \
    --criterion                     $criterion \
    --multitask_training_mode       $multitask_training_mode \
    --features_paths "/home/alta/BLTSpeaking/exp-graphemic-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXtrn06-fbk.ark" \
       "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXgrp13-fbk.ark" \
    $unpacked_lambda_options \

# "/home/alta/BLTSpeaking/exp-graphemic-kaldi-lw519/feats/htk_fbk/AMI-IHM-train-fbk.ark"
# "/home/alta/BLTSpeaking/exp-yw396/K5/kaldi-mar2017/feats/htk_fbk/BLXXXeval3-fbk.ark"
