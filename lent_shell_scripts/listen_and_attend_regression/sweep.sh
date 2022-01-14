run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_regression/run_train.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_regression/regression_sweep"

alignment_base="/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f"
# alignment_paths="${alignment_base}/decode_LM1-int_b50.BLXXXeval3/score_mbr_10/BLXXXeval3.utt.map.ctm,${alignment_base}/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"
alignment_paths="${alignment_base}/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"

# lr=$1
# architecture=$2
# num_epochs=$3
# alignment_paths=`echo "$4" | tr , " "`

#run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/audio_uncertainty_regression_scripts/train_uncertainty_regression.sh"
#txt_base=


if [ "$1" = "test_run" ]; then
    run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_regression/train.sh"
    $run_script 0.00001 default_blstm_listener_transformer_regression_architecture 10 $alignment_paths
    exit 1
fi

$run_script ${txt_base}_01.txt 0.00001 default_blstm_listener_self_attention_regression_architecture 10 $alignment_paths
$run_script ${txt_base}_02.txt 0.000001 default_blstm_listener_self_attention_regression_architecture 10 $alignment_paths
$run_script ${txt_base}_03.txt 0.0000001 default_blstm_listener_self_attention_regression_architecture 10 $alignment_paths
$run_script ${txt_base}_04.txt 0.00001 default_blstm_listener_transformer_regression_architecture 10 $alignment_paths
$run_script ${txt_base}_05.txt 0.000001 default_blstm_listener_transformer_regression_architecture 10 $alignment_paths
$run_script ${txt_base}_06.txt 0.0000001 default_blstm_listener_transformer_regression_architecture 10 $alignment_paths
