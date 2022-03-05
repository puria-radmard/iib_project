run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_regression/run_train.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_regression/regression_sweep"

alignment_base="/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f"
# alignment_paths="${alignment_base}/decode_LM1-int_b50.BLXXXeval3/score_mbr_10/BLXXXeval3.utt.map.ctm,${alignment_base}/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"
alignment_paths="${alignment_base}/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"

#run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/audio_uncertainty_regression_scripts/train_uncertainty_regression.sh"
#txt_base=


if [ "$1" = "test_run" ]; then
    crit=$2
    run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_regression/train.sh"
    $run_script 0.0001 $crit default_moving_dnn_listener_transformer_regression_architecture 10 $alignment_paths
    exit 1
fi

$run_script "${txt_base}_vanilla_bce_01.txt" 0.0001 vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_vanilla_bce_02.txt" 0.00001 vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_vanilla_bce_03.txt" 0.000001 vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_vanilla_bce_04.txt" 0.0000001 vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_vanilla_bce_05.txt" 0.00000001 vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths

$run_script "${txt_base}_jac_bce_01.txt" 0.0001 jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_jac_bce_02.txt" 0.00001 jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_jac_bce_03.txt" 0.000001 jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_jac_bce_04.txt" 0.0000001 jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_jac_bce_05.txt" 0.00000001 jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths

$run_script "${txt_base}_reweighted_jac_bce_01.txt" 0.0001 reweighted_jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_jac_bce_02.txt" 0.00001 reweighted_jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_jac_bce_03.txt" 0.000001 reweighted_jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_jac_bce_04.txt" 0.0000001 reweighted_jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_jac_bce_05.txt" 0.00000001 reweighted_jac_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths

$run_script "${txt_base}_reweighted_vanilla_bce_01.txt" 0.0001 reweighted_vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_vanilla_bce_02.txt" 0.00001 reweighted_vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_vanilla_bce_03.txt" 0.000001 reweighted_vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_vanilla_bce_04.txt" 0.0000001 reweighted_vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
$run_script "${txt_base}_reweighted_vanilla_bce_05.txt" 0.00000001 reweighted_vanilla_bce default_moving_dnn_listener_transformer_regression_architecture   15   $alignment_paths
