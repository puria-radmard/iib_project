run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification_no_ami/run_train.sh"
save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/config"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/log"

labelled_list="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.labelled_b50"
unlabelled_list="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.unlabelled_b50"

# arch=$1
# lr=$2
# dropout=$3
# labelled_list=$4
# unlabelled_list=$5
# test_prop=$6
# save_dir=$7

# /home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification_no_ami/train.sh "default_blstm_listener_transformer_regression_architecture" 0.0001 0.0 /home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.labelled_b50 /home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.unlabelled_b50 "/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/config"

# $run_script ${txt_base}_01.txt "default_blstm_listener_self_attention_regression_architecture"      0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_02.txt "default_moving_dnn_listener_self_attention_regression_architecture" 0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_03.txt "default_tdnn_listener_self_attention_regression_architecture"       0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

# $run_script ${txt_base}_04.txt "default_blstm_listener_self_attention_regression_architecture"      0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_05.txt "default_moving_dnn_listener_self_attention_regression_architecture" 0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_06.txt "default_tdnn_listener_self_attention_regression_architecture"       0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

# $run_script ${txt_base}_07.txt "default_blstm_listener_self_attention_regression_architecture"      0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_08.txt "default_moving_dnn_listener_self_attention_regression_architecture" 0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_09.txt "default_tdnn_listener_self_attention_regression_architecture"       0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir


# $run_script ${txt_base}_10.txt "default_blstm_listener_transformer_regression_architecture"         0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_11.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_12.txt "default_tdnn_listener_transformer_regression_architecture"          0.0001      0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

# $run_script ${txt_base}_13.txt "default_blstm_listener_transformer_regression_architecture"         0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_14.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_15.txt "default_tdnn_listener_transformer_regression_architecture"          0.00001     0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

# $run_script ${txt_base}_16.txt "default_blstm_listener_transformer_regression_architecture"         0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_17.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_18.txt "default_tdnn_listener_transformer_regression_architecture"          0.000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

# $run_script ${txt_base}_19.txt "default_blstm_listener_transformer_regression_architecture"         0.0000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_20.txt "default_blstm_listener_transformer_regression_architecture"         0.0000005    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_21.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.0000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_22.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.0000005    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_23.txt "default_tdnn_listener_transformer_regression_architecture"          0.0000001    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
# $run_script ${txt_base}_24.txt "default_tdnn_listener_transformer_regression_architecture"          0.0000005    0.0     $labelled_list    $unlabelled_list   0.0    $save_dir

$run_script ${txt_base}_new_25.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.00002   0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
$run_script ${txt_base}_new_26.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.00005   0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
$run_script ${txt_base}_new_27.txt "default_moving_dnn_listener_transformer_regression_architecture"    0.00007   0.0     $labelled_list    $unlabelled_list   0.0    $save_dir
