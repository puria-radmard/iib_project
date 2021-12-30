run_script="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification/run_train.sh"
save_dir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/config"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/log"

labelled_list="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.labelled_b50"
unlabelled_list="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.unlabelled_b50"

# arch=$1
# lr=$2
# decoder_dropout=$3
# labelled_list=$4
# unlabelled_list=$5
# save_dir=$6

$run_script ${txt_base}_01.txt  0.0001      0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_02.txt  0.00001     0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_03.txt  0.000001    0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_04.txt  0.0001      0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_05.txt  0.00001     0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_06.txt  0.000001    0.0     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_07.txt  0.0001      0.2     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_08.txt  0.00001     0.2     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_09.txt  0.000001    0.2     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_10.txt  0.0001      0.2     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_11.txt  0.00001     0.2     $labelled_list    $unlabelled_list    $save_dir
$run_script ${txt_base}_12.txt  0.000001    0.2     $labelled_list    $unlabelled_list    $save_dir