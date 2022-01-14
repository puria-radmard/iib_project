txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/evaluation_log_"
config_path_base_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/listen_and_attend_classification/no_test_config"


# run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification/evaluate.sh"
# $run_path "${config_path_base_base}-0" "${config_path_base_base}-0/evaluations.pkl"
# exit 1

run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification/run_evaluate.sh"

$run_path "${txt_base}00.txt" "${config_path_base_base}-0" "${config_path_base_base}-0/evaluations.pkl"
$run_path "${txt_base}01.txt" "${config_path_base_base}-1" "${config_path_base_base}-1/evaluations.pkl"
$run_path "${txt_base}02.txt" "${config_path_base_base}-2" "${config_path_base_base}-2/evaluations.pkl"
$run_path "${txt_base}03.txt" "${config_path_base_base}-3" "${config_path_base_base}-3/evaluations.pkl"
$run_path "${txt_base}04.txt" "${config_path_base_base}-4" "${config_path_base_base}-4/evaluations.pkl"
$run_path "${txt_base}05.txt" "${config_path_base_base}-5" "${config_path_base_base}-5/evaluations.pkl"
$run_path "${txt_base}06.txt" "${config_path_base_base}-6" "${config_path_base_base}-6/evaluations.pkl"
$run_path "${txt_base}07.txt" "${config_path_base_base}-7" "${config_path_base_base}-7/evaluations.pkl"
$run_path "${txt_base}08.txt" "${config_path_base_base}-8" "${config_path_base_base}-8/evaluations.pkl"
$run_path "${txt_base}09.txt" "${config_path_base_base}-9" "${config_path_base_base}-9/evaluations.pkl"
$run_path "${txt_base}10.txt" "${config_path_base_base}-10" "${config_path_base_base}-10/evaluations.pkl"
$run_path "${txt_base}11.txt" "${config_path_base_base}-11" "${config_path_base_base}-11/evaluations.pkl"
