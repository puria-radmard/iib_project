config_path_base_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/config"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/eval_log"

# run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification_no_ami/evaluate.sh"
# $run_path "${config_path_base_base}-0" "${config_path_base_base}-0/evaluations.pkl"
# exit 1

run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification_no_ami/run_evaluate.sh"

# config_path_base_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/config"
# txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/generalised_listen_and_attend_classification_no_ami/eval_log"
# run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/listen_and_attend_classification_no_ami/evaluate.sh"
# $run_path "${config_path_base_base}-0" "${config_path_base_base}-0/evaluations.pkl"

# These ones seems to be the stars for moving DNN!
#$run_path "${txt_base}_05.txt" "${config_path_base_base}-4" "${config_path_base_base}-4/evaluations.pkl"
#$run_path "${txt_base}_11.txt" "${config_path_base_base}-10" "${config_path_base_base}-10/evaluations.pkl"
#$run_path "${txt_base}_14.txt" "${config_path_base_base}-15" "${config_path_base_base}-15/evaluations.pkl"

$run_path "${txt_base}_25.txt" "${config_path_base_base}-25" "${config_path_base_base}-25/evaluations.pkl"
$run_path "${txt_base}_26.txt" "${config_path_base_base}-26" "${config_path_base_base}-26/evaluations.pkl"
$run_path "${txt_base}_27.txt" "${config_path_base_base}-24" "${config_path_base_base}-24/evaluations.pkl"
