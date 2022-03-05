run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/acquisition_distillation/train_regression_100.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_test/lc_bce_distillation_TEST_log_"
logit_bank_path="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_100/trained_logits-2/evaluations.pkl"

# "entropy_log_mse" "entropy_mse" "entropy_bce" "lc_bce" "lc_parametric_jacobian_bce" "lc_histogram_jacobian_bce"
training_mode=$1

# "no_un" "online_un"
un_option=$2

$run_path 70 $logit_bank_path "${training_mode}" "${un_option}" 0.7 3

exit 1

# TESTED:

# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_log_mse no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_log_mse online_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_mse no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_mse online_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_bce no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh entropy_bce online_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_bce  no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_bce  online_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_parametric_jacobian_bce  no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_parametric_jacobian_bce  online_un
# ./ lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_histogram_jacobian_bce   no_un
# ./lent_shell_scripts/acquisition_distillation/sweep_train_regression_test.sh lc_histogram_jacobian_bce   online_un
