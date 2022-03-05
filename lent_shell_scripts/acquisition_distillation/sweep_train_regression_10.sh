txt_base_tag=$1

run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/acquisition_distillation/run_train_regression_10.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_10/lc_bce_distillation_log_${txt_base_tag}"

# This one was trained with 15001 images
logit_bank_path="/home/alta/BLTSpeaking/exp-pr450/lent_logs/acquisition_distillation_10/trained_logits-2/evaluations.pkl"

# "entropy_log_mse" "entropy_mse" "entropy_bce" "lc_bce" "lc_parametric_jacobian_bce" "lc_histogram_jacobian_bce"
# "no_un" "online_un"

declare -a training_modes=("entropy_log_mse" "entropy_mse" "entropy_bce" "lc_bce" "lc_parametric_jacobian_bce" "lc_histogram_jacobian_bce")
declare -a uns=("no_un" "online_un")
declare -a test_props=(0.3 0.5 0.7)
declare -a layers=(70 88 130)

declare -i i=1

for training_mode in "${training_modes[@]}"
do
    for un in "${uns[@]}"
    do
        for test_prop in "${test_props[@]}"
        do
            for layer in "${layers[@]}"
            do

                # MSE cannot access objective uniformalisation
                if [[ $training_mode == *"mse"* ]] && [[ $un == "online_un" ]]
                then
                    continue
                fi

                echo $run_path "${txt_base}${i}.txt" $layer  $logit_bank_path "${training_mode}" "${un}" ${test_prop}  50
                i=$((i+1))
            done
        done
    done
done
