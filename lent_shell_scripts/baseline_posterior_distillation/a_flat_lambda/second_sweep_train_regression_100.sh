txt_base_tag=$1

run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/baseline_posterior_distillation/run_train_regression_100.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/second_baseline_posterior_distillation_100/log_${txt_base_tag}_"


# lr=$1
# training_mode=$2
# multitask_training_mode=$3
# distillation_temperature=$4
# multitask_acquisition_target_weight=$5
# byol_daf=$6

declare -a densenet_lrs=(0.01)
declare -a byol_lrs=(0.000001 0.00001 0.0001)
declare -a training_modes=("posterior_distillation_multitask" "posterior_distillation" "entropy_bce" "lc_histogram_jacobian_bce")
declare -a multitask_training_modes=( "lc_histogram_jacobian_bce" "entropy_bce" "None")
declare -a byol_dafs=("") # "--byol_daf")
declare -a distillation_temperatures=("2.5" "3.5")
declare -a multitask_acquisition_target_weights=("0.75" "0.85" "0.95")

declare -i i=1


for training_mode in "${training_modes[@]}"
do
    for mtm in "${multitask_training_modes[@]}"
    do
        for byol_daf in "${byol_dafs[@]}"
        do
            for dtemp in "${distillation_temperatures[@]}"
            do
                for matw in "${multitask_acquisition_target_weights[@]}"
                do
                    # Skip invalid combination
                    if [[ $training_mode != "posterior_distillation_multitask" && $mtm != "None" ]]
                    then
                        continue
                    fi

                    # Skip invalid combination
                    if [[ $training_mode == "posterior_distillation_multitask" && $mtm == "None" ]]
                    then
                        continue
                    fi

                    # For testing
                    if [[ $txt_base_tag == 'test' ]]
                    then
                        test_run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/baseline_posterior_distillation/train_regression_100.sh" 
                        echo $test_run_path $lr $training_mode $mtm $dtemp $matw $byol_daf
                        exit 1
                    fi
                    
                    # Select from the correct learning rates
                    if [[ $byol_daf == "--byol_daf" ]]
                    then
                        for lr in "${byol_lrs[@]}"
                        do
                            $run_path "${txt_base}_${i}_${byol_daf:2}.txt" $lr $training_mode $mtm $dtemp $matw $byol_daf
                            i=$((i+1))
                        done
                    else
                        for lr in "${densenet_lrs[@]}"
                        do
                            $run_path "${txt_base}_${i}_${byol_daf:2}.txt" $lr $training_mode $mtm $dtemp $matw $byol_daf
                            i=$((i+1))
                        done
                    fi
                done
            done
        done
    done
done

