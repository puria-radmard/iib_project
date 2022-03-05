txt_base_tag=$1

run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/baseline_posterior_distillation/b_lambda_scheduling/run_train_regression.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/third_baseline_posterior_distillation_10/log_${txt_base_tag}_"


dataset="cifar10"

declare -i i=0

declare -a training_modes=("entropy_soft_rank" "entropy_histogram_jacobian_bce" "posterior_distillation_multitask")
declare -a multitask_training_modes=("entropy_soft_rank" "entropy_histogram_jacobian_bce")

## Lambda options
declare -a lambda_scheduler_option_and_parameters=(
    # Flat lambda - include 0.0 in here for pure distillation, rather than set off a whole new set of experiments
    # 1.0 would be very wasteful though
    "--lambda_scheduler_option,no_lambda_scheduling,--multitask_acquisition_target_weight,0.75"
    "--lambda_scheduler_option,no_lambda_scheduling,--multitask_acquisition_target_weight,0.0"
    # Step lambda - just try for switching after 15 epochs for now - justified in report
    "--lambda_scheduler_option,step_pretrainer_lambda_scheduling,--multitask_acquisition_target_weight,15"
    # Exponential tempering towards acquisition - resembles switch at epoch 15
    "--lambda_scheduler_option,asymptotic_pretrainer_lambda_scheduling,--multitask_acquisition_target_weight,5"
)


for training_mode in "${training_modes[@]}"
do  
    # Multitask requires giving multitask option lambda settings
    if [[ $training_mode == *"multitask"* ]]
    then
        for multitask_training_mode in "${multitask_training_modes[@]}"
        do
            for lambda_scheduler_option_and_parameter in "${lambda_scheduler_option_and_parameters[@]}"
            do
                i=$((i+1))
                $run_path "$txt_base${i}.txt" $training_mode $multitask_training_mode $lambda_scheduler_option_and_parameter $dataset
            done
        done
    else
        # Not multitask - can leave multitask option and lambda scheduling empty
        i=$((i+1))
        $run_path "$txt_base${i}.txt" $training_mode "None" "--lambda_scheduler_option,no_lambda_scheduling,--multitask_acquisition_target_weight,1.0" $dataset
    fi
done
