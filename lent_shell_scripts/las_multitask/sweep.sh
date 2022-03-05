run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/las_multitask/run_train.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/las_multitask/log_${txt_base_tag}_"

alignment_path="/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"

declare -a criterions=('lc_histogram_jacobian_bce', 'lc_soft_rank', 'bag_of_phonemes_multitask', 'phoneme2vec_multitask')
declare -a multitask_training_modes=('lc_histogram_jacobian_bce', 'lc_soft_rank')

# Slots into "default_{}_listener_transformer_regression_architecture"
declare -a listener_names=("blstm" "moving_dnn" "convolutional")

# Try only the most basic cases for now (complete distillation and multitasking)
declare -a lambda_scheduler_option_and_parameters=(
    "\"--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 0.75\""
    "\"--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 0.0\""
)

declare -i i=0

# MAKE listener_names THE TOP ITERATOR SO WE CAN DO CONVOLUTION AFTER, SEPERATELY
for listener_type in "${listener_names[@]}"
do
    for training_mode in "${criterions[@]}"
    do  
        # Multitask requires giving multitask option lambda settings
        if [[ $training_mode == *"multitask"* ]]
        then
            for multitask_training_mode in "${multitask_training_modes[@]}"
            do
                for lambda_scheduler_option_and_parameter in "${lambda_scheduler_option_and_parameters[@]}"
                do
                    i=$((i+1))
                    
                    echo $run_path "$txt_base${i}" $listener_type $training_mode $multitask_training_mode 0.00001 $lambda_scheduler_option_and_parameter $alignment_paths
                done
            done
        else
            # Not multitask - can leave multitask option and lambda scheduling empty
            i=$((i+1))
            
            echo $run_path $training_mode "None"  0.00001 "\"--lambda_scheduler_option no_lambda_scheduling --multitask_acquisition_target_weight 0.0\"" $alignment_paths
        fi
    done
done
