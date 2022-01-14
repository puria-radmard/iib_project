run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_unsupervised_daf_recalibration/run_train.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_unsupervised_daf_recalibration/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_unsupervised_daf_recalibration/round_history"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_unsupervised_daf_recalibration/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_unsupervised_daf_recalibration/round_history"
elif [ "$dataset" = "test_run" ]; then 
    run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_unsupervised_daf_recalibration/train.sh"
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_unsupervised_daf_recalibration/test_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_unsupervised_daf_recalibration/test_round_history"
    $run_path l2   0.01   0.001    3    $save_dir_base  cifar100
    exit 1
else
    echo "Bad dataset"
    exit 1    
fi;

# metric_function=$1
# minibatch_prop=$2
# initial_lr=$3
# num_initial_epochs=$4
# save_dir=$5
# dataset=$6

#                            metric_function  minibatch_prop   lr1   epochs_per_round

# Standard without reinit, adding 1% each time
$run_path ${txt_base}_01.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset
$run_path ${txt_base}_02.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset
$run_path ${txt_base}_03.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset
$run_path ${txt_base}_04.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset
$run_path ${txt_base}_05.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset
$run_path ${txt_base}_06.txt image_reconstruction   0.01      0.001    30       $save_dir_base  $dataset

# Standard without reinit, adding 0.1% each time
$run_path ${txt_base}_07.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
$run_path ${txt_base}_08.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
$run_path ${txt_base}_09.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
$run_path ${txt_base}_10.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
$run_path ${txt_base}_11.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
$run_path ${txt_base}_12.txt image_reconstruction   0.001     0.001   30        $save_dir_base  $dataset
