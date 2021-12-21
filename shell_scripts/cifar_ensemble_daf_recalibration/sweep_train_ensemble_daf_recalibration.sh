run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/cifar_ensemble_daf_recalibration/run_train_ensemble_daf_recalibration.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/cifar_ensemble_daf_recalibration/daf_recalibration_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/cifar_ensemble_daf_recalibration/round_history"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/cifar_ensemble_daf_recalibration_100/daf_recalibration_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/cifar_ensemble_daf_recalibration_100/round_history"
else
    echo "Bad dataset"
    exit 1
fi;

# metric_function
# minibatch_prop
# total_start_prop
# initial_lr
# finetune_lr
# mult_noise
# num_initial_epochs
# num_finetune_epochs
# reinit_option
# save_dir

#                      ensemble_size metric_function  minibatch_prop total_start_prop   lr1    lr2  noise_mult   epochs1  epochs2   reinit_option ensemble_size

## All repeated twice

# Standard without reinit, adding 1% each time
$run_path ${txt_base}_01.txt  l1   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_02.txt  l2   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_03.txt  l3   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_04.txt  l1   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_05.txt  l2   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_06.txt  l3   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_07.txt  l1   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_08.txt  l2   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_09.txt  l3   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_10.txt  l1   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_11.txt  l2   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
$run_path ${txt_base}_12.txt  l3   0.01         0.05          0.001  0.001   0            50      50      5   do_reinitialise_autoencoder_ensemble    $save_dir_base  $dataset
