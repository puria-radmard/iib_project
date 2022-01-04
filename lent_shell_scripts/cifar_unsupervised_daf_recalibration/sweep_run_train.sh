run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/cifar_unsupervised_daf_recalibration/run_train.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration/round_history"
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration_100/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration_100/round_history"
elif [ "$dataset" = "test_run" ]; then 
    run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/cifar_unsupervised_daf_recalibration/train_daf_recalibration_debug.sh"
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration/test_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar_unsupervised_daf_recalibration/test_round_history"
    $run_path l2   0.01         0.05          0.001  0.001   compressed_net    3      3      do_reinitialise_autoencoder_ensemble     $save_dir_base  cifar10
    exit 1
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

#                            metric_function  minibatch_prop total_start_prop   lr1    lr2  skip_option   epochs1  epochs2   reinit_option

# WITH SKIPS - NO AUG
# Standard without reinit, adding 1% each time
$run_path ${txt_base}_01.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_with_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_02.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_with_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_03.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_with_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset

# Standard without reinit, adding 0.1% each time
$run_path ${txt_base}_04.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_with_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_05.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_with_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_06.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_with_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset


# WITHOUT SKIPS - NO AUG
# Standard without reinit, adding 1% each time
$run_path ${txt_base}_07.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_without_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_08.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_without_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_09.txt image_reconstruction   0.01         0.05          0.001  0.001   unet_without_skips    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset

# Standard without reinit, adding 0.1% each time
$run_path ${txt_base}_10.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_without_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_11.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_without_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_12.txt image_reconstruction   0.002          0.05          0.001  0.001   unet_without_skips   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset


# STAIRCASE NET - NO AUG
# Standard without reinit, adding 1% each time
$run_path ${txt_base}_13.txt image_reconstruction   0.01         0.05          0.001  0.001   compressed_net    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_14.txt image_reconstruction   0.01         0.05          0.001  0.001   compressed_net    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_15.txt image_reconstruction   0.01         0.05          0.001  0.001   compressed_net    30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset

# Standard without reinit, adding 0.1% each time
$run_path ${txt_base}_16.txt image_reconstruction   0.002          0.05          0.001  0.001   compressed_net   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_17.txt image_reconstruction   0.002          0.05          0.001  0.001   compressed_net   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset
$run_path ${txt_base}_18.txt image_reconstruction   0.002          0.05          0.001  0.001   compressed_net   30      30      do_reinitialise_autoencoder_ensemble         $save_dir_base  $dataset

