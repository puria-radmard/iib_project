run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/audio_gru_daf_recalibration/run_train_daf_recalibration.sh"

txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/audio_gru_daf_recalibration/daf_recalibration_log"
save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/audio_gru_daf_recalibration/round_history"


labelled_b50="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.labelled_b50"
unlabelled_b50="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list.unlabelled_b50"

if [ "$1" = "test_run" ]; then
    run_path="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/audio_gru_daf_recalibration/train_daf_recalibration.sh"
    txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/audio_gru_daf_recalibration/test_daf_recalibration_log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/logs/audio_gru_daf_recalibration/test_round_history"
    $run_path image_reconstruction 7200 $labelled_b50 $unlabelled_b50  0.001  0.001  unet_with_skips   0  0  do_reinitialise_autoencoder_ensemble  $save_dir_base
    exit 1
fi;

#                             metric_function   minibatch_seconds  labelled_utt_list_path   unlabelled_utt_list_path   initial_lr  finetune_lr  skip_option  num_initial_epochs    num_finetune_epochs              reinit_option            save_dir
# With skips, adding 2 hours each time
$run_path ${txt_base}_01.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_02.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_03.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base

# With skips, adding 1 hour each time
$run_path ${txt_base}_04.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_05.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_06.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_with_skips         1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base

# Without skips, adding 2 hours each time
$run_path ${txt_base}_07.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_08.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_09.txt image_reconstruction   7200          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base

# Without skips, adding 1 hour each time
$run_path ${txt_base}_10.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_11.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
$run_path ${txt_base}_12.txt image_reconstruction   3600          $labelled_b50             $unlabelled_b50              0.001        0.001     unet_without_skips      1                      1      do_reinitialise_autoencoder_ensemble         $save_dir_base
