run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/unsupervised_recalibration_scripts/run_train_unsupervised_recalibration.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/active_learning_sweep/gru_sweep"
utt_list_base="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list."

# run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/unsupervised_recalibration_scripts/train_unsupervised_recalibration.sh"
# utt_list_base="/home/alta/BLTSpeaking/active_learning-pr450/data/initial_sets/utt_list."
# $run_script  0.0000001   0.00000001  ${utt_list_base}labelled_b0     ${utt_list_base}unlabelled_b0  8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder


# initial_lr=$1
# finetune_lr=$2
# labelled_utt_list_path=$3
# unlabelled_utt_list_path=$4
# num_initial_epochs=$5
# embedding_dim=$6
# encoder_lstm_sizes=`echo "$7" | tr , " "`
# encoder_lstm_layers=`echo "$8" | tr , " "`
# encoder_fc_hidden_dims=`echo "$9" | tr , " "`
# mult_noise=${10}
# cell_type=${11}
# encoder_architecture=${12}
# decoder_architecture=${13}

$run_script  ${txt_base}_01.txt  0.0000001   0.00000001  ${utt_list_base}labelled_b0     ${utt_list_base}unlabelled_b0        8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_02.txt  0.000001    0.0000001   ${utt_list_base}labelled_b0     ${utt_list_base}labelled_BLXXXeval3  8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_03.txt  0.0000001   0.00000001  ${utt_list_base}labelled_b50    ${utt_list_base}unlabelled_b50       8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_04.txt  0.000001    0.0000001   ${utt_list_base}labelled_b50    ${utt_list_base}labelled_BLXXXeval3  8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_05.txt  0.0000001   0.000001    ${utt_list_base}labelled_b0     ${utt_list_base}unlabelled_b0        8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_06.txt  0.00001     0.0001      ${utt_list_base}labelled_b0     ${utt_list_base}labelled_BLXXXeval3  8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_07.txt  0.0000001   0.000001    ${utt_list_base}labelled_b50    ${utt_list_base}unlabelled_b50       8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_08.txt  0.00001     0.0001      ${utt_list_base}labelled_b50    ${utt_list_base}labelled_BLXXXeval3  8  40  256,156,256   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder

$run_script  ${txt_base}_09.txt  0.0000001   0.00000001  ${utt_list_base}labelled_b0     ${utt_list_base}unlabelled_b0        8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_10.txt  0.000001    0.0000001   ${utt_list_base}labelled_b0     ${utt_list_base}labelled_BLXXXeval3  8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_11.txt  0.0000001   0.00000001  ${utt_list_base}labelled_b50    ${utt_list_base}unlabelled_b50       8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_12.txt  0.000001    0.0000001   ${utt_list_base}labelled_b50    ${utt_list_base}labelled_BLXXXeval3  8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_13.txt  0.0000001   0.000001    ${utt_list_base}labelled_b0     ${utt_list_base}unlabelled_b0        8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_14.txt  0.00001     0.0001      ${utt_list_base}labelled_b0     ${utt_list_base}labelled_BLXXXeval3  8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_15.txt  0.0000001   0.000001    ${utt_list_base}labelled_b50    ${utt_list_base}unlabelled_b50       8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
$run_script  ${txt_base}_16.txt  0.00001     0.0001      ${utt_list_base}labelled_b50    ${utt_list_base}labelled_BLXXXeval3  8  40  156,256,156   3,3,3  -1  0.0    gru  only_bidirectional_LSTM  no_decoder
