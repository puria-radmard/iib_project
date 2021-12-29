run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/uncertainty_audio_regression_scripts/run_train_uncertainty_regression.sh"
txt_base="/home/alta/BLTSpeaking/exp-pr450/logs/uncertainty_regression/regression_sweep"
alignment_base="/home/alta/BLTSpeaking/active_learning-pr450/models/baseline/CTDF1_b50/tdnn-f"
alignment_paths="${alignment_base}/decode_LM1-int_b50.BLXXXeval3/score_mbr_10/BLXXXeval3.utt.map.ctm,${alignment_base}/decode_LM1-int_b50.unlabelled_b50/score_mbr_10/unlabelled_b50.utt.map.ctm"

# lr=$1
# encoder_lstm_sizes=`echo "$2" | tr , " "`
# encoder_lstm_layers=`echo "$3" | tr , " "`
# embedding_dim=$4
# decoder_fc_hidden_dims=`echo "$5" | tr , " "`
# num_epochs=$6
# cell_type=$7
# alignment_paths=`echo "$8" | tr , " "`

#run_script="/home/alta/BLTSpeaking/exp-pr450/shell_scripts/uncertainty_audio_regression_scripts/train_uncertainty_regression.sh"
#txt_base=

$run_script ${txt_base}_01.txt 0.0001 256,128 3,3 128 128,16,1 10 gru $alignment_paths
$run_script ${txt_base}_02.txt 0.00001 256,128 3,3 128 128,16,1 10 gru $alignment_paths
$run_script ${txt_base}_03.txt 0.000001 256,128 3,3 128 128,16,1 10 gru $alignment_paths
$run_script ${txt_base}_04.txt 0.0001 256,128 3,3 128 128,64,16,1 10 gru $alignment_paths
$run_script ${txt_base}_05.txt 0.00001 256,128 3,3 128 128,64,16,1 10 gru $alignment_paths
$run_script ${txt_base}_06.txt 0.000001 256,128 3,3 128 128,64,16,1 10 gru $alignment_paths
