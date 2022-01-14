queuer_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/general_daf_active_learning/queue.sh"

dataset="cifar100"
DAFpathbase="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_recalibration/round_history"
mirroredPath="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_densenet_al_baseline/round_history-1"
saveDir="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_labelled_classification_daf_active_learning/round_history"

$queuer_path $dataset "${DAFpathbase}-1" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-2" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-3" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-4" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-5" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-6" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-7" $mirroredPath $saveDir
$queuer_path $dataset "${DAFpathbase}-8" $mirroredPath $saveDir
