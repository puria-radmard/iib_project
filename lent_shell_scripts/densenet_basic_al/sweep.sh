run_path="/home/alta/BLTSpeaking/exp-pr450/lent_shell_scripts/densenet_basic_al/run_train.sh"

dataset=$1

if [ "$dataset" = "cifar10" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_densenet_al_baseline/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar10_densenet_al_baseline/round_history"
    epochs=100
elif [ "$dataset" = "cifar100" ]; then 
    txt_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_densenet_al_baseline/log"
    save_dir_base="/home/alta/BLTSpeaking/exp-pr450/lent_logs/cifar100_densenet_al_baseline/round_history"
    epochs=300
else
    echo "Bad dataset"
    exit 1
fi;

# initProp=$1
# roundProp=$2
# totalBudgetProp=$3
# acquisitionFunction=$4
# saveDir=$5
# dataset=$6

# Repeat 1
$run_path ${txt_base}_01.txt 0.05 0.05 0.6 rand $save_dir_base $epochs $dataset
$run_path ${txt_base}_02.txt 0.05 0.05 0.6 rand $save_dir_base $epochs $dataset
$run_path ${txt_base}_03.txt 0.05 0.05 0.6 rand $save_dir_base $epochs $dataset

$run_path ${txt_base}_04.txt 0.05 0.15 0.6 rand $save_dir_base $epochs $dataset
$run_path ${txt_base}_05.txt 0.05 0.15 0.6 rand $save_dir_base $epochs $dataset
$run_path ${txt_base}_06.txt 0.05 0.15 0.6 rand $save_dir_base $epochs $dataset

$run_path ${txt_base}_07.txt 0.05 0.05 0.6 maxent $save_dir_base $epochs $dataset
$run_path ${txt_base}_08.txt 0.05 0.05 0.6 maxent $save_dir_base $epochs $dataset
$run_path ${txt_base}_09.txt 0.05 0.05 0.6 maxent $save_dir_base $epochs $dataset

$run_path ${txt_base}_10.txt 0.05 0.15 0.6 maxent $save_dir_base $epochs $dataset
$run_path ${txt_base}_11.txt 0.05 0.15 0.6 maxent $save_dir_base $epochs $dataset
$run_path ${txt_base}_12.txt 0.05 0.15 0.6 maxent $save_dir_base $epochs $dataset
